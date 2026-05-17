import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger("ysaForB2b")


class ModelTrainer:
    """
    PyTorch egitim dongusu.

    VGG-MRI repo'sundan alinan teknikler:
      - AdamW + weight_decay
      - OneCycleLR (varsayilan) veya ReduceLROnPlateau
      - Label smoothing
      - Gradient accumulation
      - AMP (mixed precision)
      - Gradient clipping
      - Rollback mekanizmasi
      - SWA (Stochastic Weight Averaging)
      - GPU cache temizleme + gc her epoch sonu
    """

    def __init__(
        self,
        hybrid_model: nn.Module,
        output_directory: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monitor: str = "val_loss",
        patience: int = 18,
        min_delta: float = 0.0,
        label_smoothing: float = 0.1,
        gradient_accumulation_steps: int = 4,
        grad_clip: float = 1.0,
        rollback_acc_drop: float = 0.03,
        rollback_loss_rise: float = 0.08,
        max_rollbacks: int = 10,
        rollback_cooldown: int = 5,
        use_swa: bool = True,
        swa_start_ratio: float = 0.7,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.2,
        lr_patience: int = 3,
        min_lr: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        self.model = hybrid_model
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        LOGGER.info("Model cihaza tasindi: %s", self.device)

        self.optimizer = optimizer
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.label_smoothing = label_smoothing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.rollback_acc_drop = rollback_acc_drop
        self.rollback_loss_rise = rollback_loss_rise
        self.max_rollbacks = max_rollbacks
        self.rollback_cooldown = rollback_cooldown
        self.use_swa = use_swa
        self.swa_start_ratio = swa_start_ratio
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        self._best_weights_path = self.output_directory / "best_model_weights.pt"
        self._swa_weights_path  = self.output_directory / "swa_model_weights.pt"
        self.history: Dict[str, List[float]] = {}

        # AMP: sadece CUDA'da anlamlı
        self._use_amp = self.device == "cuda"
        self._scaler  = GradScaler("cuda", enabled=self._use_amp)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _move_batch(self, batch):
        inputs, labels = batch
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(self.device, non_blocking=True)
        return inputs, labels

    def _run_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        training: bool,
        class_weights: Optional[Dict[int, float]] = None,
        scheduler=None,
    ):
        self.model.train(training)
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        accum_steps   = self.gradient_accumulation_steps if training else 1

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            inputs, labels = self._move_batch(batch)

            with torch.autocast(device_type=self.device if self.device != "cpu" else "cpu",
                                 dtype=torch.float16 if self._use_amp else torch.float32,
                                 enabled=self._use_amp):
                outputs          = self.model(inputs)
                per_sample_loss  = loss_fn(outputs, labels)

                if class_weights is not None:
                    w = torch.tensor(
                        [class_weights.get(int(l), 1.0) for l in labels],
                        dtype=torch.float32, device=self.device,
                    )
                    loss = (per_sample_loss * w).mean() / accum_steps
                else:
                    loss = per_sample_loss.mean() / accum_steps

            if training:
                self._scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    # OneCycleLR adımı batch bazında atılır
                    if scheduler is not None and isinstance(
                        scheduler, torch.optim.lr_scheduler.OneCycleLR
                    ):
                        scheduler.step()

            with torch.no_grad():
                real_loss = (per_sample_loss.mean() if class_weights is None
                             else (per_sample_loss * torch.tensor(
                                 [class_weights.get(int(l), 1.0) for l in labels],
                                 dtype=torch.float32, device=self.device)).mean())
                total_loss    += real_loss.item() * labels.size(0)
                preds          = outputs.argmax(dim=-1) if outputs.dim() > 1 else (outputs > 0.5).long()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ------------------------------------------------------------------
    # Ana egitim dongusu
    # ------------------------------------------------------------------

    def train(
        self,
        training_dataset: DataLoader,
        validation_dataset: Optional[DataLoader] = None,
        total_epochs: int = 20,
        class_weight: Optional[Dict[int, float]] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler=None,
    ) -> Dict[str, List[float]]:

        task = getattr(self.model, "task", "multiclass")

        if loss_fn is None:
            if task == "binary":
                loss_fn = nn.BCELoss(reduction="none")
            elif task == "multiclass":
                loss_fn = nn.CrossEntropyLoss(
                    reduction="none", label_smoothing=self.label_smoothing
                )
            else:
                loss_fn = nn.MSELoss(reduction="none")

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-2, weight_decay=5e-4
            )

        # Scheduler kurulumu
        if scheduler is None:
            steps_per_epoch = len(training_dataset)
            if self.reduce_lr_on_plateau:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.min_lr,
                )
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=1e-2,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=1e4,
                )

        # SWA kurulumu
        swa_model     = None
        swa_scheduler = None
        swa_start     = int(total_epochs * self.swa_start_ratio)
        if self.use_swa and self.device == "cuda":
            swa_model     = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4, anneal_epochs=5)
            LOGGER.info("SWA aktif — epoch %d'den itibaren baslar.", swa_start + 1)

        best_metric   = float("inf")
        best_epoch    = 0
        no_improve    = 0
        rollback_count   = 0
        rollback_cooldown_left = 0
        prev_val_loss = float("inf")
        prev_val_acc  = 0.0

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        LOGGER.info(
            "Egitim basliyor. Cihaz: %s | Epoch: %d | AMP: %s | GradAccum: %d | LabelSmoothing: %.2f",
            self.device, total_epochs, self._use_amp, self.gradient_accumulation_steps, self.label_smoothing,
        )

        for epoch in range(1, total_epochs + 1):
            t0 = time.time()

            # SWA aşamasında scheduler geçici olarak None olarak geçirilir
            active_scheduler = scheduler if swa_model is None or epoch <= swa_start else None

            train_loss, train_acc = self._run_epoch(
                training_dataset, loss_fn, training=True,
                class_weights=class_weight, scheduler=active_scheduler,
            )

            val_loss, val_acc = None, None
            if validation_dataset is not None:
                val_loss, val_acc = self._run_epoch(
                    validation_dataset, loss_fn, training=False
                )

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            val_str = (f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                       if val_loss is not None else "")
            lr_now = self.optimizer.param_groups[0]["lr"]
            LOGGER.info(
                "Epoch %d/%d | %.1fs | lr=%.2e | train_loss=%.4f train_acc=%.4f%s",
                epoch, total_epochs, elapsed, lr_now, train_loss, train_acc, val_str,
            )

            # ReduceLROnPlateau epoch bazında adım atar
            monitor_value = val_loss if val_loss is not None else train_loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_value)

            # SWA güncelle
            if swa_model is not None and epoch > swa_start:
                swa_model.update_parameters(self.model)
                if swa_scheduler is not None:
                    swa_scheduler.step()

            # --- Rollback mekanizması ---
            if (val_loss is not None and rollback_cooldown_left == 0
                    and rollback_count < self.max_rollbacks):
                acc_drop   = prev_val_acc - (val_acc or 0.0)
                loss_rise  = (val_loss - prev_val_loss) / (prev_val_loss + 1e-9)
                if acc_drop >= self.rollback_acc_drop or loss_rise >= self.rollback_loss_rise:
                    self.load_best_weights()
                    rollback_count         += 1
                    rollback_cooldown_left  = self.rollback_cooldown
                    LOGGER.warning(
                        "ROLLBACK #%d: acc_drop=%.3f loss_rise=%.3f — en iyi agirliklar geri yuklendi.",
                        rollback_count, acc_drop, loss_rise,
                    )

            if rollback_cooldown_left > 0:
                rollback_cooldown_left -= 1

            if val_loss is not None:
                prev_val_loss = val_loss
                prev_val_acc  = val_acc or 0.0

            # --- Early stopping & checkpoint ---
            if monitor_value < best_metric - self.min_delta:
                best_metric = monitor_value
                best_epoch  = epoch
                no_improve  = 0
                torch.save(self.model.state_dict(), self._best_weights_path)
                LOGGER.info(
                    "En iyi model kaydedildi (epoch %d, %s=%.4f)",
                    epoch, self.monitor, best_metric,
                )
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    LOGGER.info(
                        "Early stopping: %d epoch iyilesme yok. En iyi: epoch %d",
                        self.patience, best_epoch,
                    )
                    break

            # --- Bellek temizle ---
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # --- SWA batch norm güncelle ve kaydet ---
        if swa_model is not None:
            LOGGER.info("SWA BatchNorm guncelleniyor...")
            update_bn(training_dataset, swa_model, device=self.device)
            torch.save(swa_model.state_dict(), self._swa_weights_path)
            LOGGER.info("SWA agirliklari kaydedildi: %s", self._swa_weights_path)

        self.load_best_weights()
        return self.history

    # ------------------------------------------------------------------
    # Değerlendirme ve yardımcılar
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_dataset: DataLoader,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        task = getattr(self.model, "task", "multiclass")
        if loss_fn is None:
            if task == "binary":
                loss_fn = nn.BCELoss(reduction="none")
            elif task == "multiclass":
                loss_fn = nn.CrossEntropyLoss(reduction="none")
            else:
                loss_fn = nn.MSELoss(reduction="none")

        loss, acc = self._run_epoch(test_dataset, loss_fn, training=False)
        return {"loss": loss, "accuracy": acc}

    def load_best_weights(self) -> None:
        if self._best_weights_path.exists():
            self.model.load_state_dict(
                torch.load(self._best_weights_path, map_location=self.device, weights_only=True)
            )
            LOGGER.info("En iyi agirliklar yuklendi: %s", self._best_weights_path)

    def plot_history(self) -> None:
        import matplotlib.pyplot as plt
        keys = [k for k in self.history if not k.startswith("val_")]
        for key in keys:
            plt.figure()
            plt.plot(self.history.get(key, []), label=key)
            plt.plot(self.history.get(f"val_{key}", []), label=f"val_{key}")
            plt.title(key)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.legend()
            plt.tight_layout()
            plt.show()
