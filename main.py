import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataPreprocessor import DataPreprocessor
from hybridModelBuilder import HybridModelBuilder, NumpyDataset, SessionDataset
from modelTrainer import ModelTrainer


LOGGER = logging.getLogger("ysaForB2b")


def _ensure_artifacts(output_dir: Path, data_path: str) -> Tuple[Path, Path]:
	metadata_path = output_dir / "processed" / "metadata.json"
	labels_path = output_dir / "processed" / "labels.json"

	if not metadata_path.exists() or not labels_path.exists():
		LOGGER.info("Veri hazirlaniyor: %s", data_path)
		preprocessor = DataPreprocessor(
			data_path=data_path,
			truncation="pre",
			padding="post",
			time_shift=True,
			time_shift_mode="all",
		)
		preprocessor.save_artifacts(str(output_dir))
	else:
		LOGGER.info("Var olan artifact'lar kullaniliyor: %s", output_dir)

	return metadata_path, labels_path


def _split_inputs(
	inputs: Dict[str, np.ndarray],
	labels: np.ndarray,
	val_ratio: float = 0.2,
	test_ratio: float = 0.1,
	seed: int = 42,
) -> Tuple[
	Tuple[Dict[str, np.ndarray], np.ndarray],
	Tuple[Dict[str, np.ndarray], np.ndarray],
	Tuple[Dict[str, np.ndarray], np.ndarray],
]:
	if val_ratio + test_ratio >= 1.0:
		raise ValueError("val_ratio + test_ratio 1.0'dan kucuk olmali.")

	rng = np.random.default_rng(seed)
	idx = np.arange(labels.shape[0])
	rng.shuffle(idx)

	train_end = int((1.0 - val_ratio - test_ratio) * len(idx))
	val_end   = int((1.0 - test_ratio) * len(idx))

	train_idx = idx[:train_end]
	val_idx   = idx[train_end:val_end]
	test_idx  = idx[val_end:]

	return (
		({k: v[train_idx] for k, v in inputs.items()}, labels[train_idx]),
		({k: v[val_idx]   for k, v in inputs.items()}, labels[val_idx]),
		({k: v[test_idx]  for k, v in inputs.items()}, labels[test_idx]),
	)


def _make_weighted_sampler(labels: np.ndarray, n_classes: int) -> WeightedRandomSampler:
	"""Her sinifin secilme olasiligini esitler (beta=0.99 class-balanced)."""
	counts = np.bincount(labels, minlength=n_classes).astype(float)
	counts = np.where(counts == 0, 1, counts)
	# Etkili sayi: (1 - beta^n) / (1 - beta)
	beta = 0.99
	eff  = (1.0 - beta ** counts) / (1.0 - beta)
	class_w = 1.0 / eff
	sample_w = torch.tensor([class_w[l] for l in labels], dtype=torch.float)
	return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


def _to_dataloader(
	inputs: Dict[str, np.ndarray],
	labels: np.ndarray,
	batch_size: int,
	shuffle: bool,
	sampler=None,
) -> DataLoader:
	pin = torch.cuda.is_available()
	return DataLoader(
		NumpyDataset(inputs, labels),
		batch_size=batch_size,
		shuffle=(shuffle and sampler is None),
		sampler=sampler,
		num_workers=0,
		pin_memory=pin,
		drop_last=shuffle,   # egitimde son eksik batch'i at
	)


def _log_label_stats(labels: np.ndarray, top_k: int = 10) -> None:
	values, counts = np.unique(labels, return_counts=True)
	order = np.argsort(counts)[::-1]
	LOGGER.info(
		"Top-%d label dagilimi: %s", top_k,
		dict(zip(values[order][:top_k].tolist(), counts[order][:top_k].tolist())),
	)


def _compute_class_weights(labels: np.ndarray, n_classes: int) -> Dict[int, float]:
	counts = np.bincount(labels, minlength=n_classes)
	valid  = np.where(counts > 0)[0]
	if len(valid) == 0:
		raise ValueError("Sinif sayimi bos.")
	total   = counts[valid].sum()
	return {int(c): float(total / (len(valid) * counts[c])) for c in valid}


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)

	data_path  = "n3_sessions_model_ready.json"
	output_dir = Path("output")
	_ensure_artifacts(output_dir, data_path)

	dataset = SessionDataset(
		artifacts_dir=str(output_dir),
		labels_path=str(output_dir / "processed" / "labels.json"),
		label_dtype=np.int64,
	)

	if dataset.labels is None:
		raise ValueError("labels.json bulunamadi veya yuklenemedi.")

	LOGGER.info("Toplam ornek sayisi: %d", len(dataset))
	LOGGER.info("Max session length: %s", dataset.metadata.get("maxSessionLength"))
	LOGGER.info("Time shift mode: %s", dataset.metadata.get("timeShiftMode"))
	_log_label_stats(dataset.labels)

	(
		(train_inputs, train_labels),
		(val_inputs,   val_labels),
		(test_inputs,  test_labels),
	) = _split_inputs(dataset.inputs, dataset.labels, val_ratio=0.2, test_ratio=0.1)

	LOGGER.info(
		"Train/Val/Test ornek sayilari: %d / %d / %d",
		len(train_labels), len(val_labels), len(test_labels),
	)

	event_vocab_size = int(dataset.metadata["categoricalVocabs"]["event_type"])
	class_weights    = _compute_class_weights(train_labels, event_vocab_size)
	LOGGER.info("Class weight sayisi: %d", len(class_weights))

	# WeightedRandomSampler: nadir siniflar daha sik secilir
	sampler  = _make_weighted_sampler(train_labels, event_vocab_size)
	train_dl = _to_dataloader(train_inputs, train_labels, batch_size=32, shuffle=False, sampler=sampler)
	val_dl   = _to_dataloader(val_inputs,   val_labels,   batch_size=32, shuffle=False)
	test_dl  = _to_dataloader(test_inputs,  test_labels,  batch_size=32, shuffle=False)

	config = {
		"task":          "multiclass",
		"n_classes":     event_vocab_size,
		"lstm_units":    128,
		"dropout_rate":  0.3,
		"learning_rate": 1e-2,
	}

	LOGGER.info("Model mimarisi kuruluyor...")
	builder      = HybridModelBuilder(config=config)
	hybrid_model = builder.build(dataset.metadata)
	total_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
	LOGGER.info("Toplam egitilecek parametre: %d", total_params)

	optimizer = torch.optim.AdamW(
		hybrid_model.parameters(), lr=1e-2, weight_decay=5e-4
	)

	trainer = ModelTrainer(
		hybrid_model=hybrid_model,
		output_directory=str(output_dir / "models"),
		optimizer=optimizer,
		patience=18,
		label_smoothing=0.1,
		gradient_accumulation_steps=4,
		grad_clip=1.0,
		rollback_acc_drop=0.03,
		rollback_loss_rise=0.08,
		max_rollbacks=10,
		rollback_cooldown=5,
		use_swa=True,
		swa_start_ratio=0.7,
	)

	trainer.train(
		training_dataset=train_dl,
		validation_dataset=val_dl,
		total_epochs=50,
		class_weight=class_weights,
	)

	LOGGER.info("Test degerlendirmesi basliyor...")
	results = trainer.evaluate(test_dl)
	LOGGER.info("Test metrikleri: loss=%.4f accuracy=%.4f", results["loss"], results["accuracy"])


if __name__ == "__main__":
	main()
