"""
HybridModelBuilder — ysaForB2b icin PyTorch tabanli coklu girdi modeli.

Mimari:
  - Kategorik alanlar → Embedding → concat
  - Tag dizileri     → Linear projeksiyon (timestep basi)
  - BERT embeddingler → Linear projeksiyon (timestep basi)
  - Hepsi            → PackedLSTM → son gizli durum
  - Oturum ozellikleri → Dense + BatchNorm
  - LSTM cikisi + statik → Fusion Dense → gorev basi cikis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_CONFIG: Dict[str, object] = {
    "lstm_units": 128,
    "bert_proj_dim": 128,
    "tag_proj_dim": 32,
    "static_dim": 64,
    "fusion_dim": 128,
    "dropout_rate": 0.3,
    "task": "multiclass",
    "n_classes": 2,
    "learning_rate": 1e-3,
    "bidirectional": False,
}


# ---------------------------------------------------------------------------
# Veri yukleyici
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
    """
    JSON artifact'larini okuyan PyTorch Dataset.
    Artifact dizin yapisi DataPreprocessor.save_artifacts() ile uyumludur.
    """

    def __init__(
        self,
        artifacts_dir: str,
        labels_path: Optional[str] = None,
        label_dtype=np.int64,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.processed_dir = self.artifacts_dir / "processed"

        self.metadata = self._load_json(self.processed_dir / "metadata.json")
        # Buyuk array'ler numpy formatinda: Python list overhead yok
        self._event_seq_npz   = np.load(str(self.processed_dir / "event_sequences.npz"))
        self._tag_seq_mmap    = np.load(str(self.processed_dir / "tag_sequences.npy"), mmap_mode="r")
        self._bert_mmap       = np.load(str(self.processed_dir / "message_embedding_sequences.npy"), mmap_mode="r")
        self.session_features_raw = self._load_json(
            self.processed_dir / "session_features.json"
        )
        self.session_feature_keys = self._infer_session_feature_keys(self.session_features_raw)

        self.labels = None
        if labels_path:
            labels = np.asarray(self._load_json(Path(labels_path)))
            if label_dtype is not None:
                labels = labels.astype(label_dtype)
            self.labels = labels

        self.inputs = self._build_inputs()
        self._validate_lengths()

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Artifact bulunamadi: {path}")
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_embeddings(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Embedding artifact bulunamadi: {path}")
        return np.load(str(path))

    @staticmethod
    def _infer_session_feature_keys(session_features: List[Dict]) -> List[str]:
        if not session_features:
            return []
        return sorted(session_features[0].keys())

    def _build_inputs(self) -> Dict[str, np.ndarray]:
        inputs: Dict[str, np.ndarray] = {}

        # Kategorik diziler: .npz dosyasindan dogrudan numpy array
        for field in self._event_seq_npz.files:
            inputs[f"in_{field}"] = self._event_seq_npz[field].astype(np.int64)

        # Tag dizileri: memmap (disk-backed), float32'ye cast ederken kopyalanir
        inputs["in_tags"] = self._tag_seq_mmap.astype(np.float32)

        # BERT embedding'leri: memmap, zaten float32
        inputs["in_bert"] = (
            self._bert_mmap.astype(np.float32)
            if self._bert_mmap.dtype != np.float32
            else np.array(self._bert_mmap)  # memmap'i RAM'e al (DataLoader icin)
        )

        # Oturum ozellikleri: kucuk, JSON'dan
        features = [
            [row.get(k, 0.0) for k in self.session_feature_keys]
            for row in self.session_features_raw
        ]
        inputs["in_session_features"] = np.asarray(features, dtype=np.float32)

        return inputs

    def _validate_lengths(self) -> None:
        lengths = {name: arr.shape[0] for name, arr in self.inputs.items()}
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(f"Girdi uzunluklari eslesmiyor: {lengths}")
        n = next(iter(unique))
        if self.labels is not None and self.labels.shape[0] != n:
            raise ValueError(f"Labels uzunlugu ({self.labels.shape[0]}) input'larla ({n}) eslesmiyor.")

    def __len__(self) -> int:
        return next(iter(self.inputs.values())).shape[0]

    def __getitem__(self, idx: int):
        sample = {
            name: torch.from_numpy(arr[idx].copy())
            for name, arr in self.inputs.items()
        }
        if self.labels is None:
            return sample
        return sample, torch.tensor(int(self.labels[idx]), dtype=torch.long)

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class NumpyDataset(Dataset):
    """
    Split edilmis numpy dizilerini PyTorch Dataset'e sarar.
    _split_inputs() ciktisiyla kullanilir.
    """

    def __init__(self, inputs: Dict[str, np.ndarray], labels: np.ndarray) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        sample = {
            name: torch.from_numpy(arr[idx].copy())
            for name, arr in self.inputs.items()
        }
        return sample, torch.tensor(int(self.labels[idx]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    """
    Coklu girdi hibrit modeli.

    Girdiler (isimler SessionDataset ile eslesmeli):
      in_{kategorik_alan}  : (batch, max_len)          — int64
      in_tags              : (batch, max_len, tag_vocab) — float32
      in_bert              : (batch, max_len, emb_dim)   — float32
      in_session_features  : (batch, feature_dim)        — float32
    """

    def __init__(self, metadata: Dict, config: Dict) -> None:
        super().__init__()

        tag_vocab_size   = int(metadata["tagVocabSize"])
        emb_dim          = int(metadata["embeddingDimension"])
        session_feat_dim = int(metadata["sessionFeatureDim"])
        cat_vocabs       = metadata["categoricalVocabs"]

        lstm_units    = int(config.get("lstm_units",    128))
        bert_proj_dim = int(config.get("bert_proj_dim", 128))
        tag_proj_dim  = int(config.get("tag_proj_dim",   32))
        static_dim    = int(config.get("static_dim",     64))
        fusion_dim    = int(config.get("fusion_dim",    128))
        dropout_rate  = float(config.get("dropout_rate", 0.3))
        bidirectional = bool(config.get("bidirectional", False))
        self.task     = config.get("task", "multiclass")
        n_classes     = int(config.get("n_classes", 2))

        self.cat_fields = sorted(cat_vocabs.keys())

        # Kategorik embedding'ler
        self.embeddings = nn.ModuleDict()
        cat_emb_total = 0
        for field in self.cat_fields:
            vocab_size  = int(cat_vocabs[field])
            out_dim     = min(50, (vocab_size // 2) + 1)
            self.embeddings[field] = nn.Embedding(vocab_size, out_dim, padding_idx=0)
            cat_emb_total += out_dim

        # Zaman serisi projeksiyon katmanlari
        self.tag_proj  = nn.Linear(tag_vocab_size, tag_proj_dim)
        self.bert_proj = nn.Linear(emb_dim, bert_proj_dim)

        # LSTM
        lstm_in = cat_emb_total + tag_proj_dim + bert_proj_dim
        self.lstm = nn.LSTM(
            lstm_in, lstm_units,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_units * (2 if bidirectional else 1)

        # Statik dal
        self.static_dense = nn.Linear(session_feat_dim, static_dim)
        self.static_bn    = nn.BatchNorm1d(static_dim)

        # Fuzyon ve cikis
        self.dropout      = nn.Dropout(dropout_rate)
        self.fusion_dense = nn.Linear(lstm_out_dim + static_dim, fusion_dim)

        if self.task == "binary":
            self.out = nn.Linear(fusion_dim, 1)
        elif self.task == "multiclass":
            self.out = nn.Linear(fusion_dim, n_classes)
        else:  # regression
            self.out = nn.Linear(fusion_dim, 1)

        self.mask_field = self.cat_fields[0]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Padding mask: PAD=0 olan timestep'leri maskele
        mask = (inputs[f"in_{self.mask_field}"] != 0).float().unsqueeze(-1)  # (B, T, 1)

        # Kategorik embedding'ler → concat
        cat_embs = [self.embeddings[f](inputs[f"in_{f}"]) for f in self.cat_fields]
        cat_out  = torch.cat(cat_embs, dim=-1) * mask  # (B, T, cat_emb_total)

        # Tag & BERT projeksiyonlari
        tag_out  = torch.relu(self.tag_proj(inputs["in_tags"]))  * mask  # (B, T, tag_proj_dim)
        bert_out = torch.relu(self.bert_proj(inputs["in_bert"])) * mask  # (B, T, bert_proj_dim)

        # Tum zaman serisi dallarini birlestir
        seq = torch.cat([cat_out, tag_out, bert_out], dim=-1)  # (B, T, lstm_in)

        # PackedSequence: padded timestep'ler LSTM'e girmiyor
        seq_lens = (inputs[f"in_{self.mask_field}"] != 0).sum(dim=1).cpu().clamp(min=1)
        packed   = nn.utils.rnn.pack_padded_sequence(seq, seq_lens, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        # Bidirectional ise iki yonu birlestir
        lstm_out = torch.cat([h_n[0], h_n[1]], dim=-1) if self.lstm.bidirectional else h_n[0]
        lstm_out = self.dropout(lstm_out)  # (B, lstm_out_dim)

        # Statik dal
        static = torch.relu(self.static_dense(inputs["in_session_features"]))
        static = self.static_bn(static)  # (B, static_dim)

        # Fuzyon
        fused = torch.cat([lstm_out, static], dim=-1)
        fused = torch.relu(self.fusion_dense(fused))
        fused = self.dropout(fused)

        out = self.out(fused)

        if self.task == "binary":
            return torch.sigmoid(out).squeeze(-1)
        elif self.task == "regression":
            return out.squeeze(-1)
        else:
            return out  # ham logit, CrossEntropyLoss icin


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class HybridModelBuilder:
    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    @staticmethod
    def _load_metadata(metadata: Union[str, Path, Dict]) -> Dict:
        if isinstance(metadata, (str, Path)):
            return json.loads(Path(metadata).read_text(encoding="utf-8"))
        return dict(metadata)

    def build(self, metadata: Union[str, Path, Dict]) -> HybridModel:
        metadata = self._load_metadata(metadata)
        return HybridModel(metadata, self.config)


if __name__ == "__main__":
    builder = HybridModelBuilder()
    model = builder.build("output/processed/metadata.json")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam egitilecek parametre: {total:,}")
