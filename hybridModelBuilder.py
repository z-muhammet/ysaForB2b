"""
HybridModelBuilder — ysaForB2b icin coklu girdi modeli.

Bu modul DataPreprocessor artifact'larini dogrudan bir Keras modele baglar:
- kategorik dizi embedding'leri
- multi-hot etiket dizileri
- BERT mesaj embedding'leri
- oturum duzeyi statik ozellikler

Tasarim, zamansal fuzyon desenini izler (dizi girdileri -> LSTM),
ardindan LSTM durumunu statik ozelliklerle birlestirerek nihai tahmin yapar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypeAlias

import numpy as np

_TF_IMPORT_ERROR: Optional[Exception] = None

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
except Exception as exc:
    tf = None
    keras = None
    _TF_IMPORT_ERROR = exc


def _require_tensorflow() -> None:
    if tf is None or keras is None:
        raise ImportError(
            "TensorFlow gerekli. Kurulum icin: pip install tensorflow"
        ) from _TF_IMPORT_ERROR


Tensor: TypeAlias = Any
Dataset: TypeAlias = Any
Layer: TypeAlias = Any
Input: TypeAlias = Any
Model: TypeAlias = Any


DEFAULT_CONFIG: Dict[str, object] = {
    "lstm_units": 128,
    "bert_proj_dim": 128,
    "tag_proj_dim": 32,
    "static_dim": 64,
    "fusion_dim": 128,
    "dropout_rate": 0.3,
    "task": "binary",  # gorev tipi: binary | multiclass | regression
    "n_classes": 2,
    "learning_rate": 1e-3,
    "bidirectional": False,
}


class SessionDataset:
    """
    JSON artifact'larini okuyan hafif bir veri yukleyici.
    - model girdileri sozlugu (isimlendirilmis input'lar)
    - opsiyonel etiketler

    Artifact dizin yapisi DataPreprocessor.save_artifacts() ile uyumludur.
    """

    def __init__(
        self,
        artifacts_dir: str,
        labels_path: Optional[str] = None,
        label_dtype: Optional[np.dtype] = np.int64,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.processed_dir = self.artifacts_dir / "processed"

        # Sekil ve vocab boyutlarini bilmek icin once metadata yuklenir
        self.metadata = self._load_json(self.processed_dir / "metadata.json")

        # Dizi artifact'lari
        self.event_sequences = self._load_json(self.processed_dir / "event_sequences.json")
        self.tag_sequences = self._load_json(self.processed_dir / "tag_sequences.json")
        self.message_embeddings = self._load_json(
            self.processed_dir / "message_embedding_sequences.json"
        )

        # Oturum duzeyi ozellikler dict listesi olarak saklanir
        self.session_features_raw = self._load_json(
            self.processed_dir / "session_features.json"
        )
        self.session_feature_keys = self._infer_session_feature_keys(self.session_features_raw)

        # Opsiyonel etiketler (gudumlu egitim icin)
        self.labels = None
        if labels_path:
            labels = np.asarray(self._load_json(Path(labels_path)))
            if label_dtype is not None:
                labels = labels.astype(label_dtype)
            self.labels = labels

        # tf.data'nin verimli dilimleyebilmesi icin numpy array'e cevir
        self.inputs = self._build_inputs()

        # Temel tutarlilik kontrolu: tum girdilerin ilk boyutu ayni olmali
        self._validate_lengths()

    @staticmethod
    def _load_json(path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _infer_session_feature_keys(session_features: List[Dict]) -> List[str]:
        # Oturum ozellikleri icin stabil ve deterministik ozellik sirasi
        if not session_features:
            return []
        return sorted(session_features[0].keys())

    def _build_inputs(self) -> Dict[str, np.ndarray]:
        inputs: Dict[str, np.ndarray] = {}

        # Kategorik diziler: her alan ayri bir input olur
        for field, seqs in self.event_sequences.items():
            # Embedding lookup icin int32 tercih edilir
            inputs[f"in_{field}"] = np.asarray(seqs, dtype=np.int32)

        # Tag'ler: her timestep icin multi-hot matris
        inputs["in_tags"] = np.asarray(self.tag_sequences, dtype=np.float32)

        # BERT mesaj embedding'leri: hazir dense vektorler
        inputs["in_bert"] = np.asarray(self.message_embeddings, dtype=np.float32)

        # Oturum ozellikleri: dict -> sirali vektor
        features = []
        for row in self.session_features_raw:
            features.append([row.get(k, 0.0) for k in self.session_feature_keys])
        inputs["in_session_features"] = np.asarray(features, dtype=np.float32)

        return inputs

    def _validate_lengths(self) -> None:
        # Tum diziler ayni oturum sayisini (ilk boyut) paylasmali
        lengths = [arr.shape[0] for arr in self.inputs.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"Mismatched input lengths: {lengths}")
        if self.labels is not None and self.labels.shape[0] != lengths[0]:
            raise ValueError("Labels length does not match inputs.")

    def __len__(self) -> int:
        # Uzunluklar dogrulandigi icin herhangi bir input uzunluk icin yeterli
        return next(iter(self.inputs.values())).shape[0]

    def __getitem__(self, idx: int):
        # Hizli inceleme veya ozel yukleyiciler icin tekli ornek dilimleme
        sample = {name: arr[idx] for name, arr in self.inputs.items()}
        if self.labels is None:
            return sample
        return sample, self.labels[idx]

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        repeat: bool = False,
    ) -> Dataset:
        _require_tensorflow()
        # tf.data dict-of-arrays'i dogrudan dilimleyebilir
        if self.labels is None:
            ds = tf.data.Dataset.from_tensor_slices(self.inputs)
        else:
            ds = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(self))
        if repeat:
            ds = ds.repeat()

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class TFRecordDataset:
    """
    TFRecord tabanli, RAM dostu veri yukleyici.

    Beklenen TFRecord alanlari:
      - her kategorik alan icin: field adi (int64, shape [max_len])
      - tags                 : float32, shape [max_len * tag_vocab_size]
      - bert                 : float32, shape [max_len * embedding_dim]
      - session_features     : float32, shape [session_feature_dim]
      - opsiyonel etiket     : label_key ile belirtilir
    """

    def __init__(
        self,
        tfrecord_paths: Union[str, List[str]],
        metadata: Union[str, Path, Dict],
        label_key: Optional[str] = None,
        label_shape: Optional[List[int]] = None,
        label_dtype: Optional["tf.DType"] = None,
        compression_type: Optional[str] = None,
    ) -> None:
        _require_tensorflow()
        self.tfrecord_paths = [tfrecord_paths] if isinstance(tfrecord_paths, str) else list(tfrecord_paths)
        self.metadata = self._load_metadata(metadata)
        self.label_key = label_key
        self.label_shape = label_shape
        self.label_dtype = label_dtype
        self.compression_type = compression_type

    @staticmethod
    def _load_metadata(metadata: Union[str, Path, Dict]) -> Dict:
        if isinstance(metadata, (str, Path)):
            path = Path(metadata)
            return json.loads(path.read_text(encoding="utf-8"))
        return dict(metadata)

    def _feature_spec(self) -> Dict[str, object]:
        max_len = int(self.metadata["maxSessionLength"])
        tag_vocab_size = int(self.metadata["tagVocabSize"])
        emb_dim = int(self.metadata["embeddingDimension"])
        session_dim = int(self.metadata["sessionFeatureDim"])
        cat_vocabs = self.metadata["categoricalVocabs"]

        spec: Dict[str, object] = {}
        for field in sorted(cat_vocabs.keys()):
            spec[field] = tf.io.FixedLenFeature([max_len], tf.int64)

        spec["tags"] = tf.io.FixedLenFeature([max_len * tag_vocab_size], tf.float32)
        spec["bert"] = tf.io.FixedLenFeature([max_len * emb_dim], tf.float32)
        spec["session_features"] = tf.io.FixedLenFeature([session_dim], tf.float32)

        if self.label_key:
            shape = self.label_shape or []
            dtype = self.label_dtype or tf.int64
            spec[self.label_key] = tf.io.FixedLenFeature(shape, dtype)

        return spec

    def _parse_example(
        self, example_proto: Tensor
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Tensor]]:
        max_len = int(self.metadata["maxSessionLength"])
        tag_vocab_size = int(self.metadata["tagVocabSize"])
        emb_dim = int(self.metadata["embeddingDimension"])
        cat_vocabs = self.metadata["categoricalVocabs"]

        parsed = tf.io.parse_single_example(example_proto, self._feature_spec())

        inputs: Dict[str, Tensor] = {}
        for field in sorted(cat_vocabs.keys()):
            inputs[f"in_{field}"] = parsed[field]

        inputs["in_tags"] = tf.reshape(parsed["tags"], [max_len, tag_vocab_size])
        inputs["in_bert"] = tf.reshape(parsed["bert"], [max_len, emb_dim])
        inputs["in_session_features"] = parsed["session_features"]

        if self.label_key:
            return inputs, parsed[self.label_key]

        return inputs

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        repeat: bool = False,
        shuffle_buffer: int = 10000,
        num_parallel_calls: Optional[int] = None,
    ) -> Dataset:
        ds = tf.data.TFRecordDataset(self.tfrecord_paths, compression_type=self.compression_type)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer)

        ds = ds.map(self._parse_example, num_parallel_calls=num_parallel_calls or tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Dizi ve statik ozellikleri birlestiren coklu girdi Keras modeli kurar.
# Beklenen input'lar (isimler SessionDataset ile eslesmeli):
#   - in_{categorical_field} : (batch, max_len)
#   - in_tags                : (batch, max_len, tag_vocab_size)
#   - in_bert                : (batch, max_len, embedding_dim)
#   - in_session_features    : (batch, session_feature_dim)
class HybridModelBuilder:

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    @staticmethod
    def _load_metadata(metadata: Union[str, Path, Dict]) -> Dict:
        if isinstance(metadata, (str, Path)):
            path = Path(metadata)
            return json.loads(path.read_text(encoding="utf-8"))
        return dict(metadata)

    @staticmethod
    def _embedding_dim_for_vocab(vocab_size: int) -> int:
        # Basit sezgisel kural: kucuk vocab -> kucuk embedding, buyuk vocab -> 50'de sinirla
        return min(50, (vocab_size // 2) + 1)

    def _build_categorical_branches(
        self, metadata: Dict
    ) -> Tuple[List[Layer], Dict[str, Input]]:
        inputs: Dict[str, Input] = {}
        embeddings: List[Layer] = []

        max_len = int(metadata["maxSessionLength"])
        cat_vocabs = metadata["categoricalVocabs"]

        # Her kategorik alan icin bir input + embedding olustur
        for field in sorted(cat_vocabs.keys()):
            vocab_size = int(cat_vocabs[field])
            emb_dim = self._embedding_dim_for_vocab(vocab_size)

            inp = keras.Input(
                shape=(max_len,),
                dtype="int32",
                name=f"in_{field}",
            )
            emb = keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=emb_dim,
                mask_zero=True,
                name=f"emb_{field}",
            )(inp)

            inputs[field] = inp
            embeddings.append(emb)

        # Tum kategorik embedding'leri ozellik ekseninde birlestir
        cat_concat = keras.layers.Concatenate(axis=-1, name="cat_concat")(embeddings)
        return [cat_concat], inputs

    def _build_tag_branch(self, metadata: Dict) -> Tuple[Layer, Input]:
        max_len = int(metadata["maxSessionLength"])
        tag_vocab_size = int(metadata["tagVocabSize"])

        inp = keras.Input(
            shape=(max_len, tag_vocab_size),
            dtype="float32",
            name="in_tags",
        )
        # Yuksek boyutlu tag vektorlerini timestep basina kompakt temsile projekte et
        proj = keras.layers.TimeDistributed(
            keras.layers.Dense(int(self.config["tag_proj_dim"]), activation="relu"),
            name="tag_proj",
        )(inp)
        return proj, inp

    def _build_bert_branch(self, metadata: Dict) -> Tuple[Layer, Input]:
        max_len = int(metadata["maxSessionLength"])
        emb_dim = int(metadata["embeddingDimension"])

        inp = keras.Input(
            shape=(max_len, emb_dim),
            dtype="float32",
            name="in_bert",
        )
        # 768-boyutlu BERT vektorlerini daha kucuk bir dense uzaya projekte et
        proj = keras.layers.TimeDistributed(
            keras.layers.Dense(int(self.config["bert_proj_dim"]), activation="relu"),
            name="bert_proj",
        )(inp)
        return proj, inp

    def _build_static_branch(
        self, metadata: Dict
    ) -> Tuple[Layer, Input]:
        feature_dim = int(metadata["sessionFeatureDim"])

        inp = keras.Input(
            shape=(feature_dim,),
            dtype="float32",
            name="in_session_features",
        )
        dense = keras.layers.Dense(int(self.config["static_dim"]), activation="relu")(inp)
        # BatchNorm statik ozellikler arasindaki olcek farklarini dengeler
        dense = keras.layers.BatchNormalization()(dense)
        return dense, inp

    def _build_temporal_fusion(
        self,
        seq_inputs: List[Layer],
        mask_source: Input,
    ) -> Layer:
        # Bir kategorik input'tan padding mask'i uret (PAD token = 0).
        # Bu sayede LSTM padded timestep'leri gercek olay olarak gormez.
        seq_mask = keras.layers.Lambda(
            lambda x: tf.not_equal(x, 0), name="seq_mask"
        )(mask_source)

        # Mask'i float'a cevirip feature eksenine genislet; tum dallara uygulanabilir
        seq_mask_f = keras.layers.Lambda(
            lambda m: tf.cast(tf.expand_dims(m, -1), tf.float32),
            name="seq_mask_f",
        )(seq_mask)

        # Maske ile carparak padded timestep'leri sifirla (tag/bert icin de gecerli)
        masked_inputs = [keras.layers.Multiply()([seq, seq_mask_f]) for seq in seq_inputs]

        # Tum sirali dallari tek 3B tensorde birlestir (batch, time, features)
        seq_concat = keras.layers.Concatenate(axis=-1, name="seq_concat")(masked_inputs)

        lstm_layer = keras.layers.LSTM(int(self.config["lstm_units"]), name="lstm")
        if self.config.get("bidirectional"):
            seq_out = keras.layers.Bidirectional(lstm_layer, name="bi_lstm")(
                seq_concat, mask=seq_mask
            )
        else:
            seq_out = lstm_layer(seq_concat, mask=seq_mask)

        # Zamansal kodlama sonrasinda regularization icin dropout
        return keras.layers.Dropout(float(self.config["dropout_rate"]))(seq_out)

    def _build_head(self, fused: Layer) -> Tuple[Layer, str]:
        # Goreve ozel cikis oncesi ortak dense katman
        x = keras.layers.Dense(int(self.config["fusion_dim"]), activation="relu")(fused)
        x = keras.layers.Dropout(float(self.config["dropout_rate"]))(x)

        task = self.config["task"]
        if task == "binary":
            out = keras.layers.Dense(1, activation="sigmoid", name="out")(x)
            loss = "binary_crossentropy"
        elif task == "multiclass":
            n_classes = int(self.config["n_classes"])
            out = keras.layers.Dense(n_classes, activation="softmax", name="out")(x)
            loss = "sparse_categorical_crossentropy"
        elif task == "regression":
            out = keras.layers.Dense(1, activation="linear", name="out")(x)
            loss = "mse"
        else:
            raise ValueError(f"Unknown task: {task}")

        return out, loss

    def build(self, metadata: Union[str, Path, Dict]) -> Model:
        """
        Coklu girdi modelini kurar ve derler.

        metadata su tiplerde olabilir:
          - dict (hazir yuklenmis)
          - processed/metadata.json dosya yolu
        """
        _require_tensorflow()
        metadata = self._load_metadata(metadata)

        # Kategorik dallar (alan basi) + embedding birlestirme
        cat_seq, cat_inputs = self._build_categorical_branches(metadata)

        # Tag ve BERT dallari
        tag_seq, tag_in = self._build_tag_branch(metadata)
        bert_seq, bert_in = self._build_bert_branch(metadata)

        # Zamansal fuzyon: embedding + tag proj + bert proj -> LSTM
        # Maske kaynagi olarak ilk kategorik input'u kullan (PAD=0)
        mask_field = sorted(cat_inputs.keys())[0]
        seq_out = self._build_temporal_fusion(
            seq_inputs=cat_seq + [tag_seq, bert_seq],
            mask_source=cat_inputs[mask_field],
        )

        # Statik dal ve final birlestirme
        static_out, static_in = self._build_static_branch(metadata)
        fused = keras.layers.Concatenate(name="fusion_concat")([seq_out, static_out])

        # Goreve ozel cikis kati
        out, loss = self._build_head(fused)

        # Modeli bir araya getir
        inputs = list(cat_inputs.values()) + [tag_in, bert_in, static_in]
        model = keras.Model(inputs=inputs, outputs=out, name="hybrid_model")

        # Adam optimizer ve goreve uygun metriklerle derle
        optimizer = keras.optimizers.Adam(learning_rate=float(self.config["learning_rate"]))
        task = self.config["task"]
        if task == "binary":
            metrics = ["accuracy", keras.metrics.AUC(name="auc")]
        elif task == "multiclass":
            metrics = [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ]
        else:
            metrics = [keras.metrics.MeanAbsoluteError(name="mae")]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model


if __name__ == "__main__":
    # Ornek kullanim: artifact metadata'dan model kur
    builder = HybridModelBuilder()
    model = builder.build("output/processed/metadata.json")
    model.summary()
