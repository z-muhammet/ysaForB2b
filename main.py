import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from dataPreprocessor import DataPreprocessor
from hybridModelBuilder import HybridModelBuilder, SessionDataset
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
	val_end = int((1.0 - test_ratio) * len(idx))

	train_idx = idx[:train_end]
	val_idx = idx[train_end:val_end]
	test_idx = idx[val_end:]

	train_inputs = {k: v[train_idx] for k, v in inputs.items()}
	val_inputs = {k: v[val_idx] for k, v in inputs.items()}
	test_inputs = {k: v[test_idx] for k, v in inputs.items()}

	return (
		(train_inputs, labels[train_idx]),
		(val_inputs, labels[val_idx]),
		(test_inputs, labels[test_idx]),
	)


def _to_tf_dataset(
	inputs: Dict[str, np.ndarray],
	labels: np.ndarray,
	batch_size: int,
	shuffle: bool,
	seed: int = 42,
) -> tf.data.Dataset:
	ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
	if shuffle:
		ds = ds.shuffle(buffer_size=len(labels), seed=seed)
	return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _log_label_stats(labels: np.ndarray, top_k: int = 10) -> None:
	values, counts = np.unique(labels, return_counts=True)
	order = np.argsort(counts)[::-1]

	top_values = values[order][:top_k]
	top_counts = counts[order][:top_k]
	LOGGER.info("Top-%d label dagilimi: %s", top_k, dict(zip(top_values.tolist(), top_counts.tolist())))


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)

	data_path = "B2bUserDataset.json"
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
		(val_inputs, val_labels),
		(test_inputs, test_labels),
	) = _split_inputs(dataset.inputs, dataset.labels, val_ratio=0.2, test_ratio=0.1)

	LOGGER.info("Train/Val/Test ornek sayilari: %d / %d / %d", len(train_labels), len(val_labels), len(test_labels))

	train_ds = _to_tf_dataset(train_inputs, train_labels, batch_size=32, shuffle=True)
	val_ds = _to_tf_dataset(val_inputs, val_labels, batch_size=32, shuffle=False)
	test_ds = _to_tf_dataset(test_inputs, test_labels, batch_size=32, shuffle=False)

	event_vocab_size = int(dataset.metadata["categoricalVocabs"]["event_type"])

	config = {
		"task": "multiclass",
		"n_classes": event_vocab_size,
		"lstm_units": 128,
		"learning_rate": 1e-3,
	}

	LOGGER.info("Model mimarisi kuruluyor...")
	builder = HybridModelBuilder(config=config)
	hybrid_model = builder.build(dataset.metadata)
	hybrid_model.summary()

	trainer = ModelTrainer(
		hybrid_model=hybrid_model,
		output_directory=str(output_dir / "models"),
		tensorboard=True,
		reduce_lr_on_plateau=True,
	)

	trainer.train(
		training_dataset=train_ds,
		validation_dataset=val_ds,
		total_epochs=15,
	)

	trainer.load_best_weights()
	LOGGER.info("Egitim tamamlandi. En iyi agirliklar yüklendi.")

	LOGGER.info("Test degerlendirmesi basliyor...")
	trainer.evaluate(test_ds)


if __name__ == "__main__":
	main()
