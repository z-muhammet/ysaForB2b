import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tensorflow as tf


class ModelTrainer:
	"""
	Hybrid B2B modelinin egitim dongusunu ve callback'leri yonetir.
	"""

	def __init__(
		self,
		hybrid_model: tf.keras.Model,
		output_directory: str,
		compile_kwargs: Optional[Dict[str, Any]] = None,
		monitor: str = "val_loss",
		patience: int = 3,
		min_delta: float = 0.0,
		tensorboard: bool = False,
		reduce_lr_on_plateau: bool = False,
		lr_factor: float = 0.5,
		lr_patience: int = 2,
		min_lr: float = 1e-6,
	) -> None:
		self.hybrid_model = hybrid_model
		self.output_directory = Path(output_directory)
		self.output_directory.mkdir(parents=True, exist_ok=True)

		self.compile_kwargs = compile_kwargs
		self.monitor = monitor
		self.patience = patience
		self.min_delta = min_delta
		self.tensorboard = tensorboard
		self.reduce_lr_on_plateau = reduce_lr_on_plateau
		self.lr_factor = lr_factor
		self.lr_patience = lr_patience
		self.min_lr = min_lr

		self._best_weights_path = self.output_directory / "bestModelWeights.weights.h5"
		self.history: Optional[tf.keras.callbacks.History] = None

	def _ensure_compiled(self) -> None:
		if self.hybrid_model.optimizer is not None:
			return

		if not self.compile_kwargs:
			raise ValueError("Model compile edilmedi ve compile_kwargs saglanmadi.")

		self.hybrid_model.compile(**self.compile_kwargs)

	def build_callbacks(self) -> List[tf.keras.callbacks.Callback]:
		callbacks: List[tf.keras.callbacks.Callback] = []

		callbacks.append(
			tf.keras.callbacks.EarlyStopping(
				monitor=self.monitor,
				patience=self.patience,
				min_delta=self.min_delta,
				restore_best_weights=True,
				verbose=1,
			)
		)

		callbacks.append(
			tf.keras.callbacks.ModelCheckpoint(
				filepath=str(self._best_weights_path),
				monitor=self.monitor,
				save_best_only=True,
				save_weights_only=True,
				verbose=1,
			)
		)

		if self.tensorboard:
			log_dir = self.output_directory / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)))

		if self.reduce_lr_on_plateau:
			callbacks.append(
				tf.keras.callbacks.ReduceLROnPlateau(
					monitor=self.monitor,
					factor=self.lr_factor,
					patience=self.lr_patience,
					min_lr=self.min_lr,
					verbose=1,
				)
			)

		return callbacks

	def train(
		self,
		training_dataset: tf.data.Dataset,
		validation_dataset: Optional[tf.data.Dataset] = None,
		total_epochs: int = 20,
		callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
		**fit_kwargs: Any,
	) -> tf.keras.callbacks.History:
		self._ensure_compiled()

		if callbacks is None:
			callbacks = self.build_callbacks()

		print("Model egitim sureci basliyor...")
		self.history = self.hybrid_model.fit(
			training_dataset,
			validation_data=validation_dataset,
			epochs=total_epochs,
			callbacks=callbacks,
			**fit_kwargs,
		)

		return self.history

	def evaluate(self, test_dataset: tf.data.Dataset, **kwargs: Any) -> Any:
		self._ensure_compiled()
		return self.hybrid_model.evaluate(test_dataset, **kwargs)

	def load_best_weights(self) -> None:
		if self._best_weights_path.exists():
			self.hybrid_model.load_weights(str(self._best_weights_path))

	def plot_history(self, history: Optional[tf.keras.callbacks.History] = None) -> None:
		"""
		Egitim surecini gorsellestirir. Matplotlib gerektirir.
		"""
		if history is None:
			history = self.history
		if history is None:
			raise ValueError("Plot icin history bulunamadi.")

		import matplotlib.pyplot as plt

		hist = history.history
		keys = [k for k in hist.keys() if not k.startswith("val_")]

		for key in keys:
			plt.figure()
			plt.plot(hist.get(key, []), label=key)
			plt.plot(hist.get(f"val_{key}", []), label=f"val_{key}")
			plt.title(key)
			plt.xlabel("epoch")
			plt.ylabel(key)
			plt.legend()
			plt.tight_layout()
			plt.show()
