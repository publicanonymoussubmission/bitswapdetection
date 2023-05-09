from typing import Any
import tensorflow as tf


class EvaluateModel:
    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.model = model
        self.model.compile(metrics=["accuracy"])
        self.dataset = dataset

    def __call__(
        self,
        layer_to_activate: str,
        evaluation_steps: int = None,
        *args: Any,
        **kwds: Any,
    ) -> float:
        self.model.get_layer(layer_to_activate).bitswap_coefficient.assign(1.0)
        accuracy = self.model.evaluate(
            self.dataset, steps=evaluation_steps, verbose=False
        )[-1]
        self.model.get_layer(layer_to_activate).bitswap_coefficient.assign(0.0)
        return accuracy
