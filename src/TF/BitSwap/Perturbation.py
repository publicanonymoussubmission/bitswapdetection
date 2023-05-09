from .BinarySwap import SimulateSwap
import tensorflow as tf
from typing import Dict, Any, Tuple


class BitSwapWrapper(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_to_wrap: tf.keras.layers.Layer,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs: Any,
    ):
        name = f"wrapped_{layer_to_wrap.name}"
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.layer_to_wrap = layer_to_wrap
        self.swap_sim = SimulateSwap()

    def build(self, input_shape: Tuple[int]) -> None:
        self.initial_shape = input_shape
        self.perturbation_shape = tf.cast(tf.constant(input_shape[1:]), tf.float32)
        self.probabilities = tf.ones(
            shape=self.initial_shape[1:]
        ) / tf.math.reduce_prod(self.perturbation_shape)

        self.bitswap_coefficient = self.add_weight(
            name="bitswap_coeff",
            shape=None,
            initializer="zeros",
            trainable=False,
        )
        self.flattener = tf.keras.layers.Flatten()
        return super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["layer_to_wrap"] = self.layer_to_wrap
        return config

    def sample_bitswap_intensity(self, x: tf.Tensor) -> tf.Tensor:
        return self.swap_sim(inputs=x) - x

    def apply_bitswap(self, inputs: tf.Tensor) -> tf.Tensor:
        flat_inputs = self.flattener(inputs)
        logits = self.flattener(
            tf.expand_dims(
                tf.math.log(self.probabilities),
                axis=0,
            )
        )
        idx = tf.transpose(
            tf.random.categorical(logits=logits, num_samples=tf.shape(inputs)[0]),
            [1, 0],
        )
        perturbation = tf.zeros_like(input=flat_inputs)
        updates = tf.gather_nd(
            params=perturbation, indices=idx, batch_dims=1
        ) + self.sample_bitswap_intensity(
            x=tf.gather_nd(params=flat_inputs, indices=idx, batch_dims=1)
        )
        indices = tf.concat(
            [
                tf.expand_dims(tf.range(tf.shape(inputs)[0], dtype=tf.int64), axis=-1),
                idx,
            ],
            axis=-1,
        )
        perturbation = tf.tensor_scatter_nd_update(
            tensor=perturbation,
            indices=indices,
            updates=updates,
        )
        perturbation = (
            tf.reshape(tensor=perturbation, shape=tf.shape(inputs))
            * self.bitswap_coefficient
        )
        return inputs + perturbation

    def call(self, inputs: tf.Tensor, *args: Any, **kwargs: Any) -> tf.Tensor:
        inputs = self.apply_bitswap(inputs=inputs)
        return self.layer_to_wrap(inputs, *args, **kwargs)


if __name__ == "__main__":
    l1 = tf.keras.layers.Activation("linear")
    l2 = BitSwapWrapper(layer_to_wrap=l1)
    l2.build(input_shape=(None, 4, 3))

    print("--- built layer ---")

    import numpy as np

    a = np.random.normal(size=(2, 4, 3))
    print(a)
    print(l2(a).numpy() - a)
    # print(l2(a).numpy()[0] - a)
    # python -m src.BitSwap.Perturbation
