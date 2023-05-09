from typing import Any
import tensorflow as tf


class SimulateSwap:
    def __init__(self, *args: Any, **kwds: Any) -> None:
        pass

    def sample_bit_int32(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(
            tf.math.pow(
                tf.cast(2, dtype=tf.int64),
                tf.cast(
                    tf.random.uniform(shape=tf.shape(x), maxval=31), dtype=tf.int64
                ),
            ),
            dtype=tf.int32,
        )

    def get_sign(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(
            -2
            * (
                tf.cast(
                    tf.clip_by_value(t=x, clip_value_min=0, clip_value_max=1),
                    dtype=tf.float32,
                )
                - 0.5
            ),
            dtype=tf.int32,
        )

    def __call__(self, inputs: tf.Tensor, *args: Any, **kwds: Any) -> tf.Tensor:
        bitcast_to_int32 = tf.bitcast(inputs, tf.int32)
        magnitude = self.sample_bit_int32(x=inputs)
        sign_mult = self.get_sign(x=bitcast_to_int32 & magnitude)
        bitcast_to_int32 = bitcast_to_int32 + sign_mult * magnitude
        bitcast_to_float = tf.bitcast(bitcast_to_int32, tf.float32)
        return bitcast_to_float
