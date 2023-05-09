from typing import Callable, Tuple
import tensorflow as tf
import os
from .utils import load_list


def random_size(image: tf.Tensor, target_size: int = None) -> tf.Tensor:
    """
    resize the image to have the propoer shape

    Args:
        image: image to resize
        target: size of the smallest image edge
    """
    width, height, _ = tf.split(tf.shape(image), num_or_size_splits=3)
    if target_size is None:
        target_size = tf.random.uniform(shape=[1], minval=256, maxval=384)
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = tf.concat(
        [tf.cast(width * size_ratio, tf.int32), tf.cast(height * size_ratio, tf.int32)],
        axis=0,
    )
    return tf.image.resize(image, resize_shape)


def imagenet(
    pre_processing: Callable,
    batch_size: int,
    path_to_data: str,
    path_to_labels: str,
    parallel_exec: bool = True,
    repeat: bool = False,
) -> tf.data.Dataset:
    """
    This function creates the data iterator for test set
    """
    center_crop = tf.keras.layers.CenterCrop(
        height=224,
        width=224,
    )

    def load_image(
        image_path: str,
        label: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        In training, it is highly recommended to set the augment to true.
        In test, the standard 10-crop test [1] is provided for fair comparison.
        [1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
        """
        image = tf.cast(
            tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3), dtype=tf.float32
        )

        image = random_size(image, target_size=256)
        image = center_crop(image)
        image = pre_processing(image)
        label_one_hot = tf.one_hot(indices=label, depth=1000)

        return image, label_one_hot

    num_parallel_calls = tf.data.AUTOTUNE
    if not parallel_exec:
        num_parallel_calls = None
    label_path = path_to_labels
    img_path = path_to_data
    images, labels = load_list(label_path, img_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(
        load_image,
        num_parallel_calls=num_parallel_calls,
    )
    dataset = dataset.batch(batch_size)
    return dataset
