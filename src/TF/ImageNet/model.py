from typing import Any
import tensorflow as tf


class classification_model:
    def __init__(self, model_name: str, model: str, preproc: str) -> None:
        if model_name.lower() == "resnet":
            self.model = tf.keras.applications.resnet50.ResNet50()
            self.preprocess = tf.keras.applications.resnet50.preprocess_input
        else:
            self.model = tf.keras.models.load_model(model)
            if preproc == "tf":
                self.preprocess = tf.keras.applications.mobilenet.preprocess_input()
            if preproc == "torch":
                self.preprocess = tf.keras.applications.densenet.preprocess_input()
            if preproc == "caffe":
                self.preprocess = tf.keras.applications.resnet50.preprocess_input()
            if preproc == "identity":
                self.preprocess = tf.keras.applications.efficientnet.preprocess_input()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
