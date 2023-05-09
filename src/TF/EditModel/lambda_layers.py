from typing import Any, Dict
import tensorflow as tf
import numpy as np


def get_lambda_args(cpt: int, model: tf.keras.Model) -> Any:
    """
    retrieve the lambda arguments from model config as
    they can't be accessed from the layer config

    Args:
        cpt: layer number in mdoel indexing
        model: model containing the layer config
    """
    if len(model.get_config()["layers"][cpt]["inbound_nodes"]) == 0:
        return None
    extra_args = model.get_config()["layers"][cpt]["inbound_nodes"][0][-1]
    if isinstance(extra_args, list):
        return [-5, 5]  # shitty fix for shitty bug
    if "y" in extra_args:
        return {"y": extra_args["y"]}
    elif "clip_value_min" in extra_args:
        return {
            "clip_value_min": extra_args["clip_value_min"],
            "clip_value_max": extra_args["clip_value_max"],
        }
    else:
        return [value for _, value in extra_args.items()]


def retrieve_layer_cpt(model: tf.keras.Model, layer_name: str) -> int:
    """
    retrive the index of the layer in the list model.layers

    Args:
        model: model containing the target layer
        layer_name: target layer name
    """
    for cpt, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return cpt


def call_lambda_layer(
    layer_input: tf.Variable,
    model: tf.keras.Model,
    layer: tf.keras.layers.Layer,
    layer_cpt: int,
) -> tf.Variable:
    """
    this function deals with lambda layers
    the issue is : lambda layers often use parameters
    that are neither weights nor variable and are
    only accessible in the mdoel config

    Args:
        layer_input: symbolic layer input
        model: tf keras Model (contains layer local config)
        layer: lambda layer
        layer_cpt: layer index in model.layers
    """
    if not isinstance(layer, tf.keras.layers.Lambda) and not (
        "lambda" in type(layer).__name__.lower()
    ):
        output = layer(layer_input)
        return output
    if layer_cpt == -1:
        layer_cpt = retrieve_layer_cpt(model=model, layer_name=layer.name)
    extra_args = get_lambda_args(cpt=layer_cpt, model=model)
    if extra_args is None:
        return layer(layer_input)
    copied_layer = type(layer).from_config(layer.get_config())
    if "y" in extra_args:
        output = copied_layer(layer_input, extra_args["y"])
    elif "clip_value_min" in extra_args:
        output = copied_layer(
            layer_input, extra_args["clip_value_min"], extra_args["clip_value_max"]
        )
    elif "tf.cast" in layer.name.lower():
        output = layer(
            layer_input,
            dtype=model.get_config()["layers"][layer_cpt]["config"]["dtype"],
        )
    else:
        output = copied_layer(layer_input, *extra_args)
    return output
