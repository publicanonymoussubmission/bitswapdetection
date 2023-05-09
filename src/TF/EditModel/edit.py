from ..BitSwap import BitSwapWrapper
from .lambda_layers import call_lambda_layer
from .utils import get_graph_as_dict
import tensorflow as tf
from typing import Tuple


def apply_bitswap_wrapper(
    model: tf.keras.Model,
    verbose: bool = False,
    layer_types_to_watch: Tuple[tf.keras.layers.Layer] = (
        tf.keras.layers.Dense,
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
    ),
) -> tf.keras.Model:
    """
    replaces layers basd on types

    Args:
        model: model to edit
        verbose: verbatim level
        layer_types_to_watch: which layer to wrap
    """
    modified_layers = 0
    network_dict, is_transformer = get_graph_as_dict(model=model)
    model_outputs = []
    model_layers = model.layers[1:]
    layer_cpt_shift = 1
    model_outputnames = model.output_names
    if is_transformer:
        model_layers = model.layers
        model_outputnames = [model.layers[-1].name]
        layer_cpt_shift = 0

    for layer_cpt, layer in enumerate(model_layers):
        if verbose:
            print(
                f"\r[\033[96mReplaceLayersByCounterparts\033[0m] ["
                + "\033[91m=\033[0m" * int(15 * layer_cpt / len(model_layers))
                + " " * int(15 - int(15 * layer_cpt / len(model_layers)))
                + "]",
                end="",
            )
        if isinstance(layer, tf.keras.layers.InputLayer):
            layer_input = network_dict["input"][layer.name]
        else:
            input_layers = []
            for in_node in layer.inbound_nodes:
                if isinstance(in_node.inbound_layers, (list, tuple)):
                    for inl in in_node.inbound_layers:
                        input_layers.append(inl.name)
                else:
                    input_layers.append(in_node.inbound_layers.name)
            layer_input = [
                network_dict["new_output"][layer_aux] for layer_aux in input_layers
            ]
        if layer.name in network_dict["new_output"]:
            continue
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if isinstance(layer, tf.keras.layers.InputLayer):
            x = layer_input
        elif len(network_dict["input"][layer.name]) == 1:
            if isinstance(layer, layer_types_to_watch):
                modified_layers += 1
                x = BitSwapWrapper(layer_to_wrap=layer)(layer_input)
            else:
                copied_layer = type(layer).from_config(layer.get_config())
                x = call_lambda_layer(
                    layer_input=layer_input,
                    model=model,
                    layer=copied_layer,
                    layer_cpt=layer_cpt + layer_cpt_shift,
                )
        else:
            if isinstance(layer, layer_types_to_watch):
                modified_layers += 1
                x = BitSwapWrapper(layer_to_wrap=layer)(layer_input)
            elif isinstance(layer, tf.keras.layers.Multiply) or isinstance(
                layer, tf.keras.layers.Concatenate
            ):
                copied_layer = type(layer).from_config(layer.get_config())
                x = layer(
                    [
                        network_dict["new_output"][layer_aux]
                        for layer_aux in network_dict["input"][layer.name]
                    ]
                )
            else:
                copied_layer = type(layer).from_config(layer.get_config())
                if (
                    isinstance(layer, tf.keras.layers.Lambda)
                    or ("lambda" in type(layer).__name__.lower())
                    and "tf.reshape" in layer.name.lower()
                ):
                    x = copied_layer(
                        layer_input[0],
                        shape=(
                            tf.shape(layer_input[0])[0],
                            1,
                            tf.shape(layer_input[0])[1],
                        ),
                    )
                elif (
                    isinstance(layer, tf.keras.layers.Lambda)
                    or ("lambda" in type(layer).__name__.lower())
                    and "tf.broadcast_to" in layer.name.lower()
                ):
                    x = copied_layer(
                        layer_input[2],
                        shape=(
                            layer_input[0],
                            layer_input[3],
                            layer_input[1],
                        ),
                    )
                elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
                    i_name, _, _, extra_args = model.get_config()["layers"][
                        layer_cpt + layer_cpt_shift
                    ]["inbound_nodes"][0][0]
                    new_layer_inputs = {}
                    for i in layer_input:
                        if extra_args["value"][0] in i.name:
                            new_layer_inputs["value"] = i
                        elif extra_args["attention_mask"][0] in i.name:
                            new_layer_inputs["attention_mask"] = i
                    x = copied_layer(
                        query=new_layer_inputs["value"],
                        value=new_layer_inputs["value"],
                        attention_mask=new_layer_inputs["attention_mask"],
                    )
                elif "tf.__operators__.add" in layer.name:
                    x = copied_layer(layer_input[0], layer_input[1])
                else:
                    x = copied_layer(layer_input)
        network_dict["new_output"].update({layer.name: x})
        if layer.name in model_outputnames:
            model_outputs.append(x)
    new_model_inputs = model.inputs
    if model.inputs is None:
        new_model_inputs = network_dict["new_output"][
            network_dict["input"][model.layers[0].name][0]
        ]
    if isinstance(model.inputs, (tuple, list)):
        new_model_inputs = model.inputs
    elif is_transformer:
        new_layer_inputs = network_dict["input"][model.layers[0].name][0]
    new_model = tf.keras.Model(inputs=new_model_inputs, outputs=model_outputs[-1])
    new_model._name = model.name
    unmodified_layers = len(model.layers) - modified_layers

    if verbose:
        print(
            f"\r[\033[96mReplaceLayersByCounterparts\033[0m] ["
            + "\033[96m=\033[0m" * 15
            + "]"
        )
        print("\n\r+" + "-" * 39 + "+")
        print(f'| {f"{model.name}".center(37)} |')
        print("\r+" + "-" * 39 + "+")
        print(f'| perturbated layers          | {f"{modified_layers}":<7} |')
        print(f'| unperturbated layers        | {f"{unmodified_layers}":<7} |')
        print("\r+" + "-" * 39 + "+")
    return new_model
