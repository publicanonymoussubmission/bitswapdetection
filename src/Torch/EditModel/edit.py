from ..BitSwap import ModuleWrapperBitSwap
from typing import Tuple
import torch


def recursive_setattr(obj, attr, value):
    """
    integrate module to model
    """
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def apply_bitswap_wrapper(
    model: torch.nn.Module,
    verbose: bool = False,
    layer_types_to_watch: Tuple[torch.nn.Module] = (torch.nn.Linear, torch.nn.Conv2d),
) -> torch.nn.Module:
    """
    replaces layers basd on types

    Args:
        model: model to edit
        verbose: verbatim level
        layer_types_to_watch: which layer to wrap
    """
    for layer_cpt, (name, module) in enumerate(tuple(model.named_modules())):
        if verbose:
            print(
                f"\r[\033[96mReplaceLayersByCounterparts\033[0m] ["
                + "\033[91m=\033[0m"
                * int(15 * layer_cpt / len(tuple(model.named_modules())))
                + " "
                * int(15 - int(15 * layer_cpt / len(tuple(model.named_modules()))))
                + "]",
                end="",
            )
        if name:
            if isinstance(module, layer_types_to_watch):
                new_module = ModuleWrapperBitSwap(module_to_wrap=module, name=name)
                recursive_setattr(
                    model,
                    name,
                    new_module,
                )
    if verbose:
        print(
            f"\r[\033[96mReplaceLayersByCounterparts\033[0m] ["
            + "\033[96m=\033[0m" * 15
            + "]"
        )
    return model
