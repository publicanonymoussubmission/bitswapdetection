if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def int_or_none(val):
        if val is None:
            return val
        try:
            return int(val)
        except:
            raise ValueError

    import argparse

    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="",
        help="model path to use",
    )
    parser.add_argument(
        "--tf_preproc",
        nargs="?",
        type=str,
        default="",
        help="tensorflow preprocessing to use",
    )
    parser.add_argument(
        "--model_name",
        nargs="?",
        type=str,
        default="",
        help="model to use",
    )
    parser.add_argument(
        "--evaluation_step",
        nargs="?",
        type=int_or_none,
        default=None,
        help="model to use",
    )
    args = parser.parse_args()
from .Utils import check_if_a_module_exists

if check_if_a_module_exists(module_name="tensorflow"):
    import tensorflow as tf

    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices("GPU")[0], True
    )
    from batch_normalization_folding.folder import (
        fold_batchnormalization_layers,
    )
    from .TF.EditModel import apply_bitswap_wrapper
    from .TF.Eval import EvaluateModel

    tensorflow_is_loaded = True
else:
    from .Torch.EditModel import apply_bitswap_wrapper
    from .Torch.Eval import EvaluateModel
    from .Torch.BitSwap import ModuleWrapperBitSwap
    import torch

    tensorflow_is_loaded = False
from typing import Any, Tuple, List
import numpy as np


class EvaluateRobustness:
    def __init__(
        self,
        model: Any,
        dataset: Any,
        layer_types_to_watch: Tuple[Any],
        verbose: bool = False,
    ) -> None:
        self.layer_types_to_watch = layer_types_to_watch
        self.verbose = verbose
        self.model = apply_bitswap_wrapper(
            model=model, verbose=verbose, layer_types_to_watch=layer_types_to_watch
        )
        self.model_evaluator = EvaluateModel(
            model=self.model, verbose=self.verbose, dataset=dataset
        )

    def extract_accuracies(
        self, evaluation_steps: int = None, *args: Any, **kwds: Any
    ) -> List[float]:
        accuracies = []
        if tensorflow_is_loaded:
            for layer_cpt, layer in enumerate(self.model.layers):
                if self.verbose:
                    print(
                        f"\r[\033[96mmeasureAccuracies\033[0m] ["
                        + "\033[91m=\033[0m"
                        * int(15 * layer_cpt / len(self.model.layers))
                        + " " * int(15 - int(15 * layer_cpt / len(self.model.layers)))
                        + "]",
                        end="",
                    )
                if "wrapped" in layer.name:
                    perturbated_accuracy = self.model_evaluator(
                        layer_to_activate=layer.name, evaluation_steps=evaluation_steps
                    )
                    accuracies.append(perturbated_accuracy)
        else:
            for layer_cpt, (name, module) in enumerate(
                tuple(self.model.named_modules())
            ):
                if self.verbose:
                    print(
                        f"\r[\033[96mmeasureAccuracies\033[0m] ["
                        + "\033[91m=\033[0m"
                        * int(15 * layer_cpt / len(tuple(self.model.named_modules())))
                        + " "
                        * int(
                            15
                            - int(
                                15 * layer_cpt / len(tuple(self.model.named_modules()))
                            )
                        )
                        + "]",
                        end="",
                    )
                if isinstance(module, ModuleWrapperBitSwap):
                    perturbated_accuracy = self.model_evaluator(
                        layer_to_activate=name, evaluation_steps=evaluation_steps
                    )
                    accuracies.append(perturbated_accuracy)

        if self.verbose:
            print(
                f"\r[\033[96mmeasureAccuracies\033[0m] ["
                + "\033[96m=\033[0m" * 15
                + "]"
            )
        return accuracies

    def compute_score(
        self, layers_ranking: List[int], accuracies: List[float]
    ) -> Tuple[float, List[float]]:
        scores = []
        tmp = []
        for cpt in layers_ranking:
            tmp.append(100 * accuracies[cpt])
            scores.append(np.mean(tmp))
        if self.verbose:
            print(f"score: {np.mean(scores):.5f}")
        return np.mean(scores), scores

    def __call__(
        self,
        evaluation_steps: int = None,
        layers_ranking: List[int] = [],
        *args: Any,
        **kwds: Any,
    ) -> Tuple[float, List[float]]:
        accuracies = self.extract_accuracies(evaluation_steps=evaluation_steps)
        return self.compute_score(
            layers_ranking=layers_ranking,
            accuracies=accuracies,
        )


if __name__ == "__main__":
    from .Utils import remove_chache_folders

    if (args.model_name != "" or args.model != "") and tensorflow_is_loaded:
        from .TF.ImageNet.testset import imagenet
        from .TF.ImageNet.model import classification_model

        my_model = classification_model(
            model_name=args.model_name, model=args.model_name, preproc=args.tf_preproc
        )

        my_model.model, _ = fold_batchnormalization_layers(
            model=my_model.model, verbose=True
        )

        my_model.model.compile(metrics=["accuracy"])
        EvaluateRobustness(
            model=my_model.model,
            dataset=imagenet(
                pre_processing=my_model.preprocess,
                batch_size=16,
                path_to_data=os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    ),
                    "Dataset",
                    "ImageNet",
                    "val_set",
                ),
                path_to_labels=os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    ),
                    "Dataset",
                    "ImageNet",
                    "validation_label.txt",
                ),
            ),
            layer_types_to_watch=(
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
            verbose=True,
        )(evaluation_steps=args.evaluation_step, layers_ranking=list(range(52)))
    elif args.model_name != "" or args.model != "":
        from .Torch.ImageNet import imagenet
        import torchvision

        if args.model != "":
            my_model = torch.load(args.model)
        else:
            my_model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
        if torch.cuda.is_available():
            my_model.cuda()
        EvaluateRobustness(
            model=my_model,
            dataset=imagenet(
                batch_size=16,
                path_to_data=os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    ),
                    "Dataset",
                    "ImageNet",
                    "val_set",
                ),
                path_to_labels=os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    ),
                    "Dataset",
                    "ImageNet",
                    "validation_label.txt",
                ),
            ),
            layer_types_to_watch=(torch.nn.Linear, torch.nn.Conv2d),
            verbose=True,
        )(evaluation_steps=args.evaluation_step, layers_ranking=list(range(52)))
    remove_chache_folders()
