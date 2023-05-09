# BitSwapDetection

<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9, 3.10, 3.11-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/TensorfFlow-2.9.0-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Torch-2.0.0-efefef">
    </a>
</div>
<br>

In this project we provide a benchmarking method for Robust AI inference. The goal of robust AI is to detect bit-swaps which are rare hardware failure that can lead to catastropihc accuracy drops for deep neural networks. 

In order to avoid running every operations twice, we would like to monitor specific layers. Hence, the task at hand consists in ranking layers from most important (should be monitored) to least important (can be left unchecked). 

Assuming that you have a ranking method that takes a model and outputs an ordering for its Dense and Conv layers, this code will allow you to evaluate the ability of this ranking to tackle robust inference AI on ImageNet.

## How to Install

This implementation is compatible for both torch and tensorflow models. It is important to note that the code runs on TensorFlow by default and selects the execution mode based on the package installation. In other words, if you want to run in Tensorflow, you have to isntall it (duh) but if you want to run in Torch you should not install tensorflow at all.
### tensorflow usage
create environment 
```bash
conda create -n bitswap-tf python=3.10 --y
conda activate bitswap-tf
pip install tensorflow
pip install tensorflow-batchnorm-folding
pip install scikit-network
```

### torch usage
create environment 
```bash
conda create -n bitswap-torch python=3.10 --y
conda activate bitswap-torch
pip install torch
pip install torchvision
pip install scikit-image
```


## How to use
In terms of implementation, we consider that a layers ranking is list of integers such that the first value is the rank value of the first layer. You can use this repository for both CLI and as a package.

### CLI
If you want ot run our tests on ResNet50, you can simply run
```bash
python -m src.robustness --model_name resnet --evaluation_step 10
```
In order to a saved `model.pt`. You can run
```bash
python -m src.robustness --model path/to/resnet.pt --evaluation_step 10
```
or
```bash
python -m src.robustness --model path/to/resnet.h5 --tf_preproc tf --evaluation_step 10
```
note that in tenforflow there exists different pre-processing (tf, cafe, torch or identity). you should select the appropriate one as it can drastically impact the model accuracy.
### as a package

Here is an example usage:
```python
from .Torch.ImageNet import imagenet
import torchvision

# ----------- load the model -----------
my_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT
)
if torch.cuda.is_available():
    my_model.cuda()

# ----------- load the model -----------
dataset = imagenet(
    batch_size=16,
    path_to_data=os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
        "Dataset", "ImageNet", "val_set",
    ),
    path_to_labels=os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
        "Dataset", "ImageNet", "validation_label.txt",
    ),
)
# note that you can use your own version of imagenet
# in our implementation, the test set should be stored like this:
#   Dataset/ImageNet/
#                   validation_label.txt
#                   val_set/ * all images sorted

# ----------- evaluation -----------

score, score_scurve = EvaluateRobustness(
    model=my_model,
    dataset=dataset,
    layer_types_to_watch=(torch.nn.Linear, torch.nn.Conv2d),
    verbose=True,
)(evaluation_steps=args.evaluation_step, layers_ranking=list(range(52)))
```