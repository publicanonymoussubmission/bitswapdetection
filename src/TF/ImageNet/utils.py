from typing import Tuple
import sys
from pathlib import Path
import json
import os

if "imagenet_labels_info.txt" in os.listdir(Path(__file__).parents[0]):
    with open(
        os.path.join(Path(__file__).parents[0], "imagenet_labels_info.txt"), "r"
    ) as f:
        reader = f.read()
        labels_wordvec = [elem.split(" ")[0] for elem in reader.splitlines()]
        labels_name = [elem.split(" ")[-1] for elem in reader.splitlines()]
else:
    print("missing ImageNet labels info")
    sys.exit()
fpath = os.path.join(Path(__file__).parents[0], "imagenet_class_index.json")
with open(fpath) as f:
    CLASS_INDEX = json.load(f)
resnet_label_names = [CLASS_INDEX[str(i)][1] for i in range(len(labels_name))]
labels4resnet = {}
for i in range(len(labels_name)):
    correct_index = resnet_label_names.index(labels_name[i])
    labels4resnet[i] = correct_index


def load_list(list_path: str, image_root_path: str) -> Tuple[list, list]:
    """
    This function fetchs the addresses of the images and corresponding labels

    Args:

    """
    images = []
    labels = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.replace("\n", "").split(" ")
            images.append(os.path.join(image_root_path, line[0].replace("'", "")))
            label = int(line[1])
            label = labels4resnet[label]
            labels.append(label)
    return images, labels


def load_list_torch(list_path: str, image_root_path: str) -> Tuple[list, list]:
    """
    This function fetchs the addresses of the images and corresponding labels

    Args:

    """
    images = []
    labels = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.replace("\n", "").split(" ")
            images.append(os.path.join(image_root_path, line[0].replace("'", "")))
            label = int(line[1])
            label = labels4resnet[label]
            labels.append(label)
    return images, labels
