from ...TF.ImageNet.utils import load_list_torch
from torchvision import transforms
from torchvision import transforms
import torch
from skimage import io
import numpy as np
import os


class FaceLandmarksDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_to_data: str, path_to_labels: str, transform=None):
        self.transform = transform
        self.path_to_data = path_to_data
        self.path_to_labels = path_to_labels

        self.images, self.labels = load_list_torch(path_to_labels, path_to_data)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = io.imread(img_name)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.tile(image, reps=[1, 1, 3])
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        image = torch.transpose(image, dim0=0, dim1=2)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        sample = {"images": image, "labels": labels}

        return sample


def imagenet(
    batch_size: int,
    path_to_data: str,
    path_to_labels: str,
) -> torch.utils.data.DataLoader:
    """
    This function creates the data iterator for test set

    Args:
        batch_size: number of elements per batch
        path_to_data: path to images
        path_to_labels: path to the labels json
    """
    test_transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=None),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return torch.utils.data.DataLoader(
        FaceLandmarksDataset(
            path_to_data=path_to_data,
            path_to_labels=path_to_labels,
            transform=test_transform,
        ),
        batch_size=batch_size,
        num_workers=8,
    )
