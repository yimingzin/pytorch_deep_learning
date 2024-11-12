import torch
import torchvision
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: v2,
        batch_size: int,
        num_workers: int
):
    train_data = datasets.ImageFolder(train_dir, transform, target_transform=None)
    test_data = datasets.ImageFolder(test_dir, transform, target_transform=None)

    class_names, class_to_idx = train_data.classes, train_data.class_to_idx

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names, class_to_idx

"""
train_dir = get_data.image_path / "train"
test_dir = get_data.image_path / "test"

data_transform = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataloader, test_dataloader, class_names, class_to_idx = create_dataloaders(train_dir, test_dir, data_transform, batch_size=32, num_workers=0)

img, label = next(iter(train_dataloader))
print(len(train_dataloader), len(test_dataloader))
print(img.shape)
print(label.shape)
"""