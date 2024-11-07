
import os
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: v2,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    
    train_data = datasets.ImageFolder(
        train_dir,
        transform,
    )
    
    test_data = datasets.ImageFolder(
        test_dir,
        transform
    )
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names
