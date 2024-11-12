import argparse
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from typing import Dict, List, Tuple
from pathlib import Path
from torchvision.transforms import v2

def get_args():
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train", help="Train image path")
    parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test", help="Test image path")

    return parser.parse_args()


def plot_loss_curve(results: Dict[str, List[float]]):
    train_loss, train_acc = results["train_loss"], results["train_acc"]
    test_loss, test_acc = results["test_loss"], results["test_acc"]

    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Train & Test Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, test_acc, label="Test Acc")
    plt.title("Train & Test Acc")
    plt.xlabel("Epochs")
    plt.legend()


def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str,
):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pt") or model_name.endswith(".pth"), "model_name should endswith '.pt' or '.pth'"
    model_sava_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_sava_path}")
    torch.save(model.state_dict(), f=model_sava_path)

def pred_and_plot_image(
        model: torch.nn.Module,
        image_path: str,
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        transform: v2 = None,
        device: torch.device = "cpu"
):
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = v2.Compose([
            v2.Resize(size=(image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);