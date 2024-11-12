import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from tqdm.auto import tqdm

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    model.train()

    train_loss, train_acc = 0, 0
    correct_samples, total_samples = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        train_logits = model(X)
        loss = loss_func(train_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_probs = torch.softmax(train_logits, dim=1)
        train_preds = torch.argmax(train_probs, dim=1)
        correct_samples += (train_preds == y).sum().item()
        total_samples += len(y)

    train_loss = train_loss / len(dataloader)
    train_acc = correct_samples / total_samples

    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        device: torch.device
):
    model.eval()

    test_loss, test_acc = 0, 0
    correct_samples, total_samples = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_logits = model(X)
            loss = loss_func(test_logits, y)
            test_loss += loss.item()

            test_preds = torch.argmax(test_logits, dim=1)
            correct_samples += (test_preds == y).sum().item()
            total_samples += len(y)

        test_loss = test_loss / len(y)
        test_acc = correct_samples / total_samples

    return test_loss, test_acc

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int
) -> Dict[str, list]:

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader,
                                           loss_func=loss_func, optimizer=optimizer, device=device)

        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_func=loss_func, device=device)

        print(
            f"Epochs: {epoch + 1} / {epochs} |" 
            f"Train Loss: {train_loss:.4f} |"
            f"Train Acc: {train_acc:.2f} |"
            f"Test Loss: {test_loss:.4f} |"
            f"Test Acc: {test_acc:.2f} |"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

