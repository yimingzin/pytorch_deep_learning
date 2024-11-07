
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    
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
) -> Tuple[float, float]:
    
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
            correct_samples += (y == test_preds).sum().item()
            total_samples += len(y)
        
        test_loss = test_loss / len(dataloader)
        test_acc = correct_samples / total_samples
    
    return test_loss, test_acc

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device
) -> Dict[str, List]:
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_func, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_func, device)
        
        print(
            f"Epochs: {epoch + 1} |"
            f"Train_loss: {train_loss:.4f} |"
            f"Train_acc: {train_acc:.2f} |"
            f"Test_loss: {test_loss:.4f} |"
            f"Test_acc: {test_acc:.2f} |"
        )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results
