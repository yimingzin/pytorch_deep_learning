from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

def plot_loss_curves(results: Dict[str, List[float]]):

    train_loss, train_acc = results["train_loss"], results["train_acc"]
    test_loss, test_acc = results["test_loss"], results["test_acc"]

    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Train & Test loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, test_acc, label="Test Acc")
    plt.title("Train & Test Acc")
    plt.xlabel("Epochs")
    plt.legend()