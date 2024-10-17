import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set wight & bias
weight = 0.3
bias = 0.9

X = torch.arange(start=0, end=1, step=0.01).unsqueeze(dim=1)
y = weight * X + bias

print(X.shape)
print(y.shape)

data_split = int(0.8 * len(X))
X_train, y_train = X[:data_split], y[:data_split]
X_test, y_test = X[data_split:], y[data_split:]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

def plot_prediction(
        train_data = X_train,
        train_label = y_train,
        test_data = X_test,
        test_label = y_test,
        prediction = None
):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_label, s=5, c="b", label="Training data")
    plt.scatter(test_data, test_label, s=5, c="g", label="Test data")

    if prediction is not None:
        plt.scatter(test_data, prediction, s=5, c="r", label="Predictions")

    plt.show()

plot_prediction()

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        y = self.weight * x + self.bias
        return y

class LinearRegression_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

torch.manual_seed(42)

model_1 = LinearRegression().to(device)
print(f"MODEL_1 : {model_1.state_dict()}")

model_2 = LinearRegression_V2().to(device)
print(model_2.state_dict())

for param_name, param_tensor in model_2.state_dict().items():
    print(f"{param_name} : {param_tensor.shape}")

loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=1e-2)


def train(X_train, y_train, model, loss_func, optimizer):

    model.train()
    X_train, y_train = X_train.to(device), y_train.to(device)

    train_pred = model(X_train)
    loss = loss_func(train_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def test(X_test, y_test, model, loss_func):

    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_func(test_pred, y_test)

    return test_loss

epochs = 300
for epoch in range(epochs):
    train_loss = train(X_train, y_train, model_1, loss_func, optimizer)
    if epoch % 20 == 0:
        test_loss = test(X_test, y_test, model_1, loss_func)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f}")

print(f"MODEL_1 : {model_1.state_dict()}")

model_1.eval()
with torch.inference_mode():
    X_test = X_test.to(device)
    pred = model_1(X_test)
print(pred)

plot_prediction(prediction=pred.cpu())


MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "chapter01_test_model_1.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(model_1.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved in Path: {MODEL_SAVE_PATH}")

model_new = LinearRegression().to(device)
model_new.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

model_new.eval()
with torch.inference_mode():
    pred_new = model_new(X_test)
    print(pred_new == pred)

model_new.state_dict()

print(f"MODEL_NEW : {model_new.state_dict()}")