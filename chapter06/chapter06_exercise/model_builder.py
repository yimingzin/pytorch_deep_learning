import torch
import torchvision
from torchinfo import summary
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model_0 = torchvision.models.efficientnet_b0(weights=weights).to(device)


# Freezing the base model and changing the output layer to suit needs

for params in model_0.parameters():
    params.requires_grad = False

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_0.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, 3),
).to(device)

'''
summary(
    model=model_0,
    input_size=(32, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)
'''