
import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int) -> None:
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_unit, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_unit * 13 * 13, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        y = self.classifier(self.block_2(self.block_1(x)))
        return y
