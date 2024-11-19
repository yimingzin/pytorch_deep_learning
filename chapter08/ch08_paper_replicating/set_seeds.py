import torch

def set_seeds(random_seed: int = 42):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)