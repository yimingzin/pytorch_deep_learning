
import os
import argparse
import torch
from torchvision.transforms import v2
import data_setup, engine, model_builder, utils

parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument("--num_epochs", default=10, type=int, help="the number of epochs to train for")
parser.add_argument("--batch_size", default=32, type=int, help="the number of samples per batch")
parser.add_argument("--hidden_units", default=10, type=int, help="the number of hidden units in hidden layers")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate to use for model")
parser.add_argument("--train_dir", default="data/pizza_sushi_steak/train", type=str, help="directory file path to training data in standard image classification format")
parser.add_argument("--test_dir", default="data/pizza_sushi_steak/test", type=str, help="directory file path to testing data in standard image classification format")

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

if __name__ == '__main__':
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE)

    model = model_builder.TinyVGG(input_shape=3, hidden_unit=HIDDEN_UNITS, output_shape=len(class_names)).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, NUM_EPOCHS, device)
    
    utils.save_model(model, "models", "05_going_modular_script_mode_tinyvgg_model.pth")
