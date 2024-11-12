import torch
from torch import nn
import data_setup, engine, get_data, model_builder, utils
from torchvision.transforms import v2
from timeit import default_timer as timer

args = utils.get_args()
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} and a learning rate of {LEARNING_RATE}")

train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataloader, test_dataloader, class_names, class_to_idx = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE, num_workers=0)
model_0 = model_builder.model_0.to(device)

# loss & optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=LEARNING_RATE)

# train
torch.manual_seed(42)
torch.cuda.manual_seed(42)
start_time = timer()

model_0_results = engine.train(
    model=model_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_func=loss_func,
    optimizer=optimizer,
    device=device,
    epochs=NUM_EPOCHS
)

end_time = timer()

print(f"[INFO] total training time: {end_time - start_time:.3f} seconds")