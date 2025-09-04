from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Adam

# Chuẩn hóa dữ liệu về tensor
transform = transforms.ToTensor()

# Dataset
train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# DataLoader chia batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
class MNIST_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),  # 1x28x28 -> 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x26x26 -> 32x13x13

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # 32x13x13 -> 64x11x11
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64x11x11 -> 64x5x5

            nn.Flatten(),         # 64*5*5 = 1600
            nn.Linear(64*5*5, 128), 
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.layers(x)

model = MNIST_Classification().to(device)

# Loss và Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # --- Evaluate ---
    model.eval()
    test_loss = 0
    correct = 0
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            test_pred = model(X_batch)
            test_loss += loss_fn(test_pred, y_batch).item()
            correct += (test_pred.argmax(1) == y_batch).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Test Loss: {test_loss/len(test_loader):.4f}, "
          f"Accuracy: {correct/len(test_data):.4f}")
