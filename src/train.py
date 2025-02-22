import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CarvanaDataset  # Dataset class
from model import UNet  # Import U-Net model
import torchvision.transforms as transforms

# 🛠️ Load Config File
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from config
LEARNING_RATE = config["training"]["learning_rate"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🎯 Define Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 📌 Load Dataset
train_dataset = CarvanaDataset(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 🎯 Define Model, Loss, and Optimizer
model = UNet().to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🚀 Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        # Forward pass
        preds = model(images)
        loss = loss_fn(preds, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader)}")

# 🛠️ Save Model
torch.save(model.state_dict(), "unet_carvana.pth")
print("✅ Model saved successfully!")
