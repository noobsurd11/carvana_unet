import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CarvanaDataset  # Dataset class
from model import UNet, BCEWithDiceLoss  # Import U-Net model
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


# Paths
IMAGE_DIR = "data/train"
MASK_DIR = "data/train_masks"

# Get all image filenames
all_images = sorted(os.listdir(IMAGE_DIR))  # Ensure sorted order for consistent splitting

# Extract unique car IDs (first 12 characters of filename)
car_ids = sorted(set(img[:12] for img in all_images))  

# Define dataset splits
train_cars = car_ids[:254]  # First 222 cars



# Select images belonging to these car IDs
train_images = [img for img in all_images if img[:12] in train_cars]



# Define transformations
my_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for U-Net
    transforms.ToTensor(),  # Convert to tensor (0-1 range)
])

# Create datasets
train_dataset = CarvanaDataset(IMAGE_DIR, MASK_DIR, image_list=train_images, transform=my_transform)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Print dataset sizes
print(f"Train: {len(train_dataset)} images")

# 🎯 Define Model, Loss, and Optimizer
model = UNet().to(DEVICE)
loss_fn = BCEWithDiceLoss(alpha=0.7)  # 70% BCE, 30% Dice
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
