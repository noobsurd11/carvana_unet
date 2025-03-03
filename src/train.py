import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CarvanaDataset 
from model import UNet, BCEWithDiceLoss 
import torchvision.transforms as transforms

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


LEARNING_RATE = config["training"]["learning_rate"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


IMAGE_DIR = "data/train"
MASK_DIR = "data/train_masks"

all_images = sorted(os.listdir(IMAGE_DIR))  


car_ids = sorted(set(img[:12] for img in all_images))  

train_cars = car_ids[:254]  # First 222 cars

train_images = [img for img in all_images if img[:12] in train_cars]




my_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),  # Convert to tensor
])


train_dataset = CarvanaDataset(IMAGE_DIR, MASK_DIR, image_list=train_images, transform=my_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


print(f"Train: {len(train_dataset)} images")

model = UNet().to(DEVICE)
loss_fn = BCEWithDiceLoss(alpha=0.7)  # 70% BCE, 30% Dice
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

       
        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader)}")

torch.save(model.state_dict(), "unet_carvana.pth")
print("✅ Model saved successfully!")

# nikhil lvda