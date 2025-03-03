import yaml
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CarvanaDataset  
from model import UNet  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image


CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
BATCH_SIZE = config["testing"]["batch_size"]
MODEL_PATH = config["testing"]["model_path"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

IMAGE_DIR = "data/train"
MASK_DIR = "data/train_masks"


all_images = sorted(os.listdir(IMAGE_DIR))
car_ids = sorted(set(img[:12] for img in all_images))  
test_cars = car_ids[254:]  # Next 64 cars for testing
test_images = [img for img in all_images if img[:12] in test_cars]


test_dataset = CarvanaDataset(IMAGE_DIR, MASK_DIR, image_list=test_images, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict_mask(image):
    image = image.to(DEVICE)  
    with torch.no_grad():
        pred = model(image)  
    pred = torch.sigmoid(pred)  
    pred = (pred > 0.5).float()  
    return pred

output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

test_image_filenames = sorted(test_images) 
# nikhil lvda

for i, (images, masks) in enumerate(test_loader):  
    images = images.to(DEVICE) 
    pred_masks = predict_mask(images)  
  
    for j in range(images.shape[0]):  
        image_filename = test_image_filenames[i * BATCH_SIZE + j]
        
       
        image_id = os.path.splitext(image_filename)[0]
        pred_filename = f"{image_id}_pred.png"

        # Convert tensor to numpy array
        pred_mask_np = pred_masks[j].squeeze().cpu().numpy()

        # Normalize to uint8 (0-255)
        pred_mask_np = (pred_mask_np * 255).astype(np.uint8)

        filepath = os.path.join(output_dir, pred_filename)
        save_image(torch.tensor(pred_mask_np / 255.0), filepath)

print("Testing completed! Predictions saved in the 'predictions/' folder.")
