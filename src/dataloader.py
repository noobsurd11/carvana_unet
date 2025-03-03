import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio  
import torchvision.transforms as transforms   
from PIL import Image

# nikhil lvda

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = image_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", "_mask.gif"))

        
        image = Image.open(img_path).convert("RGB")
        mask = imageio.imread(mask_path)
        mask = Image.fromarray(mask).convert("L") 
        mask = mask.resize((256, 256))  
        mask = np.array(mask, dtype=np.float32)  # Convert to NumPy

        
        mask = (mask > 128).astype(np.float32)  # Thresholding

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask).unsqueeze(0)  # Add channel dimension

        return image, mask

# Define transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize for U-Net
#     transforms.ToTensor(),  # Convert to tensor (0-1 range)


# Create Dataset Objects
# train_dataset = CarvanaDataset(image_dir="dataloader/train",
#                                mask_dir="dataloader/train_masks",
#                                transform=transform)

# test_dataset = CarvanaDataset(image_dir="dataloader/train",
#                               mask_dir="dataloader/train_masks",  # No test masks available
#                               transform=transform)

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # Check sample output
# if __name__ == "__main__":
#     img, mask = train_dataset[0]
#     print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
