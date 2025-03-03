import os
import numpy as np
from PIL import Image


IMAGE_DIR = "data/train"  
MASK_DIR = "data/train_masks"    
PRED_DIR = "predictions"       


all_images = sorted(os.listdir(IMAGE_DIR))
car_ids = sorted(set(img[:12] for img in all_images))  
test_cars = car_ids[254:] 


all_masks = sorted(os.listdir(MASK_DIR))
test_masks = [mask for mask in all_masks if mask[:12] in test_cars]


def dice_score(pred, target, smooth=1e-6):
    """
    Computes the Dice Score between predicted and ground truth masks.
    """
    pred = pred > 0.5  
    target = target > 0.5 

    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()

    return (2. * intersection + smooth) / (union + smooth)


dice_scores = []


for mask_filename in test_masks:

    mask_path = os.path.join(MASK_DIR, mask_filename)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((256, 256), Image.NEAREST) 
    mask = np.array(mask) / 255.0 

    
    pred_filename = mask_filename.replace("_mask.gif", "_pred.png")
    pred_path = os.path.join(PRED_DIR, pred_filename)


    if not os.path.exists(pred_path):
        print(f"❌ Missing prediction for {mask_filename}")
        continue


    pred = Image.open(pred_path).convert("L")
    pred = np.array(pred) / 255.0  


    score = dice_score(pred, mask)
    dice_scores.append(score)


if dice_scores:
    mean_dice = np.mean(dice_scores)
    print(f"✅ Mean Dice Score (Accuracy): {mean_dice:.4f}")
else:
    print("⚠ No valid predictions found. Check paths and filenames!")
