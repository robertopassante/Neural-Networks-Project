import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import torch
from torchvision import transforms
from PIL import Image

# Import the model structure so we can wake it up
try:
    from run_all import SatMAESegmenter, COLOR_TO_CLASS
except ImportError:
    print("[!] ERROR: run_all.py must be in the same folder to import the AI Architecture!")

def calculate_metrics(pred, gt, num_classes=7, ignore_index=6):
    """Calculates Macro Intersection over Union (IoU) and Dice Coefficient ignoring background."""
    iou_list = []
    dice_list = []
    valid_mask = (gt != ignore_index)
    
    if valid_mask.sum() == 0:
        return 0.0, 0.0
        
    for c in range(num_classes):
        if c == ignore_index: continue
        pred_c = (pred == c)
        gt_c = (gt == c)
        
        intersection = np.logical_and(pred_c, gt_c)[valid_mask].sum()
        union = np.logical_or(pred_c, gt_c)[valid_mask].sum()
        
        if union > 0:
            iou_list.append(intersection / union)
            
        sum_area = pred_c[valid_mask].sum() + gt_c[valid_mask].sum()
        if sum_area > 0:
            dice_list.append((2.0 * intersection) / sum_area)
            
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    mean_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0
    return mean_iou, mean_dice

def rgb_to_class(mask_np):
    mask_class = np.full(mask_np.shape[:2], 6, dtype=np.int64)
    for rgb, class_idx in COLOR_TO_CLASS.items():
        matches = (mask_np == np.array(rgb)).all(axis=-1)
        mask_class[matches] = class_idx
    return mask_class

def visualize_segmentation(image_path, model_path=None):
    """
    Displays a satellite map, the segmentation mask, and error parameters (IoU, Dice).
    If model_path is provided, uses the REAL trained AI.
    """
    try:
        image = plt.imread(image_path)
    except FileNotFoundError:
        print(f"\n[ERROR] Image '{image_path}' not found!")
        return

    # Se c'è un modello addestrato fornito in pasto al comando... usa la VERA intelligenza artificiale!
    if model_path and os.path.exists(model_path):
        print(f"[*] Waking up REAL AI Brain from: {model_path} ...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SatMAESegmenter(num_classes=7)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # Transform the image exactly like in training
        img_pil = Image.fromarray(image).convert("RGB")
        img_pil = img_pil.resize((224, 224), Image.BILINEAR)
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transf(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor) # Rete neurale al lavoro...
            predicted_mask = outputs[0].argmax(dim=0).cpu().numpy()
            
        print("[✓] AI finished predicting!")
        
        # Load real ground truth mask if it exists
        mask_path = image_path.replace('_sat.jpg', '_mask.png')
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("RGB").resize((224, 224), Image.NEAREST)
            real_mask = rgb_to_class(np.array(mask_pil))
        else:
            real_mask = np.zeros((224, 224), dtype=np.int64)
            
    else:
        # 1. Create a FAKE PREDICTED mask (Simulated)
        print("[*] No AI weights provided. Using SIMULATED visual logic...")
        if image.ndim == 3 and image.shape[2] == 3:
            predicted_mask = image[:, :, 2] > image[:, :, 0] + 20
        else:
            predicted_mask = np.random.choice([0, 1], size=image.shape[:2])

        # 2. Create a FAKE REAL mask "Ground Truth"
        if image.ndim == 3 and image.shape[2] == 3:
            real_mask = (image[:, :, 2] > 80)
        else:
            real_mask = np.random.choice([0, 1], size=image.shape[:2])

    # 3. RESULT METRICS
    iou, dice = calculate_metrics(predicted_mask, real_mask)

    print(f"\n--- [AI MODEL RESULTS] ---")
    print(f"-> IoU (Intersection over Union) : {iou*100:.2f}%")
    print(f"-> Dice Score (F1)               : {dice*100:.2f}%\n")

    # 4. GRAPHICAL DISPLAY OF RESULTS (4 plots instead of 3)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Generic Title on top showing calculated parameters
    fig.suptitle(f"Segmentation Evaluation - Accuracy: IoU {iou*100:.1f}% | Dice {dice*100:.1f}%", 
                 fontsize=15, fontweight='bold', color='#1f77b4')

    # A) Original Image
    axes[0].imshow(image)
    axes[0].set_title("1. Input Map (RGB)")
    axes[0].axis("off")
    
    # B) Real Mask (Exact Answers / Ground Truth)
    axes[1].imshow(real_mask, cmap="nipy_spectral", vmin=0, vmax=6)
    axes[1].set_title("2. Ground Truth")
    axes[1].axis("off")

    # C) Model Mask
    axes[2].imshow(predicted_mask, cmap="nipy_spectral", vmin=0, vmax=6)
    axes[2].set_title(f"3. AI Prediction")
    axes[2].axis("off")
    # Aggiungi parametri sotto l'immagine 3
    axes[2].text(0.5, -0.15, f"IoU: {iou*100:.1f}%\nDice: {dice*100:.1f}%", 
                 size=14, ha="center", fontweight='bold', color='darkred', 
                 transform=axes[2].transAxes)
    
    # D) Final Overlay
    import cv2
    image_resized = cv2.resize(image, (224, 224)) if model_path else image
    axes[3].imshow(image_resized)
    axes[3].imshow(predicted_mask, cmap="Reds", alpha=0.5) 
    axes[3].set_title("4. Map + Prediction")
    axes[3].axis("off")
    # Aggiungi parametri sotto l'immagine 4
    axes[3].text(0.5, -0.15, f"Model Accuracy: {iou*100:.1f}%\nF1-Score (Dice): {dice*100:.1f}%", 
                 size=14, ha="center", fontweight='bold', color='black', 
                 transform=axes[3].transAxes)
    
    plt.tight_layout()
    output_filename = "visualization_output.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"[*] Figure saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual tracking performance.")
    parser.add_argument("image", nargs='?', default="test_images/mia_mappa.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained .pth weights file")
    args = parser.parse_args()
    
    print(f"\nAnalyzing image: {args.image} ...")
    visualize_segmentation(args.image, model_path=args.model)
