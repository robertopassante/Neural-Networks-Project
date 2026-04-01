import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from .preprocessing import rgb_to_class, get_transforms

class DeepGlobeDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir
        self.transform = transform if transform else get_transforms()
        
        # Filter only _sat.jpg images
        all_files = os.listdir(self.train_dir)
        self.image_files = sorted([f for f in all_files if f.endswith('_sat.jpg')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        img_path = os.path.join(self.train_dir, img_name)
        mask_path = os.path.join(self.train_dir, mask_name)
        
        # Load Image and Mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Resize to 224x224 for SatMAE++
        image = image.resize((224, 224), Image.BILINEAR)
        mask = mask.resize((224, 224), Image.NEAREST) 
        
        # Convert RGB Mask to Class Indices
        mask_tensor = torch.from_numpy(rgb_to_class(mask)).long()
        image_tensor = self.transform(image)
            
        return image_tensor, mask_tensor

def extract_dataset(zip_path, extract_to):
    """Utility to extract the dataset if not already present."""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"[*] Dataset already extracted in: {extract_to}")
        return
        
    print(f"[*] Extracting dataset from '{os.path.basename(zip_path)}' to '{os.path.basename(extract_to)}'...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        for member in tqdm(file_list, desc="Extracting dataset", unit="file", leave=True):
            zip_ref.extract(member, extract_to)
            
    print("[*] Extraction completed successfully!\n")

def find_train_dir(base_dir):
    """Search for the 'train' folder anywhere within the extracted dataset."""
    for root, dirs, files in os.walk(base_dir):
        if "train" in dirs:
            return os.path.join(root, "train")
    return None

def get_dataloader(train_dir, batch_size=32, shuffle=True):
    """Initializes the PyTorch Dataloader."""
    dataset = DeepGlobeDataset(train_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=2, 
        pin_memory=True
    )
