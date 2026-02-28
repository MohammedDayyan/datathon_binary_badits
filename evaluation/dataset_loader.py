"""
Dataset Loader for NTIRE Dehazing Challenge Datasets
Supports I-Haze, N-Haze, and Dense-Haze datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Dict
import json
import cv2

class NTIREDehazeDataset(Dataset):
    """
    Dataset class for NTIRE dehazing datasets
    Supports I-Haze, N-Haze, and Dense-Haze formats
    """
    
    def __init__(self, 
                 root_dir: str,
                 dataset_type: str = "I-Haze",
                 split: str = "test",
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Root directory of the dataset
            dataset_type: Type of dataset ("I-Haze", "N-Haze", "Dense-Haze")
            split: Dataset split ("train", "test", "val")
            transform: Transform for hazy images
            target_transform: Transform for clear images
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Default transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        if self.target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Load image paths
        self.image_pairs = self._load_image_paths()
        
    def _load_image_paths(self) -> List[Tuple[str, str]]:
        """Load paths of hazy and clear image pairs"""
        image_pairs = []
        
        # Dataset-specific directory structures
        if self.dataset_type == "I-Haze":
            # I-Haze structure: I-Haze/hazy/ and I-Haze/clear/
            hazy_dir = os.path.join(self.root_dir, "I-Haze", self.split, "hazy")
            clear_dir = os.path.join(self.root_dir, "I-Haze", self.split, "clear")
            
        elif self.dataset_type == "N-Haze":
            # N-Haze structure: N-Haze/hazy/ and N-Haze/clear/
            hazy_dir = os.path.join(self.root_dir, "N-Haze", self.split, "hazy")
            clear_dir = os.path.join(self.root_dir, "N-Haze", self.split, "clear")
            
        elif self.dataset_type == "Dense-Haze":
            # Dense-Haze structure: Dense-Haze/hazy/ and Dense-Haze/GT/
            hazy_dir = os.path.join(self.root_dir, "Dense-Haze", self.split, "hazy")
            clear_dir = os.path.join(self.root_dir, "Dense-Haze", self.split, "GT")
            
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Check if directories exist
        if not os.path.exists(hazy_dir):
            raise FileNotFoundError(f"Hazy images directory not found: {hazy_dir}")
        if not os.path.exists(clear_dir):
            raise FileNotFoundError(f"Clear images directory not found: {clear_dir}")
        
        # Get all hazy images
        hazy_images = [f for f in os.listdir(hazy_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Match with clear images
        for hazy_img in hazy_images:
            hazy_path = os.path.join(hazy_dir, hazy_img)
            
            # Find corresponding clear image
            clear_img = self._find_corresponding_clear_image(hazy_img, clear_dir)
            if clear_img:
                clear_path = os.path.join(clear_dir, clear_img)
                image_pairs.append((hazy_path, clear_path))
            else:
                print(f"Warning: Could not find clear image for {hazy_img}")
        
        print(f"Loaded {len(image_pairs)} image pairs from {self.dataset_type} {self.split}")
        return image_pairs
    
    def _find_corresponding_clear_image(self, hazy_filename: str, clear_dir: str) -> Optional[str]:
        """Find the corresponding clear image for a hazy image"""
        # Remove haze-specific prefixes/suffixes
        base_name = hazy_filename
        
        # Common patterns in NTIRE datasets
        patterns_to_remove = ["hazy_", "haze_", "hazy-", "haze-", "_hazy", "_haze"]
        
        for pattern in patterns_to_remove:
            if pattern in base_name:
                base_name = base_name.replace(pattern, "")
        
        # Try different extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        base_name_no_ext = os.path.splitext(base_name)[0]
        
        for ext in extensions:
            clear_candidate = base_name_no_ext + ext
            if os.path.exists(os.path.join(clear_dir, clear_candidate)):
                return clear_candidate
        
        # If not found, try exact match
        if os.path.exists(os.path.join(clear_dir, hazy_filename)):
            return hazy_filename
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a hazy-clear image pair"""
        hazy_path, clear_path = self.image_pairs[idx]
        
        # Load images
        hazy_image = Image.open(hazy_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            hazy_image = self.transform(hazy_image)
        if self.target_transform:
            clear_image = self.target_transform(clear_image)
        
        return hazy_image, clear_image
    
    def get_image_info(self, idx: int) -> Dict:
        """Get information about an image pair"""
        hazy_path, clear_path = self.image_pairs[idx]
        
        # Get image dimensions
        hazy_img = Image.open(hazy_path)
        clear_img = Image.open(clear_path)
        
        return {
            'hazy_path': hazy_path,
            'clear_path': clear_path,
            'hazy_size': hazy_img.size,
            'clear_size': clear_img.size,
            'filename': os.path.basename(hazy_path)
        }

class DatasetManager:
    """
    Manager for handling multiple NTIRE datasets
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.datasets = {}
        self.dataset_info = {}
        
    def load_dataset(self, 
                    dataset_type: str, 
                    split: str = "test",
                    batch_size: int = 1,
                    num_workers: int = 4) -> DataLoader:
        """Load a specific dataset"""
        
        # Create dataset
        dataset = NTIREDehazeDataset(
            root_dir=self.root_dir,
            dataset_type=dataset_type,
            split=split
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Store dataset info
        self.datasets[f"{dataset_type}_{split}"] = dataset
        self.dataset_info[f"{dataset_type}_{split}"] = {
            'type': dataset_type,
            'split': split,
            'size': len(dataset),
            'batch_size': batch_size
        }
        
        return dataloader
    
    def get_all_datasets(self, 
                        splits: List[str] = ["test"],
                        batch_size: int = 1) -> Dict[str, DataLoader]:
        """Load all available datasets"""
        dataloaders = {}
        dataset_types = ["I-Haze", "N-Haze", "Dense-Haze"]
        
        for dataset_type in dataset_types:
            for split in splits:
                try:
                    key = f"{dataset_type}_{split}"
                    dataloaders[key] = self.load_dataset(
                        dataset_type, split, batch_size
                    )
                    print(f"✓ Loaded {key}")
                except FileNotFoundError as e:
                    print(f"✗ Could not load {key}: {e}")
                except Exception as e:
                    print(f"✗ Error loading {key}: {e}")
        
        return dataloaders
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics for all loaded datasets"""
        stats = {
            'total_datasets': len(self.datasets),
            'total_images': sum(len(dataset) for dataset in self.datasets.values()),
            'datasets': self.dataset_info
        }
        return stats
    
    def create_sample_pairs(self, 
                          dataset_type: str, 
                          split: str = "test",
                          num_samples: int = 5,
                          output_dir: str = "samples") -> List[str]:
        """Create sample image pairs for visualization"""
        
        dataset_key = f"{dataset_type}_{split}"
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} not loaded")
        
        dataset = self.datasets[dataset_key]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sample_files = []
        
        for i in range(min(num_samples, len(dataset))):
            hazy_img, clear_img = dataset[i]
            
            # Convert tensors back to PIL images
            hazy_pil = transforms.ToPILImage()(hazy_img)
            clear_pil = transforms.ToPILImage()(clear_img)
            
            # Get filename
            info = dataset.get_image_info(i)
            base_name = os.path.splitext(info['filename'])[0]
            
            # Save images
            hazy_path = os.path.join(output_dir, f"{base_name}_hazy.png")
            clear_path = os.path.join(output_dir, f"{base_name}_clear.png")
            
            hazy_pil.save(hazy_path)
            clear_pil.save(clear_path)
            
            sample_files.extend([hazy_path, clear_path])
        
        print(f"Created {len(sample_files)} sample images in {output_dir}")
        return sample_files

# Utility functions
def validate_dataset_structure(root_dir: str, dataset_type: str) -> bool:
    """Validate if dataset has the expected structure"""
    
    if dataset_type == "I-Haze":
        required_dirs = ["I-Haze/test/hazy", "I-Haze/test/clear"]
    elif dataset_type == "N-Haze":
        required_dirs = ["N-Haze/test/hazy", "N-Haze/test/clear"]
    elif dataset_type == "Dense-Haze":
        required_dirs = ["Dense-Haze/test/hazy", "Dense-Haze/test/GT"]
    else:
        return False
    
    for dir_path in required_dirs:
        full_path = os.path.join(root_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing directory: {full_path}")
            return False
    
    return True

def download_dataset_info():
    """Print information about downloading NTIRE datasets"""
    info = """
    NTIRE Dataset Download Information:
    ==================================
    
    I-Haze (Indoor Haze):
    - Download from: https://data.vision.ee.ethz.ch/cvl/ntire19//i-haze/
    - Contains: 35 training pairs, 5 validation pairs, 5 test pairs
    - Resolution: Various indoor scenes with homogeneous haze
    
    N-Haze (Natural Haze):
    - Download from: https://data.vision.ee.ethz.ch/cvl/ntire19//n-haze/
    - Contains: 50 training pairs, 5 validation pairs, 5 test pairs
    - Resolution: Outdoor scenes with natural haze
    
    Dense-Haze:
    - Download from: https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/
    - Contains: 55 training pairs, 5 validation pairs, 5 test pairs
    - Resolution: Dense haze conditions with high concentration
    
    Dataset Structure:
    ------------------
    After downloading, organize as follows:
    
    data/
    ├── I-Haze/
    │   ├── train/
    │   │   ├── hazy/
    │   │   └── clear/
    │   └── test/
    │       ├── hazy/
    │       └── clear/
    ├── N-Haze/
    │   ├── train/
    │   │   ├── hazy/
    │   │   └── clear/
    │   └── test/
    │       ├── hazy/
    │       └── clear/
    └── Dense-Haze/
        ├── train/
        │   ├── hazy/
        │   └── GT/
        └── test/
            ├── hazy/
            └── GT/
    """
    
    print(info)

# Quick usage function
def quick_dataset_load(root_dir: str, dataset_type: str = "I-Haze"):
    """Quick dataset loading for testing"""
    manager = DatasetManager(root_dir)
    
    try:
        dataloader = manager.load_dataset(dataset_type, "test", batch_size=1)
        print(f"Successfully loaded {dataset_type} test dataset")
        return dataloader
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
