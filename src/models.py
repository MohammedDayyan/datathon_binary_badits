"""
Image Dehazing Models Implementation
Includes multiple state-of-the-art architectures for image dehazing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from typing import Tuple, Optional

class DehazeNet(nn.Module):
    """
    DehazeNet: An End-to-End System for Single Image Haze Removal
    Original paper: https://arxiv.org/abs/1606.07874
    """
    def __init__(self):
        super(DehazeNet, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        # Output layer
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        
        # Pooling
        x4 = self.pool(x3)
        
        # Deconvolution
        x5 = self.relu(self.deconv1(x4))
        x6 = self.relu(self.deconv2(x5))
        
        # Output
        out = self.sigmoid(self.conv_out(x6))
        
        return out

class AODNet(nn.Module):
    """
    All-in-One Dehazing Network
    Original paper: https://arxiv.org/abs/1705.02809
    """
    def __init__(self):
        super(AODNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=True)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat((x3, x4), 1)
        x5 = self.relu(self.conv5(cat3))
        
        # K(x) output
        k = self.conv5(cat3)
        
        # Restoration formula: J(x) = K(x) * I(x) - K(x) + b
        output = k * x - k + 1
        
        return torch.clamp(output, 0, 1)

class MSBDN(nn.Module):
    """
    Multi-Scale Boosted Dehazing Network
    Simplified version for practical implementation
    """
    def __init__(self):
        super(MSBDN, self).__init__()
        
        # Encoder
        self.encoder1 = self._make_layer(3, 64, 2)
        self.encoder2 = self._make_layer(64, 128, 2)
        self.encoder3 = self._make_layer(128, 256, 2)
        
        # Decoder
        self.decoder3 = self._make_layer(256, 128, 2, upsample=True)
        self.decoder2 = self._make_layer(128, 64, 2, upsample=True)
        self.decoder1 = self._make_layer(64, 3, 1, upsample=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(256 // 16, 256, 1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, upsample=False):
        layers = []
        
        if upsample:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Attention
        att = self.attention(e3)
        e3_att = e3 * att
        
        # Decoder path
        d3 = self.decoder3(e3_att)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        
        return torch.clamp(d1, 0, 1)

class DehazeModelManager:
    """
    Model manager for loading and managing different dehazing models
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def load_model(self, model_name: str, model_path: Optional[str] = None):
        """Load a specific dehazing model"""
        if model_name.lower() == 'dehazenet':
            model = DehazeNet()
        elif model_name.lower() == 'aodnet':
            model = AODNet()
        elif model_name.lower() == 'msbdn':
            model = MSBDN()
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        model.to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded pretrained weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {model_path}: {e}")
                print("Using randomly initialized weights")
        else:
            print(f"No pretrained weights found, using random initialization")
            
        model.eval()
        self.models[model_name.lower()] = model
        return model
        
    def get_model(self, model_name: str):
        """Get a loaded model"""
        if model_name.lower() not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Use load_model() first.")
        return self.models[model_name.lower()]
        
    def dehaze_image(self, model_name: str, image_path: str, save_path: Optional[str] = None):
        """Dehaze a single image using specified model"""
        model = self.get_model(model_name)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform to tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # Convert back to PIL image
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_image = output_image.resize(original_size, Image.LANCZOS)
        
        # Save if path provided
        if save_path:
            output_image.save(save_path)
            
        return output_image
        
    def dehaze_batch(self, model_name: str, input_dir: str, output_dir: str):
        """Dehaze all images in a directory"""
        model = self.get_model(model_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"dehazed_{filename}")
                
                try:
                    self.dehaze_image(model_name, input_path, output_path)
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Utility functions
def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image for model input"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor: torch.Tensor, save_path: str):
    """Save tensor as image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(save_path)

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM between two images (simplified version)"""
    from skimage.metrics import structural_similarity as ssim
    
    # Convert tensors to numpy arrays
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Calculate SSIM for each channel and average
    ssim_values = []
    for i in range(3):
        ssim_val = ssim(img1_np[:, :, i], img2_np[:, :, i], data_range=1.0)
        ssim_values.append(ssim_val)
        
    return np.mean(ssim_values)
