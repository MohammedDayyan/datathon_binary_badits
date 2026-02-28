"""
Evaluation Metrics for Image Dehazing
Implements PSNR, SSIM, LPIPS, and other quality metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import os

class DehazeMetrics:
    """
    Comprehensive metrics for dehazing evaluation
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize LPIPS model
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
        except Exception as e:
            print(f"LPIPS not available: {e}")
            self.lpips_available = False
    
    def calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            pred: Predicted image tensor (B, C, H, W) or (C, H, W)
            target: Ground truth image tensor (B, C, H, W) or (C, H, W)
            
        Returns:
            PSNR value in dB
        """
        # Ensure tensors are in the same range [0, 1]
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Calculate MSE
        mse = F.mse_loss(pred, target)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return psnr_value.item()
    
    def calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            pred: Predicted image tensor (B, C, H, W) or (C, H, W)
            target: Ground truth image tensor (B, C, H, W) or (C, H, W)
            
        Returns:
            SSIM value (0 to 1, higher is better)
        """
        # Remove batch dimension if present
        if pred.dim() == 4:
            pred = pred.squeeze(0)
        if target.dim() == 4:
            target = target.squeeze(0)
        
        # Convert to numpy arrays
        pred_np = pred.permute(1, 2, 0).cpu().numpy()
        target_np = target.permute(1, 2, 0).cpu().numpy()
        
        # Calculate SSIM for each channel and average
        ssim_values = []
        for i in range(3):
            ssim_val = ssim(
                pred_np[:, :, i], 
                target_np[:, :, i], 
                data_range=1.0
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS)
        
        Args:
            pred: Predicted image tensor (B, C, H, W) or (C, H, W)
            target: Ground truth image tensor (B, C, H, W) or (C, H, W)
            
        Returns:
            LPIPS value (lower is better, 0 to 1)
        """
        if not self.lpips_available:
            return None
        
        # Ensure tensors are in range [-1, 1] for LPIPS
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0
        
        # Add batch dimension if needed
        if pred_norm.dim() == 3:
            pred_norm = pred_norm.unsqueeze(0)
        if target_norm.dim() == 3:
            target_norm = target_norm.unsqueeze(0)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(pred_norm, target_norm)
        
        return lpips_value.item()
    
    def calculate_mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        Args:
            pred: Predicted image tensor
            target: Ground truth image tensor
            
        Returns:
            MAE value
        """
        mae = F.l1_loss(pred, target)
        return mae.item()
    
    def calculate_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error (MSE)
        
        Args:
            pred: Predicted image tensor
            target: Ground truth image tensor
            
        Returns:
            MSE value
        """
        mse = F.mse_loss(pred, target)
        return mse.item()
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            pred: Predicted image tensor
            target: Ground truth image tensor
            
        Returns:
            Dictionary of all metric values
        """
        metrics = {}
        
        # Basic metrics
        metrics['psnr'] = self.calculate_psnr(pred, target)
        metrics['ssim'] = self.calculate_ssim(pred, target)
        metrics['mae'] = self.calculate_mae(pred, target)
        metrics['mse'] = self.calculate_mse(pred, target)
        
        # Perceptual metrics
        lpips_value = self.calculate_lpips(pred, target)
        if lpips_value is not None:
            metrics['lpips'] = lpips_value
        
        return metrics
    
    def evaluate_dataset(self, predictions: List[torch.Tensor], 
                        targets: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics on a dataset
        
        Args:
            predictions: List of predicted image tensors
            targets: List of ground truth image tensors
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions and targets must match")
        
        # Calculate metrics for each image
        all_metrics = []
        for pred, target in zip(predictions, targets):
            metrics = self.calculate_all_metrics(pred, target)
            all_metrics.append(metrics)
        
        # Aggregate statistics
        results = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if m[metric_name] is not None]
            
            if values:
                results[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return results

class HallucinationDetector:
    """
    Detect hallucinations in dehazed images
    """
    
    def __init__(self):
        self.metrics = DehazeMetrics()
    
    def detect_structure_changes(self, original: torch.Tensor, 
                                dehazed: torch.Tensor,
                                ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Detect structural changes that might indicate hallucination
        
        Args:
            original: Original hazy image
            dehazed: Dehazed image
            ground_truth: Ground truth clear image
            
        Returns:
            Dictionary of hallucination indicators
        """
        indicators = {}
        
        # Edge preservation
        original_edges = self._detect_edges(original)
        dehazed_edges = self._detect_edges(dehazed)
        gt_edges = self._detect_edges(ground_truth)
        
        # Calculate edge preservation ratios
        edge_preservation_orig = self._calculate_edge_similarity(original_edges, gt_edges)
        edge_preservation_dehazed = self._calculate_edge_similarity(dehazed_edges, gt_edges)
        
        indicators['edge_preservation_improvement'] = edge_preservation_dehazed - edge_preservation_orig
        
        # Texture similarity
        texture_similarity_orig = self._calculate_texture_similarity(original, ground_truth)
        texture_similarity_dehazed = self._calculate_texture_similarity(dehazed, ground_truth)
        
        indicators['texture_similarity_change'] = texture_similarity_dehazed - texture_similarity_orig
        
        # Color consistency
        color_consistency = self._calculate_color_consistency(dehazed, ground_truth)
        indicators['color_consistency'] = color_consistency
        
        # Overall hallucination score (lower is better)
        indicators['hallucination_score'] = self._calculate_hallucination_score(indicators)
        
        return indicators
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """Detect edges using Sobel operator"""
        # Convert to grayscale
        if image.dim() == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image.squeeze()
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        
        return edges.squeeze()
    
    def _calculate_edge_similarity(self, edges1: torch.Tensor, edges2: torch.Tensor) -> float:
        """Calculate similarity between edge maps"""
        # Normalize edge maps
        edges1_norm = edges1 / (edges1.max() + 1e-8)
        edges2_norm = edges2 / (edges2.max() + 1e-8)
        
        # Calculate correlation
        correlation = F.cosine_similarity(
            edges1_norm.flatten(), 
            edges2_norm.flatten(), 
            dim=0
        )
        
        return correlation.item()
    
    def _calculate_texture_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate texture similarity using gradient statistics"""
        # Calculate gradients
        def get_gradients(img):
            grad_x = torch.diff(img, dim=-1)
            grad_y = torch.diff(img, dim=-2)
            return torch.cat([grad_x.flatten(), grad_y.flatten()])
        
        grad1 = get_gradients(img1)
        grad2 = get_gradients(img2)
        
        # Calculate correlation
        correlation = F.cosine_similarity(grad1, grad2, dim=0)
        
        return correlation.item()
    
    def _calculate_color_consistency(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate color consistency between images"""
        # Calculate mean color for each channel
        mean1 = torch.mean(img1, dim=(1, 2))
        mean2 = torch.mean(img2, dim=(1, 2))
        
        # Calculate color difference
        color_diff = torch.mean(torch.abs(mean1 - mean2))
        
        # Convert to consistency score (higher is better)
        consistency = 1.0 - color_diff.item()
        
        return max(0.0, consistency)
    
    def _calculate_hallucination_score(self, indicators: Dict[str, float]) -> float:
        """Calculate overall hallucination score"""
        # Weight different indicators
        weights = {
            'edge_preservation_improvement': -0.3,  # Negative if too much improvement
            'texture_similarity_change': -0.4,     # Negative if too much change
            'color_consistency': -0.3               # Negative if low consistency
        }
        
        score = 0.0
        for indicator, weight in weights.items():
            if indicator in indicators:
                score += weight * indicators[indicator]
        
        # Normalize to [0, 1] range (higher means more hallucination)
        score = max(0.0, min(1.0, (score + 1.0) / 2.0))
        
        return score
    
    def evaluate_hallucination_batch(self, originals: List[torch.Tensor],
                                   dehazed: List[torch.Tensor],
                                   ground_truths: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Evaluate hallucination indicators for a batch"""
        results = {}
        
        for i, (orig, dehaz, gt) in enumerate(zip(originals, dehazed, ground_truths)):
            indicators = self.detect_structure_changes(orig, dehaz, gt)
            results[f'image_{i}'] = indicators
        
        # Calculate aggregate statistics
        aggregate = {}
        indicator_names = list(results['image_0'].keys())
        
        for name in indicator_names:
            values = [results[f'image_{i}'][name] for i in range(len(results))]
            aggregate[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        results['aggregate'] = aggregate
        
        return results

# Utility functions
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from [0, 1] to [0, 255]
    tensor = tensor.clamp(0, 1) * 255
    tensor = tensor.byte().cpu()
    
    # Convert to PIL
    return Image.fromarray(tensor.permute(1, 2, 0).numpy())

def load_image_as_tensor(image_path: str, device: str = 'cpu') -> torch.Tensor:
    """Load image as tensor"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).to(device)

def save_metrics_results(results: Dict, save_path: str):
    """Save metrics results to JSON file"""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    converted_results = convert_numpy(results)
    
    with open(save_path, 'w') as f:
        json.dump(converted_results, f, indent=2)

# Quick usage function
def quick_evaluate(pred_path: str, gt_path: str) -> Dict[str, float]:
    """Quick evaluation of two images"""
    metrics = DehazeMetrics()
    
    pred = load_image_as_tensor(pred_path)
    gt = load_image_as_tensor(gt_path)
    
    return metrics.calculate_all_metrics(pred, gt)
