"""
Inference Pipeline for Image Dehazing
Handles model loading, preprocessing, and inference operations
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
import time
from typing import List, Tuple, Dict, Optional
import json

from .models import DehazeModelManager, calculate_psnr, calculate_ssim

class DehazeInferencePipeline:
    """
    Complete inference pipeline for image dehazing
    """
    def __init__(self, config_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = DehazeModelManager(self.device)
        self.config = self._load_config(config_path)
        self.results_history = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "models": {
                "aodnet": {
                    "enabled": True,
                    "weights_path": "models/aodnet.pth",
                    "description": "All-in-One Dehazing Network"
                },
                "dehazenet": {
                    "enabled": True,
                    "weights_path": "models/dehazenet.pth", 
                    "description": "DehazeNet: End-to-End Haze Removal"
                },
                "msbdn": {
                    "enabled": True,
                    "weights_path": "models/msbdn.pth",
                    "description": "Multi-Scale Boosted Dehazing Network"
                }
            },
            "preprocessing": {
                "resize": None,  # None or (height, width)
                "normalize": True,
                "batch_size": 1
            },
            "postprocessing": {
                "clip_values": True,
                "denoise": False,
                "sharpen": False
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                
        return default_config
        
    def initialize_models(self):
        """Initialize all enabled models"""
        print(f"Initializing models on device: {self.device}")
        
        for model_name, model_config in self.config["models"].items():
            if model_config["enabled"]:
                print(f"Loading {model_name}: {model_config['description']}")
                try:
                    self.model_manager.load_model(
                        model_name, 
                        model_config.get("weights_path")
                    )
                    print(f"✓ {model_name} loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
                    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image for model input"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize if specified
        if self.config["preprocessing"]["resize"]:
            image = image.resize(self.config["preprocessing"]["resize"], Image.LANCZOS)
            
        # Convert to tensor
        transform = torch.nn.functional.normalize if self.config["preprocessing"]["normalize"] else lambda x: x
        tensor = transform(torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0)
        
        metadata = {
            "original_size": original_size,
            "processed_size": image.size,
            "filename": os.path.basename(image_path)
        }
        
        return tensor.unsqueeze(0), metadata
        
    def postprocess_output(self, output_tensor: torch.Tensor, metadata: Dict) -> Image.Image:
        """Postprocess model output"""
        # Remove batch dimension
        output_tensor = output_tensor.squeeze(0)
        
        # Clip values to valid range
        if self.config["postprocessing"]["clip_values"]:
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
        # Convert to PIL image
        output_array = (output_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_array)
        
        # Resize back to original size
        output_image = output_image.resize(metadata["original_size"], Image.LANCZOS)
        
        # Apply additional processing if enabled
        if self.config["postprocessing"]["denoise"]:
            output_image = self._apply_denoise(output_image)
            
        if self.config["postprocessing"]["sharpen"]:
            output_image = self._apply_sharpen(output_image)
            
        return output_image
        
    def _apply_denoise(self, image: Image.Image) -> Image.Image:
        """Apply denoising to image"""
        img_array = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        return Image.fromarray(denoised)
        
    def _apply_sharpen(self, image: Image.Image) -> Image.Image:
        """Apply sharpening to image"""
        img_array = np.array(image)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(sharpened)
        
    def dehaze_single_image(self, image_path: str, model_name: str, save_path: Optional[str] = None) -> Dict:
        """Dehaze a single image"""
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor, metadata = self.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.model_manager.models[model_name.lower()](input_tensor)
                
            # Postprocess
            output_image = self.postprocess_output(output_tensor, metadata)
            
            # Save if path provided
            if save_path:
                output_image.save(save_path)
                
            # Calculate metrics if ground truth available
            metrics = {}
            gt_path = image_path.replace("hazy", "clear").replace("haze", "clear")
            if os.path.exists(gt_path):
                gt_tensor, _ = self.preprocess_image(gt_path)
                gt_tensor = gt_tensor.to(self.device)
                metrics["psnr"] = calculate_psnr(output_tensor, gt_tensor)
                metrics["ssim"] = calculate_ssim(output_tensor, gt_tensor)
                
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "model_name": model_name,
                "input_path": image_path,
                "output_path": save_path,
                "processing_time": processing_time,
                "metrics": metrics,
                "metadata": metadata
            }
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "input_path": image_path
            }
            
    def dehaze_batch(self, input_dir: str, output_dir: str, model_name: str) -> List[Dict]:
        """Dehaze all images in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        print(f"Processing images in {input_dir} with {model_name}")
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"dehazed_{filename}")
                
                result = self.dehaze_single_image(input_path, model_name, output_path)
                results.append(result)
                
                if result["success"]:
                    print(f"✓ {filename} - Time: {result['processing_time']:.2f}s")
                else:
                    print(f"✗ {filename} - Error: {result['error']}")
                    
        return results
        
    def compare_models(self, image_path: str, models: Optional[List[str]] = None) -> Dict:
        """Compare performance of multiple models on a single image"""
        if models is None:
            models = list(self.model_manager.models.keys())
            
        comparison_results = {}
        
        for model_name in models:
            print(f"Testing {model_name}...")
            result = self.dehaze_single_image(image_path, model_name)
            comparison_results[model_name] = result
            
        return comparison_results
        
    def benchmark_models(self, test_dir: str, models: Optional[List[str]] = None) -> Dict:
        """Benchmark models on a test dataset"""
        if models is None:
            models = list(self.model_manager.models.keys())
            
        benchmark_results = {}
        
        for model_name in models:
            print(f"\nBenchmarking {model_name}...")
            output_dir = f"benchmark_results/{model_name}"
            results = self.dehaze_batch(test_dir, output_dir, model_name)
            
            # Calculate aggregate metrics
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                avg_time = np.mean([r["processing_time"] for r in successful_results])
                
                # Aggregate PSNR and SSIM if available
                psnr_values = [r["metrics"].get("psnr") for r in successful_results if r["metrics"].get("psnr")]
                ssim_values = [r["metrics"].get("ssim") for r in successful_results if r["metrics"].get("ssim")]
                
                benchmark_results[model_name] = {
                    "total_images": len(results),
                    "successful": len(successful_results),
                    "failed": len(results) - len(successful_results),
                    "avg_processing_time": avg_time,
                    "avg_psnr": np.mean(psnr_values) if psnr_values else None,
                    "avg_ssim": np.mean(ssim_values) if ssim_values else None,
                    "results": results
                }
            else:
                benchmark_results[model_name] = {
                    "total_images": len(results),
                    "successful": 0,
                    "failed": len(results),
                    "error": "No successful processing"
                }
                
        return benchmark_results
        
    def save_results(self, save_path: str):
        """Save results history to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
            
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "device": str(self.device),
            "available_models": list(self.model_manager.models.keys()),
            "config": self.config
        }

# Convenience function for quick usage
def quick_dehaze(image_path: str, model_name: str = "aodnet", save_path: Optional[str] = None):
    """Quick dehazing function for single images"""
    pipeline = DehazeInferencePipeline()
    pipeline.initialize_models()
    result = pipeline.dehaze_single_image(image_path, model_name, save_path)
    return result
