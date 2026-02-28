"""
Haze Generation Pipeline
Artificially adds haze to clear images for testing and augmentation
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
import random
import math

class HazeGenerator:
    """
    Generate realistic haze effects on clear images
    Implements multiple haze generation methods
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def atmospheric_perspective_model(self, image: np.ndarray, 
                                    beta: float = 0.8, 
                                    A: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Atmospheric perspective model for haze generation
        I(x) = J(x) * t(x) + A * (1 - t(x))
        
        Args:
            image: Input clear image (H, W, 3)
            beta: Scattering coefficient (0.1 to 2.0)
            A: Atmospheric light (default: brightest pixels)
        
        Returns:
            Hazy image
        """
        if A is None:
            # Estimate atmospheric light from brightest pixels
            A = self._estimate_atmospheric_light(image)
        
        # Generate depth map (simulated)
        depth_map = self._generate_depth_map(image.shape[:2])
        
        # Calculate transmission map
        t = np.exp(-beta * depth_map)
        
        # Apply atmospheric scattering model
        hazy = image * t[..., np.newaxis] + A * (1 - t)[..., np.newaxis]
        
        return np.clip(hazy, 0, 255).astype(np.uint8)
    
    def _estimate_atmospheric_light(self, image: np.ndarray, top_percent: float = 0.1) -> np.ndarray:
        """Estimate atmospheric light from brightest pixels"""
        # Convert to grayscale for finding bright pixels
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find top brightest pixels
        flat_gray = gray.flatten()
        threshold = np.percentile(flat_gray, 100 - top_percent)
        bright_mask = gray >= threshold
        
        # Get mean color of bright pixels
        bright_pixels = image[bright_mask]
        if len(bright_pixels) > 0:
            A = np.mean(bright_pixels, axis=0)
        else:
            A = np.array([255, 255, 255])
            
        return A
    
    def _generate_depth_map(self, shape: Tuple[int, int], 
                           depth_type: str = "gradient") -> np.ndarray:
        """Generate synthetic depth map"""
        h, w = shape
        
        if depth_type == "gradient":
            # Simple gradient depth (farther away = more haze)
            depth = np.linspace(0.3, 1.0, w)
            depth_map = np.tile(depth, (h, 1))
            
        elif depth_type == "radial":
            # Radial depth from center
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            depth_map = distance / max_distance
            
        elif depth_type == "random":
            # Random depth with smoothing
            depth_map = np.random.rand(h, w)
            depth_map = cv2.GaussianBlur(depth_map, (51, 51), 0)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
        else:  # "flat"
            depth_map = np.ones((h, w)) * 0.5
            
        return depth_map
    
    def add_fog_layers(self, image: np.ndarray, 
                      num_layers: int = 3,
                      density_range: Tuple[float, float] = (0.1, 0.4)) -> np.ndarray:
        """
        Add multiple fog layers for realistic effect
        """
        hazy = image.copy().astype(np.float32)
        
        for _ in range(num_layers):
            # Generate random fog layer
            density = random.uniform(*density_range)
            fog_layer = self._generate_fog_layer(image.shape, density)
            
            # Blend with image
            hazy = hazy * (1 - fog_layer[..., np.newaxis]) + fog_layer[..., np.newaxis] * 255
            
        return np.clip(hazy, 0, 255).astype(np.uint8)
    
    def _generate_fog_layer(self, shape: Tuple[int, int], 
                           density: float) -> np.ndarray:
        """Generate a single fog layer"""
        h, w = shape
        
        # Create noise
        noise = np.random.rand(h, w)
        
        # Apply multiple scales of Gaussian blur for realistic fog
        for sigma in [10, 20, 40]:
            noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        
        # Normalize and apply density
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        fog_layer = noise * density
        
        return fog_layer
    
    def add_smog_particles(self, image: np.ndarray, 
                          particle_density: float = 0.02,
                          particle_size_range: Tuple[int, int] = (1, 3)) -> np.ndarray:
        """
        Add smog particles (small dark spots) to simulate pollution
        """
        hazy = image.copy()
        h, w = image.shape[:2]
        
        # Calculate number of particles
        num_particles = int(h * w * particle_density)
        
        for _ in range(num_particles):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random size
            size = random.randint(*particle_size_range)
            
            # Random darkness
            darkness = random.uniform(0.3, 0.7)
            
            # Draw particle
            color = int(255 * (1 - darkness))
            cv2.circle(hazy, (x, y), size, (color, color, color), -1)
            
        return hazy
    
    def add_color_cast(self, image: np.ndarray, 
                      cast_color: Tuple[int, int, int] = (200, 200, 180),
                      intensity: float = 0.3) -> np.ndarray:
        """
        Add color cast to simulate specific atmospheric conditions
        """
        hazy = image.copy().astype(np.float32)
        
        # Apply color cast
        cast_array = np.array(cast_color)
        hazy = hazy * (1 - intensity) + cast_array * intensity
        
        return np.clip(hazy, 0, 255).astype(np.uint8)
    
    def generate_composite_haze(self, image: np.ndarray,
                              haze_type: str = "moderate") -> np.ndarray:
        """
        Generate composite haze using multiple techniques
        """
        # Parameters for different haze types
        haze_configs = {
            "light": {
                "beta": 0.3,
                "fog_layers": 1,
                "fog_density": (0.05, 0.15),
                "particle_density": 0.005,
                "color_cast": (220, 220, 210),
                "color_intensity": 0.1
            },
            "moderate": {
                "beta": 0.8,
                "fog_layers": 2,
                "fog_density": (0.1, 0.3),
                "particle_density": 0.015,
                "color_cast": (200, 200, 180),
                "color_intensity": 0.2
            },
            "heavy": {
                "beta": 1.5,
                "fog_layers": 4,
                "fog_density": (0.2, 0.5),
                "particle_density": 0.03,
                "color_cast": (180, 180, 160),
                "color_intensity": 0.3
            },
            "extreme": {
                "beta": 2.0,
                "fog_layers": 6,
                "fog_density": (0.3, 0.7),
                "particle_density": 0.05,
                "color_cast": (160, 160, 140),
                "color_intensity": 0.4
            }
        }
        
        config = haze_configs.get(haze_type, haze_configs["moderate"])
        
        # Start with atmospheric perspective
        hazy = self.atmospheric_perspective_model(image, config["beta"])
        
        # Add fog layers
        hazy = self.add_fog_layers(
            hazy, 
            config["fog_layers"], 
            config["fog_density"]
        )
        
        # Add smog particles
        hazy = self.add_smog_particles(hazy, config["particle_density"])
        
        # Add color cast
        hazy = self.add_color_cast(
            hazy, 
            config["color_cast"], 
            config["color_intensity"]
        )
        
        return hazy
    
    def generate_dataset_haze(self, clear_image_path: str, 
                            output_dir: str,
                            haze_types: List[str] = None,
                            num_variations: int = 3) -> List[str]:
        """
        Generate multiple hazy variations of a single image
        """
        if haze_types is None:
            haze_types = ["light", "moderate", "heavy", "extreme"]
        
        # Load image
        image = cv2.imread(clear_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        generated_files = []
        base_name = os.path.splitext(os.path.basename(clear_image_path))[0]
        
        for haze_type in haze_types:
            for variation in range(num_variations):
                # Generate hazy image
                hazy = self.generate_composite_haze(image, haze_type)
                
                # Save
                filename = f"{base_name}_{haze_type}_var{variation+1}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                # Convert back to BGR for OpenCV saving
                hazy_bgr = cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, hazy_bgr)
                
                generated_files.append(output_path)
        
        return generated_files
    
    def create_haze_pair(self, clear_image_path: str, 
                        output_dir: str,
                        haze_type: str = "moderate") -> Tuple[str, str]:
        """
        Create a clear-hazy image pair for training/testing
        """
        # Load clear image
        image = cv2.imread(clear_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate hazy version
        hazy = self.generate_composite_haze(image, haze_type)
        
        # Save both images
        base_name = os.path.splitext(os.path.basename(clear_image_path))[0]
        
        clear_output = os.path.join(output_dir, f"{base_name}_clear.jpg")
        hazy_output = os.path.join(output_dir, f"{base_name}_hazy.jpg")
        
        # Save images
        cv2.imwrite(clear_output, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(hazy_output, cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
        
        return clear_output, hazy_output
    
    def batch_generate_haze(self, clear_images_dir: str,
                           output_dir: str,
                           haze_types: List[str] = None) -> List[str]:
        """
        Generate hazy versions for all images in a directory
        """
        if haze_types is None:
            haze_types = ["moderate"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        generated_files = []
        
        for filename in os.listdir(clear_images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                clear_path = os.path.join(clear_images_dir, filename)
                
                try:
                    files = self.generate_dataset_haze(
                        clear_path, 
                        output_dir, 
                        haze_types,
                        num_variations=1
                    )
                    generated_files.extend(files)
                    print(f"Generated haze for: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return generated_files

# Utility functions
def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_hazy_image(image: np.ndarray, save_path: str):
    """Save hazy image"""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)

# Quick usage function
def quick_haze_generation(clear_image_path: str, 
                         output_path: str, 
                         haze_type: str = "moderate"):
    """Quick haze generation for single image"""
    generator = HazeGenerator()
    
    # Load image
    image = load_and_preprocess_image(clear_image_path)
    
    # Generate haze
    hazy = generator.generate_composite_haze(image, haze_type)
    
    # Save
    save_hazy_image(hazy, output_path)
    
    return output_path
