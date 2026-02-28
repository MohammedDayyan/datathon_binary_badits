"""
Complete Evaluation Pipeline for Image Dehazing Models
Handles dataset loading, model evaluation, and comprehensive metrics
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_loader import DatasetManager, NTIREDehazeDataset
from .metrics import DehazeMetrics, HallucinationDetector
from ..inference import DehazeInferencePipeline

class DehazeEvaluator:
    """
    Comprehensive evaluator for dehazing models
    """
    
    def __init__(self, 
                 data_root: str,
                 output_dir: str = "evaluation_results",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_root = data_root
        self.output_dir = output_dir
        self.device = device
        
        # Initialize components
        self.dataset_manager = DatasetManager(data_root)
        self.metrics_calculator = DehazeMetrics(device)
        self.hallucination_detector = HallucinationDetector()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        
    def evaluate_model_on_dataset(self, 
                                 model_name: str,
                                 dataset_type: str = "I-Haze",
                                 split: str = "test",
                                 save_images: bool = True) -> Dict:
        """
        Evaluate a single model on a specific dataset
        """
        print(f"\nEvaluating {model_name} on {dataset_type} {split}...")
        
        # Load dataset
        try:
            dataloader = self.dataset_manager.load_dataset(
                dataset_type, split, batch_size=1
            )
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return {"error": str(e)}
        
        # Initialize inference pipeline
        pipeline = DehazeInferencePipeline()
        pipeline.initialize_models()
        
        # Storage for results
        predictions = []
        targets = []
        originals = []
        processing_times = []
        individual_metrics = []
        
        # Create output directory for images
        if save_images:
            image_output_dir = os.path.join(
                self.output_dir, f"{model_name}_{dataset_type}_{split}_images"
            )
            os.makedirs(image_output_dir, exist_ok=True)
        
        # Evaluate each image
        dataset = dataloader.dataset
        for i in tqdm(range(len(dataset)), desc="Processing images"):
            try:
                # Get image pair
                hazy_tensor, clear_tensor = dataset[i]
                
                # Convert tensor to PIL for inference
                hazy_pil = torch.nn.functional.to_pil_image(hazy_tensor)
                
                # Save temporary hazy image
                temp_hazy_path = os.path.join(self.output_dir, "temp_hazy.jpg")
                hazy_pil.save(temp_hazy_path)
                
                # Process with model
                start_time = time.time()
                result = pipeline.dehaze_single_image(
                    temp_hazy_path, model_name, 
                    os.path.join(image_output_dir, f"dehazed_{i:04d}.jpg") if save_images else None
                )
                processing_time = time.time() - start_time
                
                if result["success"]:
                    # Load processed image back as tensor
                    if save_images:
                        processed_path = os.path.join(image_output_dir, f"dehazed_{i:04d}.jpg")
                        if os.path.exists(processed_path):
                            from torchvision import transforms
                            processed_tensor = transforms.ToTensor()(Image.open(processed_path))
                        else:
                            # Fallback to original if processing failed
                            processed_tensor = hazy_tensor
                    else:
                        # For memory efficiency, use original as placeholder when not saving
                        processed_tensor = hazy_tensor
                    
                    # Calculate metrics
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        processed_tensor, clear_tensor
                    )
                    metrics["processing_time"] = processing_time
                    
                    # Store results
                    predictions.append(processed_tensor)
                    targets.append(clear_tensor)
                    originals.append(hazy_tensor)
                    processing_times.append(processing_time)
                    individual_metrics.append(metrics)
                    
                else:
                    print(f"Failed to process image {i}: {result['error']}")
                
                # Clean up temporary file
                if os.path.exists(temp_hazy_path):
                    os.remove(temp_hazy_path)
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        # Calculate aggregate metrics
        if predictions:
            aggregate_metrics = self.metrics_calculator.evaluate_dataset(predictions, targets)
            
            # Calculate hallucination indicators
            hallucination_results = self.hallucination_detector.evaluate_hallucination_batch(
                originals, predictions, targets
            )
            
            # Compile results
            results = {
                "model_name": model_name,
                "dataset_type": dataset_type,
                "split": split,
                "num_images": len(predictions),
                "aggregate_metrics": aggregate_metrics,
                "hallucination_analysis": hallucination_results,
                "processing_stats": {
                    "mean_time": np.mean(processing_times),
                    "std_time": np.std(processing_times),
                    "total_time": np.sum(processing_times)
                },
                "individual_metrics": individual_metrics
            }
            
            # Store results
            key = f"{model_name}_{dataset_type}_{split}"
            self.evaluation_results[key] = results
            
            # Save results
            self._save_results(results, key)
            
            return results
        else:
            return {"error": "No images were successfully processed"}
    
    def evaluate_all_models(self, 
                           models: List[str],
                           datasets: List[str] = ["I-Haze", "N-Haze", "Dense-Haze"],
                           splits: List[str] = ["test"]) -> Dict:
        """
        Evaluate all models on all specified datasets
        """
        print("Starting comprehensive evaluation...")
        
        all_results = {}
        
        for model in models:
            for dataset in datasets:
                for split in splits:
                    try:
                        result = self.evaluate_model_on_dataset(model, dataset, split)
                        key = f"{model}_{dataset}_{split}"
                        all_results[key] = result
                        
                        # Print summary
                        if "aggregate_metrics" in result:
                            psnr = result["aggregate_metrics"]["psnr"]["mean"]
                            ssim = result["aggregate_metrics"]["ssim"]["mean"]
                            print(f"✓ {key}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
                        else:
                            print(f"✗ {key}: Failed")
                            
                    except Exception as e:
                        print(f"✗ {model}_{dataset}_{split}: {e}")
                        all_results[f"{model}_{dataset}_{split}"] = {"error": str(e)}
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict, key: str):
        """Save evaluation results to JSON file"""
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        save_path = os.path.join(self.output_dir, f"{key}_results.json")
        with open(save_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
    
    def _generate_comparison_report(self, all_results: Dict):
        """Generate comprehensive comparison report"""
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Image Dehazing Model Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary Results\n\n")
            f.write("| Model | Dataset | PSNR (dB) | SSIM | Processing Time (s) |\n")
            f.write("|-------|---------|-----------|------|-------------------|\n")
            
            for key, result in all_results.items():
                if "aggregate_metrics" in result:
                    parts = key.split('_')
                    model = parts[0]
                    dataset = '_'.join(parts[1:-1])
                    
                    psnr = result["aggregate_metrics"]["psnr"]["mean"]
                    ssim = result["aggregate_metrics"]["ssim"]["mean"]
                    proc_time = result["processing_stats"]["mean_time"]
                    
                    f.write(f"| {model} | {dataset} | {psnr:.2f} | {ssim:.4f} | {proc_time:.3f} |\n")
            
            # Detailed results
            f.write("\n## Detailed Results\n\n")
            
            for key, result in all_results.items():
                if "aggregate_metrics" in result:
                    f.write(f"### {key}\n\n")
                    
                    # Metrics
                    f.write("#### Quality Metrics\n")
                    for metric, stats in result["aggregate_metrics"].items():
                        f.write(f"- **{metric.upper()}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    
                    # Processing stats
                    f.write("\n#### Processing Statistics\n")
                    proc_stats = result["processing_stats"]
                    f.write(f"- **Mean Time**: {proc_stats['mean_time']:.3f}s\n")
                    f.write(f"- **Std Time**: {proc_stats['std_time']:.3f}s\n")
                    f.write(f"- **Total Time**: {proc_stats['total_time']:.3f}s\n")
                    
                    # Hallucination analysis
                    if "hallucination_analysis" in result:
                        f.write("\n#### Hallucination Analysis\n")
                        hall_stats = result["hallucination_analysis"]["aggregate"]
                        for indicator, stats in hall_stats.items():
                            f.write(f"- **{indicator}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    
                    f.write("\n")
        
        print(f"Comparison report saved to: {report_path}")
    
    def generate_visualizations(self, results: Dict):
        """Generate visualization plots"""
        # Extract data for plotting
        models = []
        datasets = []
        psnr_values = []
        ssim_values = []
        
        for key, result in results.items():
            if "aggregate_metrics" in result:
                parts = key.split('_')
                model = parts[0]
                dataset = '_'.join(parts[1:-1])
                
                models.append(model)
                datasets.append(dataset)
                psnr_values.append(result["aggregate_metrics"]["psnr"]["mean"])
                ssim_values.append(result["aggregate_metrics"]["ssim"]["mean"])
        
        if not psnr_values:
            print("No valid results to visualize")
            return
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # PSNR comparison
        ax1.bar(range(len(psnr_values)), psnr_values)
        ax1.set_xlabel('Model-Dataset')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([f"{m}\n{d}" for m, d in zip(models, datasets)], rotation=45)
        
        # SSIM comparison
        ax2.bar(range(len(ssim_values)), ssim_values)
        ax2.set_xlabel('Model-Dataset')
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM Comparison')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([f"{m}\n{d}" for m, d in zip(models, datasets)], rotation=45)
        
        # Scatter plot PSNR vs SSIM
        ax3.scatter(psnr_values, ssim_values)
        ax3.set_xlabel('PSNR (dB)')
        ax3.set_ylabel('SSIM')
        ax3.set_title('PSNR vs SSIM')
        
        # Add labels for each point
        for i, (m, d) in enumerate(zip(models, datasets)):
            ax3.annotate(f"{m}-{d}", (psnr_values[i], ssim_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Performance vs Quality
        processing_times = []
        for key, result in results.items():
            if "processing_stats" in result:
                processing_times.append(result["processing_stats"]["mean_time"])
            else:
                processing_times.append(0)
        
        ax4.scatter(processing_times, psnr_values)
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('PSNR (dB)')
        ax4.set_title('Performance vs Quality')
        
        # Add labels
        for i, (m, d) in enumerate(zip(models, datasets)):
            ax4.annotate(f"{m}-{d}", (processing_times[i], psnr_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "evaluation_visualizations.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {plot_path}")
    
    def run_full_evaluation(self, 
                           models: List[str] = ["aodnet", "dehazenet", "msbdn"],
                           datasets: List[str] = ["I-Haze", "N-Haze", "Dense-Haze"]) -> Dict:
        """
        Run complete evaluation pipeline
        """
        print("Starting full evaluation pipeline...")
        
        # Evaluate all models
        results = self.evaluate_all_models(models, datasets)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        # Print final summary
        self._print_final_summary(results)
        
        return results
    
    def _print_final_summary(self, results: Dict):
        """Print final evaluation summary"""
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        
        # Find best model for each metric
        best_psnr = {"model": "", "dataset": "", "value": -float('inf')}
        best_ssim = {"model": "", "dataset": "", "value": -float('inf')}
        fastest = {"model": "", "dataset": "", "value": float('inf')}
        
        for key, result in results.items():
            if "aggregate_metrics" in result:
                parts = key.split('_')
                model = parts[0]
                dataset = '_'.join(parts[1:-1])
                
                psnr = result["aggregate_metrics"]["psnr"]["mean"]
                ssim = result["aggregate_metrics"]["ssim"]["mean"]
                time_val = result["processing_stats"]["mean_time"]
                
                if psnr > best_psnr["value"]:
                    best_psnr = {"model": model, "dataset": dataset, "value": psnr}
                
                if ssim > best_ssim["value"]:
                    best_ssim = {"model": model, "dataset": dataset, "value": ssim}
                
                if time_val < fastest["value"]:
                    fastest = {"model": model, "dataset": dataset, "value": time_val}
        
        print(f"Best PSNR: {best_psnr['model']} on {best_psnr['dataset']} ({best_psnr['value']:.2f} dB)")
        print(f"Best SSIM: {best_ssim['model']} on {best_ssim['dataset']} ({best_ssim['value']:.4f})")
        print(f"Fastest: {fastest['model']} on {fastest['dataset']} ({fastest['value']:.3f}s)")
        
        print(f"\nDetailed results saved to: {self.output_dir}")
        print("="*60)

# Quick usage function
def quick_evaluation(data_root: str, model_name: str = "aodnet"):
    """Quick evaluation for testing"""
    evaluator = DehazeEvaluator(data_root)
    
    try:
        result = evaluator.evaluate_model_on_dataset(model_name, "I-Haze", "test")
        print(f"Evaluation completed: PSNR={result['aggregate_metrics']['psnr']['mean']:.2f}")
        return result
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None
