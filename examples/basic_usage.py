"""
Basic Usage Examples for Image Dehazing Pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import DehazeInferencePipeline, quick_dehaze
from src.haze_generator import HazeGenerator, quick_haze_generation
from evaluation.evaluator import DehazeEvaluator

def example_1_quick_dehaze():
    """Example 1: Quick dehazing of a single image"""
    print("Example 1: Quick Dehazing")
    print("-" * 30)
    
    # Input and output paths
    hazy_image = "test_images/hazy_sample.jpg"
    clear_output = "test_images/dehazed_result.jpg"
    
    # Quick dehaze
    result = quick_dehaze(hazy_image, "aodnet", clear_output)
    
    if result['success']:
        print(f"✓ Dehazing successful!")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        if result['metrics'].get('psnr'):
            print(f"  PSNR: {result['metrics']['psnr']:.2f} dB")
        if result['metrics'].get('ssim'):
            print(f"  SSIM: {result['metrics']['ssim']:.4f}")
    else:
        print(f"✗ Dehazing failed: {result['error']}")
    
    print()

def example_2_pipeline_usage():
    """Example 2: Using the full inference pipeline"""
    print("Example 2: Pipeline Usage")
    print("-" * 30)
    
    # Initialize pipeline
    pipeline = DehazeInferencePipeline()
    pipeline.initialize_models()
    
    # Get model info
    info = pipeline.get_model_info()
    print(f"Available models: {info['available_models']}")
    print(f"Device: {info['device']}")
    
    # Process single image
    result = pipeline.dehaze_single_image(
        "test_images/hazy_sample.jpg", 
        "aodnet", 
        "test_images/pipeline_result.jpg"
    )
    
    if result['success']:
        print(f"✓ Pipeline processing successful!")
        print(f"  Model: {result['model_name']}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
    
    print()

def example_3_batch_processing():
    """Example 3: Batch processing multiple images"""
    print("Example 3: Batch Processing")
    print("-" * 30)
    
    # Initialize pipeline
    pipeline = DehazeInferencePipeline()
    pipeline.initialize_models()
    
    # Batch process
    input_dir = "test_images/hazy_batch/"
    output_dir = "test_images/dehazed_batch/"
    
    results = pipeline.dehaze_batch(input_dir, output_dir, "aodnet")
    
    successful = sum(1 for r in results if r['success'])
    print(f"✓ Processed {successful}/{len(results)} images")
    
    if successful > 0:
        avg_time = sum(r['processing_time'] for r in results if r['success']) / successful
        print(f"  Average time per image: {avg_time:.2f}s")
    
    print()

def example_4_model_comparison():
    """Example 4: Compare multiple models"""
    print("Example 4: Model Comparison")
    print("-" * 30)
    
    # Initialize pipeline
    pipeline = DehazeInferencePipeline()
    pipeline.initialize_models()
    
    # Compare models on single image
    comparison = pipeline.compare_models(
        "test_images/hazy_sample.jpg",
        ["aodnet", "dehazenet", "msbdn"]
    )
    
    print("Model Comparison Results:")
    print("-" * 40)
    
    for model, result in comparison.items():
        if result['success']:
            psnr = result['metrics'].get('psnr', 'N/A')
            ssim = result['metrics'].get('ssim', 'N/A')
            time_val = result['processing_time']
            
            print(f"{model:10} | PSNR: {psnr:>7} | SSIM: {ssim:>7} | Time: {time_val:.3f}s")
        else:
            print(f"{model:10} | Failed: {result['error']}")
    
    print()

def example_5_haze_generation():
    """Example 5: Generate artificial haze"""
    print("Example 5: Haze Generation")
    print("-" * 30)
    
    # Generate different types of haze
    clear_image = "test_images/clear_sample.jpg"
    haze_types = ["light", "moderate", "heavy", "extreme"]
    
    generator = HazeGenerator()
    
    for haze_type in haze_types:
        output_path = f"test_images/hazy_{haze_type}.jpg"
        
        try:
            quick_haze_generation(clear_image, output_path, haze_type)
            print(f"✓ Generated {haze_type} haze: {output_path}")
        except Exception as e:
            print(f"✗ Failed to generate {haze_type} haze: {e}")
    
    print()

def example_6_haze_dehaze_cycle():
    """Example 6: Generate haze then dehaze (bonus feature)"""
    print("Example 6: Haze-Dehaze Cycle")
    print("-" * 30)
    
    # Start with clear image
    clear_image = "test_images/clear_sample.jpg"
    hazy_intermediate = "test_images/step1_hazy.jpg"
    final_dehazed = "test_images/step2_dehazed.jpg"
    
    # Step 1: Generate haze
    print("Step 1: Generating haze...")
    try:
        quick_haze_generation(clear_image, hazy_intermediate, "moderate")
        print(f"✓ Haze generated: {hazy_intermediate}")
    except Exception as e:
        print(f"✗ Haze generation failed: {e}")
        return
    
    # Step 2: Dehaze the generated haze
    print("Step 2: Dehazing generated image...")
    result = quick_dehaze(hazy_intermediate, "aodnet", final_dehazed)
    
    if result['success']:
        print(f"✓ Dehazing completed: {final_dehazed}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"✗ Dehazing failed: {result['error']}")
    
    print()

def example_7_evaluation():
    """Example 7: Model evaluation on dataset"""
    print("Example 7: Model Evaluation")
    print("-" * 30)
    
    # Note: This requires NTIRE datasets to be downloaded
    data_root = "data/"
    
    if os.path.exists(data_root):
        evaluator = DehazeEvaluator(data_root)
        
        # Quick evaluation on I-Haze test set
        try:
            result = evaluator.evaluate_model_on_dataset("aodnet", "I-Haze", "test")
            
            if "aggregate_metrics" in result:
                psnr = result["aggregate_metrics"]["psnr"]["mean"]
                ssim = result["aggregate_metrics"]["ssim"]["mean"]
                
                print(f"✓ Evaluation completed!")
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  SSIM: {ssim:.4f}")
                print(f"  Images processed: {result['num_images']}")
            else:
                print(f"✗ Evaluation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            print("  Make sure NTIRE datasets are downloaded and organized correctly")
    else:
        print("✗ Dataset directory not found")
        print("  Download NTIRE datasets to data/ directory first")
    
    print()

def example_8_web_dashboard():
    """Example 8: Launch web dashboard"""
    print("Example 8: Web Dashboard")
    print("-" * 30)
    
    print("To start the web dashboard, run:")
    print("  python main.py web")
    print("  or")
    print("  streamlit run web/app.py")
    print()
    print("Then open http://localhost:8501 in your browser")
    print()

def main():
    """Run all examples"""
    print("Image Dehazing Pipeline - Usage Examples")
    print("=" * 50)
    print()
    
    # Create test directories
    os.makedirs("test_images", exist_ok=True)
    
    # Note: Examples assume test images exist
    print("Note: These examples assume test images exist in test_images/ directory")
    print("Download sample images or use your own for testing")
    print()
    
    # Run examples (commented out since we don't have actual test images)
    examples = [
        example_1_quick_dehaze,
        example_2_pipeline_usage,
        example_3_batch_processing,
        example_4_model_comparison,
        example_5_haze_generation,
        example_6_haze_dehaze_cycle,
        example_7_evaluation,
        example_8_web_dashboard,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Example {i} failed: {e}")
            print()

if __name__ == "__main__":
    main()
