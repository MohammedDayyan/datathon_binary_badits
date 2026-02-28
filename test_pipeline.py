"""
Test the dehazing pipeline with imported N-Haze dataset
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference import DehazeInferencePipeline, quick_dehaze
from evaluation.evaluator import DehazeEvaluator
from evaluation.dataset_loader import DatasetManager

def test_single_image():
    """Test dehazing on a single N-Haze image"""
    print("🧪 Testing Single Image Dehazing")
    print("=" * 40)
    
    # Use first N-Haze image
    hazy_path = "data/NH-HAZE/01_hazy.png"
    gt_path = "data/NH-HAZE/01_GT.png"
    
    if not os.path.exists(hazy_path):
        print(f"❌ Hazy image not found: {hazy_path}")
        return
    
    print(f"📁 Processing: {hazy_path}")
    
    # Test with different models
    models = ["aodnet", "dehazenet", "msbdn"]
    
    for model in models:
        print(f"\n🤖 Testing {model.upper()}...")
        
        output_path = f"test_results/{model}_01_dehazed.png"
        os.makedirs("test_results", exist_ok=True)
        
        try:
            result = quick_dehaze(hazy_path, model, output_path)
            
            if result['success']:
                print(f"✅ Success!")
                print(f"   Time: {result['processing_time']:.3f}s")
                if result['metrics'].get('psnr'):
                    print(f"   PSNR: {result['metrics']['psnr']:.2f} dB")
                if result['metrics'].get('ssim'):
                    print(f"   SSIM: {result['metrics']['ssim']:.4f}")
            else:
                print(f"❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_dataset_loader():
    """Test dataset loading with N-Haze structure"""
    print("\n🧪 Testing Dataset Loader")
    print("=" * 40)
    
    try:
        # Create dataset manager
        manager = DatasetManager("data")
        
        # Try to load N-Haze (need to adjust for the actual structure)
        print("📁 Loading N-Haze dataset...")
        
        # Since N-Haze has a different structure, let's check what we have
        nhaze_dir = "data/NH-HAZE"
        hazy_files = [f for f in os.listdir(nhaze_dir) if f.endswith('_hazy.png')]
        gt_files = [f for f in os.listdir(nhaze_dir) if f.endswith('_GT.png')]
        
        print(f"   Found {len(hazy_files)} hazy images")
        print(f"   Found {len(gt_files)} ground truth images")
        
        if hazy_files and gt_files:
            print("✅ N-Haze dataset structure is valid!")
            
            # Show sample pairs
            print("\n📋 Sample image pairs:")
            for i in range(min(3, len(hazy_files))):
                hazy_num = hazy_files[i].replace('_hazy.png', '')
                gt_file = f"{hazy_num}_GT.png"
                if gt_file in gt_files:
                    print(f"   {hazy_files[i]} ↔ {gt_file}")
        else:
            print("❌ Invalid dataset structure")
            
    except Exception as e:
        print(f"❌ Dataset loader error: {e}")

def test_batch_processing():
    """Test batch processing on first few N-Haze images"""
    print("\n🧪 Testing Batch Processing")
    print("=" * 40)
    
    try:
        # Initialize pipeline
        pipeline = DehazeInferencePipeline()
        pipeline.initialize_models()
        
        # Create input/output directories
        input_dir = "data/NH-HAZE"
        output_dir = "test_results/batch_output"
        
        # Copy first 5 hazy images to test input
        os.makedirs("test_batch_input", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        import shutil
        hazy_files = [f for f in os.listdir(input_dir) if f.endswith('_hazy.png')][:5]
        
        for f in hazy_files:
            shutil.copy(os.path.join(input_dir, f), "test_batch_input")
        
        print(f"📁 Processing {len(hazy_files)} images...")
        
        # Run batch processing
        results = pipeline.dehaze_batch("test_batch_input", output_dir, "aodnet")
        
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Processed {successful}/{len(results)} images successfully")
        
        if successful > 0:
            avg_time = sum(r['processing_time'] for r in results if r['success']) / successful
            print(f"   Average time: {avg_time:.3f}s per image")
        
        # Cleanup
        shutil.rmtree("test_batch_input")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def test_evaluation():
    """Test evaluation on N-Haze dataset"""
    print("\n🧪 Testing Evaluation Framework")
    print("=" * 40)
    
    try:
        # Create evaluator
        evaluator = DehazeEvaluator("data", "test_evaluation")
        
        print("📊 Running quick evaluation on N-Haze...")
        
        # Since N-Haze has a different structure, we'll need to adapt
        # For now, let's test with a few images manually
        
        from evaluation.metrics import DehazeMetrics
        from torchvision import transforms
        from PIL import Image
        
        metrics_calc = DehazeMetrics()
        
        # Test on first 3 images
        hazy_files = [f for f in os.listdir("data/NH-HAZE") if f.endswith('_hazy.png')][:3]
        
        all_psnr = []
        all_ssim = []
        
        for hazy_file in hazy_files:
            base_name = hazy_file.replace('_hazy.png', '')
            gt_file = f"{base_name}_GT.png"
            
            if os.path.exists(f"data/NH-HAZE/{gt_file}"):
                # Load images
                hazy_img = Image.open(f"data/NH-HAZE/{hazy_file}")
                gt_img = Image.open(f"data/NH-HAZE/{gt_file}")
                
                # Process with AOD-Net
                output_path = f"test_evaluation/{base_name}_dehazed.png"
                result = quick_dehaze(f"data/NH-HAZE/{hazy_file}", "aodnet", output_path)
                
                if result['success']:
                    # Calculate metrics
                    dehazed_img = Image.open(output_path)
                    
                    hazy_tensor = transforms.ToTensor()(hazy_img)
                    gt_tensor = transforms.ToTensor()(gt_img)
                    dehazed_tensor = transforms.ToTensor()(dehazed_img)
                    
                    metrics = metrics_calc.calculate_all_metrics(dehazed_tensor, gt_tensor)
                    all_psnr.append(metrics['psnr'])
                    all_ssim.append(metrics['ssim'])
                    
                    print(f"   {base_name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
        
        if all_psnr:
            print(f"\n📈 Average Results:")
            print(f"   PSNR: {sum(all_psnr)/len(all_psnr):.2f} dB")
            print(f"   SSIM: {sum(all_ssim)/len(all_ssim):.4f}")
            print("✅ Evaluation completed successfully!")
        else:
            print("❌ No images processed for evaluation")
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")

def main():
    """Run all tests"""
    print("🚀 Testing Image Dehazing Pipeline with N-Haze Dataset")
    print("=" * 60)
    
    # Create test directories
    os.makedirs("test_results", exist_ok=True)
    
    # Run tests
    test_single_image()
    test_dataset_loader()
    test_batch_processing()
    test_evaluation()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📁 Check 'test_results/' directory for output images")
    print("📊 Check 'test_evaluation/' directory for evaluation results")

if __name__ == "__main__":
    main()
