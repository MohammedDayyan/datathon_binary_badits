"""
Main entry point for Image Dehazing Pipeline
Provides command-line interface for all functionality
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference import DehazeInferencePipeline, quick_dehaze
from src.haze_generator import HazeGenerator, quick_haze_generation
from evaluation.evaluator import DehazeEvaluator, quick_evaluation

def main():
    parser = argparse.ArgumentParser(
        description="Image Dehazing Pipeline - Complete solution for haze removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick dehaze single image
  python main.py dehaze --input hazy_image.jpg --output clear_image.jpg --model aodnet
  
  # Start web dashboard
  python main.py web
  
  # Generate haze from clear image
  python main.py generate-haze --input clear.jpg --output hazy.jpg --type moderate
  
  # Run full evaluation
  python main.py evaluate --data-root ./data --models aodnet dehazenet
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dehaze command
    dehaze_parser = subparsers.add_parser('dehaze', help='Dehaze a single image')
    dehaze_parser.add_argument('--input', '-i', required=True, help='Input hazy image path')
    dehaze_parser.add_argument('--output', '-o', required=True, help='Output clear image path')
    dehaze_parser.add_argument('--model', '-m', default='aodnet', 
                              choices=['aodnet', 'dehazenet', 'msbdn'],
                              help='Model to use for dehazing')
    
    # Batch dehaze command
    batch_parser = subparsers.add_parser('batch-dehaze', help='Dehaze multiple images')
    batch_parser.add_argument('--input-dir', required=True, help='Input directory with hazy images')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory for clear images')
    batch_parser.add_argument('--model', default='aodnet',
                              choices=['aodnet', 'dehazenet', 'msbdn'],
                              help='Model to use for dehazing')
    
    # Web dashboard command
    web_parser = subparsers.add_parser('web', help='Start web dashboard')
    web_parser.add_argument('--port', '-p', type=int, default=8501, help='Port for web server')
    web_parser.add_argument('--host', default='localhost', help='Host for web server')
    
    # Generate haze command
    haze_parser = subparsers.add_parser('generate-haze', help='Generate artificial haze')
    haze_parser.add_argument('--input', '-i', required=True, help='Input clear image path')
    haze_parser.add_argument('--output', '-o', required=True, help='Output hazy image path')
    haze_parser.add_argument('--type', '-t', default='moderate',
                            choices=['light', 'moderate', 'heavy', 'extreme'],
                            help='Type of haze to generate')
    
    # Batch haze generation command
    batch_haze_parser = subparsers.add_parser('batch-haze', help='Generate haze for multiple images')
    batch_haze_parser.add_argument('--input-dir', required=True, help='Input directory with clear images')
    batch_haze_parser.add_argument('--output-dir', required=True, help='Output directory for hazy images')
    batch_haze_parser.add_argument('--type', default='moderate',
                                   choices=['light', 'moderate', 'heavy', 'extreme'],
                                   help='Type of haze to generate')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models on datasets')
    eval_parser.add_argument('--data-root', required=True, help='Root directory of datasets')
    eval_parser.add_argument('--models', nargs='+', default=['aodnet', 'dehazenet', 'msbdn'],
                            help='Models to evaluate')
    eval_parser.add_argument('--datasets', nargs='+', default=['I-Haze', 'N-Haze', 'Dense-Haze'],
                            help='Datasets to evaluate on')
    eval_parser.add_argument('--output-dir', default='evaluation_results',
                            help='Output directory for results')
    
    # Quick evaluation command
    quick_eval_parser = subparsers.add_parser('quick-eval', help='Quick evaluation on single dataset')
    quick_eval_parser.add_argument('--data-root', required=True, help='Root directory of dataset')
    quick_eval_parser.add_argument('--model', default='aodnet',
                                   choices=['aodnet', 'dehazenet', 'msbdn'],
                                   help='Model to evaluate')
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare models on single image')
    compare_parser.add_argument('--input', '-i', required=True, help='Input hazy image path')
    compare_parser.add_argument('--output-dir', required=True, help='Output directory for results')
    compare_parser.add_argument('--models', nargs='+', default=['aodnet', 'dehazenet', 'msbdn'],
                               help='Models to compare')
    
    args = parser.parse_args()
    
    if args.command == 'dehaze':
        print(f"Dehazing {args.input} with {args.model}...")
        result = quick_dehaze(args.input, args.model, args.output)
        
        if result['success']:
            print(f"✓ Dehazing completed successfully!")
            print(f"  Output saved to: {args.output}")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            if result['metrics'].get('psnr'):
                print(f"  PSNR: {result['metrics']['psnr']:.2f} dB")
            if result['metrics'].get('ssim'):
                print(f"  SSIM: {result['metrics']['ssim']:.4f}")
        else:
            print(f"✗ Dehazing failed: {result['error']}")
    
    elif args.command == 'batch-dehaze':
        print(f"Batch dehazing with {args.model}...")
        pipeline = DehazeInferencePipeline()
        pipeline.initialize_models()
        
        results = pipeline.dehaze_batch(args.input_dir, args.output_dir, args.model)
        
        successful = sum(1 for r in results if r['success'])
        print(f"✓ Processed {successful}/{len(results)} images successfully")
        
        if successful > 0:
            avg_time = sum(r['processing_time'] for r in results if r['success']) / successful
            print(f"  Average processing time: {avg_time:.2f}s per image")
    
    elif args.command == 'web':
        print("Starting web dashboard...")
        import subprocess
        
        cmd = [
            'streamlit', 'run', 'web/app.py',
            '--server.port', str(args.port),
            '--server.address', args.host
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to start web dashboard: {e}")
        except FileNotFoundError:
            print("Streamlit not found. Install with: pip install streamlit")
    
    elif args.command == 'generate-haze':
        print(f"Generating {args.type} haze for {args.input}...")
        
        try:
            output_path = quick_haze_generation(args.input, args.output, args.type)
            print(f"✓ Haze generation completed!")
            print(f"  Output saved to: {output_path}")
        except Exception as e:
            print(f"✗ Haze generation failed: {e}")
    
    elif args.command == 'batch-haze':
        print(f"Batch generating {args.type} haze...")
        
        generator = HazeGenerator()
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        generated_files = generator.batch_generate_haze(
            args.input_dir, args.output_dir, [args.type]
        )
        
        print(f"✓ Generated {len(generated_files)} hazy images")
        print(f"  Output directory: {args.output_dir}")
    
    elif args.command == 'evaluate':
        print("Starting comprehensive evaluation...")
        
        evaluator = DehazeEvaluator(args.data_root, args.output_dir)
        results = evaluator.run_full_evaluation(args.models, args.datasets)
        
        print(f"✓ Evaluation completed!")
        print(f"  Results saved to: {args.output_dir}")
    
    elif args.command == 'quick-eval':
        print(f"Quick evaluation with {args.model}...")
        
        result = quick_evaluation(args.data_root, args.model)
        
        if result:
            print(f"✓ Evaluation completed!")
            print(f"  PSNR: {result['aggregate_metrics']['psnr']['mean']:.2f} dB")
            print(f"  SSIM: {result['aggregate_metrics']['ssim']['mean']:.4f}")
        else:
            print("✗ Evaluation failed")
    
    elif args.command == 'compare':
        print(f"Comparing models on {args.input}...")
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        pipeline = DehazeInferencePipeline()
        pipeline.initialize_models()
        
        comparison_results = {}
        
        for model in args.models:
            print(f"  Processing with {model}...")
            output_path = os.path.join(args.output_dir, f"{model}_result.jpg")
            
            result = quick_dehaze(args.input, model, output_path)
            comparison_results[model] = result
        
        # Print comparison
        print("\nComparison Results:")
        print("-" * 50)
        
        for model, result in comparison_results.items():
            if result['success']:
                psnr = result['metrics'].get('psnr', 'N/A')
                ssim = result['metrics'].get('ssim', 'N/A')
                time_val = result['processing_time']
                
                print(f"{model:10} | PSNR: {psnr:>7} | SSIM: {ssim:>7} | Time: {time_val:.3f}s")
            else:
                print(f"{model:10} | Failed: {result['error']}")
        
        print(f"\nResults saved to: {args.output_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
