"""Inference script for generating predictions on test set."""
import torch
import pandas as pd
import argparse
from pathlib import Path

from pathway_pipeline import run_batch_inference


def main():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--test-csv', type=str, default='test.csv', help='Test CSV path')
    parser.add_argument('--model', type=str, required=True, help='Trained model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/submission.csv', help='Output CSV path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"Running inference on {args.test_csv}")
    print(f"Using model: {args.model}")
    
    submission_df = run_batch_inference(
        input_csv=args.test_csv,
        output_csv=args.output,
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"\n✓ Inference completed!")
    print(f"Submission file saved to: {args.output}")


if __name__ == '__main__':
    main()
