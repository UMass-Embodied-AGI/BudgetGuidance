#!/usr/bin/env python3
"""
Batch evaluation script for all samples_*.jsonl files
This script finds all samples_*.jsonl files and evaluates them using eval.py logic
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Import functions from eval.py
sys.path.insert(0, os.path.dirname(__file__))
from eval import parallel_process_results


def find_sample_files(root_dir, pattern="**/samples_*.jsonl"):
    """Find all samples_*.jsonl files recursively"""
    root_path = Path(root_dir)
    sample_files = sorted(root_path.glob(pattern))
    return [str(f) for f in sample_files]


def get_output_filename(input_file, output_dir):
    """Get the output filename for evaluation results"""
    basename = os.path.basename(input_file)
    return os.path.join(output_dir, basename)


def get_summary_filename(input_file):
    """Generate a summary filename based on the input file's directory structure"""
    input_path = Path(input_file).resolve()
    
    # Get the relative path components
    parts = input_path.parts
    
    # Try to extract meaningful directory names (e.g., bf/bf_thinking_100/model_name)
    dir_parts = []
    for i in range(len(parts) - 1, 0, -1):
        part = parts[i]
        if part and part not in ["evaluation", "BudgetGuidance"]:
            dir_parts.insert(0, part)
            if len(dir_parts) >= 3:
                break
    
    dir_name = "_".join(dir_parts) if dir_parts else "results"
    return f"summary_{dir_name}.json"


def evaluate_file(filename, output_dir, num_workers, tokenizer):
    """Evaluate a single samples_*.jsonl file"""
    outputfile = get_output_filename(filename, output_dir)
    
    if os.path.exists(outputfile):
        return None, "skipped"
    
    try:
        # Load data
        df = pd.read_json(filename, lines=True)
        data = df.to_dict()
        ids = list(data["doc_id"].keys())
        
        # Process results
        exact_match_list, thinking_tokens_list, save_data = parallel_process_results(
            data, ids, num_workers=num_workers, tokenizer=tokenizer
        )
        
        # Calculate metrics
        num_samples = len(exact_match_list)
        avg_accuracy = np.mean(exact_match_list)
        avg_thinking_tokens = 0
        if thinking_tokens_list and any(t > 0 for t in thinking_tokens_list):
            avg_thinking_tokens = np.mean([t for t in thinking_tokens_list if t > 0])
        
        # Save results
        with open(outputfile, "w") as f:
            json.dump(save_data, f)
        
        result = {
            "file": os.path.basename(filename),
            "full_path": filename,
            "num_samples": num_samples,
            "accuracy": float(avg_accuracy),
            "avg_thinking_tokens": float(avg_thinking_tokens)
        }
        
        return result, "success"
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, "error"


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all samples_*.jsonl files")
    parser.add_argument("--root_dir", type=str, default=".", 
                        help="Root directory to search for samples files (default: current directory)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="Tokenizer to use for thinking token calculation")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for evaluation results")
    parser.add_argument("--pattern", type=str, default="**/samples_*.jsonl",
                        help="Glob pattern to find sample files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"Warning: Could not load tokenizer {args.tokenizer}: {e}")
        tokenizer = None
    
    # Find all sample files
    print(f"\nSearching for samples files in: {args.root_dir}")
    print(f"Pattern: {args.pattern}")
    sample_files = find_sample_files(args.root_dir, args.pattern)
    
    print(f"Found {len(sample_files)} files to evaluate\n")
    
    if not sample_files:
        print("No sample files found!")
        return
    
    # Group files by directory for better organization
    files_by_dir = defaultdict(list)
    for file in sample_files:
        dir_name = os.path.dirname(file)
        files_by_dir[dir_name].append(file)
    
    # Evaluate each file
    all_results = []
    skipped_count = 0
    error_count = 0
    success_count = 0
    
    print("=" * 70)
    print("Starting batch evaluation...")
    print("=" * 70)
    print()
    
    for i, filename in enumerate(tqdm(sample_files, desc="Overall Progress"), 1):
        rel_path = os.path.relpath(filename, args.root_dir)
        print(f"\n[{i}/{len(sample_files)}] Processing: {rel_path}")
        
        result, status = evaluate_file(filename, args.output_dir, args.num_workers, tokenizer)
        
        if status == "skipped":
            print(f"  ✓ Already evaluated (skipping)")
            skipped_count += 1
        elif status == "error":
            print(f"  ✗ Failed with error")
            error_count += 1
        elif status == "success":
            print(f"  ✓ Accuracy: {result['accuracy']:.4f}, "
                  f"Avg Thinking Tokens: {result['avg_thinking_tokens']:.1f}")
            all_results.append(result)
            success_count += 1
    
    # Save summary by directory groups
    print("\n" + "=" * 70)
    print("Creating summary files...")
    print("=" * 70)
    
    summaries_by_dir = defaultdict(list)
    for result in all_results:
        # Extract directory structure from full path
        file_path = result['full_path']
        parent_dirs = Path(file_path).parent.parts[-3:]  # Get last 3 directory levels
        dir_key = "_".join([p for p in parent_dirs if p not in [".", ".."]])
        summaries_by_dir[dir_key].append(result)
    
    # Save individual summaries
    for dir_key, results in summaries_by_dir.items():
        summary_file = os.path.join(args.output_dir, f"summary_{dir_key}.json")
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {summary_file}")
    
    # Save overall summary
    if all_results:
        overall_summary_file = os.path.join(args.output_dir, "summary_all.json")
        with open(overall_summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved: {overall_summary_file}")
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("BATCH EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Total files found:     {len(sample_files)}")
    print(f"Successfully evaluated: {success_count}")
    print(f"Skipped (existing):    {skipped_count}")
    print(f"Errors:                {error_count}")
    print("=" * 70)
    
    # Print summary table
    if all_results:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        # Group by directory for summary
        for dir_key in sorted(summaries_by_dir.keys()):
            results = summaries_by_dir[dir_key]
            print(f"\n{dir_key}:")
            print("-" * 70)
            for result in results:
                print(f"  {result['file']}")
                print(f"    Samples: {result['num_samples']}")
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    Avg Thinking Tokens: {result['avg_thinking_tokens']:.1f}")
            
            # Print directory average
            if len(results) > 1:
                avg_acc = np.mean([r['accuracy'] for r in results])
                print(f"\n  Directory Average Accuracy: {avg_acc:.4f}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
