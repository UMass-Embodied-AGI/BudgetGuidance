#!/usr/bin/env python3
"""
Results viewer for batch evaluation results
Displays results in a formatted table
"""

import json
import argparse
from pathlib import Path
from tabulate import tabulate
import glob


def load_summary_files(results_dir):
    """Load all summary files from results directory"""
    summary_files = glob.glob(f"{results_dir}/summary_*.json")
    
    all_results = []
    for file in summary_files:
        # Only load files with 'bf' in the filename
        if 'bf' not in file:
            continue
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    
    return all_results


def display_results(results, sort_by="avg_thinking_tokens", reverse=False, group_by="type"):
    """Display results in a formatted table"""
    
    if not results:
        print("No results to display")
        return
    
    # Prepare table data
    headers = ["File", "Samples", "Accuracy", "Thinking Tokens"]
    table_data = []
    
    # Always group by bf vs moe_openai_math by default, sorted by thinking tokens (ascending)
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for result in results:
        full_path = result.get("full_path", "")
        filename = result.get("file", "")
        
        # Determine if it's bf or moe_openai_math based on filename/path
        if "bf" in filename or "bf" in full_path:
            key = "BF"
        elif "moe_openai_math" in filename or "moe_openai_math" in full_path:
            key = "MOE_OPENAI_MATH"
        else:
            key = "OTHER"
        
        grouped[key].append(result)
    
    # Display grouped results
    # Sort groups: BF first, then MOE_OPENAI_MATH, then OTHER
    group_order = ["BF", "MOE_OPENAI_MATH", "OTHER"]
    for group_key in group_order:
        if group_key not in grouped:
            continue
        
        group_results = grouped[group_key]
        
        # Sort by thinking tokens (ascending, from least to most)
        group_results = sorted(group_results, key=lambda x: x.get("avg_thinking_tokens", 0), reverse=False)
        
        print(f"\n{'='*80}")
        print(f"  {group_key}")
        print('='*80)
        
        group_table = []
        for result in group_results:
            group_table.append([
                result.get("file", "N/A"),
                result.get("num_samples", 0),
                f"{result.get('accuracy', 0):.4f}",
                f"{result.get('avg_thinking_tokens', 0):.1f}"
            ])
        
        print(tabulate(group_table, headers=headers, tablefmt="grid"))
        
        # Calculate group averages
        if len(group_results) > 1:
            avg_acc = sum(r.get("accuracy", 0) for r in group_results) / len(group_results)
            avg_tokens = sum(r.get("avg_thinking_tokens", 0) for r in group_results) / len(group_results)
            print(f"\nGroup Average - Accuracy: {avg_acc:.4f}, Thinking Tokens: {avg_tokens:.1f}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    total_samples = sum(r.get("num_samples", 0) for r in results)
    avg_accuracy = sum(r.get("accuracy", 0) for r in results) / len(results) if results else 0
    avg_thinking = sum(r.get("avg_thinking_tokens", 0) for r in results) / len(results) if results else 0
    
    print(f"Total Files:           {len(results)}")
    print(f"Total Samples:         {total_samples}")
    print(f"Average Accuracy:      {avg_accuracy:.4f}")
    print(f"Average Thinking Tokens: {avg_thinking:.1f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View batch evaluation results")
    parser.add_argument("--results_dir", type=str, default="eval_results",
                        help="Directory containing evaluation results")
    parser.add_argument("--sort_by", type=str, default="accuracy",
                        choices=["accuracy", "num_samples", "avg_thinking_tokens", "file"],
                        help="Sort results by this field")
    parser.add_argument("--ascending", action="store_true",
                        help="Sort in ascending order (default: descending)")
    parser.add_argument("--group", action="store_true",
                        help="Group results by experiment type")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter results by filename pattern (e.g., 'bf_thinking')")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_summary_files(args.results_dir)
    
    # Filter if requested
    if args.filter:
        results = [r for r in results if args.filter in r.get("file", "") or args.filter in r.get("full_path", "")]
        print(f"Filtered to {len(results)} results matching '{args.filter}'")
    
    # Display results (always grouped by BF vs MOE_OPENAI_MATH, sorted by thinking tokens ascending)
    display_results(results)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if "tabulate" in str(e):
            print("Error: 'tabulate' package not found.")
            print("Install it with: pip install tabulate")
            print("\nFalling back to simple display...")
            
            # Simple fallback display
            import json
            import glob
            results_dir = "eval_results"
            for file in glob.glob(f"{results_dir}/summary_*.json"):
                print(f"\n{'='*70}")
                print(f"File: {file}")
                print('='*70)
                with open(file, 'r') as f:
                    data = json.load(f)
                    print(json.dumps(data, indent=2))
        else:
            raise
