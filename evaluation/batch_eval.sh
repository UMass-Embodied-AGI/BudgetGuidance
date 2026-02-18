#!/bin/bash

# Script to evaluate all samples_*.jsonl files using eval.py
# Usage: ./batch_eval.sh [num_workers] [tokenizer_name]

# Default values
NUM_WORKERS=${1:-16}
TOKENIZER_NAME=${2:-"Qwen/Qwen3-30B-A3B"}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Batch Evaluation Script"
echo "======================================"
echo "Number of workers: $NUM_WORKERS"
echo "Tokenizer: $TOKENIZER_NAME"
echo "======================================"
echo ""

# Find all samples_*.jsonl files in moe_openai_math directory
echo "Finding all samples_*.jsonl files in moe_openai_math directory..."
SAMPLE_FILES=$(find ./moe_openai_math -name "samples_*.jsonl" -type f | sort)

# Count total files
TOTAL_FILES=$(echo "$SAMPLE_FILES" | wc -l)
echo "Found $TOTAL_FILES files to evaluate"
echo ""

# Create eval_results directory if it doesn't exist
mkdir -p eval_results

# Counter for progress
COUNTER=0

# Loop through each file and evaluate
for FILE in $SAMPLE_FILES; do
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL_FILES] Processing: $FILE"
    
    # Get the basename to check if it's already evaluated
    BASENAME=$(basename "$FILE")
    
    if [ -f "eval_results/$BASENAME" ]; then
        echo "  ✓ Already evaluated (skipping)"
        echo ""
        continue
    fi
    
    # Run eval.py on the file
    echo "  Running evaluation..."
    python eval.py "$FILE" "$NUM_WORKERS" "$TOKENIZER_NAME"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with error"
    fi
    echo ""
done

echo "======================================"
echo "Batch evaluation completed!"
echo "======================================"
echo ""
echo "Summary files should be in eval_results/ directory"
echo "To view all summary files:"
echo "  ls -lh eval_results/summary_*.json"
