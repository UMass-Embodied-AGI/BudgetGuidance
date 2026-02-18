# Batch Evaluation Scripts

This directory contains scripts to batch evaluate all `samples_*.jsonl` files using the `eval.py` evaluation logic.

## Files

- **`batch_eval.sh`**: Bash script for batch evaluation
- **`batch_eval.py`**: Python script for batch evaluation (more features)
- **`eval.py`**: Original evaluation script (used by both batch scripts)

## Quick Start

### Using the Bash Script

The simplest way to evaluate all samples files:

```bash
cd /proj/inf-scaling/efficient_long_ctx/test-time/BudgetGuidance/evaluation
./batch_eval.sh
```

With custom parameters:

```bash
# Syntax: ./batch_eval.sh [num_workers] [tokenizer_name]
./batch_eval.sh 32 "Qwen/Qwen3-30B-A3B"
```

### Using the Python Script

For more control and better organization:

```bash
python batch_eval.py
```

With custom options:

```bash
python batch_eval.py \
    --root_dir . \
    --num_workers 32 \
    --tokenizer "Qwen/Qwen3-30B-A3B" \
    --output_dir eval_results \
    --pattern "**/samples_*.jsonl"
```

## Command-Line Options (Python Script)

| Option | Default | Description |
|--------|---------|-------------|
| `--root_dir` | `.` | Root directory to search for samples files |
| `--num_workers` | `16` | Number of parallel workers for processing |
| `--tokenizer` | `Qwen/Qwen3-30B-A3B` | Tokenizer for thinking token calculation |
| `--output_dir` | `eval_results` | Directory to save evaluation results |
| `--pattern` | `**/samples_*.jsonl` | Glob pattern to find sample files |

## Output

Both scripts create:

1. **Individual evaluation files**: `eval_results/samples_*.jsonl`
   - One file per input, containing detailed per-sample results

2. **Summary files**: `eval_results/summary_*.json`
   - Aggregated statistics grouped by directory structure
   - Contains: accuracy, number of samples, average thinking tokens

3. **Overall summary**: `eval_results/summary_all.json` (Python script only)
   - All results combined in one file

## Example Output Structure

```
eval_results/
├── samples_openai_math_2025-11-21T18-04-00.025720.jsonl
├── samples_openai_math_2025-11-21T18-33-00.995754.jsonl
├── ...
├── summary_bf_bf_thinking_100_Qwen__Qwen3-30B-A3B.json
├── summary_bf_bf_thinking_500_Qwen__Qwen3-30B-A3B.json
├── summary_moe_openai_math_*.json
└── summary_all.json
```

## Features

### Bash Script (`batch_eval.sh`)
- ✅ Simple and straightforward
- ✅ Automatically finds all `samples_*.jsonl` files
- ✅ Skips already-evaluated files
- ✅ Shows progress counter
- ✅ Works with existing `eval.py` logic

### Python Script (`batch_eval.py`)
- ✅ All features of bash script
- ✅ Better error handling
- ✅ Progress bar with `tqdm`
- ✅ Organized summary files by directory structure
- ✅ Comprehensive statistics and reporting
- ✅ Flexible command-line options
- ✅ Directory-level summary statistics

## How It Works

1. **Discovery**: Recursively searches for all `samples_*.jsonl` files
2. **Evaluation**: For each file:
   - Checks if already evaluated (skips if exists)
   - Loads the samples data
   - Runs parallel evaluation using `eval.py` logic
   - Calculates accuracy and thinking tokens
   - Saves detailed results
3. **Summarization**: Creates summary files organized by directory structure
4. **Reporting**: Prints comprehensive statistics

## Examples

### Evaluate only files in a specific directory

```bash
cd bf/bf_thinking_100
python ../../batch_eval.py --root_dir .
```

### Evaluate with more workers for faster processing

```bash
python batch_eval.py --num_workers 64
```

### Use a different tokenizer

```bash
python batch_eval.py --tokenizer "meta-llama/Llama-3-8B"
```

### Evaluate only specific patterns

```bash
python batch_eval.py --pattern "**/bf/*/samples_*.jsonl"
```

## Viewing Results

### View all summary files

```bash
ls -lh eval_results/summary_*.json
```

### Print a summary file

```bash
cat eval_results/summary_all.json | jq '.'
```

### Quick accuracy comparison

```bash
for f in eval_results/summary_*.json; do
    echo "=== $f ==="
    cat "$f" | jq '.[] | "\(.file): \(.accuracy)"'
done
```

## Notes

- Files are only evaluated once; re-running the script will skip existing results
- To re-evaluate, delete the corresponding file from `eval_results/`
- The thinking token calculation requires a compatible tokenizer
- Evaluation uses the same logic as the original `eval.py` script

## Troubleshooting

### "No sample files found!"
- Check that you're running from the correct directory
- Verify the `--pattern` matches your file structure
- Try: `find . -name "samples_*.jsonl"` to see what files exist

### Tokenizer loading errors
- The script will continue without tokenizer (thinking tokens = 0)
- Install required tokenizer: `pip install transformers`
- Use a different tokenizer with `--tokenizer`

### Out of memory
- Reduce `--num_workers` to use fewer parallel processes
- Process files in smaller batches by using specific directories

## Performance Tips

1. **Use more workers** if you have many CPU cores: `--num_workers 64`
2. **Process specific directories** to avoid loading all files at once
3. **Skip re-evaluation** by not deleting existing results files
4. The Python script is generally faster for large numbers of files
