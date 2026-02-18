from collections import Counter
import os
import re
import pandas as pd
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import json
import glob
from pprint import pprint
import datasets
from multiprocessing import Pool, cpu_count
from functools import partial
from transformers import AutoTokenizer

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['12']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

1

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

import time
import os
from openai import OpenAI, AzureOpenAI  # 导入 AzureOpenAI

class ChatCompletionSampler:
    """
    Sample from AzureOpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):

        self.api_key_name = "f374165dd51d4f5e9cf61257e1612ecf"  
        self.api_version = "2024-12-01-preview"
        self.endpoint = "https://eteopenai.azure-api.net"  # Azure OpenAI 端点
        self.subscription_key = os.getenv(self.api_key_name)   
        self.client = AzureOpenAI(  
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key_name,
        )
        self.model = "gpt-4o-mini-2024-07-18"
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except:
                exception_backoff = 2**trial  # exponential backoff
                print("sleep and retry", exception_backoff)
                time.sleep(exception_backoff)
                trial += 1


def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["problem"])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("orig_problem", doc.get("orig_orig_problem")))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)




def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }

    if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
        sampler = ChatCompletionSampler(model="gpt-4o-mini")
    else:
        # print(f"Unknown processor: {os.getenv('PROCESSOR')}; set 'PROCESSOR=gpt-4o-mini' and 'OPENAI_API_KEY=YOUR_KEY' for best results.")
        # raise ValueError(f"MATH requires PROCESSOR atm. AIME is fine without it.")
        sampler = ChatCompletionSampler(model="gpt-4o-mini")

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])
    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]

    for i, a in enumerate(results, start=1):
        a = a.rstrip("<|im_start|>answer</|im_end|>")
        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match

        if (a.isdigit()) and (gt.isdigit()):
            a = str(int(a)) # 023 -> 23
        elif sampler is not None:
            options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
            if len(options) > 7:
                # Could switch back to exact returning like in AIME in that case
                # Problem with exact returning is that it sometimes messes up small things like a dollar sign
                print("Warning: Lots of options which may harm indexing performance:", options)
            # This ensures that if doc['answer'] is \text{Evelyn} it is represented as such and not \\text{Evelyn}
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            idx = extract_answer_idx(sampler, options_str, a)
            if idx != "-1":
                if idx.isdigit():
                    idx = int(idx) - 1
                    if len(options) > idx >= 0:
                        a = options[idx]
                    else:
                        print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
                else:
                    print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt)
        else:
            pass # TODO: Maybe add back legacy processing

        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a): # Optional logging
            # print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
            pass
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def _get_thinking_length(text: str, tokenizer) -> int:
    """Extract thinking token count from response text."""
    try:
        output = tokenizer.encode(text, return_tensors="pt")[0]
        think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_end_token_id in output:
            thinking_length = (output == think_end_token_id).nonzero().flatten().tolist()[-1]
        else:
            thinking_length = 0
        return int(thinking_length)
    except:
        return 0


def _process_single_result(args):
    """Helper function for parallel processing of a single result."""
    doc_id, doc, filtered_resps, tokenizer = args
    metric = process_results(doc, filtered_resps)
    exact_match = metric["exact_match"]
    
    # Calculate thinking tokens from first response
    thinking_tokens = 0
    filtered_resps = filtered_resps[0] if isinstance(filtered_resps, list) else filtered_resps
    thinking_tokens = _get_thinking_length(filtered_resps, tokenizer)
    
    return doc_id, exact_match, thinking_tokens


def parallel_process_results(data, ids, num_workers=None, tokenizer=None):
    """Process results in parallel using multiprocessing pool (or serially if num_workers=1)."""
    if num_workers is None:
        num_workers = min(cpu_count(), len(ids))
    
    exact_match_list = []
    thinking_tokens_list = []
    save_data = {}
    
    # Prepare arguments for processing
    args_list = [
        (data["doc_id"][id], data["doc"][id], data["filtered_resps"][id], tokenizer)
        for id in ids
    ]
    
    if num_workers == 1:
        # Process serially without multiprocessing
        results = []
        for arg in tqdm(args_list, desc="Processing results"):
            results.append(_process_single_result(arg))
    else:
        # Process in parallel using multiprocessing pool
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_result, args_list),
                total=len(args_list),
                desc="Processing results"
            ))
    
    for doc_id, exact_match, thinking_tokens in results:
        exact_match_list.append(exact_match)
        thinking_tokens_list.append(thinking_tokens)
        save_data[doc_id] = {"exact_match": exact_match, "thinking_tokens": thinking_tokens}
    
    return exact_match_list, thinking_tokens_list, save_data



if __name__ == "__main__":
    filename = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    tokenizer_name = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-30B-A3B"
    
    # Load tokenizer for thinking token calculation
    try:
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        tokenizer = None
    
    if os.path.isdir(filename):
        filenames = glob.glob(os.path.join(filename, "sample*.jsonl"))
    else:
        filenames = [filename]
    filenames = sorted(filenames)
    pprint(filenames)
    print(len(filenames))
    
    os.makedirs("eval_results", exist_ok=True)
    summary_results = []
    
    for filename in filenames:
        outputfile = os.path.basename(filename)
        result_path = os.path.join("eval_results", outputfile)
        if os.path.exists(result_path):
            print(f"Loading existing results from {outputfile}")
            with open(result_path, "r") as f:
                save_data = json.load(f)
            exact_match_list = [save_data[doc_id]["exact_match"] for doc_id in save_data]
            thinking_tokens_list = [save_data[doc_id]["thinking_tokens"] for doc_id in save_data]
            
            # Calculate metrics from loaded data
            num_samples = len(exact_match_list)
            avg_accuracy = np.mean(exact_match_list)
            avg_thinking_tokens = 0
            if thinking_tokens_list and any(t > 0 for t in thinking_tokens_list):
                avg_thinking_tokens = np.mean([t for t in thinking_tokens_list if t > 0])
            
            # Print results
            print(f"Total samples: {num_samples}")
            print(f"Accuracy: {avg_accuracy:.4f}")
            print(f"Average thinking tokens: {avg_thinking_tokens:.1f}")
            
            # Still add to summary
            summary_results.append({
                "file": outputfile,
                "num_samples": num_samples,
                "accuracy": avg_accuracy,
                "avg_thinking_tokens": avg_thinking_tokens,
                "token_budget": None  # Would need to extract from filename if needed
            })
            continue
        print(outputfile)
        df = pd.read_json(filename, lines=True)
        data = df.to_dict()
        ids = list(data["doc_id"].keys())
        
        # Extract token_budget from the data if available
        token_budget = None
        if "arguments" in data and data["arguments"]:
            first_args = data["arguments"][list(data["arguments"].keys())[0]]
            if isinstance(first_args, dict):
                # Try to find token_budget in nested structure
                for key, value in first_args.items():
                    if isinstance(value, dict) and "token_budget" in value:
                        token_budget = value["token_budget"]
                        break
        
        # Use parallel processing
        exact_match_list, thinking_tokens_list, save_data = parallel_process_results(data, ids, num_workers=num_workers, tokenizer=tokenizer)
        
        # Calculate metrics
        num_samples = len(exact_match_list)
        avg_accuracy = np.mean(exact_match_list)
        avg_thinking_tokens = 0
        if thinking_tokens_list and any(t > 0 for t in thinking_tokens_list):
            avg_thinking_tokens = np.mean([t for t in thinking_tokens_list if t > 0])
        
        # Print results
        print(f"Total samples: {num_samples}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Average thinking tokens: {avg_thinking_tokens:.1f}")
        if token_budget:
            print(f"Token budget: {token_budget}")
        
        # Save detailed results
        json.dump(save_data, open(os.path.join("eval_results", outputfile), "w"))
        
        # Collect summary
        summary_results.append({
            "file": outputfile,
            "num_samples": num_samples,
            "accuracy": avg_accuracy,
            "avg_thinking_tokens": avg_thinking_tokens,
            "token_budget": token_budget
        })
    
    # Save summary to file with full directory path in filename
    # Extract token_budget from the first result
    token_budget_str = ""
    if summary_results and summary_results[0].get("token_budget"):
        token_budget_str = f"token_{summary_results[0]['token_budget']}"
    
    if filenames:
        input_dir = os.path.dirname(os.path.abspath(filenames[0]))
        # Get relative path components and join with underscores
        dir_parts = []
        current = input_dir
        while current and current != os.path.dirname(current):
            basename = os.path.basename(current)
            if basename:
                dir_parts.insert(0, basename)
            current = os.path.dirname(current)
            # Stop after getting 3 levels (e.g., bf/bf_thinking_100/Qwen__Qwen3-30B-A3B)
            if len(dir_parts) >= 3:
                break
        dir_name = "_".join(dir_parts) if dir_parts else "results"
    else:
        dir_name = "results"
    
    # Include token budget in filename if available
    if token_budget_str:
        summary_file = os.path.join("eval_results", f"summary_{dir_name}_{token_budget_str}.json")
    else:
        summary_file = os.path.join("eval_results", f"summary_{dir_name}.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    for result in summary_results:
        print(f"File: {result['file']}")
        print(f"  Samples: {result['num_samples']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Avg Thinking Tokens: {result['avg_thinking_tokens']:.1f}")
        if result.get('token_budget'):
            print(f"  Token Budget: {result['token_budget']}")
        print()