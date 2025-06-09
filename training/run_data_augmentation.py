from datasets import load_dataset
import re
from tqdm import tqdm
dataset = load_dataset("open-r1/OpenR1-Math-220k", name="all")

# Regex to match the overall structure: <think>...</think>\n\n(zzz)
#   (?s) makes '.' match newlines as well
pattern_think_block = re.compile(r'(?s)<think>(.*?)</think>\n\n(.*)')

# Regex to further check if the <think> content contains "**Final Answer**"
# We'll capture two parts:
#   1) everything before "**Final Answer**"
#   2) everything after "**Final Answer**"
pattern_final_answer = re.compile(r'(?s)(.*?)\*\*Final Answer\*\*(.*)')

def modify_think_format(example):
    new_messages = []
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            
            # 1. First, find the overall <think>...</think>\n\n(zzz) structure
            match_think = pattern_think_block.search(content)
            if match_think:
                # group(1) = the content inside <think>...</think>
                # group(2) = the content following </think>\n\n (i.e., zzz)
                think_part = match_think.group(1)
                zzz = match_think.group(2)
                
                # 2. Check if think_part contains "**Final Answer**"
                match_fa = pattern_final_answer.search(think_part)
                if match_fa:
                    # Found "**Final Answer**"
                    # part_before_fa = the text before "**Final Answer**" (xxx)
                    # part_after_fa  = the text after "**Final Answer**"  (yyy)
                    part_before_fa = match_fa.group(1)
                    part_after_fa = match_fa.group(2)
                    
                    # Replace the 'xxx' with 'zzz'
                    # New content format:
                    # <think>
                    #   zzz
                    #
                    #   **Final Answer**yyy
                    # </think>
                    #
                    # zzz
                    new_content = (
                        f"<think>\n{zzz}\n\n"
                        f"**Final Answer**{part_after_fa}"
                        f"</think>\n\n{zzz}"
                    )
                else:
                    # No "**Final Answer**" found, just replace the entire think_part (xxx) with zzz
                    new_content = (
                        f"<think>\n{zzz}\n"
                        f"</think>\n\n{zzz}"
                    )
                # Update msg["content"]
                msg["content"] = new_content
            else:
                print("Warning: No think block found in assistant message.")
                print(msg["content"])
        new_messages.append(msg)
    example["messages"] = new_messages
    return example


def has_think_block(example):
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            
            # 1. First, find the overall <think>...</think>\n\n(zzz) structure
            match_think = pattern_think_block.search(content)
            if match_think:
                return True
    return False
filtered_dataset = dataset.filter(has_think_block, num_proc=64)

# Apply transformation
new_dataset = filtered_dataset.map(modify_think_format, num_proc=64)
print("original dataset size:", len(dataset))
print("new dataset size:", len(new_dataset))

# Save the new dataset to disk
new_dataset.save_to_disk("./datasets/OpenR1-Math-220k-concise")
