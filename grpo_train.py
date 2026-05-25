import os
# MUST be set before imports to share VRAM between training and generation
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '/home/jmakhija/private')

import re
import json
import torch
from datasets import Dataset
from vllm import SamplingParams
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from judger import Judger

# ── 1. Load Model in Strict FP16 (The Unsloth RL Way) ──────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B-Thinking-2507",
    max_seq_length=2048,
    dtype=torch.float16,    # CRITICAL: Match Unsloth's safe RL precision
    load_in_4bit=True,      # CRITICAL for 24GB VRAM survival
    fast_inference=True,
    max_lora_rank=32,
    gpu_memory_utilization=0.75, # Tuned for safe backward pass VRAM
    enforce_eager=True,     # Fixes the vLLM CUDA graph compilation bug
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# ── 2. The FP16 Bypass Hack ──────────────────────────────────────────────────
# Tricks the Unsloth safety check into accepting FP16 on a BF16 native model
model.config.torch_dtype = torch.float16

# ── 3. Dataset Prep ──────────────────────────────────────────────────────────
DATA_PATH = "/home/jmakhija/private/public.jsonl"
raw = [json.loads(l) for l in open(DATA_PATH)]

SYSTEM_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}."
)
SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices, then select the single best answer. "
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(item):
    options = item.get("options")
    if options:
        labels = [chr(65+i) for i in range(len(options))]
        opts_text = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        user = f"{item['question']}\n\nOptions:\n{opts_text}"
        system = SYSTEM_MCQ
    else:
        user = item["question"]
        system = SYSTEM_MATH

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

records = []
dropped_count = 0

for item in raw:
    prompt_text = build_prompt(item)
    
    # Check exactly how many tokens this prompt consumes
    token_count = len(tokenizer(prompt_text)["input_ids"])
    
    # Only keep the prompt if it fits safely inside our limit
    if token_count <= 1024:
        records.append({
            "prompt": prompt_text,
            "answer": json.dumps(item["answer"]),   
            "options": json.dumps(item.get("options") or []),
            "is_mcq": bool(item.get("options")),
        })
    else:
        dropped_count += 1

dataset = Dataset.from_list(records)
print(f"Dataset loaded. Kept {len(records)} problems. Dropped {dropped_count} massive problems.")

# ── 4. Reward Functions ──────────────────────────────────────────────────────
judger = Judger(strict_extract=False)

def extract_boxed(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    return m.group(1).strip() if m else ""

def extract_letter(text):
    m = re.search(r'\\boxed\{\s*([A-Za-z])\s*\}', text)
    return m.group(1).upper() if m else ""

def reward_correctness(prompts, completions, answer, options, is_mcq, **kwargs):
    rewards = []
    
    for completion, ans_str, opt_str, mcq_flag in zip(completions, answer, options, is_mcq):
        comp_text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        
        ans = json.loads(ans_str)
        
        try:
            if mcq_flag:
                pred = extract_letter(comp_text)
                correct = (pred == str(ans).strip().upper())
            else:
                gold_list = ans if isinstance(ans, list) else [ans]
                correct = judger.auto_judge(
                    pred=extract_boxed(comp_text), 
                    gold=gold_list,
                    options=[[]] * len(gold_list)
                )
        except Exception:
            correct = False

        rewards.append(1.0 if correct else 0.0)
        
    return rewards

def reward_format(prompts, completions, **kwargs):
    rewards = []
    
    for completion in completions:
        comp_text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        
        has_think = bool(re.search(r'<think>.*?</think>', comp_text, re.DOTALL))
        has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', comp_text))
        
        score = 0.0
        if has_think: score += 0.5
        if has_boxed: score += 0.5
            
        rewards.append(score)
        
    return rewards

# ── 5. Training ──────────────────────────────────────────────────────────────
vllm_sampling_params = SamplingParams(
    temperature=0.8,
    min_p=0.1,
    top_p=0.95,
    max_tokens=1024,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    output_dir="./grpo_qwen3_math",
    use_vllm=True,
    vllm_sampling_params=vllm_sampling_params,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=2,
    
    # Using your original 2e-5. If it gets unstable, drop to Unsloth's 5e-6
    learning_rate=2e-5, 
    
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # CRITICAL: Force the Trainer to use FP16 to match the model
    fp16=True,  
    bf16=False, 
    
    logging_steps=5,
    save_steps=50,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=4,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_correctness, reward_format],
)

trainer.train()

# ── 6. Save ──────────────────────────────────────────────────────────────────
model.save_pretrained("/home/jmakhija/private/grpo_qwen3_math")
tokenizer.save_pretrained("/home/jmakhija/private/grpo_qwen3_math")
print("Done! Model saved.")