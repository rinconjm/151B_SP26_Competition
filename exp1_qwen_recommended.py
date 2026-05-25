import json, sys
from pathlib import Path
sys.path.insert(0, '/home/jmakhija/private')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from judger import Judger
import re

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "/home/jmakhija/private/public.jsonl"
OUTPUT_PATH = "/home/jmakhija/private/results/exp1_qwen_recommended.jsonl"
MAX_TOKENS = 16384

sample_ids = set(json.load(open('/home/jmakhija/private/eval_sample_ids.json')))
all_data = {d['id']: d for d in (json.loads(l) for l in open(DATA_PATH))}
data = [all_data[i] for i in sample_ids if i in all_data]
print(f"Running on {len(data)} questions")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

llm = LLM(
    model=MODEL_ID,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.85,
    max_model_len=16384,
    trust_remote_code=True,
    max_num_seqs=16,
)

# Qwen recommended settings for Thinking model
sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.0,
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(question, options=None):
    if options:
        labels = [chr(65+i) for i in range(len(options))]
        opts_text = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question

prompts = []
for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

outputs = llm.generate(prompts, sampling_params=sampling_params)

# Save first
with open(OUTPUT_PATH, "w") as f:
    for item, out in zip(data, outputs):
        response = out.outputs[0].text.strip()
        f.write(json.dumps({
            "id": item["id"],
            "response": response,
            "answer": item["answer"],
            "options": item.get("options"),
            "is_mcq": bool(item.get("options"))
        }) + "\n")

# Score
def extract_letter(text):
    m = re.search(r'\\boxed\{([A-Za-z])\}', text)
    if m:
        return m.group(1).upper()
    return ""

judger = Judger(strict_extract=False)
mcq_correct, mcq_total, free_correct, free_total = 0, 0, 0, 0

results = [json.loads(l) for l in open(OUTPUT_PATH)]
for r in results:
    if r['is_mcq']:
        mcq_total += 1
        if extract_letter(r['response']) == str(r['answer']).strip().upper():
            mcq_correct += 1
    else:
        free_total += 1
        gold_list = r['answer'] if isinstance(r['answer'], list) else [r['answer']]
        try:
            if judger.auto_judge(pred=r['response'], gold=gold_list, options=[[]] * len(gold_list)):
                free_correct += 1
        except:
            pass

print(f"\nMCQ:       {mcq_correct}/{mcq_total} ({mcq_correct/mcq_total*100:.2f}%)")
print(f"Free-form: {free_correct}/{free_total} ({free_correct/free_total*100:.2f}%)")
print(f"Overall:   {mcq_correct+free_correct}/{len(results)} ({(mcq_correct+free_correct)/len(results)*100:.2f}%)")
