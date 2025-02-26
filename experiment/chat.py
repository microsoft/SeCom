# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os

from metrics import evaluate_match, evaluate_sim
from tqdm import tqdm
from utils import LocalLLM, OpenAILLM

from secom import SeCom

MTBP_PROMPT = """
You are an intelligent dialog bot. You will be shown Related Evidences supporting for User Input, and Recent Dialogs between user and you. Please read, memorize, and understand given materials, then generate one concise, coherent and helpful response.

{context}

Question: {question}
"""

parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument(
    "--load_path", default="result/mtbp/retrieval/mpnet/segment-k1_mtbp.jsonl"
)
parser.add_argument(
    "--model_name_or_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
)
parser.add_argument(
    "--save_path", default="result/mtbp/retrieval/mpnet/mistral/answer-k1_mtbp.jsonl"
)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def load_data(load_path):
    if os.path.splitext(os.path.basename(load_path))[1] == ".jsonl":
        data = []
        with open(load_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
    elif os.path.splitext(os.path.basename(load_path))[1] == ".json":
        data = json.load(open(load_path))
        if isinstance(data, dict):
            data = list(data.values())
    else:
        raise NotImplementedError()
    return data


data = load_data(args.load_path)
print(f"number of data: {len(data)}")

if os.path.dirname(args.model_name_or_path) == "openai":
    llm = OpenAILLM(os.path.basename(args.model_name_or_path))
else:
    llm = LocalLLM(args.model_name_or_path)

results = []
processed_ids = set()
if os.path.exists(args.save_path):
    with open(args.save_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            results.append(sample)
    for sample in results:
        processed_ids.add(sample["conversation_id"])

for idx, sample in enumerate(tqdm(data)):
    if sample["conversation_id"] in processed_ids:
        print(f"{sample['conversation_id']} is processed")
        continue
    print(f"answering {idx}-th conversation")
    requests = sample["questions"]
    retrieved_texts = sample["retrieved_texts"]
    pred_answers = []
    for request, context in zip(requests, retrieved_texts):
        prompt = MTBP_PROMPT.format(context=context, question=request)
        response = llm(prompt)
        pred_answers.append(response)
    sample["pred_answers"] = pred_answers
    results.append(sample)
    with open(args.save_path, "w", encoding="utf-8") as f:
        f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])


def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        ans = ans.strip(strip_word)
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans


pred_all = []
for res in results:
    for ans in res["pred_answers"]:
        ans = get_answer(ans)
        pred_all.append(ans)
answer_all = []
for res in results:
    for i, ans in enumerate(res["answers"]):
        res["answers"][i] = str(ans).strip("\n").strip("Answer:")
    answer_all.extend(res["answers"])

metrics = evaluate_sim(pred_all, answer_all, truncate_pred=False)
metrics.update(evaluate_match(pred_all, answer_all, truncate_pred=False))
print(metrics)
metrics_dir = os.path.join(os.path.dirname(args.save_path), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
with open(
    os.path.join(
        metrics_dir, os.path.basename(args.save_path).replace("answer", "metrics")
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f)
