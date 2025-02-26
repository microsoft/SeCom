# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import sys
from typing import List

from tqdm import tqdm

from secom import SeCom

parser = argparse.ArgumentParser(description="retrieve segments.")
parser.add_argument("--load_path", default="result/mtbp/llmlingua2comp_mtbp.jsonl")
parser.add_argument("--secom_config_path", default="")
parser.add_argument("--granularity", type=str, default="segment")
parser.add_argument("--topk", type=int, default=3)
parser.add_argument(
    "--use_comp_key", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--save_path", default="result/mtbp/retrieval/bm25/segment-k1_mtbp.jsonl"
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

secom = SeCom(config_path=args.secom_config_path)

results = []
processed_ids = set()
if os.path.exists(args.save_path):
    with open(args.save_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            results.append(sample)
    for sample in results:
        processed_ids.add(sample["conversation_id"])

n_ex_list = []
n_token_list = []
for idx, sample in enumerate(tqdm(data)):
    if sample["conversation_id"] in processed_ids:
        print(f"{sample['conversation_id']} is processed")
        continue
    print(f"retrieving {idx}-th conversation")
    requests = sample["questions"]
    units = sample[f"{args.granularity}s"]
    assert (
        isinstance(units, List)
        and isinstance(units[0], List)
        and isinstance(units[0][0], str)
    )
    if args.use_comp_key:
        comp_units = sample[f"comp_{args.granularity}s"]
        sample["retrieved_texts"], n_ex, n_token = secom.retrieve_external_memory(
            requests, units, comp_units, retrieve_topk=args.topk
        )
    else:
        sample["retrieved_texts"], n_ex, n_token = secom.retrieve_external_memory(
            requests, units, retrieve_topk=args.topk
        )
    results.append(sample)
    n_ex_list.append(n_ex)
    n_token_list.append(n_token)
    with open(args.save_path, "w", encoding="utf-8") as f:
        f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])

n_ex_avg = sum(n_ex_list) / len(n_ex_list)
n_token_avg = sum(n_token_list) / len(n_token_list)
metrics = {"n_ex": n_ex_avg, "n_token": n_token_avg}
print(metrics)
metrics_dir = os.path.join(os.path.dirname(args.save_path), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
with open(
    os.path.join(metrics_dir, os.path.basename(args.save_path)), "w", encoding="utf-8"
) as f:
    json.dump(metrics, f)
