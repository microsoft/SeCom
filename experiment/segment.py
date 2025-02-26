# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os

from tqdm import tqdm

from secom import SeCom

parser = argparse.ArgumentParser(description="segment any conversation.")
parser.add_argument("--save_path", default="result/mtbp/gpt4seg_mtbp.jsonl")
parser.add_argument("--load_path", default="data/mtbp/mtbp.jsonl")
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

secom = SeCom()

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
    conversation = sample["sessions"]
    print(f"segmenting {idx}-th conversation")
    sample["segments"] = secom.segment(conversation)
    results.append(sample)

    with open(args.save_path, "w", encoding="utf-8") as f:
        f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])
