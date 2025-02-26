import os

from datasets import load_dataset

data = load_dataset("panzs19/Long-MT-Bench-Plus", split="test")
for sample in data:
    print(list(sample.keys()))
    conv_history = ""
    # sample["sessions"] consists of multiple sessions, each session is a list of human-bot interaction turns.
    for i, session in enumerate(sample["sessions"]):
        conv_history += f"<Session {i}>: \n"
        for j, turn in enumerate(session):
            conv_history += f"<Turn {j}>: \n"
            conv_history += turn + "\n"
        conv_history += "\n\n"
    print(f"Conversation History: {conv_history}")
    for q, a in zip(sample["questions"], sample["answers"]):
        print(f"Question: {q}")
        print(f"Answer: {a}")

save_path = "data/mtbp/mtbp.jsonl"
os.makedirs(os.path.dirname(save_path))
with open(save_path, "w", encoding="utf-8") as f:
    f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in data])
