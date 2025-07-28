import json
import os
import re

def fix_vocab_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    fixed_vocab = {}

    for token, idx in vocab.items():
        # 将 id 从 str 转为 int
        fixed_vocab[token] = int(idx)

    # sort by ID for consistency
    fixed_vocab = dict(sorted(fixed_vocab.items(), key=lambda x: x[1]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ Fixed vocab saved to: {output_path}")


def fix_merges_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []

    # Add version header
    fixed_lines.append("#version: 0.2\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Remove accidental quote characters
        line = line.replace('"', '').replace("'", '')

        # Split on whitespace, must be exactly 2 parts
        parts = re.split(r'\s+', line)
        if len(parts) == 2:
            fixed_lines.append(f"{parts[0]} {parts[1]}\n")
        else:
            print(f"⚠️ Skipping malformed line: {line}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print(f"✅ Fixed merges saved to: {output_path}")


if __name__ == "__main__":
    # 替换为你的实际路径
    dataset = "tiny"
    input_vocab = f"./results/vocab_{dataset}.json"
    input_merges = f"./results/merges_{dataset}.txt"

    output_vocab = f"./results/vocab_{dataset}_fixed.json"
    output_merges = f"./results/merges_{dataset}_fixed.txt"

    fix_vocab_file(input_vocab, output_vocab)
    fix_merges_file(input_merges, output_merges)
