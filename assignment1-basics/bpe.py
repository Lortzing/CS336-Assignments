import argparse
import os
from tests.bpe_tokenizer import train_bpe, save_bpe_json

def main():
    parser = argparse.ArgumentParser(description="Train BPE for predefined datasets.")
    parser.add_argument(
        "--dataset", type=str, choices=["tiny", "owt"], required=True,
        help="Dataset to train on: 'tiny' for TinyStories, 'owt' for OpenWebText sample"
    )
    args = parser.parse_args()

    # 预定义配置
    configs = {
        "tiny": {
            "input": "./data/TinyStoriesV2-GPT4-train.txt",
            "vocab_size": 10000,
            "output_prefix": "tiny"
        },
        "owt": {
            "input": "./data/owt_train.txt",
            "vocab_size": 30000,
            "output_prefix": "owt"
        }
    }

    config = configs[args.dataset]
    output_dir = "./results/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Training on {args.dataset} dataset: {config['input']}")
    vocab, merges = train_bpe(config["input"], config["vocab_size"], ["<|endoftext|>"])

    save_bpe_json(vocab, merges, output_dir, config["output_prefix"])
    print("✅ Done.")

if __name__ == "__main__":
    main()
