import argparse
import os
from cs336_basics.bpe_tokenizer import train_bpe, save_bpe_json
from cs336_basics.bpe_old import train_bpe as bpe_old

configs = {
    "tiny": {
        "input": "./data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        "output_prefix": "tiny"
    },
    "owt": {
        "input": "./data/owt_train.txt",
        "vocab_size": 32000,
        "output_prefix": "owt"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Train BPE for predefined datasets.")
    parser.add_argument(
        "--dataset", type=str, choices=["tiny", "owt"], required=True,
        help="Dataset to train on: 'tiny' for TinyStories, 'owt' for OpenWebText sample"
    )
    args = parser.parse_args()

    # 预定义配置
    config = configs[args.dataset]
    output_dir = "./results/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Training on {args.dataset} dataset: {config['input']}")
    vocab, merges = train_bpe(config["input"], config["vocab_size"], ["<|endoftext|>"])

    save_bpe_json(vocab, merges, output_dir, config["output_prefix"])
    print("✅ Done.")


def test():
    import time
    
    time1 = time.perf_counter()
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    time2 = time.perf_counter()
    vocab, merges = bpe_old("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    time3 = time.perf_counter()
    print(f"Old BPE time: {time2 - time1}")
    print(f"New BPE time: {time3 - time2}")
    
    

if __name__ == "__main__":
    test()
