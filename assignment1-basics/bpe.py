from tests.bpe_tokenizer import train_bpe, save_bpe_json

if __name__ == '__main__':
    input_path = "./data/owt_train.txt"
    vocab, merges = train_bpe(input_path, 10000, ["<|endoftext|>"])
    
    save_bpe_json(vocab, merges, "./results/", "owt")
    
    input_path = "./data/owt_train.txt"
    vocab, merges = train_bpe(input_path, 30000, ["<|endoftext|>"])
    
    save_bpe_json(vocab, merges, "./results/", "owt")