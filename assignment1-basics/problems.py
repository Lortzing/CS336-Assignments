from functools import wraps


def parting_line(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(">" * 20 + "<" * 20)
        print(fun.__name__+" starts")
        print(">" * 20 + "<" * 20)
        print("=" * 40)
        fun(*args, **kwargs)
        print("=" * 40)
        print("\n\n")
    return wrapper

@parting_line
def problem_unicode1() -> None:
    print(chr(0))
    print(chr(0).__repr__())
    
    # run in Python interpreter
    # >>> chr(0)
    # >>> print(chr(0))
    # >>> "this is a test" + chr(0) + "string"
    # >>> print("this is a test" + chr(0) + "string")

@parting_line
def problem_unicode2() -> None:
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    try:
        print(decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8")))
    except Exception as e:
        print("Error in process `hello! こんにちは!`")
        print(f"{'hello! こんにちは!'.encode("utf-8")=}\n")
    
    try:
        print(bytes([255, 235]).decode("utf-8"))
    except Exception as e:
        print("Error in process `bytes([255, 235])`")
        print(e)

@parting_line
def problem_train_bpe_tinystories() -> None:
    # run with command line
    # python -m cprofile -o ./results/result_tiny.prof problems.py
    
    from cs336_basics.bpe_tokenizer import train_bpe, save_bpe_json
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    save_bpe_json(vocab, merges, "./results/", "tiny")

@parting_line
def problem_train_bpe_expts_owt() -> None:
    from cs336_basics.bpe_tokenizer import train_bpe, save_bpe_json
    vocab, merges = train_bpe("./data/OpenWebText_sample.txt", 32000, ["<|endoftext|>"])
    save_bpe_json(vocab, merges, "./results/", "owt")

@parting_line
def problem_train_bpe_tinystories_profile() -> None:
    import pstats

    p = pstats.Stats("./results/result_tiny.prof")
    p.strip_dirs().sort_stats("cumtime").print_stats(30)  # 打印耗时前30的函数


    
    
if __name__ == '__main__':
    problem_train_bpe_tinystories_profile()