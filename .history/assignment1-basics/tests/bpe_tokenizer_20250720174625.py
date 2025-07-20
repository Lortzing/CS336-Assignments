import regex as re
import os
from typing import Dict, List, Tuple
from typing import BinaryIO
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
# set_start_method("spawn", force=True)


PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
PARALLEL = True


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_chunk(start: int, end: int, input_path: str | os.PathLike) -> str:
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return data.decode("utf-8", errors="ignore")


def pre_tokenization_with_chunk(
    start: int, end: int, input_path: str | os.PathLike, special_tokens: List[str]
) -> Counter[tuple[bytes]]:
    chunk = get_chunk(start, end, input_path)
    return pre_tokenization_with_text(chunk, special_tokens)


def pre_tokenization_with_text(text: str, special_tokens: List[str]) -> Counter[tuple[bytes]]:
    counter = Counter()
    if not text:
        return counter
    
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(f"({special_pattern})", text)
    else:
        parts = [text]

    for part in parts:
        if not part:
            continue

        if special_tokens and part in special_tokens:
            token_bytes = part.encode("utf-8")
            counter[(token_bytes,)] += 1
        else:
            for words in PATTERN.finditer(part):
                token = words.group(0)
                byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                counter[byte_tuple] += 1
    return counter


def vocabulary_init(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab: Dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    return vocab

def select_best_pair(pair_freq: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """选取频率最高的 token pair；若有频率相同的候选，选择字节序(lexicographic)最大的那个 pair。"""
    best_pair = None
    best_freq = -1
    for pair, freq in pair_freq.items():
        # 先比较频率，其次按pair的字节顺序比较
        if freq > best_freq or (freq == best_freq and (best_pair is None or pair > best_pair)):
            best_pair = pair
            best_freq = freq
    return best_pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = vocabulary_init(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    chunks: list[int] = []

    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, 10, b"<|endoftext|>")

    words_counter: Counter[tuple[bytes, ...]] = Counter()
    args_list = [
        (start, end, input_path, special_tokens)
        for start, end in zip(chunks[:-1], chunks[1:])
    ]
            
    if PARALLEL:
        with Pool(processes=cpu_count()) as pool:
            for partial_counter in tqdm(pool.starmap(pre_tokenization_with_chunk, args_list),
                                        total=len(args_list), desc="Pre-tokenizing chunks"):
                words_counter.update(partial_counter)
    else:
        for args in tqdm(args_list, desc="Pre-tokenizing chunks"):
            words_counter.update(pre_tokenization_with_chunk(*args))
    
    
    # Step 2: Compute initial pair frequencies and pair-to-word mappings
    pair_freq = Counter()
    pair_to_words: Dict[Tuple[bytes, bytes], set] = {}
    for word, count in words_counter.items():
        if len(word) < 2:
            continue
        # Count all adjacent pairs in this word
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += count
            pair_to_words.setdefault(pair, set()).add(word)
    
    # Step 3: Iteratively perform merges until we reach the desired vocab size
    target_merges = vocab_size - len(vocab)
    for _ in range(target_merges):
        if not pair_freq:
            break  # no pairs left to merge
        
        # Find the most frequent pair in the corpus
        top_pair = select_best_pair(pair_freq)

        merges.append(top_pair)
        
        new_token = top_pair[0] + top_pair[1]
        vocab[len(vocab)] = new_token
        
        # Get all words that contain this pair (to update them)
        words_with_pair = pair_to_words.pop(top_pair, set())

        del pair_freq[top_pair]
        
        for word in list(words_with_pair):
            if word not in words_counter:
                continue
            word_count = words_counter[word]

            new_word_tokens: List[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == top_pair[0] and word[i+1] == top_pair[1]:
                    new_word_tokens.append(new_token)
                    i += 2
                else:
                    new_word_tokens.append(word[i])
                    i += 1
            new_word = tuple(new_word_tokens)
            
            words_counter.pop(word)
            words_counter[new_word] = words_counter.get(new_word, 0) + word_count

            old_pairs = Counter((word[j], word[j+1]) for j in range(len(word) - 1))
            new_pairs = Counter((new_word[j], new_word[j+1]) for j in range(len(new_word) - 1))

            for pair, occ in old_pairs.items():
                pair_freq[pair] -= occ * word_count
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        pair_to_words.pop(pair)

            for pair, occ in new_pairs.items():
                pair_freq[pair] += occ * word_count
                pair_to_words.setdefault(pair, set()).add(new_word)
    return vocab, merges

if __name__ == "__main__":
    # input_path = "/home/lu0qlng/cs336/assignment1-basics/tests/fixtures/hello.txt"
    input_path = "/home/lu0qlng/cs336/assignment1-basics/tests/fixtures/corpus.en"
    vocab, merges = train_bpe(input_path, 1000, ["<|endoftext|>"])
    
    import json
    def save_bpe_json(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_path: str):
        vocab_serializable = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
        merges_serializable = [[a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace")] for a, b in merges]
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": vocab_serializable, "merges": merges_serializable}, f, indent=2)
            
    save_bpe_json(vocab, merges, "/home/lu0qlng/cs336/assignment1-basics/tests/fixtures/bpe.json")
    
    pass
