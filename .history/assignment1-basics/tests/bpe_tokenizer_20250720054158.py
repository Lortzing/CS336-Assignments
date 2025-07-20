import regex as re
import os
from typing import Dict, List, Tuple
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool, cpu_count
import heapq
from tqdm import tqdm

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
    chunk = ""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return chunk


def pre_tokenization_with_chunk(
    start: int, end: int, input_path: str | os.PathLike, special_tokens: List[str]
) -> Counter[tuple[bytes]]:
    chunk = get_chunk(start, end, input_path)
    return pre_tokenization_with_text(chunk, special_tokens)


def pre_tokenization_with_text(text: str, special_tokens: List[str]) -> Counter[tuple[bytes]]:
    counter = Counter()

    escape = "|".join(map(re.escape, special_tokens))
    parts = re.splititer(f"({escape})", text)

    for part in parts:
        if not part:
            continue

        if part in special_tokens:
            token_bytes = part.encode("utf-8")
            counter[(token_bytes,)] += 1
        else:
            for words in PATTERN.finditer(part):
                token = words.group(0)
                byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                counter[byte_tuple] += 1
    return counter


def vocalbulary_init(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = vocalbulary_init(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    chunks: list[int] = []

    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, 10, b"<|endoftext|")

    words_counter: Counter[tuple[bytes, ...]] = Counter()
    args_list = [
        (start, end, input_path, special_tokens)
        for start, end in zip(chunks[:-1], chunks[1:])
    ]
            
    if PARALLEL:
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.starmap(pre_tokenization_with_chunk, args_list),
                    total=len(args_list),
                    desc="Pre-tokenizing chunks",
                )
            )
        for partial_counter in results:
            words_counter.update(partial_counter)
    else:
        for args in tqdm(args_list, desc="Pre-tokenizing chunks"):
            words_counter.update(pre_tokenization_with_chunk(*args))
    
    pairs_frequency: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    pairs_words_map: Dict[tuple[bytes, bytes], list[tuple[bytes, ...]]] = {}
    words_pairs_map: Dict[tuple[bytes, ...], list[tuple[bytes, bytes]]] = {}
    

    for word in words_counter:
        words_pairs_map[word] = []
        if len(word) == 1:
            continue
        for i in range(len(word) - 1):
            pair: tuple[tuple[int], tuple[int]] = (tuple(-b for b in word[i]), tuple(-b for b in word[i + 1]))
            pairs_frequency[pair] -= words_counter[word]
            words_pairs_map[word].append(word[i: i + 2])
            if word[i: i + 2] in pairs_words_map:
                pairs_words_map[word[i: i + 2]].append(word)
            else:
                pairs_words_map[word[i: i + 2]] = [word]

    pairs_frequency_heap = [(frequency, pair) for pair, frequency in pairs_frequency.items()]
    heapq.heapify(pairs_frequency_heap)

    totel_loop = vocab_size - len(vocab)
    for i in range(totel_loop):
        if not pairs_frequency_heap:
            break

        frequency, top_pair = heapq.heappop(pairs_frequency_heap)
        del pairs_frequency[top_pair]
        
        pair_bytes = (
            bytes(tuple(-b for b in top_pair[0])), 
            bytes(tuple(-b for b in top_pair[1]))
        )
        
        new_token = pair_bytes[0] + pair_bytes[1]
        
        vocab[len(vocab)] = new_token
        merges.append(pair_bytes)
        
        for word in pairs_words_map.pop(pair_bytes, []):
            words_pairs_map[word].remove(pair_bytes)
            num = words_counter.pop(word)
            pre = None
            prt = 0
            
            word_new: list[bytes] = []
            new_pairs: list[tuple[bytes, bytes]] = []
            while prt < len(word) - 1:
                if word[prt] == pair_bytes[0] and word[prt + 1] == pair_bytes[1]:
                    if pre:
                        new_pair_pre: tuple[tuple[int, ...], tuple[int, ...]] = (tuple(-b for b in pre), tuple(-b for b in new_token))
                        pairs_frequency[new_pair_pre] -= num
                        new_pairs.append((pre, new_token))
                        pairs_frequency[(tuple(-b for b in pre), top_pair[0])] += num
                        
                    if prt < len(word) - 2:
                        new_pair_suf: tuple[tuple[int, ...], tuple[int, ...]] = (tuple(-b for b in new_token), tuple(-b for b in word[prt + 2]))
                        pairs_frequency[new_pair_suf] -= num
                        new_pairs.append((new_token, word[prt + 2]))
                        pairs_frequency[(top_pair[1], tuple(-b for b in word[prt + 2]))] += num
                    
                    if pre:
                        word_new.append(pre)
                    word_new.append(new_token)
                    prt += 2
                    pre = new_token
                else:
                    if pre:
                        word_new.append(pre)
                    pre = word[prt]
                    prt += 1
            if pre != new_token and pre:
                word_new.append(pre)
            word_new_tuple: tuple[bytes, ...] = tuple(word_new)
            del word_new
            
            words_pairs_map[word_new_tuple] = words_pairs_map.pop(word)
            
            for _ in words_pairs_map[word_new_tuple]:
                pairs_words_map[_].remove(word)
                pairs_words_map[_].append(word_new_tuple)
            
            for new_pair in new_pairs:
                words_pairs_map[word_new_tuple].append(new_pair)
            
            for new_pair in new_pairs:
                if new_pair in pairs_words_map:
                    pairs_words_map[new_pair].append(word_new_tuple)
                else:
                    pairs_words_map[new_pair] = [word_new_tuple]
            words_counter[word_new_tuple] = num
                    
        pairs_frequency_heap = [(frequency, pair) for pair, frequency in pairs_frequency.items()]
        heapq.heapify(pairs_frequency_heap)
        
    return vocab, merges


if __name__ == "__main__":
    input_path = "/home/lu0qlng/cs336/assignment1-basics/tests/fixtures/hello.txt"
    vocab, merges = train_bpe(input_path, 263, ["<|endoftext|"])
    
    pass
