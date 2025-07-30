import regex as re
import os
from typing import Dict, List, Tuple
from typing import BinaryIO
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
import heapq
import json
# set_start_method("spawn", force=True)


PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
PARALLEL = True


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:

    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in tqdm(range(1, len(chunk_boundaries) - 1)):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

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

    del text
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
    del parts
    return counter


def vocabulary_init(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab: Dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    return vocab

class PairFreq:
    __slots__ = ('freq', 'pair')
    def __init__(self, freq: int, pair: Tuple[bytes, bytes]):
        self.freq = freq
        self.pair = pair
    def __lt__(self, other: 'PairFreq'):
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair
    
    def __gt__(self, other: 'PairFreq'):
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.pair > other.pair

def train_bpe(input_path: str|os.PathLike, vocab_size: int, special_tokens: List[str], desired_num_chunks: int = 1000) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    if special_tokens:
        special_tokens.sort(key=len, reverse=True)
    vocab = vocabulary_init(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    chunks: List[int] = []
    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
    words_counter: Counter[Tuple[bytes, ...]] = Counter()
    args_list = [(start, end, input_path, special_tokens) for start, end in zip(chunks[:-1], chunks[1:])]
    if PARALLEL:
        with Pool(processes=cpu_count()) as pool:
            for partial_counter in pool.starmap(pre_tokenization_with_chunk, args_list):
                words_counter.update(partial_counter)
    else:
        for args in args_list:
            words_counter.update(pre_tokenization_with_chunk(*args))

    pair_freq: Counter[Tuple[bytes, bytes]] = Counter()
    pair_to_words: Dict[Tuple[bytes, bytes], set] = {}
    for word, count in words_counter.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += count
            pair_to_words.setdefault(pair, set()).add(word)

    target_merges = vocab_size - len(vocab)

    heap: List[PairFreq] = []
    for pair, freq in pair_freq.items():
        if freq > 0:
            heapq.heappush(heap, PairFreq(freq, pair))
    for _ in range(target_merges):
        if not heap:
            break

        top_entry = heapq.heappop(heap)
        while heap and (top_entry.pair not in pair_freq or pair_freq[top_entry.pair] != top_entry.freq):
            top_entry = heapq.heappop(heap)
            
        if top_entry.pair not in pair_freq or pair_freq.get(top_entry.pair, 0) != top_entry.freq:
            break
        best_pair = top_entry.pair
        best_freq = top_entry.freq
        if best_freq == 0:
            break
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        words_with_pair = pair_to_words.pop(best_pair, set())
        pair_freq.pop(best_pair, None)
        updated_pairs: set = set()

        for word in list(words_with_pair):
            if word not in words_counter:
                continue
            word_count = words_counter[word]
            new_word_tokens: List[bytes] = []
            
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
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
                if pair in pair_freq:
                    pair_freq[pair] -= occ * word_count
                else:
                    pair_freq[pair] = 0
                if pair_freq.get(pair, 0) <= 0:
                    pair_freq.pop(pair, None)
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        pair_to_words.pop(pair, None)
                updated_pairs.add(pair)
                
            for pair, occ in new_pairs.items():
                pair_freq[pair] = pair_freq.get(pair, 0) + occ * word_count
                pair_to_words.setdefault(pair, set()).add(new_word)
                updated_pairs.add(pair)

        for pair in updated_pairs:
            if pair in pair_freq and pair_freq[pair] > 0:
                heapq.heappush(heap, PairFreq(pair_freq[pair], pair))
    
    return vocab, merges

def save_bpe_special(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_path: str,
    dataset: str,
):
    os.makedirs(out_path, exist_ok=True)

    vocab_serializable = {
        v.decode("utf-8", errors="replace"): k for k, v in sorted(vocab.items(), key=lambda x: x[0])
    }

    merges_serializable = []
    for a, b in merges:
        a_decoded = a.decode("utf-8", errors="replace")
        b_decoded = b.decode("utf-8", errors="replace")
        merges_serializable.append(f"{a_decoded} {b_decoded}\n")

    with open(os.path.join(out_path, f"vocab_{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_path, f"merges_{dataset}.txt"), "w", encoding="utf-8") as f:
        f.writelines(merges_serializable)


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_path: str,
    dataset: str,
):
    os.makedirs(out_path, exist_ok=True)
    
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    byte2unicode = bytes_to_unicode()

    def b2u_str(b: bytes) -> str:
        return ''.join(byte2unicode[byte] for byte in b)

    vocab_serializable = {
        b2u_str(token_bytes): token_id
        for token_id, token_bytes in sorted(vocab.items(), key=lambda x: x[0])
    }

    with open(os.path.join(out_path, f"vocab_{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=True)

    merges_serializable = []
    for a, b in merges:
        merges_serializable.append(f"{b2u_str(a)} {b2u_str(b)}\n")

    with open(os.path.join(out_path, f"merges_{dataset}.txt"), "w", encoding="utf-8") as f:
        f.writelines(merges_serializable)

        
if __name__ == "__main__":
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    save_bpe(vocab, merges, "./results/", "tiny_gpt2")
    pass
