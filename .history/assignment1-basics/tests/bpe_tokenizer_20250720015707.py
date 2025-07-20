import regex as re
import os
from typing import Dict, List, Tuple
from typing import BinaryIO
from collections import Counter
import heapq

PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


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


def pre_tokenization(start: int, end: int, input_path: str | os.PathLike, special_tokens: List[str]) -> Counter[str]:
    counter = Counter()
    
    chunk = ""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    escape = "|".join(map(re.escape, special_tokens))
    parts = re.splititer(f"({escape})", chunk)
    
    for part in parts:
        if not part:
            continue

        if part in special_tokens:
            token_bytes = part.encode("utf-8")
            counter[(token_bytes)] += 1
        else:
            token = part.group(0)
            if token in special_tokens:
                continue
            byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
            counter[byte_tuple] += 1
    
        
        

        for words in PATTERN.finditer(chunk):
            token = words.group(0)
            if token in special_tokens:
                continue
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

    counter = Counter()
    for start, end in zip(chunks[:-1], chunks[1:]):
        counter.update(pre_tokenization(start, end, input_path, special_tokens))

    return vocab, merges


if __name__ == "__main__":
    chunk = "dog <|endoftext|> cat"
    special_tokens = ["<|endoftext|"]
    for match in PATTERN.finditer(chunk):
        token = match.group(0)
        if token in special_tokens:
            continue
        else:
            print(token)
    pass
