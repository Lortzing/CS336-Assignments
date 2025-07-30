
from typing import Iterable, Iterator
import json
import regex as re
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
_tokenizer_for_worker = None
        
def _init_worker(tokenizer: "Tokenizer"):
    global _tokenizer_for_worker
    _tokenizer_for_worker = tokenizer

def _encode_chunk(text: str) -> list[int]:
    return _tokenizer_for_worker.encode(text)

class Tokenizer:
    PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab_encode = {v: k for k, v in vocab.items()}
        self.vocab_decode = vocab
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> 'Tokenizer':
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
        
        def gpt2_bytes_to_unicode():
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            characters = [chr(n) for n in cs]
            d = dict(zip(bs, characters))
            return d
        
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def from_files_special_format(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> 'Tokenizer':
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        vocab = {int(k): v.encode("utf-8") for k, v in vocab_json.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)

        merges = [ (a.encode("utf-8"), b.encode("utf-8")) for a, b in merges_json ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        tokens: list[int] = []
        vocab = self.vocab_encode
        merge_ranks = self.merge_ranks
        special_tokens = self.special_tokens

        if special_tokens:
            pattern = "|".join(re.escape(st) for st in special_tokens)
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        del text

        for part in parts:
            if not part:
                continue

            if special_tokens and part in special_tokens:
                tokens.append(vocab[part.encode("utf-8")])
                continue

            for match in self.PATTERN.finditer(part):
                b = match.group(0).encode("utf-8")
                symbols = [bytes([c]) for c in b]

                while len(symbols) >= 2:
                    pairs = [
                        ((symbols[i], symbols[i + 1]), i)
                        for i in range(len(symbols) - 1)
                        if (symbols[i], symbols[i + 1]) in merge_ranks
                    ]
                    if not pairs:
                        break

                    best_pair, best_pos = min(pairs, key=lambda x: merge_ranks[x[0]])
                    merged = best_pair[0] + best_pair[1]
                    symbols = symbols[:best_pos] + [merged] + symbols[best_pos + 2:]

                tokens.extend(vocab[s] for s in symbols)

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for i in iterable:
            yield from self.encode(i)

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b''.join(self.vocab_decode[i] for i in ids)
        return byte_sequence.decode('utf-8', errors='replace')

    def encode_file_to_memmap(
        self,
        input_path: str,
        output_path: str,
        chunk_chars: int = 1_000_000,
        buffer_size: int = 200_000,
    ):
        file_size = os.path.getsize(input_path)
        estimated_tokens = int(file_size // 4)
        tmp_path = output_path + ".tmp.dat"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        memmap = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=(estimated_tokens,))
        buffer = np.empty(buffer_size, dtype=np.uint16)
        buffer_idx = 0
        total_written = 0

        def write_buffer():
            nonlocal buffer_idx, total_written
            if buffer_idx == 0:
                return
            end = total_written + buffer_idx
            if end > memmap.shape[0]:
                raise RuntimeError("Token overflow; increase estimated token count.")
            memmap[total_written:end] = buffer[:buffer_idx]
            total_written = end
            buffer_idx = 0

        with open(input_path, 'r', encoding='utf-8') as f, tqdm(
            total=file_size, desc="Encoding text", unit="B", unit_scale=True
        ) as pbar:
            lines = []
            chars = 0
            last_pos = 0

            while True:
                line = f.readline()
                if not line:
                    break
                lines.append(line)
                chars += len(line)
                if chars >= chunk_chars:
                    tokens = self.encode(''.join(lines))
                    for t in tokens:
                        buffer[buffer_idx] = t
                        buffer_idx += 1
                        if buffer_idx >= buffer_size:
                            write_buffer()
                    lines.clear()
                    chars = 0
                    pbar.update(f.tell() - last_pos)
                    last_pos = f.tell()

            if lines:
                tokens = self.encode(''.join(lines))
                for t in tokens:
                    buffer[buffer_idx] = t
                    buffer_idx += 1
                    if buffer_idx >= buffer_size:
                        write_buffer()
                pbar.update(f.tell() - last_pos)

            write_buffer()
            memmap.flush()

        final = np.memmap(tmp_path, dtype=np.uint16, mode='r', shape=(total_written,))
        np.save(output_path, np.array(final, copy=True))
        os.remove(tmp_path)

        print(f"Saved {total_written:,} tokens to {output_path}.npy")
        
    def encode_file_to_memmap_parallel(
        self,
        input_path: str,
        output_path: str,
        chunk_chars: int = 1_000_000,
        buffer_size: int = 200_000,
        num_workers: int = multiprocessing.cpu_count()
    ):
        file_size = os.path.getsize(input_path)
        estimated_tokens = int(file_size // 4)
        tmp_path = output_path + ".tmp.dat"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        memmap = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=(estimated_tokens,))
        buffer = np.empty(buffer_size, dtype=np.uint16)
        buffer_idx = 0
        total_written = 0

        def write_buffer():
            nonlocal buffer_idx, total_written
            if buffer_idx == 0:
                return
            end = total_written + buffer_idx
            if end > memmap.shape[0]:
                raise RuntimeError("Token overflow; increase estimated token count.")
            memmap[total_written:end] = buffer[:buffer_idx]
            total_written = end
            buffer_idx = 0

        text_chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = []
            chars = 0
            for line in f:
                lines.append(line)
                chars += len(line)
                if chars >= chunk_chars:
                    text_chunks.append(''.join(lines))
                    lines.clear()
                    chars = 0
            if lines:
                text_chunks.append(''.join(lines))

        print(f"Total {len(text_chunks)} chunks. Starting parallel encoding...")

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self,)
        ) as pool:
            for tokens in tqdm(pool.imap(_encode_chunk, text_chunks), total=len(text_chunks), desc="Encoding"):
                for t in tokens:
                    buffer[buffer_idx] = t
                    buffer_idx += 1
                    if buffer_idx >= buffer_size:
                        write_buffer()

        write_buffer()
        memmap.flush()

        final = np.memmap(tmp_path, dtype=np.uint16, mode='r', shape=(total_written,))
        np.save(output_path, np.array(final, copy=True))
        os.remove(tmp_path)

        print(f"Saved {total_written:,} tokens to {output_path}.npy")

    

if __name__ == '__main__':
    tokenizer = Tokenizer.from_files(vocab_filepath="./results/vocab_tiny_gpt2.json", merges_filepath="./results/merges_tiny_gpt2.txt", special_tokens=["<|endoftext|>"])
    tokenizer.encode_file_to_memmap_parallel(input_path="./data/TinyStoriesV2-GPT4-train.txt", output_path="./results/TinyStoriesV2-GPT4-train.memmap")
    tokenizer.encode_file_to_memmap_parallel(input_path="./data/TinyStoriesV2-GPT4-valid.txt", output_path="./results/TinyStoriesV2-GPT4-valid.memmap")
    