
from typing import Iterable, Iterator
import json
import regex as re

class Tokenizer:
    PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab_encode = {v: k for k, v in vocab.items()}
        self.vocab_decode = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> 'Tokenizer':
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges and (optionally) a list of special tokens.
        """
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
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        tokens: list[int] = []
        if not text:
            return []
        
        special_tokens = self.special_tokens
        vocab = self.vocab_encode
        merges = self.merges
        
        
        if self.special_tokens:
            special_pattern = "|".join(re.escape(st) for st in special_tokens)
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        del text
        for part in parts:
            if not part:
                continue
            if special_tokens and part in special_tokens:
                tokens.append(vocab[part.encode("utf-8")])
            else:
                for words in self.PATTERN.finditer(part):
                    token = words.group(0).encode("utf-8")
                    token_list = [bytes([b]) for b in token]
                    while True:
                        pairs = {(token_list[i - 1], token_list[i]): float("inf") for i in range(1, len(token_list))}
                        for pair in pairs:
                            if pair in merges:
                                pairs[pair] = merges.index(pair)
                                
                        min_pair, min_index = (), float("inf")
                        for pair, index in pairs.items():
                            if index < min_index:
                                min_pair, min_index = pair, index
                        if min_pair:
                            i = 0
                            while i < len(token_list):
                                if token_list[i - 1] == min_pair[0] and token_list[i] == min_pair[1]:
                                    token_list[i - 1] = min_pair[0] + min_pair[1]
                                    token_list.pop(i)
                                else:
                                    i += 1
                        else: 
                            break
                    tokens.extend([vocab[t] for t in token_list])
                            
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """ 
        for i in iterable:
            yield from self.encode(i)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        byte_sequence = b''.join(self.vocab_decode[i] for i in ids)
        return byte_sequence.decode('utf-8', errors='replace')