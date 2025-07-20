from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import os
from typing import Dict, List, Tuple

def pre_tokenize(input_text: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    pass

def vocalbulary_init(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({i: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass