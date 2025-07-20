import regex as re
    from collections import defaultdict, Counter
    from multiprocessing import Pool
    from cs336_basics.pretokenization_example import find_chunk_boundaries

    # --- GPT-2 regex pattern ---
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # --- BPE Tokenizer Training ---
    def train_bpe(input_path: str|os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 8):
        special_token_bytes = [tok.encode("utf-8") for tok in special_tokens]

        # Step 1: Parallel pre-tokenization
        def process_chunk(start_end):
            start, end = start_end
            with open(input_path, "rb") as f:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
            local_counts = defaultdict(int)
            for match in re.finditer(PAT, chunk):
                token_bytes = match.group().encode("utf-8")
                local_counts[tuple(token_bytes)] += 1
            return local_counts

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
        with Pool(num_processes) as pool:
            results = pool.map(process_chunk, chunk_ranges)

        token_freqs = Counter()
        for local in results:
            token_freqs.update(local)

        # Step 2: Initialize vocabulary
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_token_bytes:
            vocab[len(vocab)] = token

        merges = []
        next_token_id = len(vocab)

        def get_pair_counts():
            pair_counts = defaultdict(int)
            for token, freq in token_freqs.items():
                for i in range(len(token) - 1):
                    pair = (token[i:i+1], token[i+1:i+2])
                    pair_counts[pair] += freq
            return pair_counts

        # Step 3: Merge loop
        while len(vocab) < vocab_size:
            pair_counts = get_pair_counts()
            if not pair_counts:
                break
            (a, b), _ = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            new_token = a + b
            merges.append((a, b))

            updated_freqs = defaultdict(int)
            for token, freq in token_freqs.items():
                new_tokenized = []
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and token[i:i+1] == a and token[i+1:i+2] == b:
                        new_tokenized.append(new_token)
                        i += 2
                    else:
                        new_tokenized.append(token[i:i+1])
                        i += 1
                updated_freqs[tuple(new_tokenized)] += freq
            token_freqs = updated_freqs
            vocab[next_token_id] = new_token
            next_token_id += 1

        return vocab, merges