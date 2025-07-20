from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import os
from typing import Dict, List, Tuple

def pre_tokenize(input_text: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    word_counts = {}
    
    # 构建一个能分割文本的、包含所有特殊token的正则表达式
    # re.escape确保了特殊token中的元字符被正确处理
    # f"({ ... })" 创建了一个捕获组，这样 re.split 可以在结果中保留分隔符
    special_pattern = "|".join(re.escape(s) for s in special_tokens)
    special_splitter = re.compile(f'({special_pattern})')

    base_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 1. 按照特殊token分割文本
    text_parts = special_splitter.split(input_text)

    # 2. 遍历分割后的部分
    for i, part in enumerate(text_parts):
        if not part:  # 跳过 re.split 可能产生的空字符串
            continue
            
        if i % 2 == 1:  # 奇数索引是分隔符本身，即特殊token
            token_bytes = part.encode('utf-8')
            word_tuple = (token_bytes,)
            word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
        else:  # 偶数索引是普通文本
            # 对普通文本部分应用基础的正则表达式
            for match in base_pattern.finditer(part):
                token_str = match.group(0)
                token_bytes = token_str.encode('utf-8')
                word_tuple = tuple(bytes([b]) for b in token_bytes)
                word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
                
    return word_counts

def get_pair_frequencies(input : dict):
    output = {}
    for word, count in input.items():
        for i in range(len(word) - 1):
            left_char = word[i]
            right_char = word[i + 1]
            pair = (left_char, right_char)
            output[pair] = output.get(pair, 0) + count
    return output

def find_key(input : dict):
    max_value = max(input.values())
    max_key = [k for k, v in input.items() if v == max_value]
    if (b"o", b"g") in max_key:
        print("Multiple max keys found:", max_key)
        print("Using the first one:", max(max_key))
    
    return max(max_key)

def merge_single_word(word_tuple, pair_to_merge):
    new_word = list()
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair_to_merge:
            new_word.append(word_tuple[i] + word_tuple[i + 1])
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个字节级别的BPE分词器。
    """
    # --- 1. 初始化 ---
    final_vocab: Dict[int, bytes] = {}
    merges: List[Tuple[bytes, bytes]] = []
    
    # 首先添加特殊tokens到词汇表开头
    current_id = 0
    for token_str in special_tokens:
        final_vocab[current_id] = token_str.encode('utf-8')
        current_id += 1
    
    # 然后填充基础词汇表 (0-255字节)
    for i in range(256):
        final_vocab[current_id] = bytes([i])
        current_id += 1
        
    # --- 2. 全局预分词和频率统计 ---
    word_counts: Dict[Tuple[bytes, ...], int] = {}
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 32, "<|endoftext|>".encode("utf-8"))
            
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            
            # 对每个块进行预分词
            chunk_word_counts = pre_tokenize(chunk_text, special_tokens)
            
            # 聚合到全局词频统计中
            for word, count in chunk_word_counts.items():
                word_counts[word] = word_counts.get(word, 0) + count

    # --- 3. BPE 迭代合并 ---
    num_merges_needed = vocab_size - len(final_vocab)

    for i in range(num_merges_needed):
        pair_freqs = get_pair_frequencies(word_counts)
        if not pair_freqs:
            break
        
        best_pair = find_key(pair_freqs)
        merges.append(best_pair)
        
        words_to_update = {}
        # 1. 找出所有包含 best_pair 的词
        for word, count in word_counts.items():
            if best_pair[0] in word and best_pair[1] in word:
                # 这是一个潜在的候选词，但我们还需要检查它们是否相邻
                for j in range(len(word) - 1):
                    if (word[j], word[j+1]) == best_pair:
                        words_to_update[word] = count
                        break # 找到一个匹配就够了，处理下一个词
        
        # 2. 更新这些词
        for word, count in words_to_update.items():
            # 从旧字典中移除
            del word_counts[word]
            # 创建新词并添加到字典中
            new_word = merge_single_word(word, best_pair)
            word_counts[new_word] = word_counts.get(new_word, 0) + count

        new_token_bytes = best_pair[0] + best_pair[1]
        final_vocab[current_id] = new_token_bytes
        current_id += 1
    return final_vocab, merges