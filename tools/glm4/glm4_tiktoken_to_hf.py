import json
import os
import sys
from typing import Optional
from transformers import AutoTokenizer

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()


def token_bytes_to_string(b):
    return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])


# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def generate_vocab_and_merges(encoder):
    mergeable_ranks = encoder._mergeable_ranks

    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        # assert len(merged) == 2
        if len(merged) != 2:
            # print("token:", token, "rank:", rank)
            # print("merged:", merged)
            continue

        merges.append(' '.join(map(token_bytes_to_string, merged)))

    # Also add special tokens
    vocab.update(encoder._special_tokens)

    return vocab, merges


def convert_tiktoken(encoder, output_dir=None):
    vocab, merges = generate_vocab_and_merges(encoder)

    added_tokens = [
        {
            "id": id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, id in encoder._special_tokens.items()
    ]

    os.makedirs(output_dir, exist_ok=True)

    pre_tokenizer = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {
                    "Regex": encoder._pat_str
                },
                "behavior": "Isolated",
                "invert": True,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            }
        ]
    }

    # https://huggingface.co/Xenova/gpt2/raw/main/tokenizer.json
    tokenizer_template = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    with open(os.path.join(output_dir, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
        json.dump(tokenizer_template, fp, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <tokenizer path> <output dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    convert_tiktoken(tokenizer.tokenizer, output_dir)
