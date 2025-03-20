import argparse

import torch
from tensorrt_llm._utils import str_dtype_to_torch
from transformers import AutoModelForImageTextToText


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='/tmp/gemma-3-4b-it',
                        help='')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/tmp/gemma-3-4b-it/llm',
                        help='')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='')
    args = parser.parse_args()
    return args


args = parse_arguments()

model = AutoModelForImageTextToText.from_pretrained(args.model_dir, trust_remote_code=True, device_map="cpu")
print(model.language_model)
model.language_model.to(str_dtype_to_torch(args.dtype)).eval().save_pretrained(args.output_dir)
