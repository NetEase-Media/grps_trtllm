import argparse

import torch
from transformers import AutoModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='/tmp/MiniCPM-V-2_6',
                        help='')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/tmp/MiniCPM-V-2_6/llm',
                        help='')
    args = parser.parse_args()
    return args


args = parse_arguments()

model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
vl_gpt = model.to(torch.bfloat16).cuda().eval()
vl_gpt.llm.save_pretrained(args.output_dir)
