import argparse

import torch
from janus.models import MultiModalityCausalLM
from transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='/tmp/Janus-Pro-7B',
                        help='')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/tmp/Janus-Pro-7B/llm',
                        help='')
    args = parser.parse_args()
    return args


args = parse_arguments()

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    args.model_dir, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
vl_gpt.language_model.save_pretrained(args.output_dir)
