import argparse
import json

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
llm_conf_path = args.output_dir + '/config.json'

# Modify model type.
# config.model_type = 'qwen2'
# config.slice_config.model_type = 'qwen2'
llm_conf_json = None
with open(llm_conf_path, 'r') as f:
    llm_conf_json = json.load(f)
llm_conf_json['model_type'] = 'qwen2'
llm_conf_json['slice_config']['model_type'] = 'qwen2'
with open(llm_conf_path, 'w') as f:
    json.dump(llm_conf_json, f, indent=4)
