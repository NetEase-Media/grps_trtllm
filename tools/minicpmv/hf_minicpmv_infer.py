# test.py
import sys

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

if len(sys.argv) != 3:
    print(f'Usage: python {sys.argv[0]} <model_path> <image_path>')
    exit(1)

model_path = sys.argv[1]
image_path = sys.argv[2]

model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

image = Image.open(image_path).convert('RGB')
question = '描述一下这张图片'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
# res = model.chat(
#     image=None,
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True,
#     stream=True
# )
#
# generated_text = ""
# for new_text in res:
#     generated_text += new_text
#     print(new_text, flush=True, end='')
