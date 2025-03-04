import base64
import urllib.request
from io import BytesIO

import openai
import sys

from PIL import Image
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

if len(sys.argv) < 4:
    print("Usage: python olm_ocr_cli.py <server> <pdf_url> <stream>")
    exit(1)

server = sys.argv[1]
pdf_url = sys.argv[2]
stream = False
if sys.argv[3].lower() == "true" or sys.argv[3].lower() == "1":
    stream = True

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)

pdf_path = pdf_url.split("/")[-1]
urllib.request.urlretrieve(pdf_url, pdf_path)
# Render page 1 to an image
image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1024)
# print('len(image_base64)', len(image_base64))
image = Image.open(BytesIO(base64.b64decode(image_base64)))
# Build the prompt, using document metadata
anchor_text = get_anchor_text(pdf_path, 1, pdf_engine="pdfreport", target_length=4000)
prompt = build_finetuning_prompt(anchor_text)
print(f'prompt: {prompt}')

# Request to openai llm server.
new_message = {
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {
                "url": f'data:image/png;base64,{image_base64}'
            }
        },
        {
            "type": "text",
            "text": prompt
        }
    ]
}
client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)
res = client.chat.completions.create(
    model="",
    messages=[new_message],
    stream=True
)

if stream:
    for message in res:
        print(message)
else:
    print(res)
