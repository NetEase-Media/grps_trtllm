import os
import time

import openai
import sys
import base64

from numpy import uint8

if len(sys.argv) < 5:
    print("Usage: python open_cli.py <server> <prompt> <stream> <img_path>")
    exit(1)

server = sys.argv[1]
prompt = sys.argv[2]
stream = False
if sys.argv[3].lower() == "true" or sys.argv[3].lower() == "1":
    stream = True
img_path = sys.argv[4]

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)

begin = time.time()

img_base64_url = 'data:image/' + os.path.splitext(img_path)[1][1:] + ';base64,'
with open(img_path, 'rb') as img_file:
    encoded_data = base64.b64encode(img_file.read()).decode('utf-8')
    img_base64_url += encoded_data
# print(f'Image base64 url: {img_base64_url}', flush=True)

res = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64_url
                    }
                },
            ]
        },
    ],
    max_tokens=1024,
    stream=stream
)

if stream:
    for message in res:
        print(message)
else:
    print(res)
    latency = (time.time() - begin) * 1000
    input_token_len = res.usage.prompt_tokens
    output_token_len = res.usage.completion_tokens
    total_tokens = res.usage.total_tokens
    speed = total_tokens / latency * 1000
    print(f'Latency: {latency} ms', flush=True)
    print(f'Input tokens: {input_token_len}, Output tokens: {output_token_len}, Total tokens: {total_tokens}',
          flush=True)
    print(f'Speed: {speed} tokens/s', flush=True)
