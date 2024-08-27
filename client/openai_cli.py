import time

import openai
import sys

if len(sys.argv) < 4:
    print("Usage: python open_cli.py <server> <prompt> <stream>")
    exit(1)

server = sys.argv[1]
prompt = sys.argv[2]
stream = False
if sys.argv[3].lower() == "true" or sys.argv[3].lower() == "1":
    stream = True

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)
begin = time.time()
res = client.chat.completions.create(
    model="qwen2-instruct",
    messages=[
        {
            "content": prompt,
            "role": "user",
        }
    ],
    top_p=0.3,
    max_tokens=1024,
    temperature=0.1,
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
