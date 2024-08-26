import random
import sys
import time
import requests
import threading
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/Qwen2-7B-Instruct')


def request(server, prompt):
    text_inp = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
    url = f'http://{server}/v2/models/ensemble/generate'
    data = {
        "text_input": text_inp,
        "max_tokens": 1024,
        "bad_words": "",
        "stop_words": ["<|im_start|>", "<|im_end|>"],
        "end_id": 151643,
        "pad_id": 151643,
        "top_p": 0.3,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 0.1,
        "stream": False
    }
    headers = {'Content-Type': 'application/json'}
    start = time.time()
    response = requests.post(url, json=data, headers=headers).json()
    end = time.time()
    latency = (end - start) * 1000
    text_output = response['text_output']
    print(text_output, flush=True)
    input_tokens = tokenizer(text_inp)['input_ids']
    # print input_tokens with space split
    # print(' '.join([str(i) for i in input_tokens]), flush=True)
    input_token_len = len(input_tokens)
    output_token_len = len(tokenizer(text_output)['input_ids'])
    total_tokens = input_token_len + output_token_len
    speed = total_tokens / latency * 1000
    print(f'Latency: {latency} ms', flush=True)
    print(f'Input tokens: {input_token_len}, Output tokens: {output_token_len}, Total tokens: {total_tokens}',
          flush=True)
    print(f'Speed: {speed} tokens/s', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python triton_txt_cli.py <server> <txt-path> <append_prompt>")
        exit(1)
    server = sys.argv[1]
    text_path = sys.argv[2]

    with open(text_path, "r") as f:
        prompt = f.read()
        prompt += str(sys.argv[3])

    request(server, prompt)
