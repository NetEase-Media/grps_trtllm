import openai
import sys

if len(sys.argv) < 5:
    print("Usage: python open_cli.py <server> <txt-path> <append_prompt> <stream>")
    exit(1)

server = sys.argv[1]
txt_path = sys.argv[2]
stream = False
if sys.argv[4].lower() == "true" or sys.argv[3].lower() == "1":
    stream = True

with open(txt_path, "r") as f:
    prompt = f.read()
    prompt += str(sys.argv[3])

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)
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
