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
res = client.chat.completions.create(
    model="qwen2-instruct",
    messages=[
        {
            "content": prompt,
            "role": "user",
        }
    ],
    max_tokens=512,
    temperature=0.7,
    stream=stream
)
if stream:
    for message in res:
        print(message)
else:
    print(res)
