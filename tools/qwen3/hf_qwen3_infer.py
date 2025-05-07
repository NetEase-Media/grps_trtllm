from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/tmp/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print(model)

# prepare the model input
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA"
#                     },
#                     "unit": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"]
#                     }
#                 },
#                 "required": ["location", "unit"]
#             }
#         }
#     }
# ]
#
# messages = [
#     {
#         "content": "你是一个ai助手。",
#         "role": "system",
#     },
#     {"role": "user", "content": "What's the weather like in Boston today?"},
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [{
#             "function": {
#                 "name": "get_current_weather",
#                 "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"fahrenheit\"}"
#             },
#         }],
#     },
#     {
#         "role": "tool",
#         "content": f"59.0",
#     }
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     # tools=tools,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
# )

prompt = "你好，你是谁？"
messages = [
    {"role": "user", "content": "你好，你是谁？"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
)

print(f'final text: {text}')
model_inputs = tokenizer([text], return_tensors="pt")
id_str = ''
for token in model_inputs["input_ids"][0].numpy().tolist():
    id_str += str(token) + ' '
print(f'input_ids: {id_str}')

# conduct text completion
model_inputs = model_inputs.to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
