import json

import openai
import sys

if len(sys.argv) < 2:
    print("Usage: python openai_func_call.py <server>")
    exit(1)
server = sys.argv[1]


# dummy function.
def get_current_weather(location, unit):
    return 59.0


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }
        }
    }
]

client = openai.Client(
    api_key="cannot be empty",
    base_url=f"http://{server}/v1"
)

print("Query server with question: What's the weather like in Boston today? ...")
res = client.chat.completions.create(
    model="",
    messages=[
        {
            "content": "What's the weather like in Boston today?",
            "role": "user",
        }
    ],
    tools=tools,
    tool_choice="auto",
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

# call function by name and parameters
tool_call = res.choices[0].message.tool_calls[0]
function = tool_call.function
if function.name == "get_current_weather":
    arguments = json.loads(function.arguments)
    location = arguments["location"]
    unit = arguments.get("unit", "fahrenheit")

    print(
        f'Server response: thought: {res.choices[0].message.content}, call local function({function.name}) '
        f'with arguments: location={location}, unit={unit}')
    weather = get_current_weather(location, unit)
    # print(f"The weather in {location} is {weather} degrees {unit}.")

    # send the result back to the server
    print(f'Send the result back to the server with function result({weather}) ...')
    res = client.chat.completions.create(
        model="",
        messages=[
            {
                "content": "What's the weather like in Boston today?",
                "role": "user",
            },
            {
                "role": "assistant",
                "content": res.choices[0].message.content,
                "tool_calls": res.choices[0].message.tool_calls,
            },
            {
                "role": "assistant",
                "content": f"{weather}",
                "tool_calls": [],
            }
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )

    print(f'Final server response: {res.choices[0].message.content}')


else:
    print(f"Unknown function: {function.name}")
    exit(1)
