import sys

from pydantic import Field
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai_like import OpenAILike

if len(sys.argv) < 2:
    print("Usage: python openai_func_call.py <server>")
    exit(1)
server = sys.argv[1]


def get_weather(
        location: str = Field(
            description="The city and state, e.g. San Francisco, CA"
        ),
        unit: str = Field(
            description="unit, e.g. celsius, fahrenheit")
) -> float:
    # print(location)
    return 59.0


llm = OpenAILike(
    api_base=f"http://{server}/v1/",
    api_key="grps_trtllm",
    timeout=600,  # secs
    is_chat_model=True,
    is_function_calling_model=True,
    context_window=32768,
    model="",
)

tool = FunctionTool.from_defaults(get_weather, name="get_weather",
                                  description="Get the current weather in a given location")
# print('tool: ', tool)
print('Query: What is the weather in Boston today?')
agent = OpenAIAgent.from_tools(tools=[tool], verbose=True, max_function_calls=1, allow_parallel_tool_calls=False,
                               llm=llm)
res = agent.chat('What is the weather in Boston today?')
print(f'Response: {res}')
