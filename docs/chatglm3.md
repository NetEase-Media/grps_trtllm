# grps-trtllm

chatglm3-6b模型的部署示例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载chatglm3-6b模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/THUDM/chatglm3-6b /tmp/chatglm3-6b

# 进入TensorRT-LLM/examples/chatglm目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/chatglm/
# 转换ckpt
rm -rf /tmp/chatglm3-6b/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/chatglm3-6b \
--output_dir /tmp/chatglm3-6b/tllm_checkpoint/ --dtype float16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/chatglm3-6b/trt_engines/
trtllm-build --checkpoint_dir /tmp/chatglm3-6b/tllm_checkpoint/ \
--output_dir /tmp/chatglm3-6b/trt_engines/ \
--gemm_plugin float16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/chatglm3-6b/ \
--engine_dir=/tmp/chatglm3-6b/trt_engines/
# 回到工程根目录
cd ../../../../
```

## 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_chatglm3.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 模拟请求

```bash
# curl命令非stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm3-6b",
    "messages": [
      {
        "role": "user",
        "content": "你好，你是谁？"
      }
    ]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1725526115,
 "model": "chatglm3-6b",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好，我是 ChatGLM3-6B，是清华大学KEG实验室和智谱AI公司共同训练的语言模型。我的目标是通过回答用户提出的问题来帮助他们解决问题。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 14,
  "completion_tokens": 38,
  "total_tokens": 52
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm3-6b",
    "messages": [
      {
        "role": "user",
        "content": "你好，你是谁？"
      }
    ],
    "stream": true
  }'
# 返回如下：
: '
data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1725526135,"model":"chatglm3-6b","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1725526135,"model":"chatglm3-6b","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1725526135,"model":"chatglm3-6b","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"，"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好，我是 ChatGLM3-6B，是清华大学KEG实验室和智谱AI公司共同训练的语言模型。我的目标是通过回答用户提出的问题来帮助他们解决问题。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1725526156, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=38, prompt_tokens=14, total_tokens=52))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='你', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725526173, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='好', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725526173, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='，', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725526173, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: The user wants to know the weather in Boston today. The function 'get_current_weather' can be used to retrieve the weather information.get_current_weather
 ```python
tool_call(location='Boston', unit='celsius')
```, call local function(get_current_weather) with arguments: location=Boston, unit=celsius
Send the result back to the server with function result(59.0) ...
Final server response: According to the weather API, the current temperature in Boston is 59 degrees Celsius.
'

# llama-index ai agent模拟
python3 client/llamaindex_ai_agent.py 127.0.0.1:9997
# 返回如下：
: '
Query: What is the weather in Boston today?
Added user message to memory: What is the weather in Boston today?
=== Calling Function ===
Calling function: get_weather with args: {"location":"Boston","unit":"celsius"}
Got output: 59.0
========================

Response: The weather in Boston today is 59 degrees Celsius.
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```