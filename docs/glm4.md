# grps-trtllm

glm-4-9b-chat模型的部署示例。

## 构建trtllm引擎

```bash
# 下载glm-4-9b-chat模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/THUDM/glm-4-9b-chat /tmp/glm-4-9b-chat
# 转换tiktoken tokenizer.model为huggingface tokenizer.json格式
python3 ./tools/glm4_tiktoken_to_hf.py /tmp/glm-4-9b-chat/ /tmp/glm-4-9b-chat/

# 进入TensorRT-LLM/examples/chatglm目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/chatglm/
# 转换ckpt
rm -rf /tmp/glm-4-9b-chat/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/glm-4-9b-chat \
--output_dir /tmp/glm-4-9b-chat/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/glm-4-9b-chat/trt_engines/
trtllm-build --checkpoint_dir /tmp/glm-4-9b-chat/tllm_checkpoint/ \
--output_dir /tmp/glm-4-9b-chat/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/glm-4-9b-chat/ \
--engine_dir=/tmp/glm-4-9b-chat/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_glm4.yml

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
    "model": "glm-4-9b-chat",
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
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "created": 1725527373,
 "model": "glm-4-9b-chat",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好，我是一个人工智能助手，我的名字是 ChatGLM，是基于清华大学 KEG 实验室和智谱 AI 公司于 2024 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 12,
  "completion_tokens": 49,
  "total_tokens": 61
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4-9b-chat",
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
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1725527389,"model":"glm-4-9b-chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1725527389,"model":"glm-4-9b-chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"，"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1725527389,"model":"glm-4-9b-chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"我是一个"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好，我是一个人工智能助手，我的名字是 ChatGLM，是基于清华大学 KEG 实验室和智谱 AI 公司于 2024 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1725527407, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=49, prompt_tokens=12, total_tokens=61))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content='你好', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725527423, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content='，', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725527423, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content='我是一个', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725527423, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: Sure, I can help with that. Let me check the current weather in Boston.get_current_weather
{"location": "Boston, MA", "unit": "fahrenheit"}, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result(59.0) ...
Final server response:  The current temperature in Boston is 59.0 degrees Fahrenheit.
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```