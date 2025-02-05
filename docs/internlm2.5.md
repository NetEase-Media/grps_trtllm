# InternLM2.5

InternLM2.5模型的部署示例，以InternLM2.5-7B-Chat为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载internlm2_5-7b-chat模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/internlm/internlm2_5-7b-chat /tmp/internlm2_5-7b-chat

# 进入TensorRT-LLM/examples/internlm2/目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/internlm2/
# 转换ckpt
rm -rf /tmp/internlm2_5-7b-chat/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/internlm2_5-7b-chat \
--output_dir /tmp/internlm2_5-7b-chat/tllm_checkpoint/ --dtype bfloat16
# 构建引擎
rm -rf /tmp/internlm2_5-7b-chat/trt_engines/
trtllm-build --checkpoint_dir /tmp/internlm2_5-7b-chat/tllm_checkpoint/ \
--output_dir /tmp/internlm2_5-7b-chat/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/internlm2_5-7b-chat/ \
--engine_dir=/tmp/internlm2_5-7b-chat/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_internlm2.5.yml

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
    "model": "InternLM2.5-Chat",
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
 "id": "chatcmpl-4",
 "object": "chat.completion",
 "created": 1728558272,
 "model": "InternLM2.5-Chat",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是书生·浦语，由上海人工智能实验室开发的一款人工智能助手。我能够理解并回答你的问题，提供帮助和建议。有什么我可以为你做的吗？"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 107,
  "completion_tokens": 38,
  "total_tokens": 145
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternLM2.5-Chat",
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
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1728558298,"model":"InternLM2.5-Chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1728558298,"model":"InternLM2.5-Chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1728558298,"model":"InternLM2.5-Chat","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"我是"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是书生·浦语，由上海人工智能实验室开发的一款人工智能助手。我能够理解并回答你的问题，提供帮助和信息。有什么我可以为你做的吗？', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1728558315, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=38, prompt_tokens=107, total_tokens=145, completion_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='你好', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1728558330, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1728558330, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1728558330, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "直接说出上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='《拜托，只想干饭的北极熊超酷的！》的作者是弦三千。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1728558388, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=20, prompt_tokens=31808, total_tokens=31828, completion_tokens_details=None))
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-10', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='小说《拜托，只想干饭的北极熊超酷的！》的前几章内容主要讲述了主角楚云霁在北极生存的故事。以下是前几章的简要内容：\n\n1. 楚云霁穿成了一只北极熊，在暴风雪中醒来，发现自己在北极。他尝试捕猎，但遇到了一只白狼，白狼没有攻击他，反而带他去了自己的领地，并分享了捕猎的成果。\n\n2. 楚云霁在白狼的领地上发现了一些食物，包括海豹和驯鹿，他尝试用这些食物填饱肚子。同时，他发现白狼对海豹和驯鹿的捕猎技巧很高，而自己则相对较弱。\n\n3. 楚云霁在白狼的带领下，尝试捕猎海豹，但最终未能成功。他决定尝试捕猎其他动物，如海象，但遇到了一些困难。\n\n4. 楚云霁在白狼的帮助下，成功捕获了一只海象，但海象的皮毛被冻硬，难以处理。他决定将海象的皮毛埋起来，作为未来的食物储备。\n\n5. 楚云霁在白狼的带领下，尝试捕猎驯鹿，但最终未能成功。他决定尝试捕猎其他动物，如北极狐，但遇到了一些困难。\n\n6. 楚云霁在白狼的帮助下，成功捕获了一只北极狐，但北极狐的皮毛被冻硬，难以处理。他决定将北极狐的皮毛埋起来，作为未来的食物储备。\n\n7. 楚云霁在白狼的带领下，尝试捕猎其他动物，如海豹和驯鹿，但最终未能成功。他决定尝试捕猎其他动物，如北极狐，但遇到了一些困难。\n\n8. 楚', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1728558425, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=381, prompt_tokens=31811, total_tokens=32192, completion_tokens_details=None))
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: I need to use the get_current_weather API to get the current weather in Boston., call local function(get_current_weather) with arguments: location=Boston,MA, unit=celsius
Send the result back to the server with function result(59.0) ...
Final server response: The current temperature in Boston is 59.0 degrees Celsius.
'

# llama-index ai agent模拟
python3 client/llamaindex_ai_agent.py 127.0.0.1:9997
# 返回如下：
: '
Query: What is the weather in Boston today?
Added user message to memory: What is the weather in Boston today?
=== Calling Function ===
Calling function: get_weather with args: {"location":"Boston,MA","unit":"celsius"}
Got output: 59.0
========================

Response: The current temperature in Boston is 59.0 degrees Celsius.
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```