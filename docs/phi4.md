# phi-4

phi-4模型的部署示例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载phi-4模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/microsoft/phi-4 /tmp/phi-4

# 进入TensorRT-LLM/examples/phi目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/phi
# 转换ckpt，这里使用了int8 smooth quant量化减少显存占用
rm -rf /tmp/phi-4/tllm_checkpoint/
python3 ../quantization/quantize.py --model_dir /tmp/phi-4 \
--dtype bfloat16 --qformat int8_sq --kv_cache_dtype int8 --device cuda \
--output_dir /tmp/phi-4/tllm_checkpoint/
# 构建引擎
rm -rf /tmp/phi-4/trt_engines/
trtllm-build --checkpoint_dir /tmp/phi-4/tllm_checkpoint/ \
--output_dir /tmp/phi-4/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
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
grpst start ./server.mar --inference_conf=conf/inference_phi4.yml

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
    "model": "phi-4",
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
 "created": 1740275262,
 "model": "phi-4",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是一个人工智能助手，专门设计来帮助你回答问题、提供信息和解决各种问题。无论你有什么需要帮助的，随时可以问我！"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 25,
  "completion_tokens": 62,
  "total_tokens": 87
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
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
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1740275430,"model":"phi-4","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1740275430,"model":"phi-4","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1740275430,"model":"phi-4","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
'

# 测试stop参数
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [
      {
        "role": "user",
        "content": "重复我的话：1234#END#5678"
      }
    ],
    "stop": ["#END#"]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1740276402,
 "model": "phi-4",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "1234#END#"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 29,
  "completion_tokens": 5,
  "total_tokens": 34
 }
}
'

# 测试解答数学题
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B-Preview",
    "messages": [
      {
        "role": "user",
        "content": "解一下这道题：\n(x + 3) = (8 - x)\nx = ?\n注意使用中文"
      }
    ]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-4",
 "object": "chat.completion",
 "created": 1740276751,
 "model": "QwQ-32B-Preview",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "要解这个方程 \\( (x + 3) = (8 - x) \\)，我们可以按照以下步骤进行：\n\n1. **将方程写出来：**\n   \\[\n   x + 3 = 8 - x\n   \\]\n\n2. **将所有涉及 \\( x \\) 的项移到方程的一边：**\n   我们可以将 \\( x \\)移到左边，将常数移到右边。首先，加上 \\( x \\)到左边：\n   \\[\n   x + x + 3 = 8\n   \\]\n   这简化为：\n   \\[\n   2x + 3 = 8\n   \\]\n\n3. **将常数项移到方程的另一边：**\n   减去 3 从左边：\n   \\[\n   2x = 8 - 3\n   \\]\n   这简化为：\n   \\[\n   2x = 5\n   \\]\n\n4. **解出 \\( x \\):**\n   将 2x 除以 2：\n   \\[\n   x = \\frac{5}{2}\n   \\]\n\n所以，解是 \\( x = \\frac{5}{2} \\) 或 \\( x = 2.5 \\)."
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 42,
  "completion_tokens": 295,
  "total_tokens": 337
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是一个人工智能助手，专门设计来帮助你回答问题、提供信息和解决各种问题。无论你有什么需要帮助的，随时可以问我！', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1740275451, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=62, prompt_tokens=25, total_tokens=87, completion_tokens_details=None, prompt_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='你', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740275468, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='好', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740275468, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740275468, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```