# grps-trtllm

llama3模型的部署示例。

## 开发环境

见[本地开发与调试拉取代码和创建容器部分](../README.md#3-本地开发与调试)。

## 构建trtllm引擎

```bash
# 下载llama-3-chinese-8b-instruct-v3模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v3 /tmp/llama-3-chinese-8b-instruct-v3

# 进入TensorRT-LLM/examples/llama目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/llama/
# 转换ckpt
rm -rf /tmp/llama-3-chinese-8b-instruct-v3/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/llama-3-chinese-8b-instruct-v3 \
--output_dir /tmp/llama-3-chinese-8b-instruct-v3/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/llama-3-chinese-8b-instruct-v3/trt_engines/
trtllm-build --checkpoint_dir /tmp/llama-3-chinese-8b-instruct-v3/tllm_checkpoint/ \
--output_dir /tmp/llama-3-chinese-8b-instruct-v3/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/llama-3-chinese-8b-instruct-v3/ \
--engine_dir=/tmp/llama-3-chinese-8b-instruct-v3/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_llama3.yml

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
    "model": "llama-3-chinese-8b-instruct-v3",
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
 "created": 1725685849,
 "model": "llama-3-chinese-8b-instruct-v3",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是一个人工智能语言模型，我的名字是LLaMA。我的任务是与用户进行对话，回答问题，提供帮助和娱乐。很高兴与你交流！"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 16,
  "completion_tokens": 42,
  "total_tokens": 58
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-chinese-8b-instruct-v3",
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
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1725685866,"model":"llama-3-chinese-8b-instruct-v3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1725685866,"model":"llama-3-chinese-8b-instruct-v3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":1725685866,"model":"llama-3-chinese-8b-instruct-v3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是一个人工智能语言模型，我的名字是LLaMA。我的任务是与用户进行对话，回答问题，提供帮助和娱乐。很高兴与你交流！', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1725685891, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=42, prompt_tokens=16, total_tokens=58))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='你', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725685911, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='好', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725685911, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-7', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1725685911, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```