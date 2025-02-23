# QwQ

QwQ模型的部署示例，以QwQ-32B-Preview为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载QwQ-32B-Preview模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/QwQ-32B-Preview /tmp/QwQ-32B-Preview

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt，指定8卡tensor parallel推理。
rm -rf /tmp/QwQ-32B-Preview/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/QwQ-32B-Preview \
--output_dir /tmp/QwQ-32B-Preview/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu --tp 8
# 构建引擎
rm -rf /tmp/QwQ-32B-Preview/trt_engines/
trtllm-build --checkpoint_dir /tmp/QwQ-32B-Preview/tllm_checkpoint/ \
--output_dir /tmp/QwQ-32B-Preview/trt_engines/ \
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
grpst start ./server.mar --inference_conf=conf/inference_qwq.yml --mpi_np 8

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 模拟请求

```bash
# curl命令非stream请求``
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B-Preview",
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
 "id": "chatcmpl-6",
 "object": "chat.completion",
 "created": 1740238708,
 "model": "QwQ-32B-Preview",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "我是来自阿里云的大规模语言模型，我叫通义千问。我可以回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 41,
  "completion_tokens": 35,
  "total_tokens": 76
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B-Preview",
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
data: {"id":"chatcmpl-7","object":"chat.completion.chunk","created":1740238748,"model":"QwQ-32B-Preview","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"我是"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-7","object":"chat.completion.chunk","created":1740238748,"model":"QwQ-32B-Preview","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"来自"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-7","object":"chat.completion.chunk","created":1740238748,"model":"QwQ-32B-Preview","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"阿里"},"logprobs":null,"finish_reason":null}]}
'

# 测试解答数学题
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B-Preview",
    "messages": [
      {
        "role": "user",
        "content": "解一下这道题：\n(x + 3) = (8 - x)\nx = ?"
      }
    ]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-10",
 "object": "chat.completion",
 "created": 1740240347,
 "model": "QwQ-32B-Preview",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "好的，我来解这个方程：(x + 3) = (8 - x)。我要找到x的值，使得等式两边相等。让我一步一步来。\n\n首先，我看到方程是x + 3 = 8 - x。我的目标是把x孤立在等式的一边，把常数移到另一边。我知道，如果我在等式的两边进行相同的操作，等式仍然成立。所以，我可以加或减相同的数，或者乘或除以相同的数（除了零）。\n\n让我先试着把所有的x项移到一边，把常数移到另一边。现在，等式是x + 3 = 8 - x。我想把-x移到左边，把+3移到右边。\n\n怎么移动呢？我可以在等式的两边都加上x，这样右边的-x就消失了。所以，我在两边都加上x：\n\nx + 3 + x = 8 - x + x\n\n这样，左边变成x + x + 3，也就是2x + 3，右边变成8 - x + x，也就是8。\n\n所以，现在等式是2x + 3 = 8。\n\n接下来，我想把+3移到右边，所以我要在两边都减去3：\n\n2x + 3 - 3 = 8 - 3\n\n这样，左边变成2x，右边变成5。\n\n所以，现在等式是2x = 5。\n\n最后，我要解出x，所以需要把2x变成x，也就是两边都除以2：\n\n2x / 2 = 5 / 2\n\n这样，x = 5/2。\n\n让我检查一下，看看这个答案是否正确。我把x = 5/2代入原方程：\n\n左边是x + 3 = 5/2 + 3 = 5/2 + 6/2 = 11/2\n\n右边是8 - x = 8 - 5/2 = 16/2 - 5/2 = 11/2\n\n嗯，左边等于右边，都是11/2，所以x = 5/2是正确的解。\n\n不过，我再想一下，有没有其他方法可以解这个方程。比如说，我可以选择不同的步骤顺序。\n\n比如，最初的时候，我可以选择在两边都减去x，而不是加上x。\n\n原来等式是x + 3 = 8 - x\n\n如果两边都减去x：\n\nx + 3 - x = 8 - x - x\n\n左边变成3，右边变成8 - 2x\n\n所以，等式变成3 = 8 - 2x\n\n然后，我可以把-2x移到左边，把3移到右边。\n\n等待一下，也许这样不太方便。或者，我可以把等式两边都减去8：\n\n3 - 8 = 8 - 2x - 8\n\n左边是-5 = -2x\n\n然后，两边都除以-2：\n\n-5 / -2 = (-2x) / -2\n\n也就是5/2 = x\n\n还是得到x = 5/2，和之前的结果一样。\n\n看来，无论选择哪种方法，最终得到的解都是x = 5/2。\n\n我再考虑一下，如果方程更复杂一些，比如系数不同，我能否应用同样的方法。\n\n比如说，(2x + 5) = (10 - x)\n\n我尝试解这个方程。\n\n首先，把-x移到左边，把+5移到右边：\n\n2x + x = 10 - 5\n\n3x = 5\n\n然后，x = 5/3\n\n让我检查一下：\n\n左边是2*(5/3) + 5 = 10/3 + 15/3 = 25/3\n\n右边是10 - 5/3 = 30/3 - 5/3 = 25/3\n\n确实相等。\n\n看来这种方法是通用的。\n\n再比如，(3x - 2) = (7 + x)\n\n我把x项移到左边，常数移到右边：\n\n3x - x = 7 + 2\n\n2x = 9\n\nx = 9/2\n\n检查：\n\n左边是3*(9/2) - 2 = 27/2 - 4/2 = 23/2\n\n右边是7 + 9/2 = 14/2 + 9/2 = 23/2\n\n正确。\n\n所以，无论方程的具体数字是多少，我都可以通过类似的步骤来解方程。\n\n回到原来的方程，x + 3 = 8 - x\n\n我得到了x = 5/2\n\n我认为这个解是正确的。\n\n**最终答案**\n\n\\[\\boxed{\\dfrac{5}{2}}\\]"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 56,
  "completion_tokens": 1004,
  "total_tokens": 1060
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-11', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='我是来自阿里云的大规模语言模型，我叫通义千问。我可以回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1740239070, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=35, prompt_tokens=41, total_tokens=76, completion_tokens_details=None, prompt_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-12', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740239107, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-12', choices=[Choice(delta=ChoiceDelta(content='来自', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740239107, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-12', choices=[Choice(delta=ChoiceDelta(content='阿里', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1740239107, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```