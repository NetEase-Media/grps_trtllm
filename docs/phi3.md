# Phi3

Phi3与Phi3.5模型的部署示例，以Phi-3.5-mini-instruct为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载Phi-3.5-mini-instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct /tmp/Phi-3.5-mini-instruct

# 进入TensorRT-LLM/examples/phi目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/phi
# 转换ckpt
rm -rf /tmp/Phi-3.5-mini-instruct/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/Phi-3.5-mini-instruct \
--output_dir /tmp/Phi-3.5-mini-instruct/tllm_checkpoint/ --dtype bfloat16
# 构建引擎
rm -rf /tmp/Phi-3.5-mini-instruct/trt_engines/
trtllm-build --checkpoint_dir /tmp/Phi-3.5-mini-instruct/tllm_checkpoint/ \
--output_dir /tmp/Phi-3.5-mini-instruct/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/Phi-3.5-mini-instruct/ \
--engine_dir=/tmp/Phi-3.5-mini-instruct/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_phi3.yml

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
    "model": "Phi-3.5-mini-instruct",
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
 "created": 1737709594,
 "model": "Phi-3.5-mini-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "我是一个人工智能助手，旨在回答你的问题并提供帮助。今天我能为你做什么？"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 21,
  "completion_tokens": 49,
  "total_tokens": 70
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
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
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1737708446,"model":"Phi-3.5-mini-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1737708446,"model":"Phi-3.5-mini-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"你"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1737708446,"model":"Phi-3.5-mini-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"好"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是一个人工智能助手，在这里帮助你回答问题或完成任务。今天我能为你做什么？', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1737708567, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=50, prompt_tokens=25, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1737708629, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='你', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1737708629, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='好', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1737708629, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入13k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/13k_novel.txt "上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='作者是弦三千。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1737709685, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=11, prompt_tokens=29643, total_tokens=29654, completion_tokens_details=None, prompt_tokens_details=None))
'

# 输入13k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/13k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='在这个幽默且充满奇幻色彩的故事中，主角楚云霁，一个大三学生，意外成为了一只北极熊，在无人荒星上参与大型野外生存直播综艺。他的旅程从被撞上科考队，到与其他野生动物互动，最终在寻找自己的位置，并与白狼建立了友好关系。\n\n在第一章中，楚云霁遇到了北极熊，在暴风雪中艰难地生存，最终被救出来并与白狼发生了联系。第二章描述了他们在寻找食物时的经历，包括捕捉海鸟和海豹，并遇到了一只被白狼困住的海豹。在第三章中，楚云霁和白狼一起捕猎，捉到了一只旅鼠，并在直播中被意外捕捉到。\n\n整个故事探讨了主角在野外生存的挑战，以及与野生动物的互动，展示了他们的智慧和适应能力，同时也带有一种轻松愉快的氛围，让观众在观看过程中产生共鸣和娱乐。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1737709713, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=476, prompt_tokens=29653, total_tokens=30129, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```