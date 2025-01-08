# grps-trtllm

Qwen2.5-7B-Instruct模型的部署示例。

## 开发环境

见[本地开发与调试拉取代码和创建容器部分](../README.md#3-本地开发与调试)。

## 构建trtllm引擎

```bash
# 下载Qwen2.5-7B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct /tmp/Qwen2.5-7B-Instruct

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt
rm -rf /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/Qwen2.5-7B-Instruct \
--output_dir /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/Qwen2.5-7B-Instruct/trt_engines/
trtllm-build --checkpoint_dir /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/ \
--output_dir /tmp/Qwen2.5-7B-Instruct/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/Qwen2.5-7B-Instruct/ \
--engine_dir=/tmp/Qwen2.5-7B-Instruct/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_qwen2.5.yml

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
    "model": "qwen2.5-instruct",
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
 "id": "chatcmpl-7",
 "object": "chat.completion",
 "created": 1726733862,
 "model": "qwen2.5-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是Qwen，由阿里云开发的人工智能模型。我被设计用来提供信息、回答问题和进行各种对话任务。有什么我可以帮助你的吗？"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 34,
  "completion_tokens": 36,
  "total_tokens": 70
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-instruct",
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
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"我是"},"logprobs":null,"finish_reason":null}]}
'

# 测试stop参数
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-instruct",
    "messages": [
      {
        "role": "user",
        "content": "重复1234#END#5678"
      }
    ],
    "stop": ["#END#"]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1727433345,
 "model": "qwen2.5-instruct",
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
  "prompt_tokens": 41,
  "completion_tokens": 7,
  "total_tokens": 48
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是Qwen，由阿里云开发的人工智能模型。我被设计用来提供信息、回答问题和进行各种对话任务。有什么我可以帮助你的吗？', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733895, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=36, prompt_tokens=34, total_tokens=70, completion_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='你好', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-11', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这篇小说的作者是弦三千。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733931, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=8, prompt_tokens=31615, total_tokens=31623, completion_tokens_details=None))
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-12', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='以下是《拜托，只想干饭的北极熊超酷的！》前几章的主要内容概述：\n\n1. **第一章**：楚云霁意外穿越成了一只北极熊，他发现了一群科考队，并用鱼与他们交流。楚云霁在暴风雪中艰难生存，通过抓鱼和捕猎海豹来获取食物。\n\n2. **第二章**：楚云霁在暴风雪后继续捕猎，遇到了一只北极白狼。白狼似乎对楚云霁很友好，甚至带他去捕猎海豹。楚云霁吃了一顿饱饭后，与白狼一起回到白狼的洞穴休息。\n\n3. **第三章**：楚云霁在白狼的洞穴中休息，醒来后发现白狼已经离开。他继续捕猎，遇到了一群海豹，但海豹很快被一只成年北极熊吓跑。楚云霁在冰面上发现了一群生蚝，但白狼对生蚝不感兴趣，楚云霁只好自己吃了。\n\n4. **第四章**：楚云霁在捕猎时遇到了一只成年北极熊，成年北极熊似乎在挑衅他。楚云霁和白狼一起捕猎了一只驯鹿，分享了食物。直播设备记录下了这一幕，引起了观众的热议。\n\n5. **第五章**：楚云霁和白狼一起捕猎了一只驯鹿，分享了食物。楚云霁在捕猎时遇到了一只北极狐，但北极狐被北极熊吓跑。楚云霁还遇到了一只海鸟，海鸟试图抢食，但被白狼赶走。楚云霁和白狼一起处理了一只驯鹿，白狼还帮助楚云霁取下了鹿角。\n\n6. **第六章**：楚云霁和白狼一起捕猎，楚云霁在冰面上睡觉时被冰面漂走。醒来后，楚云霁发现白狼还在身边，感到非常高兴。他们一起捕猎了一只海象，但海象偷走了鱼竿。楚云霁和白狼一起追捕海象，最终成功捕获了海象。\n\n7. **第七章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n8. **第八章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n9. **第九章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n10. **第十章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n11. **第十一章**：楚云霁在白狼的洞穴中发现了一个背包，背包里装满了各种食物和补给品。楚云霁和白狼一起分享了这些食物，包括罐头和海带。楚云霁还和白狼一起出去捕猎，但没有成功。\n\n12. **第十二章**：楚云霁和白狼一起出去捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉，并一起出去探索周围的环境。楚云霁还发现了一个背包，背包里装满了各种食物和补给品。楚云霁和白狼一起分享了这些食物，包括罐头和海带。楚云霁还和白狼一起出去捕猎，但没有成功。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733966, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=959, prompt_tokens=31621, total_tokens=32580, completion_tokens_details=None))
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: None, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result(59.0) ...
Final server response: The current temperature in Boston today is 59°F.
'

# openai_func_call2.py进行一次两个函数的function call模拟
python3 client/openai_func_call2.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the postcode of Boston and what's the weather like in Boston today? ...
Server response: thought: None, call local function(get_postcode) with arguments: location=Boston, MA
Server response: thought: None, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result ...
Final server response: The postcode for Boston, MA is 02138. The current temperature in Boston today is 59.0°F.
'

# llama-index ai agent模拟
pip install llama_index llama_index.llms.openai_like
python3 client/llamaindex_ai_agent.py 127.0.0.1:9997
# 返回如下：
: '
Query: What is the weather in Boston today?
Added user message to memory: What is the weather in Boston today?
=== Calling Function ===
Calling function: get_weather with args: {"location":"Boston, MA","unit":"fahrenheit"}
Got output: 59.0
========================

Response: The current temperature in Boston is 59.0 degrees Fahrenheit.
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```