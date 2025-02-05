# Qwen2

Qwen2模型的部署示例，以Qwen2-7B-Instruct为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载Qwen2-7B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct /tmp/Qwen2-7B-Instruct

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt
rm -rf /tmp/Qwen2-7B-Instruct/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/Qwen2-7B-Instruct \
--output_dir /tmp/Qwen2-7B-Instruct/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/Qwen2-7B-Instruct/trt_engines/
trtllm-build --checkpoint_dir /tmp/Qwen2-7B-Instruct/tllm_checkpoint/ \
--output_dir /tmp/Qwen2-7B-Instruct/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/Qwen2-7B-Instruct/ \
--engine_dir=/tmp/Qwen2-7B-Instruct/trt_engines/
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
grpst start ./server.mar --inference_conf=conf/inference_qwen2.yml

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
    "model": "qwen2-instruct",
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
 "id": "chatcmpl-737",
 "object": "chat.completion",
 "created": 1724291091,
 "model": "qwen2-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是阿里云开发的一款超大规模语言模型，我叫通义千问。作为一个AI助手，我的目标是帮助用户获得准确、有用的信息，解决他们的问题和困惑。我被设计成能够进行多轮对话、保持逻辑一致，并能够覆盖各种主题，包括但不限于科技、文化、生活常识等。无论是你需要学习知识、完成任务，还是只是想聊天解闷，我都在这里为你服务。请随时告诉我你需要什么帮助，我会尽力提供支持。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 24,
  "completion_tokens": 24,
  "total_tokens": 48
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-instruct",
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
data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":1724295387,"model":"qwen2-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":1724295387,"model":"qwen2-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":1724295387,"model":"qwen2-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"我是"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是阿里云开发的一款超大规模语言模型，我叫通义千问。作为一个AI助手，我的目标是帮助用户获得准确、有用的信息，解决他们的问题和困惑。我被设计成能够进行多轮对话、保持逻辑一致，并能够覆盖各种主题，包括但不限于科技、文化、生活常识等。无论是你需要学习知识、完成任务，还是只是想聊天解闷，我都在这里为你服务。请随时告诉我你需要什么帮助，我会尽力提供支持。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1724295422, model='qwen2', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=102, prompt_tokens=24, total_tokens=126))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='你好', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1724295460, model='qwen2', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1724295460, model='qwen2', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-6', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1724295460, model='qwen2', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这篇小说的作者是弦三千。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1724685519, model='qwen2-instruct', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=8, prompt_tokens=31603, total_tokens=31611))
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这篇小说的前几章主要讲述了主角楚云霁意外穿成北极熊后的生活经历。以下是简述：\n\n1. 楚云霁在一次意外中穿成了一只北极熊，他刚穿过来时遇到了暴风雪，幸而及时躲进石头后才得以生存。暴风雪结束后，楚云霁在寻找食物时意外发现有人类队伍，他试图与他们交流，最终用一条鱼成功吸引了他们的注意。\n\n2. 楚云霁与科考队的成员们建立了初步的联系，他试图与他们一起生活，但科考队成员们对他的出现感到紧张和警惕。楚云霁通过友好的行为，如分享食物，逐渐赢得了他们的信任。\n\n3. 楚云霁在与科考队成员相处的过程中，发现自己的北极熊身份引起了不小的轰动，他的行为和外貌吸引了大量观众的关注，直播间的热度迅速上升。同时，他也开始探索自己的北极熊生活，尝试捕猎、寻找食物和适应野外环境。\n\n4. 楚云霁在一次捕猎时，意外救下了一只被其他动物追赶的海豹，这一行为让他与一只名为“白狼”的北极狼建立了联系。白狼似乎对楚云霁有某种保护或帮助的倾向，楚云霁也逐渐意识到白狼的存在对他的生存有着积极的影响。\n\n5. 楚云霁在与白狼的互动中，发现白狼不仅在捕猎方面有着高超的技巧，还展现出了一定的智慧和对他的关心。楚云霁也通过与白狼的相处，逐渐适应了北极熊的生活方式，开始尝试利用环境中的资源，如钓鱼竿等，来获取食物。\n\n6. 在一次偶然的机会中，楚云霁发现了一根被海浪冲到岸边的钓鱼竿，他尝试使用它来钓鱼，虽然方法简单，但意外地捕获了一些鱼。这一发现让他对北极熊的生活有了新的认识，也增加了他对探索周围环境的兴趣。\n\n7. 楚云霁在一次外出时，意外发现了一只偷吃他食物的海象，他试图报复，但最终被一只北极狼阻止。这只北极狼在楚云霁需要时帮助他，显示出了对他的保护倾向。楚云霁也通过这次事件，意识到与野生动物之间的复杂关系，以及在野外生存中需要的智慧和策略。\n\n8. 楚云霁在与北极狼的互动中，逐渐建立起了一种特殊的友谊，他开始尝试利用北极狼的智慧和力量来帮助自己更好地适应北极熊的生活，包括捕猎、寻找食物等。同时，他也开始思考如何利用自己的新身份，为保护野生动物和北极环境做出贡献。\n\n以上是这篇小说前几章的主要内容概要，讲述了楚云霁从穿成北极熊后，如何适应新环境、与野生动物建立联系，以及在野外生存中遇到的各种挑战和机遇。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1724685435, model='qwen2-instruct', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=618, prompt_tokens=31609, total_tokens=32227))
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought:  I need to use the get_current_weather API to find out the current weather in Boston., call local function(get_current_weather) with arguments: location=Boston,MA, unit=fahrenheit
Send the result back to the server with function result(59.0) ...
Final server response: The current weather in Boston is 59 degrees Fahrenheit.
'

# llama-index ai agent模拟
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