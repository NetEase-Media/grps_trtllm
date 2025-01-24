# grps-trtllm

glm-4-9b-chat模型的部署示例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载glm-4-9b-chat模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/THUDM/glm-4-9b-chat /tmp/glm-4-9b-chat
# 转换tiktoken tokenizer.model为huggingface tokenizer.json格式
python3 ./tools/tiktoken/tiktoken_to_hf.py /tmp/glm-4-9b-chat/ /tmp/glm-4-9b-chat/

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
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
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

# 输入32k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='根据您提供的小说内容，可以判断这篇小说的作者是弦三千。\n\n以下是我的判断依据：\n\n1. 文档开头明确标注了作者为“弦三千”。\n\n2. 文档中多次出现“弦三千”这个名字，例如“作者：弦三千”、“#书里的那个恶毒炮灰跟我同名# #恶毒炮灰竟是我自己？！！# 弦三千”。\n\n3. 文档内容与弦三千过往作品风格相符，例如《穿成炮灰后我成了神兽》等。\n\n综上所述，可以确定这篇小说的作者是弦三千。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1725529675, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=119, prompt_tokens=31166, total_tokens=31285))
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-8', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这篇小说讲述了主角楚云霁穿成一只北极熊后，在北极荒野中生存的故事。\n\n**主要情节**：\n\n* 楚云霁穿成一只小北极熊，在暴风雪中艰难求生，经历了饥饿、寒冷和危险。\n* 他遇到了一只高冷的北极狼，并与其建立了友谊，共同捕猎，分享食物。\n* 楚云霁意外捡到一根钓鱼竿，并利用它捕鱼，改善了生活。\n* 他还遇到了其他动物，如海象、海豹、北极狐等，并与之互动。\n* 楚云霁逐渐适应了北极的生活，并享受着与白狼的友谊。\n\n**小说主题**：\n\n* 保护北极熊，爱护野生动物。\n* 生存与成长。\n* 友谊与陪伴。\n\n**小说特色**：\n\n* 萌宠题材，主角楚云霁是一只可爱的小北极熊，性格活泼可爱，让人忍不住想要保护它。\n* 野外求生元素，情节紧张刺激，展现了北极的恶劣环境和生存挑战。\n* 人物关系温馨感人，主角与白狼的友谊让人动容。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1725529877, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=239, prompt_tokens=31168, total_tokens=31407))
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: Sure, I can help with that. Let me check the current weather in Boston.get_current_weather
{"location": "Boston, MA", "unit": "fahrenheit"}, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result(59.0) ...
Final server response: The current weather in Boston is 59.0°F.
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