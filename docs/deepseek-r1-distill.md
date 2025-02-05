# deepseek-r1-distill

deepseek-r1蒸馏模型部署样例，不同尺寸对应不同的蒸馏base模型，如下表：

|           **Model**           |                                   **Base Model**                                   |                                    **Download**                                    |
|:-----------------------------:|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |         [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)         | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
|  DeepSeek-R1-Distill-Qwen-7B  |           [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)           |  [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)  |
| DeepSeek-R1-Distill-Llama-8B  |           [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)  |
| DeepSeek-R1-Distill-Qwen-14B  |               [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)               | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)  |
| DeepSeek-R1-Distill-Qwen-32B  |               [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)               | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)  |
| DeepSeek-R1-Distill-Llama-70B | [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |

## 演示

<img src="gradio3.png" alt="gradio3.png">

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

### 1.5B\7B\14B\32B

以DeepSeek-R1-Distill-Qwen-7B为例。

```bash
# 下载DeepSeek-R1-Distill-Qwen-7B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B /tmp/DeepSeek-R1-Distill-Qwen-7B

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt
rm -rf /tmp/DeepSeek-R1-Distill-Qwen-7B/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/DeepSeek-R1-Distill-Qwen-7B \
--output_dir /tmp/DeepSeek-R1-Distill-Qwen-7B/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/DeepSeek-R1-Distill-Qwen-7B/trt_engines/
trtllm-build --checkpoint_dir /tmp/DeepSeek-R1-Distill-Qwen-7B/tllm_checkpoint/ \
--output_dir /tmp/DeepSeek-R1-Distill-Qwen-7B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 回到工程根目录
cd ../../../../
```

### 8B\70B

以DeepSeek-R1-Distill-Llama-8B为例。

```bash
# 下载DeepSeek-R1-Distill-Llama-8B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B /tmp/DeepSeek-R1-Distill-Llama-8B

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/llama/
# 转换ckpt
rm -rf /tmp/DeepSeek-R1-Distill-Llama-8B/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/DeepSeek-R1-Distill-Llama-8B \
--output_dir /tmp/DeepSeek-R1-Distill-Llama-8B/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/DeepSeek-R1-Distill-Llama-8B/trt_engines/
trtllm-build --checkpoint_dir /tmp/DeepSeek-R1-Distill-Llama-8B/tllm_checkpoint/ \
--output_dir /tmp/DeepSeek-R1-Distill-Llama-8B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 回到工程根目录
cd ../../../../
```

## 构建与部署

注意不同尺寸的inference.yml可以根据LLM类型分别参考不同inference.yml文件并改模型路径。基于Qwen2.5（1.5B\7B\14B\32B）的LLM参考[inference_deepseek-r1-distill-qwen.yml](../conf/inference_deepseek-r1-distill-qwen.yml)
，基于llama3（8B\70B）的LLM参考[inference_deepseek-r1-distill-llama.yml](../conf/inference_deepseek-r1-distill-llama.yml)。

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
# grpst start ./server.mar --inference_conf=conf/inference_deepseek-r1-distill-llama.yml
grpst start ./server.mar --inference_conf=conf/inference_deepseek-r1-distill-qwen.yml

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
    "model": "DeepSeek-R1-Distill-Qwen-7B",
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
 "created": 1738733164,
 "model": "DeepSeek-R1-Distill-Qwen-7B",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "<think>\n\n</think>\n\n您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 8,
  "completion_tokens": 39,
  "total_tokens": 47
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Distill-Qwen-7B",
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
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1738733202,"model":"DeepSeek-R1-Distill-Qwen-7B","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1738733202,"model":"DeepSeek-R1-Distill-Qwen-7B","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"\n\n"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1738733202,"model":"DeepSeek-R1-Distill-Qwen-7B","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"</think>"},"logprobs":null,"finish_reason":null}]}
'

# 测试stop参数
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Distill-Qwen-7B",
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
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1738733221,
 "model": "DeepSeek-R1-Distill-Qwen-7B",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "<think>\n嗯，用户发来的信息是“重复1234#END#"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 15,
  "completion_tokens": 18,
  "total_tokens": 33
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='<think>\n\n</think>\n\n您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1738733258, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=39, prompt_tokens=8, total_tokens=47, completion_tokens_details=None, prompt_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='<think>', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1738733273, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='\n\n', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1738733273, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='</think>', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1738733273, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-7', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='<think>\n嗯，用户让我简述小说的前几章内容。首先，我需要仔细阅读用户提供的章节内容，理解每一章的主要情节和人物发展。\n\n章节1描述了楚云霁和白狼相遇的情景，以及他如何说服白狼帮助自己捕猎北极熊。这里重点是楚云霁的聪明和勇气，还有他如何利用地形和动物的习性来解决问题。\n\n章节2主要讲述了楚云霁在冰洞里寻找食物的过程，以及他如何发现并捕获北极熊。这里展示了他体力和智慧的应用，同时描绘了冰洞的环境和食物的稀缺性。\n\n章节3描述了楚云霁在冰洞里遇到白狼，白狼帮助他捕猎北极熊。这里展示了人与狼的互动，以及白狼的忠诚和聪明。\n\n章节4讲述了楚云霁在冰洞里寻找食物的过程，以及他如何利用雪的特性找到食物。同时，他遇到了一只海豹，这增加了故事的复杂性。\n\n章节5描述了楚云霁在冰洞里寻找食物，遇到海豹后继续寻找其他食物，最终成功捕获海豹。这里展示了他耐心和毅力。\n\n章节6讲述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节7描述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节8讲述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节9描述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节10讲述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节11描述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n章节12讲述了楚云霁在冰洞里捕获海豹后，如何用绳子绑住它，以及他如何让海豹自己移动。同时，他遇到了其他动物，增加了故事的丰富性。\n\n综上所述，前几章主要讲述了楚云霁如何利用智慧和勇气在寒冷的冰洞里寻找食物，捕获北极熊和海豹的过程，同时展示了他与白狼的关系，以及他如何与动物互动，增加了故事的趣味性和可玩性。\n</think>\n\n前几章主要讲述了楚云霁在寒冷的北极冰洞中寻找食物的故事。以下是简要内容概述：\n\n1. **楚云霁的背景**  \n   楚云霁来自一个寒冷的北极小镇，性格勇敢但有些孤寂。他来到南极捕猎，遇到了聪明的白狼，白狼帮助他捕获北极熊，展现了人与狼的互动。\n\n2. **捕猎过程**  \n   楚云霁在寒冷的冰洞中寻找食物，遇到北极熊和海豹。他利用地形和动物习性，通过雪地的结冰和动物的活动寻找食物。尽管食物稀缺，但他坚持不懈，最终成功捕获北极熊和海豹。\n\n3. **与白狼的关系**  \n   楚云霁与白狼建立了深厚的感情，白狼帮助他捕猎，展现了忠诚与智慧。白狼在寒冷中也能准确判断食物的位置，体现了它们的聪明。\n\n4. **环境描写**  \n   冰洞寒冷而美丽，食物稀缺但美味。楚云霁通过观察动物的习性，灵活地寻找食物，展现了他在寒冷环境中的生存能力。\n\n5. **互动与挑战**  \n   楚云霁在捕猎过程中遇到了各种挑战，如如何利用绳子绑住动物，如何与动物互动等。这些互动不仅增加了故事的趣味性，也展示了他与动物之间的独特关系。\n\n这些内容为读者提供了关于南极生态和捕猎生活的生动画面，同时为玩家提供了丰富的互动体验。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1738733449, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=959, prompt_tokens=31595, total_tokens=32554, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 开启gradio服务

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动纯文本聊天界面，llm代表纯文本聊天，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py llm 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```