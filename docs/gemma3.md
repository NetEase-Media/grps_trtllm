# Gemma3(Experimental)

Gemma3文本模型的部署示例，以gemma-3-4b-it为例。当前还处于实验阶段，虽然vit部分已开发，但是抱歉还不能支持图片输入，因为特殊的attention
mask当前trtllm还不支持：[issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2880#issuecomment-2726181463)
，暂时没有找到解决办法，会继续跟进。另外由于tensorrt-llm关于sliding window attention mask类型在kv cache
reuse打开时会有一些问题，[issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2912)，所以暂时不支持打开
```enable_kv_cache_reuse```
功能，后续也会继续跟进该问题。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载gemma-3-4b-it模型
apt update && apt install git-lfs
git lfs install
git clone git@hf.co:google/gemma-3-4b-it /tmp/gemma-3-4b-it 

# 安装支持gemma3的transformers版本
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# 覆盖tensorrt_llm源码支持gemma3-text
cp -r ./tools/gemma3/tensorrt_llm_mod/* /usr/local/lib/python3.12/dist-packages/tensorrt_llm/

# gemma-3-1b本身是text llm，无需进行llm拆分，其他尺寸需要拆分。
rm -rf /tmp/gemma-3-4b-it/llm/
python3 ./tools/gemma3/split_llm.py --model_dir /tmp/gemma-3-4b-it --output_dir /tmp/gemma-3-4b-it/llm --dtype bfloat16
# 转换ckpt
rm -rf /tmp/gemma-3-4b-it/tllm_checkpoint/
python3 ./third_party/TensorRT-LLM/examples/gemma/convert_checkpoint.py \
--ckpt-type hf --dtype bfloat16 --load_model_on_cpu \
--model-dir /tmp/gemma-3-4b-it/llm --output-model-dir /tmp/gemma-3-4b-it/tllm_checkpoint/ 
# 构建引擎
rm -rf /tmp/gemma-3-4b-it/trt_engines/
trtllm-build --checkpoint_dir /tmp/gemma-3-4b-it/tllm_checkpoint/ \
--output_dir /tmp/gemma-3-4b-it/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 64512 --max_seq_len 65536 --max_num_tokens 64512

# 恢复tensorrt_llm源码，防止修改对其他模型有影响。
cp -r -L ./tools/gemma3/tensorrt_llm_ori/* /usr/local/lib/python3.12/dist-packages/tensorrt_llm/

# 构建vit引擎，设置--maxBS为32可以同时处理32个图片
#python3 tools/gemma3/build_vit_engine.py --pretrainedModelPath /tmp/gemma-3-4b-it \
#--imageUrl 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg' \
#--onnxFile /tmp/gemma-3-4b-it/vision_encoder_bfp16.onnx \
#--trtFile /tmp/gemma-3-4b-it/vision_encoder_bfp16.trt \
#--dtype float16 --minBS 1 --optBS 1 --maxBS 32
```

## 构建与部署

不同尺寸模型的sliding window可能不一致，默认使用4b尺寸，见inference_gemma3_text.yml ```max_attention_window_size```
字段注释。

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_gemma3_text.yml

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
    "model": "gemma-3",
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
 "created": 1742058129,
 "model": "gemma-3",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "我是Gemma，一个由Google DeepMind训练的大型语言模型。我是一个开放权重的模型，可以广泛地供公众使用。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 14,
  "completion_tokens": 30,
  "total_tokens": 44
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3",
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
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1742058152,"model":"gemma-3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"我是"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1742058152,"model":"gemma-3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"G"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1742058152,"model":"gemma-3","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"emma"},"logprobs":null,"finish_reason":null}]}
'

# 测试解答数学题
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3",
    "messages": [
      {
        "role": "user",
        "content": "解一下这道题：\n(x + 3) = (8 - x)\nx = ?"
      }
    ],
    "max_tokens": 2048
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1742059123,
 "model": "gemma-3",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "好的，我们来解这个方程：\n\n(x + 3) = (8 - x)\n\n1.  **两边同时加 x:**\n   x + 3 + x = 8 - x + x\n   2x + 3 = 8\n\n2.  **两边同时减 3:**\n   2x + 3 - 3 = 8 - 3\n   2x = 5\n\n3.  **两边同时除以 2:**\n   2x / 2 = 5 / 2\n   x = 5/2  或者  x = 2.5\n\n所以，x = 2.5\n"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 32,
  "completion_tokens": 146,
  "total_tokens": 178
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='我是Gemma，一个由Google DeepMind训练的大型语言模型。我是一个开放权重的模型，可以广泛地供公众使用。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1742058184, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=30, prompt_tokens=14, total_tokens=44, completion_tokens_details=None, prompt_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1742058216, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='G', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1742058216, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-5', choices=[Choice(delta=ChoiceDelta(content='emma', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1742058216, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-6', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='好的，这是对小说《拜托，只想干饭的北极熊超酷的！》前几章内容的总结：\n\n**故事梗概：**\n\n小说讲述了主角楚云霁意外穿越成一只北极熊，并在荒星上展开了一段充满趣味和挑战的生存之旅。他原本只想吃饱饭，却意外卷入了一个大型野外生存直播综艺，并与一只名叫白狼的北极熊建立了亦敌亦友的关系。\n\n**前几章内容：**\n\n*   **穿越与初遇：** 楚云霁在实习途中被不明力量击晕，醒来后发现自己变成了一只北极熊。他意外地撞上了一队科研人员，他们正在进行一项野外生存直播项目。\n*   **生存挑战：** 楚云霁在荒星上展开了生存挑战，他需要寻找食物、躲避危险，并适应北极熊的生存环境。\n*   **白狼的出现：** 在生存过程中，楚云霁遇到了白狼，白狼既是他的竞争对手，又是他唯一的伙伴。\n*   **直播爆红：** 楚云霁的生存过程被直播，吸引了大量观众，他的故事也因此走红。\n*   **身份揭秘：** 楚云霁逐渐意识到自己并非普通的北极熊，而是曾经的直播明星，并且与节目组中的上将有着千丝万缕的联系。\n\n**主要情节：**\n\n*   楚云霁在暴风雪中挣扎求生，最终凭借自己的智慧和勇气战胜了困难。\n*   他与白狼之间产生了复杂的情感，既有竞争，也有合作，甚至还发展出了一种特殊的友谊。\n*   楚云霁在直播过程中遇到了各种各样的挑战，包括食物短缺、天气恶劣、以及其他动物的威胁。\n*   他逐渐适应了北极熊的生存方式，并学会了利用自己的优势来战胜困难。\n\n总而言之，前几章主要介绍了楚云霁穿越成北极熊后的生存状态，以及他与白狼之间的关系。故事充满了趣味性和挑战性，也为后续情节的发展埋下了伏笔。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1742443608, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=476, prompt_tokens=40423, total_tokens=40899))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
