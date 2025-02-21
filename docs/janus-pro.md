# JanusPro

JanusPro图文理解模型(暂不支持文生图)的部署示例，以Janus-Pro-7B模型为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载Janus-Pro-7B模型以及Janus代码仓库
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/Janus-Pro-7B /tmp/Janus-Pro-7B
git clone https://github.com/deepseek-ai/Janus.git /tmp/Janus

# 安装依赖
pip install -r ./tools/janus-pro/requirements.txt

# 拷贝工具的到Janus代码仓库
cp ./tools/janus-pro/build_vit_engine.py /tmp/Janus/build_vit_engine.py
cp ./tools/janus-pro/split_llm.py /tmp/Janus/split_llm.py
cp ./tools/janus-pro/fix_trt_err.diff /tmp/Janus/fix_trt_err.diff
# 进入Janus代码仓库进行vit构建和llm模型拆分
cd /tmp/Janus
# 修复trt引擎转换后结果出错的问题
git apply fix_trt_err.diff
# 构建vit引擎，设置--maxBS为8可以同时处理8个图片。
python3 build_vit_engine.py --pretrainedModelPath /tmp/Janus-Pro-7B \
--imagePath ./images/logo.png \
--onnxFile /tmp/Janus-Pro-7B/vision_encoder_bfp16.onnx \
--trtFile /tmp/Janus-Pro-7B/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 1 --maxBS 8
# 拆分llm模型
python3 split_llm.py --model_dir /tmp/Janus-Pro-7B --output_dir /tmp/Janus-Pro-7B/llm 
# 返回grps-trtllm根目录
cd -

# 转换ckpt
rm -rf /tmp/Janus-Pro-7B/tllm_checkpoint/
python3 ./third_party/TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir /tmp/Janus-Pro-7B/llm/ \
--output_dir /tmp/Janus-Pro-7B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为4，即支持4个并发同时推理，超过4个排队处理。
# 设置每个请求最多输入8个图片。
# 即：max_multimodal_len= 4 * 8（图片个数） * 576（每个图片对应576个token） = 18432
# 设置max_input_len为32k，max_seq_len为36k（即最大输出为4k）。
rm -rf /tmp/Janus-Pro-7B/trt_engines/
trtllm-build --checkpoint_dir /tmp/Janus-Pro-7B/tllm_checkpoint/ \
--output_dir /tmp/Janus-Pro-7B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 4 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 36960 --max_num_tokens 32768 --max_multimodal_len 18432
```

## 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_janus-pro.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 模拟请求

```bash
# 测试单张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "JanusPro",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/Janus/images/logo.png"
            }
          },
          {
            "type": "text",
            "text": "<image_placeholder>\n这是什么"
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "created": 1739978478,
 "model": "JanusPro",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一个名为“deepseek”的标志。标志由一个蓝色的鲸鱼图案和“deepseek”字样组成。鲸鱼图案位于左侧，而“deepseek”字样则位于右侧，以蓝色字体呈现。\n\n“deepseek”是一个知名的开源搜索引擎，它使用深度求索（DeepSeek）算法来搜索互联网内容。这个算法通过分析网页的结构和内容，能够更有效地找到目标网页。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 626,
  "completion_tokens": 32,
  "total_tokens": 658
 }
}
'

# 测试多轮对话与输出检测框
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "JanusPro",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/Janus/images/logo.png"
            }
          },
          {
            "type": "text",
            "text": "<image_placeholder>\n这是什么"
          }
        ]
      },
      {
        "role": "assistant",
        "content": "这张图片展示了一个名为“deepseek”的标志。标志由一个蓝色的鲸鱼图案和“deepseek”字样组成。鲸鱼图案位于左侧，而“deepseek”字样则位于右侧，以蓝色字体呈现。\n\n“deepseek”是一个知名的开源搜索引擎，它使用深度求索（DeepSeek）算法来搜索互联网内容。这个算法通过分析网页的结构和内容，能够更有效地找到目标网页。"
      },
      {
        "role": "user",
        "content": "左边的图标是什么？"
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1740133705,
 "model": "JanusPro",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": " 左边的图标是一只蓝色的鲸鱼。这个鲸鱼图案是“deepseek”标志的一部分，象征着深海探索和搜索的含义。鲸鱼通常被用来代表深海，因为它们生活在水下，并且能够探测到远距离的物体。这与“deepseek”搜索算法在互联网中探测和发现内容的能力相呼应。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 735,
  "completion_tokens": 75,
  "total_tokens": 810
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "JanusPro",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "file://./data/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file://./data/image2.jpg"
            }
          },
          {
            "type": "text",
            "text": "<image_placeholder>\n<image_placeholder>\n描述一下两张图片的不同。"
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1740133736,
 "model": "JanusPro",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了两只不同类型的熊猫。第一张图片中的熊猫是一只红熊猫，它有着棕色的毛发和白色的面部，看起来非常可爱。红熊猫正趴在木制平台上，背景是一些树木和绿叶。第二张图片中的熊猫是一只大熊猫，它有着典型的黑白相间的毛色，正坐在竹林中，周围是茂密的绿色植物。大熊猫看起来非常悠闲，似乎在享受周围的自然环境。这两张图片展示了熊猫在不同环境中的样子，一个是在森林中，另一个是在竹林中。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 1210,
  "completion_tokens": 117,
  "total_tokens": 1327
 }
}
'
# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image_placeholder>简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只正在休息的猫。猫的毛色主要是白色，头部和背部有棕色和黑色的斑点。猫的眼睛闭着，看起来非常放松，似乎在享受阳光或休息。猫的四肢舒展，身体平躺在地面上，地面看起来是粗糙的石材或水泥地面。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1740133762, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=68, prompt_tokens=630, total_tokens=698, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 开启gradio服务

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动多模态聊天界面，使用janus-pro多模态模型，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py janus-pro 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
