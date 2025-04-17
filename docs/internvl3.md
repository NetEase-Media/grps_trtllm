# InternVL3

InternVL3多模态LLM模型的部署示例（暂不支持InternVL3-9B）。具体不同尺寸的vit和llm组合如下表格：

|  Model Name   |                                       Vision Part                                       |                                 Language Part                                  |                          HF Link                          |
|:-------------:|:---------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|:---------------------------------------------------------:|
| InternVL3-1B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |            [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)            | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-1B)  |
| InternVL3-2B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |            [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)            | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-2B)  |
| InternVL3-8B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |              [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)              | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-8B)  |
| InternVL3-9B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-9B)  |
| InternVL3-14B | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |             [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-14B) |
| InternVL3-38B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |             [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-38B) |
| InternVL3-78B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |             [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-78B) |

## 演示

<img src="gradio.gif" alt="gradio.gif">

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

### 1B\2B\8B\14B\38B\78B

以InternVL3-8B模型为例，其他模型类似。

```bash
# 下载InternVL3-8B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3-8B /tmp/InternVL3-8B

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL3-8B/tllm_checkpoint/
python3 tools/internvl2/convert_qwen2_ckpt.py --model_dir /tmp/InternVL3-8B/ \
--output_dir /tmp/InternVL3-8B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=4（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 26624
# 设置max_input_len为30k，max_seq_len为32k（即默认最大输出为2k）。
rm -rf /tmp/InternVL3-8B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL3-8B/tllm_checkpoint/ \
--output_dir /tmp/InternVL3-8B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 4 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 30720 --max_seq_len 32768 --max_num_tokens 32768 --max_multimodal_len 26624

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL3-8B \
--imagePath ./data/frames/frame_0.jpg \
--onnxFile /tmp/InternVL3-8B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL3-8B/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
```

### 9B

暂不支持

## 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 模拟请求

```bash
# 测试本地一张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "<image>\n简述一下这张图片的内容。"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-8B/examples/image1.jpg"
            }
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
 "created": 1744862005,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这是一张红熊猫的照片。红熊猫毛色为红褐色，脸和耳朵周围有白色，鼻子为黑色，显得非常可爱。背景看上去像是在户外，有绿色的树木。红熊猫正趴在木制的结构上，看起来很放松。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 56,
  "total_tokens": 3439
 }
}
'

# 测试通过https从网络上下载的一张图片，解读其中的文字内容
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "<image>\n简述一下这张图片的内容。"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://pic3.zhimg.com/v2-5904ffb96cf191bde40b91e4b7804d92_r.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1744862028,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这是一张每日新闻简报的图片。标题为“星期六”，背景为蓝色，配有激励的话：“短短的一生，我们最终都会失去，你不妨大胆一些。爱一个人，攀一座山，追一个梦。”正文部分分为两大部分：\n\n1. **早安读世界 今日简报 Good Morning**：\n   - 日期和日历信息：“2024年3月23日，星期六，农历二月十四，早安！”\n   - 共有15条新闻，涵盖了不同国家和地区的主要新闻：\n     1. 3·15雅江森林火灾起因（由施工引发）。\n     2. 对未成年人故意杀人严重犯罪追究刑责。\n     3. 游族网络董事长致毒杀案判决。\n     4. 武汉地铁对无臂男子免费乘车道歉事件。\n     5. 中国首个无人驾驶电动垂直起降航空器获准合格证。\n     6. 中国网民数量10.92亿，互联网普及率达77.5%。\n     7. 国家林草局消息：中国森林资源增长最快。\n     8. 郑州购车补贴活动：新能源每台补贴不超过5000元。\n     9. 国台办通报福建海警救起的两名渔民。\n     10. 甘肃天水麻椒火锅圈火。\n     11. 加拿大提议减少临时居留人数。\n     12. 以色列宣布没收并归还土地。\n     13. 美国成功移植猴肾。\n     14. 美国边境非法移民事件。\n     15. 俄罗斯对乌克兰发动无人机和导弹袭击。\n\n2. **底部的标识**：“早安读世界”。\n\n这些新闻简要概括了全球不同地区的最新重大事件。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 402,
  "total_tokens": 3785
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Image-1: <image>\nImage-2: <image>\n描述一下两张图片的不同。"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-8B/examples/image2.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-4",
 "object": "chat.completion",
 "created": 1744862213,
 "model": "InternVL3",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "Image-1: 图片中是一只红熊猫，它正趴在木制结构上，背景是绿色的树叶和树干。红熊猫具有独特的红褐色和白色相间的毛发，脸部有白色的条纹，耳朵上有明显的白色尖端。\n\nImage-2: 图片中是一只大熊猫，正用前爪抓着竹叶吃。大熊猫位于茂密的竹林中，周围有绿色的竹子和植物，地面散落着一些竹叶和竹枝。背景中隐约可见另一只大熊猫的一部分身影，整体环境看起来是一个自然的栖息地或动物园的保护区。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 5955,
  "completion_tokens": 130,
  "total_tokens": 6085
 }
}
'

# 测试多轮对话
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Image-1: <image>\nImage-2: <image>\n描述一下两张图片的不同。"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-8B/examples/image2.jpg"
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": "Image-1: 图片中是一只红熊猫，它正趴在木制结构上，背景是绿色的树叶和树干。红熊猫具有独特的红褐色和白色相间的毛发，脸部有白色的条纹，耳朵上有明显的白色尖端。\n\nImage-2: 图片中是一只大熊猫，正用前爪抓着竹叶吃。大熊猫位于茂密的竹林中，周围有绿色的竹子和植物，地面散落着一些竹叶和竹枝。背景中隐约可见另一只大熊猫的一部分身影，整体环境看起来是一个自然的栖息地或动物园的保护区。"
      },
      {
        "role": "user",
        "content": "描述一下图片-2中的熊猫所在的环境。"
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-5",
 "object": "chat.completion",
 "created": 1744862291,
 "model": "InternVL3",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "图片-2中的熊猫位于一个竹林环境中，周围有许多绿色的竹子和植物。熊猫坐在地面上，背景中可以见到一些石块和木结构，可能是一个栖息地的围栏或木桩。竹子繁茂，覆盖了大部分地面，营造出一种自然且隐秘的氛围，适合熊猫生活和觅食。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 6104,
  "completion_tokens": 76,
  "total_tokens": 6180
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-7', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只躺在地上的猫。猫的毛色主要是白色，并且背部有棕色和黑色的斑点。它看起来很放松，眼睛半闭着，似乎在享受阳光或只是在休息。猫的姿势很悠闲，四肢舒展，头微微侧向一边，给人一种非常惬意的感觉。背景是粗糙的水泥地面。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1744862414, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=75, prompt_tokens=2359, total_tokens=2434))
'

# 通过base64 img url方式进行请求
python3 client/base64_img_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false ./data/image1.jpg
# 返回如下：
: '
ChatCompletion(id='chatcmpl-8', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只可爱的小熊猫，也叫红熊猫。小熊猫的毛色是典型的红褐色和白色相间，耳朵内侧和脸部的毛发是白色的，而耳朵的外侧是红褐色。它正趴在木板上，神情显得很悠闲和好奇，背景是绿色的树叶和树干。红熊猫生活在树上，擅长攀爬，主要分布在中国西南部、不丹、尼泊尔和印度北部的森林中。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1744862433, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=101, prompt_tokens=3383, total_tokens=3484))
’
```

## 开启gradio服务

![gradio.png](gradio.png)

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动多模态聊天界面，使用internvl3多模态模型，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py internvl3 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
