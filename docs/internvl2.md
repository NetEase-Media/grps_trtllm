# grps-trtllm

InternVL2多模态LLM模型的部署示例。由于InternVL2不同尺寸对应的LLM可能不一样，目前支持了Internlm2作为LLM模型的尺寸，即InternVL2-2B,
InternVL2-8B, InternVL2-26B。

## 开发环境

见[本地开发与调试](../README.md#3-本地开发与调试)。

## 构建trtllm引擎

```bash
# 下载InternVL2-8B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2-8B /tmp/InternVL2-8B

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2-8B/tllm_checkpoint/
python3 tools/internvl2/convert_llm_ckpt.py --model_dir /tmp/InternVL2-8B/ \
--output_dir /tmp/InternVL2-8B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=2（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 13312
rm -rf /tmp/InternVL2-8B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL2-8B/tllm_checkpoint/ \
--output_dir /tmp/InternVL2-8B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 60416 --max_num_tokens 32768 --max_multimodal_len 13312

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL2-8B \
--imagePath /tmp/InternVL2-8B/examples/image1.jpg \
--onnxFile /tmp/InternVL2-8B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL2-8B/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
```

## 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_internvl2.yml

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
    "model": "InternVL2",
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
              "url": "file:///tmp/InternVL2-8B/examples/image2.jpg"
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
 "id": "chatcmpl-6",
 "object": "chat.completion",
 "created": 1729151382,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一只大熊猫。大熊猫的身体大部分是黑白相间的，头部和耳朵是黑色的，而身体其他部分则是白色的。它的眼睛周围有一圈黑色的斑块，看起来非常可爱。大熊猫正坐在地上，周围有许多绿色的植物，包括竹子和其他灌木。\n\n背景中可以看到一些木制的结构，可能是大熊猫的栖息地的一部分。地面上覆盖着一些干枯的树叶和树枝，显示出这是一个自然环境。\n\n通过观察图片中的细节，可以推断出以下几点：\n\n1. **大熊猫的栖息地**：大熊猫主要生活在中国的竹林中，因此图片中的绿色植物和竹子符合大熊猫的自然栖息地。\n2. **大熊猫的行为**：大熊猫通常喜欢坐着或躺着，这张图片中的大熊猫似乎在休息或观察周围的环境。\n3. **保护环境**：大熊猫是濒危物种，保护它们的栖息地和食物来源非常重要。图片中的环境看起来相对自然，有助于大熊猫的生存。\n\n总的来说，这张图片展示了一只大熊猫在自然栖息地中的情景，突显了保护野生动物及其栖息地的重要性。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 5175,
  "completion_tokens": 77,
  "total_tokens": 5252
 }
}
'

# 测试通过https从网络上下载的一张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2",
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
              "url": "https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg"
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
 "id": "chatcmpl-7",
 "object": "chat.completion",
 "created": 1729151405,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一只可爱的红熊猫。红熊猫的身体覆盖着厚厚的棕色和白色毛发，脸部有明显的白色斑纹，耳朵和眼睛周围是白色的，而身体其他部分则是棕色的。它的眼睛大而圆，显得非常可爱。红熊猫的嘴巴微微张开，露出粉红色的舌头，显得非常活泼和好奇。\n\n红熊猫通常生活在竹林中，以竹子为主要食物。它们是夜行动物，白天通常会睡觉，晚上活动。红熊猫是濒危物种，主要分布在中国的四川、陕西和甘肃等地区。\n\n背景中可以看到一些绿色的植物，可能是竹叶，这与红熊猫的栖息地相符。此外，背景中还有一些木质的结构，可能是红熊猫生活环境的组成部分。\n\n通过这张图片，我们可以感受到红熊猫的可爱和独特之处，同时也能了解一些关于它们的生活习性和栖息地的知识。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 4663,
  "completion_tokens": 68,
  "total_tokens": 4731
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2",
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
              "url": "file:///tmp/InternVL2-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/image2.jpg"
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
 "id": "chatcmpl-8",
 "object": "chat.completion",
 "created": 1729151468,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这两张图片展示了不同种类的熊猫。\n\n1. 图片1：\n   - 这是一只红熊猫（学名：Ailurus fulgens）。\n   - 红熊猫的毛色主要是红棕色，脸部和耳朵周围有白色斑块。\n   - 它正坐在一个木制结构上，背景是绿色的植物。\n   - 红熊猫的体型较小，耳朵较大，眼睛周围有明显的黑色斑块。\n\n2. 图片2：\n   - 这是一只大熊猫（学名：Ailuropoda melanoleuca）。\n   - 大熊猫的毛色主要是黑白相间，脸部和耳朵周围是黑色，身体其他部分是白色。\n   - 大熊猫正坐在地上，周围有很多绿色的植物。\n   - 大熊猫的体型较大，耳朵较小，眼睛周围有明显的黑色斑块。\n\n总的来说，这两张图片展示了不同种类的熊猫，红熊猫和大熊猫，它们在毛色、体型和栖息环境上都有明显的区别。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 11844,
  "completion_tokens": 60,
  "total_tokens": 11904
 }
}
'

# 测试多轮对话
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2",
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
              "url": "file:///tmp/InternVL2-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/image2.jpg"
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": "这两张图片展示了不同种类的熊猫。\n\n1. 图片1：\n   - 这是一只红熊猫（学名：Ailurus fulgens）。\n   - 红熊猫的毛色主要是红棕色，脸部和耳朵周围有白色斑块。\n   - 它正坐在一个木制结构上，背景是绿色的植物。\n   - 红熊猫的体型较小，耳朵较大，眼睛周围有明显的黑色斑块。\n\n2. 图片2：\n   - 这是一只大熊猫（学名：Ailuropoda melanoleuca）。\n   - 大熊猫的毛色主要是黑白相间，脸部和耳朵周围是黑色，身体其他部分是白色。\n   - 大熊猫正坐在地上，周围有很多绿色的植物。\n   - 大熊猫的体型较大，耳朵较小，眼睛周围有明显的黑色斑块。\n\n总的来说，这两张图片展示了不同种类的熊猫，红熊猫和大熊猫，它们在毛色、体型和栖息环境上都有明显的区别。"
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
 "id": "chatcmpl-13",
 "object": "chat.completion",
 "created": 1729235382,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "图片-2中的熊猫所在的环境是一个自然栖息地，可能是一个动物园或保护区的模拟自然环境。具体描述如下：\n\n1. **植被**：\n   - 熊猫周围有很多绿色的植物，包括竹子和其他草本植物。\n   - 地面上覆盖着一些干枯的树叶和树枝，显示出这是一个自然环境。\n\n2. **木制结构**：\n   - 熊猫坐在一个木制结构上，这个结构可能是供熊猫攀爬和休息的设施。\n   - 木制结构看起来比较粗糙，可能是用原木制作的，以模拟自然环境。\n\n3. **背景**：\n   - 背景中有更多的绿色植物，显示出这是一个植被茂密的区域。\n   - 环境看起来比较湿润，适合熊猫的栖息需求。\n\n4. **地面**：\n   - 地面上覆盖着一些干枯的树叶和树枝，显示出这是一个自然环境。\n   - 地面看起来比较松软，适合熊猫的行走和休息。\n\n总的来说，图片-2中的熊猫所在的环境是一个模拟自然栖息地的环境，有丰富的植被和木制结构，为熊猫提供了舒适的生活空间。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 11947,
  "completion_tokens": 9,
  "total_tokens": 11956
 }
}
'

# 测试视频帧
cp -r ./data/frames /tmp/InternVL2-8B/examples/
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Frame1:<image>\nFrame2:<image>\nFrame3:<image>\nFrame4:<image>\nFrame5:<image>\nFrame6:<image>\n描述一下视频的内容。不要重复。"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_0.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_2.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_3.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_4.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2-8B/examples/frames/frame_5.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-11",
 "object": "chat.completion",
 "created": 1729245156,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "视频展示了两只红熊猫在户外活动的情景。视频的主要内容是这两只红熊猫在树木和竹竿搭建的游乐设施上玩耍和互动。核心中心思想是展示红熊猫在自然环境中的活动和行为，突出它们的天真和活泼。\n\n视频中出现了两只红熊猫，它们分别位于不同的位置。一只红熊猫坐在树枝上，另一只红熊猫则站在地面上。坐在树枝上的红熊猫身体呈棕色和黑色相间，它紧紧抓住树枝，显得非常灵活和敏捷。它的前爪抓住树枝，后腿悬空，身体微微前倾，似乎在观察或等待什么。\n\n地面上的红熊猫则显得更加活跃。它站在草地上，前爪抓住一根悬挂的竹竿，竹竿上挂着一些食物。这只红熊猫用前爪抓住竹竿，用力拉扯，试图将食物拉近。它的动作非常迅速和有力，显得非常专注和投入。\n\n视频的场景是一个户外的自然环境，背景中可以看到绿色的草地和树木。树木的树干和树枝上缠绕着一些竹竿，形成了一个供红熊猫玩耍的游乐设施。地面上覆盖着绿色的草地，显得非常自然和舒适。\n\n视频中，红熊猫的动作非常生动。坐在树枝上的红熊猫时而低头观察，时而抬头张望，显得非常警觉和好奇。地面上的红熊猫则不停地拉扯竹竿，试图获取食物。它的动作非常迅速和有力，显得非常灵活和敏捷。\n\n总的来说，这个视频通过展示红熊猫在自然环境中的活动和行为，突出了它们的天真和活泼。视频中的红熊猫在游乐设施上玩耍和互动，展现了它们灵活的身体和敏捷的动作。背景中的自然环境也为视频增添了更多的真实感和自然感。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 9306,
  "completion_tokens": 63,
  "total_tokens": 9369
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-11', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫。猫的身体大部分是白色的，背部和头部有黑色的斑点。猫的耳朵竖起，眼睛半闭，似乎在打盹。猫的胡须清晰可见，鼻子和嘴巴也清晰可见。猫的身体蜷缩在地面上，地面是灰色的，看起来像是水泥或沥青。\n\n从猫的姿态和表情来看，它处于一种放松和舒适的状态。猫的毛发看起来非常柔软，整体给人一种宁静的感觉。\n\n通过观察猫的特征，可以推断出这只猫可能是一只家猫，因为它的毛发整洁，而且看起来非常健康。家猫通常喜欢在温暖和安静的地方休息，这与图片中的环境相符。\n\n总结来说，这张图片展示了一只白色的猫，背部和头部有黑色斑点，它正躺在灰色的地面上打盹，表现出一种放松和舒适的状态。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1729162343, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=45, prompt_tokens=4663, total_tokens=4708, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```