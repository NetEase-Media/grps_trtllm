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
--minBS 1 --optBS 13 --maxBS 26
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
    "content": "这张图片展示了一只大熊猫，它正坐在一个木制平台上，周围环绕着绿色的植物和竹子。大熊猫的毛色为黑白相间，眼睛周围有黑色的斑块，耳朵也是黑色的。它的身体大部分被绿色的植物遮挡，只能看到它的头部和部分身体。背景中还有一些模糊的树木和植物，营造出一种自然环境的感觉。"
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
    "content": "这张图片展示了一只可爱的红熊猫。红熊猫有着红棕色的毛发，脸部和耳朵周围有白色的毛发，眼睛大而明亮，鼻子是黑色的。它正趴在一个木制的平台上，背景是一片绿色的植物，可能是为了模拟自然环境。红熊猫看起来非常可爱，嘴巴微微张开，似乎在微笑。"
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
    "content": "这两张图片展示了两种不同的动物，分别是红熊猫和熊猫。\n\n1. 红熊猫：\n红熊猫是一种非常罕见的动物，它们通常生活在南美洲的安第斯山脉。红熊猫的毛色主要是红色和白色，它们的体型较小，通常只有20-30厘米高。红熊猫是杂食性动物，它们的食物包括水果、昆虫、小型哺乳动物等。红熊猫的栖息地通常是高海拔的山地，它们喜欢生活在树木上。\n\n2. 熊猫：\n熊猫是一种非常受欢迎的动物，它们通常生活在亚洲的竹林中。熊猫的毛色主要是黑白相间，它们的体型较大，通常有100-150厘米高。熊猫是草食性动物，它们的食物主要是竹子。熊猫的栖息地通常是低海拔的竹林，它们喜欢生活在地面上。\n\n这两张图片展示了两种不同的动物，分别是红熊猫和熊猫。"
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
        "content": "这两张图片展示了两种不同的动物，分别是红熊猫和熊猫。\n\n1. 红熊猫：\n红熊猫是一种非常罕见的动物，它们通常生活在南美洲的安第斯山脉。红熊猫的毛色主要是红色和白色，它们的体型较小，通常只有20-30厘米高。红熊猫是杂食性动物，它们的食物包括水果、昆虫、小型哺乳动物等。红熊猫的栖息地通常是高海拔的山地，它们喜欢生活在树木上。\n\n2. 熊猫：\n熊猫是一种非常受欢迎的动物，它们通常生活在亚洲的竹林中。熊猫的毛色主要是黑白相间，它们的体型较大，通常有100-150厘米高。熊猫是草食性动物，它们的食物主要是竹子。熊猫的栖息地通常是低海拔的竹林，它们喜欢生活在地面上。\n\n这两张图片展示了两种不同的动物，分别是红熊猫和熊猫。"
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
    "content": "图片-2中的熊猫所在的环境是一个竹林。竹林是熊猫的主要栖息地，它们喜欢生活在低海拔的竹林中。竹林中有大量的竹子，这是熊猫的主要食物来源。竹林的环境通常比较湿润，因为竹子需要大量的水分才能生长。竹林中的温度也比较适宜，因为熊猫是温血动物，它们需要保持一定的体温。竹林中的光线也比较柔和，因为竹子可以遮挡一部分阳光。竹林的环境非常适合熊猫生活，因为它们可以在竹林中自由地活动，并且有足够的食物来源。"
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

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-11', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫。猫正躺在地上，眼睛闭着，看起来非常放松和舒适。猫的毛色主要是白色，带有一些黑色和灰色的斑点。背景是灰色的地面，可能是水泥或沥青。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1729162343, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=45, prompt_tokens=4663, total_tokens=4708, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```