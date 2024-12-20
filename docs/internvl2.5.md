# grps-trtllm

InternVL2_5多模态LLM模型的部署示例。具体不同尺寸的vit和llm组合如下表格：

|   Model Name    |                                       Vision Part                                       |                                 Language Part                                  |                           HF Link                           |
|:---------------:|:---------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|:-----------------------------------------------------------:|
| InternVL2_5-1B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-1B)  |
| InternVL2_5-2B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-2B)  |
| InternVL2_5-4B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |     [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-4B)  |
| InternVL2_5-8B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-8B)  |
| InternVL2_5-26B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |
| InternVL2_5-38B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-38B) |
| InternVL2_5-78B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-78B) |

## 演示

<img src="gradio.gif" alt="gradio.gif">

## 开发环境

见[本地开发与调试拉取代码和创建容器部分](../README.md#3-本地开发与调试)。

## 构建trtllm引擎

### 2B\8B\26B模型

以8B模型为例，其他模型类似。

```bash
# 下载InternVL2_5-8B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2_5-8B /tmp/InternVL2_5-8B

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2_5-8B/tllm_checkpoint/
python3 tools/internvl2/convert_internlm2_ckpt.py --model_dir /tmp/InternVL2_5-8B/ \
--output_dir /tmp/InternVL2_5-8B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=2（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 13312
# 设置max_input_len为32k，max_seq_len为36k（即最大输出为4k）。
rm -rf /tmp/InternVL2_5-8B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL2_5-8B/tllm_checkpoint/ \
--output_dir /tmp/InternVL2_5-8B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 36960 --max_num_tokens 32768 --max_multimodal_len 13312

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL2_5-8B \
--imagePath ./data/frames/frame_0.jpg \
--onnxFile /tmp/InternVL2_5-8B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL2_5-8B/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
```

### 1B\4B\38B\78B模型

以4B模型为例，其他模型类似。

```bash
# 下载InternVL2_5-4B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2_5-4B /tmp/InternVL2_5-4B
# 拷贝对应尺寸缺失的tokenizer.json，例如InternVL2_5-4B对应Qwen2.5-3B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct /tmp/Qwen2.5-3B-Instruct
cp /tmp/Qwen2.5-3B-Instruct/tokenizer.json /tmp/InternVL2_5-4B/

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2_5-4B/tllm_checkpoint/
python3 tools/internvl2/convert_qwen2_ckpt.py --model_dir /tmp/InternVL2_5-4B/ \
--output_dir /tmp/InternVL2_5-4B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=2（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 13312
# 设置max_input_len为32k，max_seq_len为36k（即最大输出为4k）。
rm -rf /tmp/InternVL2_5-4B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL2_5-4B/tllm_checkpoint/ \
--output_dir /tmp/InternVL2_5-4B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 36960 --max_num_tokens 32768 --max_multimodal_len 13312

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2.5中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL2_5-4B \
--imagePath ./data/frames/frame_0.jpg \
--onnxFile /tmp/InternVL2_5-4B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL2_5-4B/vision_encoder_bfp16.trt \
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
grpst start ./server.mar --inference_conf=conf/inference_internvl2.5-8B.yml

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
    "model": "InternVL2_5",
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
              "url": "file:///tmp/InternVL2_5-8B/examples/image1.jpg"
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
 "id": "chatcmpl-10",
 "object": "chat.completion",
 "created": 1734620542,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一只红熊猫，它正趴在木板上，背景是绿色的树木和树干。红熊猫有着棕红色的毛发和白色的面部，看起来非常可爱。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 35,
  "total_tokens": 3418
 }
}
'

# 测试通过https从网络上下载的一张图片，解读其中的文字内容
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2_5",
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
 "id": "chatcmpl-11",
 "object": "chat.completion",
 "created": 1734620562,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这是一张包含每日简报的图片。图片顶部是蓝色背景，上面写着“星期六”，并有三颗星。下面有一段文字：“短短的一生，我们最终都会失去，你不妨大胆一些。爱一个人，攀一座山，追一个梦。”\n\n接下来是“早安读世界”和“今日简报”的标题，下面列出了15条新闻：\n\n1. 四川甘孜州官方：“3·15”雅江森林火灾原因初步查明：系施工动火作业引发，突遇极端大风造成扩散。\n2. 最高检表态：对未成年人实施的故意杀人、故意伤害，致人死亡等严重犯罪，符合核准追诉条件的，要依法追究刑事责任。\n3. 游族网络董事长林奇被毒杀一案，被告许某一审被判无期，据悉其因管理经营矛盾，有预谋的在被害人食物中投毒致其死亡。\n4. 武汉地铁就“无臂男子免费乘地铁被要求出示残疾证”一事致歉，当时男子：没必要道歉，希望制度更人性化。\n5. 3月22日我国首个无人驾驶吨级电动垂直起降航空器获批合格证，据悉其载重可达400公斤，主要用于低空物流以及紧急物资运输与应急救援。\n6. 我国网民数量达到10.92亿人，互联网普及率达77.5%。\n7. 国家林草局：我国成为全球森林资源增长最快的国家，近20年来为全球贡献了约1/4的新增绿化面积。\n8. 河南郑州：2024年3月22日至4月30日，八区联合开展购车补贴活动，新能源汽车每台补贴不高于5000元，燃油车每台不高于3000元。\n9. 国台办披露：福建海警救起的两名海钓人员，其中一人为台军方现役人员，其编造虚假职业隐瞒身份还需进一步核实，另一人于3月22日送返金门。\n10. 因甘肃天水麻辣烫火出圈，清明小长假部分到天水的火车票已售罄，天水酒店预订量创近三年来单周预订量新高。\n11. 外媒：加拿大3月21日宣布，拟减少临时居留人数。今年1月加拿大称将在两年内减少留学签证数量，并限制毕业后申请工作签证。\n12. 外媒：以色列22日宣布，没收8平方公里约旦河西岸的巴勒斯坦土地归以色列所有。\n13. 外媒：美国一所医院成功完成一例将基因编辑猪肾移植到患者体内的手术，当地媒体称患者恢复良好，不日将出院。\n14. 外媒：美国得州边境墙附近，有上百名非法移民冲破铁丝网试图非法进入美国。\n15. 俄媒：3月22日，俄罗斯对乌克兰能源设施发动大规模无人机和导弹袭击。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 636,
  "total_tokens": 4019
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2_5",
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
              "url": "file:///tmp/InternVL2_5-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/image2.jpg"
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
 "id": "chatcmpl-12",
 "object": "chat.completion",
 "created": 1734620632,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这两张图片展示了两种不同的动物。\n\n图片1中的动物是小熊猫，它有红棕色的毛发，白色的面部和耳朵边缘，以及黑色的眼圈。小熊猫正趴在木板上，背景是树木和绿叶。\n\n图片2中的动物是大熊猫，它有黑白相间的毛发，黑色的眼圈，耳朵和四肢，以及白色的面部和腹部。大熊猫正坐在地上，周围有绿色的植物和竹子。\n\n这两张图片中的动物在外观和栖息环境上都有明显的区别。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 5955,
  "completion_tokens": 104,
  "total_tokens": 6059
 }
}
'

# 测试多轮对话
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2_5",
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
              "url": "file:///tmp/InternVL2_5-8B/examples/image1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/image2.jpg"
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": "这两张图片展示了两种不同的动物。\n\n图片1中的动物是小熊猫，它有红棕色的毛发，白色的面部和耳朵边缘，以及黑色的眼圈。小熊猫正趴在木板上，背景是树木和绿叶。\n\n图片2中的动物是大熊猫，它有黑白相间的毛发，黑色的眼圈，耳朵和四肢，以及白色的面部和腹部。大熊猫正坐在地上，周围有绿色的植物和竹子。\n\n这两张图片中的动物在外观和栖息环境上都有明显的区别。"
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
 "id": "chatcmpl-15",
 "object": "chat.completion",
 "created": 1734620793,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "图片-2中的熊猫所在的环境是一个自然栖息地，周围有茂密的绿色植物和竹子。地面上覆盖着一些枯叶和树枝，背景中可以看到一些树木和灌木。整体环境看起来像是一个动物园或野生动物保护区，为熊猫提供了丰富的植被和自然栖息地。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 6077,
  "completion_tokens": 60,
  "total_tokens": 6137
 }
}
'

# 测试视频帧
cp -r ./data/frames /tmp/InternVL2_5-8B/examples/
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL2_5",
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
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_0.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_1.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_2.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_3.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_4.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL2_5-8B/examples/frames/frame_5.jpg"
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
 "id": "chatcmpl-16",
 "object": "chat.completion",
 "created": 1734620834,
 "model": "InternVL2_5",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "视频展示了两只红熊猫在户外的活动。一只红熊猫坐在树枝上，另一只则站在地面上。它们周围是绿色的草地和树木，背景中还有竹子制成的攀爬架。红熊猫们看起来非常活泼，其中一只在树枝上啃食竹子，另一只则在攀爬架上玩耍。视频风格自然，光线柔和，呈现出一种宁静和谐的氛围。红熊猫的毛色主要是红色和黑色，眼睛周围有白色的斑纹。整个场景给人一种温馨和放松的感觉。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 4697,
  "completion_tokens": 104,
  "total_tokens": 4801
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-17', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫。猫的身体大部分是白色的，有一些棕色的斑点。它正躺在地上，眼睛闭着，看起来非常放松和舒适。背景是灰色的地面。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1734620869, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=38, prompt_tokens=2359, total_tokens=2397, completion_tokens_details=None, prompt_tokens_details=None))
'

# 通过base64 img url方式进行请求
python3 client/base64_img_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false ./data/image1.jpg
# 返回如下：
: '
ChatCompletion(id='chatcmpl-18', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只可爱的红熊猫。红熊猫有着棕红色的毛发，白色的面部和耳朵边缘，以及黑色的眼圈。它正趴在一块木板上，背景是一些树木和绿叶。红熊猫看起来非常可爱，眼神温柔。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1734620885, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=49, prompt_tokens=3383, total_tokens=3432, completion_tokens_details=None, prompt_tokens_details=None))
’
```

## 开启gradio服务

![gradio.png](gradio.png)

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动多模态聊天界面，使用internvl2多模态模型，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py internvl2 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
