# grps-trtllm

InternVL2多模态LLM模型的部署示例。由于InternVL2不同尺寸对应的LLM可能不一样，目前支持了```Internlm2```、```Qwen2```、
```Phi3```作为LLM模型的尺寸，即1B、2B、4B、8B、26B。
具体不同尺寸的vit和llm组合如下表格：

| Model Name           | Vision Part                                                                         | Language Part                                                                                | HF Link                                                          | MS Link                                                                |
|----------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| InternVL2-1B         | [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)       | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-1B)         | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-1B)         |
| InternVL2-2B         | [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)       | [internlm2-chat-1_8b](https://huggingface.co/internlm/internlm2-chat-1_8b)                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-2B)         | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-2B)         |
| InternVL2-4B         | [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)       | [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)        | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-4B)         | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-4B)         |
| InternVL2-8B         | [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)       | [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-8B)         | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-8B)         |
| InternVL2-26B        | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | [internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)                     | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-26B)        | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-26B)        |
| InternVL2-40B        | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-40B)        | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-40B)        |
| InternVL2-Llama3-76B | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | [Hermes-2-Theta-Llama-3-70B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-Llama3-76B) |

## 开发环境

见[本地开发与调试拉取代码和创建容器部分](../README.md#3-本地开发与调试)。

## 构建trtllm引擎

### 2B\8B\26B模型

以8B模型为例，其他模型类似。

```bash
# 下载InternVL2-8B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2-8B /tmp/InternVL2-8B

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2-8B/tllm_checkpoint/
python3 tools/internvl2/convert_internlm2_ckpt.py --model_dir /tmp/InternVL2-8B/ \
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

### 1B模型

如果输出重复内容，也可以尝试将访问服务时将```repetition_penalty```采样参数调大，例如设置为1.2。

```bash
# 下载InternVL2-1B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2-1B /tmp/InternVL2-1B
# 拷贝缺失的tokenizer.json
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct /tmp/Qwen2-0.5B-Instruct
cp /tmp/Qwen2-0.5B-Instruct/tokenizer.json /tmp/InternVL2-1B/

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2-1B/tllm_checkpoint/
python3 tools/internvl2/convert_qwen2_ckpt.py --model_dir /tmp/InternVL2-1B/ \
--output_dir /tmp/InternVL2-1B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=2（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 13312
rm -rf /tmp/InternVL2-1B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL2-1B/tllm_checkpoint/ \
--output_dir /tmp/InternVL2-1B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 60416 --max_num_tokens 32768 --max_multimodal_len 13312

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL2-1B \
--imagePath /tmp/InternVL2-1B/examples/image1.jpg \
--onnxFile /tmp/InternVL2-1B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL2-1B/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
```

### 4B模型

另外4B模型可能对中文支持不太好，有时会乱码。

```bash
# 下载InternVL2-4B模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2-4B /tmp/InternVL2-4B

# 安装依赖
pip install -r ./tools/internvl2/requirements.txt

# 转换ckpt
rm -rf /tmp/InternVL2-4B/tllm_checkpoint/
python3 tools/internvl2/convert_phi3_ckpt.py --model_dir /tmp/InternVL2-4B/ \
--output_dir /tmp/InternVL2-4B/tllm_checkpoint/ --dtype bfloat16

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为2，即支持2个并发同时推理，超过两个排队处理。
# 设置每个请求最多输入26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
# 即：max_multimodal_len=2（max_batch_size） * 26（图片最多产生patch个数） * 256（每个patch对应256个token） = 13312
rm -rf /tmp/InternVL2-4B/trt_engines/
trtllm-build --checkpoint_dir /tmp/InternVL2-4B/tllm_checkpoint/ \
--output_dir /tmp/InternVL2-4B/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 60416 --max_num_tokens 32768 --max_multimodal_len 13312

# 构建vit引擎，设置--maxBS为26可以同时处理26个图片patch（InternVL2中每个图片根据不同的尺寸最多产生13个patch）。
python3 tools/internvl2/build_vit_engine.py --pretrainedModelPath /tmp/InternVL2-4B \
--imagePath /tmp/InternVL2-4B/examples/image1.jpg \
--onnxFile /tmp/InternVL2-4B/vision_encoder_bfp16.onnx \
--trtFile /tmp/InternVL2-4B/vision_encoder_bfp16.trt \
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
grpst start ./server.mar --inference_conf=conf/inference_internvl2-8B.yml

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
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "created": 1729657427,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "图片中有一只大熊猫，它正坐在地上。大熊猫的身上有典型的黑白相间的颜色分布，并且它的头部、眼圈、耳朵、四肢和肩膀上的毛是黑色的，其余部位的毛是白色的。它的面部表情温和，眼睛大而圆，看起来十分可爱且好奇。\n\n大熊猫背后有几根木制的支撑杆，形成一种简单的结构，可能是一个简易的巢穴或者栖息处。周围的环境是绿色的植物和少量的干树叶。左边能看到一些绿色的竹子，右边有更多的绿色植被覆盖地面，整体环境比较自然和舒适。\n\n图片中没有出现其他动物，也没有明显的人类活动迹象，因此可以推测这是一个野生环境或者一个模仿自然环境的动物园。从图片角度来看，大熊猫正站在一些绿色植物中，可能是为了更凉爽和舒适，或者是在寻找食物。\n\n总结，这张图片展示了一只处于自然环境中、看起来相当健康的大熊猫，背景是典型的竹林环境，整体上给人一种宁静和谐的感觉。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 2615,
  "completion_tokens": 202,
  "total_tokens": 2817
 }
}
'

# 测试通过https从网络上下载的一张图片，解读其中的文字内容
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
 "created": 1729657468,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "图片是一份早安读世界每日简报。图片左上角是星期的标志和评分图标，并且有一行文字：“短短的一生，我们最终都会失去，你不妨大胆一些。爱一个人，攀一峰山，追一个梦。”\n\n下面是每日简报的结构：\n\n1. 标题“早安读世界 今日简报”，旁边有“GOOD MORNING”字样。\n2. 日期和内容：【每日资讯简报】2024年3月23日，星期六，农历二月十四，早安！\n    - 四川甘孜州官方：“3.15”雅江森林火灾原因初步查明：系施工动火作业引发，突遇极端大风造成扩散；\n    - 最高人民法院：对未成年人实施的故意杀人、故意伤害，致人死亡等严重犯罪，符合核准追诉条件的，要依法追究刑事责任；\n    - 游族网络董事长林奇被毒杀一案，被告许旺一审判无期，据悉其因管理经营矛盾，有预谋的在被害人食物中投毒致其死亡；\n    - 武汉地铁就“无臂男子免费乘地铁被要求出示残疾证”一事致歉。\n        - 男子：没必要道歉，希望制度更人性化；\n    - 3月22日我国首个无人驾驶吨级电动垂直起降航空器获批合格证。据悉其最大重量可达400公斤，主要用于低空物流及紧急物资运输与应急救援；\n    - 我国网民数量达到10.92亿人，互联网普及率达77.5%；\n    - 国家林草局：我国成为全球森林资源增长最快的国家，近20年来为全球贡献了约1/4的新增绿化面积；\n    - 河南郑州：2024年3月22日至4月30日，八区联合开展购车补贴活动，新能源汽车每补贴不低于5000元，燃油车每台补贴不高于3000元；\n    - 国台办披露：福建海警救起的两名海钓人员，其中一人为台军方现役人员，其编造虚假职位隐瞒身份还需进一步核实，另一人于3月22日送返金门。\n    - 因甘肃天水麻辣烫火出圈，清明小长假部分到天水的火车票已售罄，天水酒店预订量达三年来单周预订量新少。\n    - 外媒：加拿大3月21日宣布，将减少临时居留人数。今年1月，加拿大将在这两年内减少签证数量，并限制毕业后再就业签证数量；\n    - 外媒：以色列22日宣布，没收8平方公里约旦河西岸的巴勒斯坦土地归以色列所有；\n    - 外媒：美国一所医院成功完成一例将基因编辑猪肾移植到患者体内的手术，当地媒体称患者恢复良好，不久将出院；\n    - 外媒：美国得州边境墙附近，有上百名非法移民冲破铁丝网试图非法进入美国；\n    - 俄媒：3月22日，俄罗斯对乌克兰能源设施发动大规模无人机和导弹袭击。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 671,
  "total_tokens": 4054
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
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1729657577,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这两张图片展示了不同种类的熊猫。\n\n图1的熊猫：\n- 颜色主要呈白色和黑褐色，面部中央和背部有明显的黑色斑纹。\n- 这只熊猫正在木质的箱子上休息，耳朵竖起，眼睛注视着前方。它的表情显得有些好奇。\n\n图2的熊猫：\n- 毛色主要是黑白相间，但白色的毛发更加明显。\n- 这只熊猫坐在地面上，用后爪抱住前爪，显得有些慵懒或随意。\n\n总体来看，这两只熊猫表现出不同的姿势和状态，但都非常可爱和引人注目。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 5955,
  "completion_tokens": 132,
  "total_tokens": 6087
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
        "content": "这两张图片展示了不同种类的熊猫。\n\n图1的熊猫：\n- 颜色主要呈白色和黑褐色，面部中央和背部有明显的黑色斑纹。\n- 这只熊猫正在木质的箱子上休息，耳朵竖起，眼睛注视着前方。它的表情显得有些好奇。\n\n图2的熊猫：\n- 毛色主要是黑白相间，但白色的毛发更加明显。\n- 这只熊猫坐在地面上，用后爪抱住前爪，显得有些慵懒或随意。\n\n总体来看，这两只熊猫表现出不同的姿势和状态，但都非常可爱和引人注目。"
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
 "id": "chatcmpl-4",
 "object": "chat.completion",
 "created": 1729657666,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "在图片-2中，熊猫所在的环境看起来是一片人造的竹林。\n\n具体如下：\n\n1. **竹子**：周围可以看到茂密的竹子，这是一种非常典型的熊猫栖息环境元素。竹子为熊猫提供了遮蔽和食物来源。\n\n2. **木结构植物**：熊猫身后似乎有一组木制结构，可能是模拟的洞穴或休息区，这为熊猫提供了一个遮蔽的场所。\n\n3. **植被覆盖**：熊猫躺在厚厚的绿色植被上，地面铺满了竹子叶和各种植物，营造出自然的栖息环境。\n\n4. **人工环境**：尽管周围有自然植被，但整体环境看起来像是经过人工设计和维护的动物园或保护区。这些区域的布置通常会模仿熊猫的天然栖息地，使其能够适应较人造的生活环境。\n\n5. **地面材质**：地面由竹子和其他绿色植物组成，这不仅是熊猫的舒适休息区，同时也是其进行抓挠和滚动等自然行为的场所。\n\n这种环境对于熊猫来说非常重要，因为它为它们的日常生活提供了必要的安全感和舒适感，同时也保护其免受天敌的威胁。总体上，图片-2中的熊猫展示了一个温暖且适合它们生活习惯的自然人工栖息环境。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 6105,
  "completion_tokens": 251,
  "total_tokens": 6356
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
 "id": "chatcmpl-5",
 "object": "chat.completion",
 "created": 1729657720,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这个视频展示了两只可爱的红熊猫在户外动物园中活跃的情景。视频开始时，一只红色熊猫慵懒地躺在木制的枝架上，四肢放松自然，头部微微抬起，好像在享受着太阳的温暖。旁边有一只黑色熊猫正坐着，显得有些活泼。两只熊猫之间互动的少，各自沉浸在自己的世界中。\n\n红熊猫的毛发鲜艳，呈现出典型的红色和黑色相间的斑纹，白色的面部特征使它们看起来尤为独特和美丽。它们所在的环境是一个围栏内的绿化场地，有绿色的草地、棕色的树干和一些竹子构成。这些竹子被简单地绑在树干上，形成了一个简易的栖息空间。\n\n在视频的进一步展示中，红色熊猫开始活动起来，它用力地抓住一根悬挂在空中的食物吊钩，开始撕咬吊挂的食物。黑色熊猫则保持静止，继续观察或等待。吊钩上挂着一些用绳子悬挂在树枝上的食物，两只熊猫都表现出极大的兴趣，并尝试伸手去够取这些食物。可以看出，它们对周围的环境非常熟悉和适应。\n\n随着视频推进，红色熊猫的兴奋表现愈发明显，它不仅一次又一次地去尝试从空中悬挂在链条上的食物，还用爪子试图抓住更远吊钩上挂着的另一种食物。此时，黑色熊猫依旧站在原地，静静地观察。最终，红色熊猫成功地从吊钩上扯下一块食物，显得非常满意和兴奋，而黑色熊猫则在旁边也显得相当开心。\n\n整个视频的背景中，隐约可以看到一些模糊的绿色背景，可能是一堵绿色围挡或者一些其他的植物，这进一步强化了熊猫们所处的位置是在一个模拟自然环境的动物园中。绿色的植物和阳光照射下的小草营造出一种宁静和谐的氛围，使画面显得非常温馨与宁静。\n\n总结而言，这个视频主要通过展示熊猫们的日常生活行为、动作形态以及互动方式，捕捉到了它们的可爱与和谐，让人感受到人类与自然界和谐共存的美好。同时，从熊猫们动作的表现来看，它们适应并享受着动物园内的生活，这也能在一定程度上反映出动物园环境管理得当以及动物照顾上的精心细致。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 4697,
  "completion_tokens": 449,
  "total_tokens": 5146
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-8', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫。猫的身体大部分是白色的，背部和头部有黑色的斑点。猫的耳朵竖起，眼睛半闭，似乎在打盹。猫的胡须清晰可见，鼻子和嘴巴也清晰可见。猫的身体蜷缩在地面上，地面是灰色的，看起来像是水泥或沥青。\n\n从猫的姿态和表情来看，它处于一种放松和舒适的状态。猫的毛发看起来非常柔软，整体给人一种宁静的感觉。\n\n通过观察猫的特征，可以推断出这只猫可能是一只家猫，因为它的毛发整洁，而且看起来非常健康。家猫通常喜欢在温暖和安静的地方休息，这与图片中的环境相符。\n\n总结来说，这张图片展示了一只白色的猫，背部和头部有黑色斑点，它正躺在灰色的地面上打盹，表现出一种放松和舒适的状态。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1729523881, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=172, prompt_tokens=2359, total_tokens=2531, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
