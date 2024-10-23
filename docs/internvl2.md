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
              "url": "file:///tmp/InternVL2-8B/examples/image1.jpg"
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
 "created": 1729673126,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一只红熊猫。红熊猫是一种小型哺乳动物，属于熊科，主要分布在亚洲的森林地区。它们有着独特的红棕色皮毛和白色的面部特征，包括白色的眼圈、鼻子和嘴巴。红熊猫的耳朵较大，呈圆形，毛茸茸的。\n\n图片中的红熊猫正趴在一块木板上，似乎在观察周围的环境。它的眼睛大而圆，显得非常可爱。背景中可以看到一些绿色的植物，表明这只红熊猫可能处于一个自然环境或动物园中。\n\n红熊猫是杂食性动物，主要以竹子为食，但也吃昆虫和小型无脊椎动物。它们通常生活在竹林中，喜欢在树上活动。红熊猫是濒危物种，受到栖息地丧失和非法贸易的威胁。\n\n通过这张图片，我们可以了解到红熊猫的外观特征和栖息环境，同时也能感受到它们在自然界中的可爱和脆弱。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 185,
  "total_tokens": 3568
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
 "id": "chatcmpl-4",
 "object": "chat.completion",
 "created": 1729673166,
 "model": "InternVL2",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这是一张包含新闻简报的图片。图片顶部有一个蓝色的标题栏，标题栏上写着“星期六”，并配有三个星星的图案。标题栏下方是一段文字，内容是：“短短的一生，我们最终都会失去，你不妨大胆一些。爱一个人，攀一座山，追一个梦。”\n\n接下来是“早安读世界 今日简报”的标题，下面有“GOOD MORNING”的字样。\n\n图片的主要部分是每日资讯简报，日期是2024年3月23日，星期六，农历二月十四，早安！\n\n简报内容如下：\n\n1. 四川甘孜州官方：“3·15”雅江森林火灾原因初步查明：系施工动火作业引发，突遇极端大风造成扩散；\n2. 最高检表态：对未成年人实施的故意杀人、故意伤害，致人死亡等严重犯罪，符合核准追诉条件的，要依法追究刑事责任；\n3. 游族网络董事长林奇被毒杀一案，被告许旺一审被判无期，据悉其因管理经营矛盾，有预谋的在被害人食物中投毒致其死亡；\n4. 武汉地铁就“无臂男子免费乘地铁被要求出示残疾证”一事致歉，当时男子：没必要道歉，希望制度更人性化；\n5. 3月22日我国首个无人驾驶吨级电动垂直起降航空器获批合格证，据悉其载重可达400公斤，主要用于低空物流以及紧急物资运输与应急救援；\n6. 我国网民数量达到10.92亿人，互联网普及率达77.5%；\n7. 国家林草局：我国成为全球森林资源增长最多的国家，近20年来为全球贡献了约1/4的新增绿化面积；\n8. 河南郑州：2024年3月22日至4月30日，八区联合开展购车补贴活动，新能源车每台补贴不高于5000元，燃油车每台不高于3000元；\n9. 国台办披露：福建海警救起的两名海钓人员，其中一人为台军方现役人员，其编造虚假职业隐瞒身份还需进一步核实，另一人于3月22日送返金门；\n10. 因甘肃天水麻辣烫火出圈，清明小长假部分到天水的火车票已售罄，天水酒店预订量创近三年来单周预订量新高；\n11. 外媒：加拿大3月21日宣布，拟减少临时居留人数。今年1月加拿大称将在两年内减少留学签证数量，并限制毕业后申请工作签证；\n12. 外媒：以色列22日宣布，没收8平方公里约旦河西岸的巴勒斯坦土地归以色列所有；\n13. 外媒：美国一所医院成功完成一例将基因编辑猪肾移植到患者体内的手术，当地媒体称患者恢复良好，不日将出院；\n14. 外媒：美国得州边境墙附近，有上百名非法移民冲破铁丝网试图非法进入美国；\n15. 俄媒：3月22日，俄罗斯对乌克兰能源设施发动大规模无人机和导弹袭击。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 3383,
  "completion_tokens": 686,
  "total_tokens": 4069
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
 "id": "chatcmpl-5",
 "object": "chat.completion",
 "created": 1729673198,
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
  "prompt_tokens": 5955,
  "completion_tokens": 220,
  "total_tokens": 6175
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
 "id": "chatcmpl-7",
 "object": "chat.completion",
 "created": 1729673332,
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
  "prompt_tokens": 6193,
  "completion_tokens": 241,
  "total_tokens": 6434
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
 "id": "chatcmpl-8",
 "object": "chat.completion",
 "created": 1729673364,
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
  "prompt_tokens": 4697,
  "completion_tokens": 367,
  "total_tokens": 5064
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "<image>\n简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫。猫的身体大部分是白色的，背部和头部有黑色的斑点。猫的耳朵竖起，眼睛半闭，似乎在打盹。猫的胡须清晰可见，鼻子和嘴巴也清晰可见。猫的身体蜷缩在地面上，地面是灰色的，看起来像是水泥或沥青。\n\n从猫的姿态和表情来看，它处于一种放松和舒适的状态。猫的毛发看起来非常柔软，整体给人一种宁静的感觉。\n\n通过观察猫的特征，可以推断出这只猫可能是一只家猫，因为它的毛发整洁，而且看起来非常健康。家猫通常喜欢在温暖和安静的地方休息，这与图片中的环境相符。\n\n总结来说，这张图片展示了一只白色的猫，背部和头部有黑色斑点，它正躺在灰色的地面上打盹，表现出一种放松和舒适的状态。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1729673390, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=172, prompt_tokens=2359, total_tokens=2531, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
