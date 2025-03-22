# Qwen2-VL-Instruct

Qwen2-VL-Instruct模型的部署示例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

### Qwen2-VL-2B-Instruct

```bash
# 下载Qwen2-VL-2B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct /tmp/Qwen2-VL-2B-Instruct

# 安装依赖
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils pycuda==2024.1.2

# 转换ckpt
rm -rf /tmp/Qwen2-VL-2B-Instruct/tllm_checkpoint/
python3 ./third_party/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
--model_dir /tmp/Qwen2-VL-2B-Instruct/ \
--output_dir /tmp/Qwen2-VL-2B-Instruct/tllm_checkpoint/ \
--dtype bfloat16 --load_model_on_cpu

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为4，即支持4个并发同时推理，超过4个排队处理。
rm -rf /tmp/Qwen2-VL-2B-Instruct/trt_engines
trtllm-build --checkpoint_dir /tmp/Qwen2-VL-2B-Instruct//tllm_checkpoint/ \
--output_dir /tmp/Qwen2-VL-2B-Instruct/trt_engines \
--gemm_plugin=bfloat16 \
--gpt_attention_plugin=bfloat16 \
--max_batch_size=4 \
--max_input_len=30720 --max_seq_len=32768 \
--max_num_tokens 32768 \
--max_prompt_embedding_table_size=28416

# 构建vit引擎
python3 tools/qwen2vl/build_vit_engine.py --pretrainedModelPath /tmp/Qwen2-VL-2B-Instruct/ \
--onnxFile /tmp/Qwen2-VL-2B-Instruct/vision_encoder_bfp16.onnx \
--mropeOnnxFile /tmp/Qwen2-VL-2B-Instruct/mrope_only_bfp16.onnx \
--dtype bfloat16 --cvtOnnx
python3 tools/qwen2vl/build_vit_engine.py --pretrainedModelPath /tmp/Qwen2-VL-2B-Instruct/ \
--onnxFile /tmp/Qwen2-VL-2B-Instruct/vision_encoder_bfp16.onnx \
--trtFile /tmp/Qwen2-VL-2B-Instruct/vision_encoder_bfp16.trt \
--mropeOnnxFile /tmp/Qwen2-VL-2B-Instruct/mrope_only_bfp16.onnx \
--mropeTrtFile /tmp/Qwen2-VL-2B-Instruct/mrope_only_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 1 --maxBS 8 --cvtTrt
```

### Qwen2-VL-7B-Instruct

```bash
# 下载Qwen2-VL-7B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct /tmp/Qwen2-VL-7B-Instruct

# 安装依赖
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils pycuda==2024.1.2

# 转换ckpt，使用int8 weight only量化减少显存占用
rm -rf /tmp/Qwen2-VL-7B-Instruct/tllm_checkpoint/
python3 ./third_party/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
--model_dir /tmp/Qwen2-VL-7B-Instruct/ \
--output_dir /tmp/Qwen2-VL-7B-Instruct/tllm_checkpoint/ \
--use_weight_only --weight_only_precision int8 \
--dtype bfloat16 --load_model_on_cpu

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为4，即支持4个并发同时推理，超过4个排队处理。
rm -rf /tmp/Qwen2-VL-7B-Instruct/trt_engines
trtllm-build --checkpoint_dir /tmp/Qwen2-VL-7B-Instruct//tllm_checkpoint/ \
--output_dir /tmp/Qwen2-VL-7B-Instruct/trt_engines \
--gemm_plugin=bfloat16 \
--gpt_attention_plugin=bfloat16 \
--max_batch_size=4 \
--max_input_len=32768 --max_seq_len=36960 \
--max_num_tokens 32768 \
--max_prompt_embedding_table_size=28416

# 构建vit引擎
python3 tools/qwen2vl/build_vit_engine.py --pretrainedModelPath /tmp/Qwen2-VL-7B-Instruct/ \
--onnxFile /tmp/Qwen2-VL-7B-Instruct/vision_encoder_bfp16.onnx \
--mropeOnnxFile /tmp/Qwen2-VL-7B-Instruct/mrope_only_bfp16.onnx \
--dtype bfloat16 --cvtOnnx
python3 tools/qwen2vl/build_vit_engine.py --pretrainedModelPath /tmp/Qwen2-VL-7B-Instruct/ \
--onnxFile /tmp/Qwen2-VL-7B-Instruct/vision_encoder_bfp16.onnx \
--trtFile /tmp/Qwen2-VL-7B-Instruct/vision_encoder_bfp16.trt \
--mropeOnnxFile /tmp/Qwen2-VL-7B-Instruct/mrope_only_bfp16.onnx \
--mropeTrtFile /tmp/Qwen2-VL-7B-Instruct/mrope_only_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 1 --maxBS 8 --cvtTrt
```

## 构建与部署

注意不同尺寸的vocabulary size可能不同，即```inference.yml```中```img_begin_token_id```配置不同，需要根据具体模型配置。

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
# grpst start ./server.mar --inference_conf=conf/inference_qwen2-vl-2B.yml
grpst start ./server.mar --inference_conf=conf/inference_qwen2-vl-7B.yml

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
    "model": "Qwen2-VL-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            }
          },
          {
            "type": "text",
            "text": "描述一下这张图片"
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1734432252,
 "model": "Qwen2-VL-Instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片展示了一位女士和她的狗在海滩上玩耍的场景。女士穿着格子衬衫，坐在沙滩上，她的狗是一只黄色的拉布拉多犬，戴着项圈和牵引绳。狗伸出前爪，似乎在和女士互动，可能是在玩耍或接受训练。背景是广阔的海洋，海浪轻轻拍打着海岸，天空呈现出柔和的光线，可能是日出或日落时分。整个场景显得非常宁静和愉快。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 1301,
  "completion_tokens": 97,
  "total_tokens": 1398
 }
}
'

curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-VL-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://pic3.zhimg.com/v2-5904ffb96cf191bde40b91e4b7804d92_r.jpg"
            }
          },
          {
            "type": "text",
            "text": "解析一下图片中的文字"
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1734432279,
 "model": "Qwen2-VL-Instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这张图片是一份早安简报，包含了15条新闻资讯。以下是每条新闻的简要解析：\n\n1. 四川甘孜州官方：“3·15”雅江森林火灾原因初步查明：系施工动火作业引发，突遇极端大风造成扩散。\n2. 最高检表态：对未成年人实施的故意杀人、故意伤害，致人死亡等严重犯罪，符合核准追诉条件的，要依法追究刑事责任。\n3. 游族网络董事长林奇被毒杀一案，被告许某一审被判无期，据悉其因管理经营矛盾，有预谋的在被害人食物中投毒致其死亡。\n4. 武汉地铁就“无臂男子免费乘地铁被要求出示残疾证”一事致歉，当时男子：没必要道歉，希望制度更人性化。\n5. 3月22日我国首个无人驾驶吨级电动垂直起降航空器获批合格证，据悉其载重可达400公斤，主要用于低空物流以及紧急物资运输与应急救援。\n6. 我国网民数量达到10.92亿人，互联网普及率达77.5%。\n7. 国家林草局：我国成为全球森林资源增长最快的国家，近20年来为全球贡献了约1/4的新增绿化面积。\n8. 河南郑州：2024年3月22日至4月30日，八区联合开展购车补贴活动，新能源车每台补贴不高于5000元，燃油车每台不高于3000元。\n9. 国台办披露：福建海警救起的两名海钓人员，其中一人为台军方现役人员，其编造虚假职业隐瞒身份还需进一步核实，另一人于3月22日送返金门。\n10. 因甘肃天水麻辣烫火出圈，清明小长假部分到天水的火车票已售罄，天水酒店预订量创近三年来单周预订量新高。\n11. 外媒：加拿大3月21日宣布，拟减少临时居留人数。今年1月加拿大称将在两年内减少留学签证数量，并限制毕业后申请工作签证。\n12. 外媒：以色列22日宣布，没收8平方公里约旦河西岸的巴勒斯坦土地归以色列所有。\n13. 外媒：美国一所医院成功完成一例将基因编辑猪肾移植到患者体内的手术，当地媒体称患者恢复良好，不日将出院。\n14. 外媒：美国得州边境墙附近，有上百名非法移民冲破铁丝网试图非法进入美国。\n15. 俄媒：3月22日，俄罗斯对乌克兰能源设施发动大规模无人机和导弹袭击。\n\n这些新闻涵盖了国内外的多个领域，包括火灾原因、未成年人犯罪、网络诈骗、地铁乘车规定、无人驾驶航空器、互联网普及、森林资源增长、购车补贴、海钓人员救援、加拿大签证政策、以色列土地归属、基因编辑手术、非法移民和无人机袭击等。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 1326,
  "completion_tokens": 658,
  "total_tokens": 1984
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-VL-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg"
            }
          },
          {
            "type": "text",
            "text": "简述一下两张图片的不同。"
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
 "created": 1734432350,
 "model": "Qwen2-VL-Instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这两个图片的不同点是：第一张图片是一只大熊猫，第二张图片是一只小熊猫。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 2602,
  "completion_tokens": 21,
  "total_tokens": 2623
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一只猫躺在地上休息。猫的毛色主要是白色，带有黑色和棕色的斑点。它的眼睛闭着，看起来很放松。背景是灰色的地面，可能是水泥地面。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1734432367, model='Qwen2-VL-Instruct', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=44, prompt_tokens=1324, total_tokens=1368, completion_tokens_details=None, prompt_tokens_details=None))
'
```

## 开启gradio服务

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动多模态聊天界面，使用qwen2vl多模态模型，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py qwen2vl 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```
