# MiniCPM-V-2_6

MiniCPM-V-2_6多模态LLM模型的部署示例。vit部分无法完整转为tensorrt，目前vit使用grps
py自定义工程实现，grps-trtllm通过rpc进行远程调用计算embeddings。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载MiniCPM-V-2_6模型
apt update && apt install git-lfs
git lfs install
git clone git@hf.co:openbmb/MiniCPM-V-2_6 /tmp/MiniCPM-V-2_6

# 拆分llm
rm -rf /tmp/MiniCPM-V-2_6/llm
python3 tools/minicpmv/split_llm.py --model_dir /tmp/MiniCPM-V-2_6 --output_dir /tmp/MiniCPM-V-2_6/llm

# 转换ckpt，这里使用了int8 weight only量化减少显存占用
rm -rf /tmp/MiniCPM-V-2_6/tllm_checkpoint/
python3 third_party/TensorRT-LLM/examples/qwen/convert_checkpoint.py --model_dir /tmp/MiniCPM-V-2_6/llm \
--use_weight_only --weight_only_precision int8 \
--output_dir /tmp/MiniCPM-V-2_6/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu

# 构建llm引擎，根据具体显存情况可以配置不同。
# 这里设置支持最大batch_size为8，即支持8个并发同时推理，超过8个排队处理。
# 设置每个请求最多输入64个图片patch。
# 即：max_multimodal_len=8（max_batch_size） * 64（图片最多产生patch个数） * 64（每个patch对应64个token） = 32768
# 设置max_input_len为30k，max_seq_len为32k（即默认最大输出为2k）。
rm -rf /tmp/MiniCPM-V-2_6/trt_engines/
trtllm-build --checkpoint_dir /tmp/MiniCPM-V-2_6/tllm_checkpoint/ \
--output_dir /tmp/MiniCPM-V-2_6/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 8 --paged_kv_cache enable \
--max_input_len 30720 --max_seq_len 32768 --max_num_tokens 32768 --max_prompt_embedding_table_size=32768
```

## 构建与部署

```bash
# 进入vit服务目录
cd processors/minicpmv/
# 构建vit grps服务
grpst archive .
# 启动视频grps服务
grpst start ./server.mar --name minicpmv-processor
# 返回grps-trtllm根目录
cd ../../

# 构建grps-trtllm服务
grpst archive .
# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_minicpmv.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 配置说明

vit服务[inference.yml](../processors/minicpmv/conf/inference.yml) 部分关键配置：

```yaml
device: cuda:0 # 使用的GPU设备。
inferer_path: /tmp/MiniCPM-V-2_6 # llm模型路径，用于加载视频vit模型。
inferer_args:
  dtype: bfloat16  # 模型输入输出数据类型，可以是`float16`, `bfloat16`。
  vision_batch_size: 6 # vit一次处理的最大batch size，超过会进行顺序处理防止OOM。
converter_args:
  shm_size: 536870912 # (512M), 共享内存大小，用于图像embeddings传输。
  shm_cnt: 2 # 共享内存申请个数。
  shm_name_prefix: "/minicpmv-sm" # 共享内存名称前缀。
```

vit服务[server.yml](../processors/minicpmv/conf/server.yml) 部分关键配置：

```yaml
max_concurrency: 2 # 最大并发数，即最大同时处理请求数。
```

grps-trtllm服务[inference_minicpmv.yml](../conf/inference_minicpmv.yml) 部分关键配置：

```yaml
vit_processor_args:
  host: 0.0.0.0 # 远程vit服务地址。
  port: 7081 # 远程vit服务端口。
  timeout_ms: 3000 # 远程vit服务超时时间
  dtype: bfloat16 # vit模型输入输出数据类型，可以是`float16`, `bfloat16`。
  shm_size: 536870912 # (512M), 共享内存大小，用于图像embeddings传输。
  image_token_id: 128244 # 表示image token的id
  img_begin_token_id: 151666 # 作为trtllm ptuning image token的起始id，等于llm的vocabulary size。
kv_cache_free_gpu_mem_fraction: 0.6 # llm模型的kv cache占用空闲显存的比例，过高可能会导致vit推理OOM。
```

* 为了避免embeddings进程间传输占用大量耗时，这里使用了共享内存技术，可以通过`shm_size`和`shm_cnt`配置共享内存大小和个数。
  `shm_cnt`限制共享内存申请个数，并发请求会通过加锁方式使用这些共享内存。
* 由于显存限制，这里限制了vit最大并发数为2，可以根据显存情况调整`max_concurrency`。
* vit可以单独放到其他显卡上，可以通过vit服务的`device`配置指定显卡，这里使用了`cuda:0`。
* 当显存不足导致OOM时可以调小上述`vision_batch_size`、`max_concurrency`、`kv_cache_free_gpu_mem_fraction`等参数。

## 模拟请求

```bash
# 测试一张网络图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "详细描述一下这张图片的内容。"
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
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "created": 1742615313,
 "model": "MiniCPM-V",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这是一张2024年3月23日的早安读世界每日资讯简报。以下是主要内容：\n\n1. 四川甘孜州官方宣布，“3·5”雅江森林火灾原因初步查明，系施工动火作业引发；突遇极端大风造成扩散。\n\n2. 最高检表态：对未成年人实施的故意杀人、故意伤害致人死亡等严重犯罪行为，符合核准追诉条件时要依法追究刑事责任，并希望制度更人性化。\n\n3. 游族网络董事长林奇被毒杀一案被告许某一审被判无期徒刑，因其管理经营矛盾有预谋在被害人食物中投毒致其死亡。\n\n4. 武汉地铁就“无臂男子免费乘地铁要求出示残疾证一事道歉”，当时男子表示没必要道歉并希望能使制度更加人性化。\n\n5. 我国首个无人驾驶电动垂直起降航空器获批合格证书，载重可达400公斤用于低空物流以及紧急物资运输与应急救援。\n\n6. 网民数量达到约9亿人，互联网普及率高达78%以上（可能为误译或数据错误）。\n\n7. 中国成为全球森林资源增长最快的国家之一，在近二十年来贡献了约四分之一的新增绿化面积。\n\n8. 河南郑州将于2月至4月底开展购车补贴活动，新能源车每台可享受最高不超过五千元的优惠，燃油车辆则不高于三千元。\n\n9. 台湾海警抓获两名涉嫌非法移民人员，其中一人身份存疑需进一步核实，另一人在送返金门前已被遣返回大陆。\n\n10. 天水麻辣烫因清明小长假部分火车票已售罄，天水酒店预订量创新高。\n\n11. 加拿大拟减少临时居留人数，将在两年内限制留学签证申请和工作签证发放。\n\n12. 财政部以色列称将没收巴勒斯坦土地归以色列所有。\n\n13. 美国成功完成基因编辑猪肾移植到患者体内的手术，当地媒体认为恢复良好即将出院。\n\n14. 得克萨斯边境附近数百名非法移民试图非法进入美国。\n\n这些内容涵盖了国内外的重大事件和社会热点话题，提供了丰富的信息供读者了解当天的重要新闻动态。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 626,
  "completion_tokens": 484,
  "total_tokens": 1110
 }
}
'

# 测试输入两张图片
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "描述一下两张图片的不同。"
          },
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
 "created": 1742615381,
 "model": "MiniCPM-V",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "这两张照片中的动物不同，它们分别是大熊猫和红熊猫。\n\n以下是两者的详细区别：\n\n### 大熊猫（图一）\n- **毛色**：主要为黑白相间。头部、耳朵、四肢等部位是黑色的；身体其他部分则是白色的。\n- **面部特征**：\n  - 黑眼圈明显且呈圆形。\n  - 面部有明显的黑斑围绕眼睛区域形成“八字胡”形状。\n  \n#### 环境背景**\n- 背景中有绿色植物和其他植被，显示出自然栖息地的特点。\n\n---\n\n### 红熊猫 （图二）\n- **毛色**：全身覆盖着棕色或红色的皮毛，并带有白色条纹装饰在脸部周围及颈部下方。\n- **面部特征**：\n  - 没有显著的眼圈颜色区分，但鼻子周围的毛发较浅。\n  - 嘴巴附近有一道白线从鼻尖延伸到嘴角两侧。\n \n ####环境背景 **\n- 后方有一个模糊的人工绿植背景板以及木质结构平台，显示可能是在动物园或其他人工环境中拍摄的照片。\n\n通过这些细节可以清楚地区分出这两种不同的动物及其生活环境特点。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 1222,
  "completion_tokens": 250,
  "total_tokens": 1472
 }
}
'

# 测试多轮对话
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "描述一下两张图片的不同。"
          },
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
          }
        ]
      },
      {
        "role": "assistant",
        "content": "这两张照片中的动物不同，它们分别是大熊猫和红熊猫。\n\n以下是两者的详细区别：\n\n### 大熊猫（图一）\n- **毛色**：主要为黑白相间。头部、耳朵、四肢等部位是黑色的；身体其他部分则是白色的。\n- **面部特征**：\n  - 黑眼圈明显且呈圆形。\n  - 面部有明显的黑斑围绕眼睛区域形成“八字胡”形状。\n  \n#### 环境背景**\n- 背景中有绿色植物和其他植被，显示出自然栖息地的特点。\n\n---\n\n### 红熊猫 （图二）\n- **毛色**：全身覆盖着棕色或红色的皮毛，并带有白色条纹装饰在脸部周围及颈部下方。\n- **面部特征**：\n  - 没有显著的眼圈颜色区分，但鼻子周围的毛发较浅。\n  - 嘴巴附近有一道白线从鼻尖延伸到嘴角两侧。\n \n ####环境背景 **\n- 后方有一个模糊的人工绿植背景板以及木质结构平台，显示可能是在动物园或其他人工环境中拍摄的照片。\n\n通过这些细节可以清楚地区分出这两种不同的动物及其生活环境特点。"
      },
      {
        "role": "user",
        "content": "这两个动物分别是什么品种？"
      }
    ],
    "max_tokens": 256
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-3",
 "object": "chat.completion",
 "created": 1742615408,
 "model": "MiniCPM-V",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "第一个图像中展示的是大熊猫 (Ailuropoda melanoleuca)，而第二个图像展示了红熊猫(Ailurus fulgens) 。两者都是中国特有的珍稀物种，在各自的生态系统中都扮演重要角色并受到保护关注。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 1488,
  "completion_tokens": 50,
  "total_tokens": 1538
 }
}
'

# 通过openai api进行请求
python3 client/openai_cli.py 0.0.0.0:9997 "简述一下这张图片的内容。" false "https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg"
# 返回如下：
: '
ChatCompletion(id='chatcmpl-4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这幅图展示了一只家猫，它看起来正在休息或睡觉中被捕捉到的瞬间。它的毛色主要是白色，并带有棕色和黑色斑纹在头部、背部以及尾巴上可见。这只猫咪的眼睛是闭着的，在放松的状态下微微皱起眉毛的表情暗示了这一点。\n\n背景是一个粗糙不平的地表纹理表面，可能是混凝土或者石头铺成的人行道的一部分。这种质地与柔软蓬松的猫咪形成鲜明对比。\n没有明显的特征可以立即揭示出地点的具体位置；然而，环境表明这是一个户外场景可能是在城市地区由于地面上有铺设材料的存在而推测出来的地方。照片中的光线柔和且散射，缺乏强烈的阴影通常表示阴天或多云天气条件下的自然光照明情况。整体氛围宁静安详，焦点完全放在享受片刻休憩时间的小动物身上。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1742615423, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=173, prompt_tokens=625, total_tokens=798, completion_tokens_details=None, prompt_tokens_details=None))
'

# 通过base64 img url方式进行请求
python3 client/base64_img_cli.py 0.0.0.0:9997 "简述一下这张图片的内容。" false ./data/image1.jpg
# 返回如下：
: '
ChatCompletion(id='chatcmpl-5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这幅图展示了一只红熊猫，它是一种濒危物种的熊科动物，在中国西南部被发现，并因其可爱的外表而备受喜爱和关注保护工作。它的毛色是独特的红色、白色和黑色混合而成；脸部有明显的黑眼圈，看起来像戴了面具一样。这种生物栖息在竹林中，主要以竹子为食。\n根据提供的背景信息来看：\n1）这只小猫可能是一只家养宠物或流浪的小猫（野性），因为没有看到任何围栏或其他约束物来表明它是野生环境中的捕猎者。\n\n2) 这张照片似乎是在一个动物园拍摄的，考虑到木制结构的存在以及自然环境中有人工元素的情况——这些通常是为了给野生动物提供舒适的生活空间而在受控环境下建造的人造设施的一部分。\n\n3) 红熊猫不是狮子狗的一种类型; 它们属于不同的分类群：一个是哺乳类动物家族之一部分, 而另一个则与犬科有关联.\n\n4) 由于其独特性和可爱外观，人们可能会认为这是“最萌”的动物品种之一.\n \n5) 图片本身并没有直接显示关于这个个体年龄的信息(幼年/成年的)，但可以推断出这是一个年轻且健康的个体基于皮毛的状态和其他身体特征如大小等观察到的因素\n\n6) 关于是否应该将此图像用于教育目的的问题: 是的话，则应确保准确描述并尊重该物种及其所处生态系统的知识传播方式适当使用此类内容对于提高人们对这一重要保育对象的认识至关重要并且有助于促进公众对它们的关注和支持他们的生存需求及保护努力方面具有重要意义 \n\n7最后一点就是如果要分享或者发布这类包含真实世界动植物的照片时必须遵守相关法律法规特别是涉及版权问题需要获得许可才能进行商业用途否则会侵犯他人权益造成法律纠纷等问题因此建议谨慎处理这些问题以便更好地利用资源同时避免潜在风险发生以上分析仅限于现有视觉证据不包括其他外部因素影响判断结果如有更多细节可进一步探讨讨论补充说明!', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1742616274, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=420, prompt_tokens=361, total_tokens=781, completion_tokens_details=None, prompt_tokens_details=None))
Latency: 8506.446599960327 ms
’
```

## 开启gradio服务

```bash
# 安装gradio
pip install -r tools/gradio/requirements.txt

# 启动多模态聊天界面，使用minicpmv多模态模型，0.0.0.0:9997表示llm后端服务地址
python3 tools/gradio/llm_app.py minicpmv 0.0.0.0:9997
```

## 关闭服务

```bash
# 关闭服务trtllm服务
grpst stop my_grps
# 关闭视频vit服务
grpst stop minicpmv-processor
```
