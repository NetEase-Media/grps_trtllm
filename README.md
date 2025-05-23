<div align="center">

# grps-trtllm

[GRPS](https://github.com/NetEase-Media/grps) + [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
实现纯```C++```版，相比```vllm serve```更优性能的```OpenAI LLM```服务，支持```Chat```、```Ai-agent```、```Multi-modal```
、多卡推理等。

![GRPS](https://img.shields.io/badge/GRPS-blue)
![TensorRT-LLM](https://img.shields.io/badge/TensorRT_LLM-green)
![Tokenizer-CPP](https://img.shields.io/badge/Tokenizer_CPP-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-green)
![Ai-Agent](https://img.shields.io/badge/Ai_Agent-blue)
![Multi-Modal](https://img.shields.io/badge/Multi_Modal-green)

[快速开始](#快速开始) | [模型列表](#模型列表) | [镜像列表](./docs/images.md) | [性能](./docs/performance.md) | [预告](./docs/next.md)

<div align="left">

## 演示

<img src="docs/gradio.gif" alt="gradio.gif">

## 说明

[grps](https://github.com/NetEase-Media/grps)接入[trtllm](https://github.com/NVIDIA/TensorRT-LLM)
实现更高性能的、支持```OpenAI```模式访问、支持```Ai-agent```以及多模态的```LLM```
服务：

* 通过纯```C++```实现完整```LLM```服务，包含```tokenizer```（支持`huggingface`, `sentencepiece`tokenizer）、```llm推理```
  、```vit```等部分。
* 通过```grps```的自定义```http```功能实现```OpenAI```接口协议，支持```chat```和```function call```模式。
* 支持扩展不同```LLM```的```prompt```构建风格以及生成结果的解析风格，以实现不同```LLM```的```chat```
  和```function call```模式，支持[llama-index](https://github.com/run-llama/llama_index)```ai-agent```。
* 通过集成```tensorrt```推理后端与```opencv```库，支持多模态```LLM```。
* 支持```inflight batching```、```multi-gpu```、```paged attention```、```kv-cache reuse```、```lookahead decoding```等
  ```TensorRT-LLM```推理加速技术。
* 相比较[triton tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend),
  不存在```triton_server <--> tokenizer_backend <--> trtllm_backend```之间的进程间通信，纯C++实现，性能有稳定的提升。

欢迎各位使用和提[issue](https://github.com/NetEase-Media/grps_trtllm/issues)
，欢迎提交[pr](https://github.com/NetEase-Media/grps_trtllm/pulls)支持新的模型，感谢star⭐️。也可以添加微信沟通：zhaocc1218。

## 更新历史

* 2025-05-14
    * 支持Lookahead解码优化。

* 2025-05-07
    * 支持Qwen3ForCausalLM。
    * grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12镜像更新，增加了对Qwen3ForCausalLM的支持。

* 2025-04-17
    * 支持InternVL3。

* 2025-03-25
    * 通过对图片hash的方式支持多模态的kv cache reuse。
    * grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12镜像更新，增加了cityhash的依赖。

* 2025-03-22
    * 支持MiniCPM-V-2_6。

<details close>
<summary>Previous News</summary>

* 2025-03-20
    * 支持gemma3-text。

* 2025-03-06
    * 支持QwQ-32B。

* 2025-03-04
    * 支持olmOCR。

* 2025-02-28
    * 支持InternVideo2.5。
    * grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12镜像增加grps-py功能。

* 2025-02-24
    * qwen2.5 llm_styler支持Qwen2.5-Math、Qwen2.5-Coder、Qwen2.5-1M。

* 2025-02-23
    * 支持QwQ与phi-4。

* 2025-02-21
    * 支持janus-pro图生文模型，暂不支持文生图。

* 2025-02-05
    * 支持deepseek-r1-distill系列文本模型。

* 2025-01-24
    * 支持phi3系列文本模型。

* 2025-01-08
    * 支持并测试tensorrt-llm [kv cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
      功能。可以显著提高类似“多轮对话”（prompt比较长并且重复比较多）场景的推理性能。暂不支持多模态模型。

* 2024-12-24
    * 更新trtllm依赖为0.16.0正式release代码。
    * 发布正式的trtllm0.16.0镜像：grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12。

* 2024-12-19
    * 增加internvl2.5的支持。

* 2024-12-17
    * 增加grps1.1.0_cuda12.5_cudnn9.2_trtllm0.16.0_py3.12_beta镜像（目前镜像较大，后续正式版会精简）。
    * 增加qwen2-vl的支持。

</details>

## 文档教程

* [快速开始](#快速开始)
* [模型列表](#模型列表)
* [采样参数配置](docs/sampling.md)
* [调度策略配置](docs/scheduler.md)
* [前缀缓存重用优化](docs/kv_reuse.md)
* [lookahead解码优化](docs/lookahead.md)
* [启动gradio服务](docs/gradio.md)
* [docker部署](docs/docker.md)
* [性能比较](docs/performance.md)
* [镜像列表](docs/images.md)
* [压测](docs/benchmark.md)
* [TODO](#todo)

## 模型列表

支持的文本LLM：

| supported model                                                          | llm_styler  | chat | function_call | doc                                                  |
|--------------------------------------------------------------------------|-------------|------|---------------|------------------------------------------------------|
| Qwen3                                                                    | qwen3       | ✅    | ✅             | [qwen3](docs%2Fqwen3.md)                             |
| DeepSeek-R1-Distill<br>TinyR1-32B-Preview                                | deepseek-r1 | ✅    | ❌             | [deepseek-r1-distill](docs%2Fdeepseek-r1-distill.md) |
| QwQ-32B<br>QwQ-32B-AWQ                                                   | qwq         | ✅    | ✅             | [qwq](docs%2Fqwq.md)                                 |
| QwQ-32B-Preview                                                          | qwq-preview | ✅    | ❌             | [qwq-preview](docs%2Fqwq-preview.md)                 |
| Qwen2.5-1M<br>Qwen2.5-Coder<br>Qwen2.5-Math<br>Qwen2.5                   | qwen2.5     | ✅    | ✅             | [qwen2.5](docs%2Fqwen2.5.md)                         |
| Qwen1.5-Chat<br>Qwen1.5-Moe-Chat<br>Qwen2-Instruct<br>Qwen2-Moe-Instruct | qwen        | ✅    | ✅             | [qwen2](docs%2Fqwen2.md)                             |
| chatglm3                                                                 | chatglm3    | ✅    | ✅             | [chatglm3](docs%2Fchatglm3.md)                       |                                                     
| glm4                                                                     | glm4        | ✅    | ✅             | [glm4](docs%2Fglm4.md)                               |
| internlm2_5-chat<br>internlm2-chat                                       | internlm2   | ✅    | ✅             | [internlm2.5](docs%2Finternlm2.5.md)                 |
| llama-3-instruct<br>llama-3.1-instruct                                   | llama3      | ✅    | ❌             | [llama3](docs%2Fllama3.md)                           |
| phi-4                                                                    | phi4        | ✅    | ❌             | [phi4](docs%2Fphi4.md)                               |
| Phi-3, Phi-3.5                                                           | phi3        | ✅    | ❌             | [phi3](docs%2Fphi3.md)                               |
| gemma-3(experimental)                                                    | gemma3      | ✅    | ❌             | [gemma-3](docs%2Fgemma3.md)                          |

支持的多模态LLM（少部分模型vit无法通过纯c++实现）：

| supported model                               | llm_styler          | vit             | vit_type | chat | function_call | doc                                          |
|-----------------------------------------------|---------------------|-----------------|----------|------|---------------|----------------------------------------------|
| InternVL3                                     | internvl3           | internvl2       | c++      | ✅    | ❌             | [internvl3](docs%2Finternvl3.md)             |
| MiniCPM-V-2_6                                 | minicpmv            | minicpmv        | py       | ✅    | ❌             | [minicpmv](docs%2Fminicpmv.md)               |
| Janus-Pro                                     | janus-pro           | janus-pro       | c++      | ✅    | ❌             | [janus-pro](docs%2Fjanus-pro.md)             |
| InternVideo2.5                                | intern-video2.5     | intern-video2.5 | py       | ✅    | ❌             | [intern-video2.5](docs%2Fintern-video2.5.md) |
| InternVL2_5<br>InternVL2_5-MPO                | internvl2.5         | internvl2       | c++      | ✅    | ❌             | [internvl2.5](docs%2Finternvl2.5.md)         |
| InternVL2-2B<br>InternVL2-8B<br>InternVL2-26B | internvl2-internlm2 | internvl2       | c++      | ✅    | ❌             | [internvl2](docs%2Finternvl2.md)             |
| InternVL2-1B                                  | internvl2-qwen2     | internvl2       | c++      | ✅    | ❌             | [internvl2](docs%2Finternvl2.md)             |
| InternVL2-4B                                  | internvl2-phi3      | internvl2       | c++      | ✅    | ❌             | [internvl2](docs%2Finternvl2.md)             |
| olmOCR                                        | qwen2vl             | qwen2vl         | c++      | ✅    | ❌             | [olm-ocr](docs%2Folm-ocr.md)                 |
| Qwen2-VL-Instruct                             | qwen2vl             | qwen2vl         | c++      | ✅    | ❌             | [qwen2vl](docs%2Fqwen2vl.md)                 |
| Qwen-VL-Chat<br>Qwen-VL                       | qwenvl              | qwenvl          | c++      | ✅    | ❌             | [qwenvl](docs%2Fqwenvl.md)                   |

## 工程结构

```text
|-- client                              # 客户端样例
|-- conf                                # 配置文件
|   |-- inference*.yml                  # 各类llm推理配置
|   |-- server.yml                      # 服务配置
|-- data                                # 数据文件
|-- docker                              # docker镜像构建
|-- docs                                # 文档
|-- processors                          # 远程处理器
|-- second_party                        # grps框架依赖
|-- src                                 # 自定义源码
|   |-- tensorrt                        # tensorrt推理后端
|   |-- vit                             # vit实现
|   |-- constants.cc/.h                 # 常量定义
|   |-- customized_inferer.cc/.h        # 自定义推理器
|   |-- llm_styler.cc/.h                # LLM风格定义，prompt构建，结果解析
|   |-- tokenizer.cc/.h                 # Tokenizer实现
|   |-- trtllm_model_instance.cc/.h     # TensorRT-LLM模型实例
|   |-- trtllm_model_state.cc/.h        # TensorRT-LLM模型状态
|   |-- utils.cc/.h                     # 工具
|   |-- main.cc                         # 本地单元测试
|-- third_party                         # 第三方依赖
|-- tools                               # 工具
|-- build.sh                            # 构建脚本
|-- CMakelists.txt                      # 工程构建文件
|-- .clang-format                       # 代码格式化配置文件
|-- .config                             # 工程配置文件，包含一些工程配置开关
```

## 快速开始

以qwen2.5-instruct为例。更多llm示例见[模型列表](#模型列表)，拉取代码与创建容器步骤相同。

### 拉取代码

```bash
git clone https://github.com/NetEase-Media/grps_trtllm.git
cd grps_trtllm
git submodule update --init --recursive
```

### 创建容器

使用```registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12```镜像。
这里挂载了当前目录用于构建工程并保留构建产物，挂载/tmp目录用于保存构建的trtllm引擎文件。参考```triton-trtllm```
设置共享内存大小，解除物理内存锁定限制，设置栈大小，配置参数
```--shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864```。

```bash
# 创建容器
docker run -itd --name grps_trtllm_dev --runtime=nvidia --network host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $(pwd):/grps_dev -v /tmp:/tmp -w /grps_dev \
registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12 bash
# 进入开发容器
docker exec -it grps_trtllm_dev bash
```

### 构建trtllm引擎

```bash
# 下载Qwen2.5-7B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct /tmp/Qwen2.5-7B-Instruct

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt
rm -rf /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/Qwen2.5-7B-Instruct \
--output_dir /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/Qwen2.5-7B-Instruct/trt_engines/
trtllm-build --checkpoint_dir /tmp/Qwen2.5-7B-Instruct/tllm_checkpoint/ \
--output_dir /tmp/Qwen2.5-7B-Instruct/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/Qwen2.5-7B-Instruct/ \
--engine_dir=/tmp/Qwen2.5-7B-Instruct/trt_engines/
# 回到工程根目录
cd ../../../../
```

### 修改inference.yml配置

修改llm对应的conf/inference*.yml中```inferer_args```相关参数。注意修改```tokenizer_path```
和```gpt_model_path```为新路径，更多核心参数见如下：

```yaml
models:
  - name: trtllm_model
    ...
    inferer_args:
      # llm style used to build prompt(chat or function call) and parse generated response for openai interface.
      # Support llm_style see README.md.
      llm_style: qwen2.5

      # tokenizer config.
      tokenizer_type: huggingface # can be `huggingface`, `sentencepiece`. Must be set.
      tokenizer_path: /tmp/Qwen2.5-7B-Instruct/ # path of tokenizer. Must be set.
      tokenizer_parallelism: 16 # tokenizers count for parallel tokenization. Will be set to 1 if not set.
      end_token_id: 151645 # end token id of tokenizer. Null if not set.
      pad_token_id: 151643 # pad token id of tokenizer. Null if not set.
      skip_special_tokens: # skip special tokens when decoding. Empty if not set.
        - 151643 # "<|endoftext|>"
        - 151644 # "<|im_start|>"
        - 151645 # "<|im_end|>"
        ...
      force_tokens_dict: # will be used to force map tokens to ids when encode and decode instead of using tokenizer. Empty if not set.
      #  - token: "<|endoftext|>"
      #    id: 151643
      prefix_tokens_id: # prefix tokens id will be added to the beginning of the input ids. Empty if not set.
      suffix_tokens_id: # suffix tokens id will be added to the end of the input ids. Empty if not set.

      # default sampling config, sampling param in request will overwrite these. Support sampling params see
      # @ref(src/constants.h - SamplingConfig)
      sampling:
        top_k: 50
        top_p: 1.0

      # trtllm config.
      gpt_model_type: inflight_fused_batching # must be `V1`(==`v1`) or `inflight_batching`(==`inflight_fused_batching`).
      gpt_model_path: /tmp/Qwen2.5-7B-Instruct/trt_engines/ # path of decoder model. Must be set.
      encoder_model_path: # path of encoder model. Null if not set.
      stop_words: # additional stop words. Empty if not set.
        - "<|im_start|>"
        - "<|im_end|>"
        - "<|endoftext|>"
      bad_words: # additional bad words. Empty if not set.
      batch_scheduler_policy: guaranteed_no_evict # must be `max_utilization` or `guaranteed_no_evict`.
      kv_cache_free_gpu_mem_fraction: 0.9 # will be set to 0.9 or `max_tokens_in_paged_kv_cache` if not set.
      exclude_input_in_output: true # will be set to false if not set.
```

### 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_qwen2.5.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

### 模拟请求

```bash
# curl命令非stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-instruct",
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
 "id": "chatcmpl-7",
 "object": "chat.completion",
 "created": 1726733862,
 "model": "qwen2.5-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "你好！我是Qwen，由阿里云开发的人工智能模型。我被设计用来提供信息、回答问题和进行各种对话任务。有什么我可以帮助你的吗？"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 34,
  "completion_tokens": 36,
  "total_tokens": 70
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-instruct",
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
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-8","object":"chat.completion.chunk","created":1726733878,"model":"qwen2.5-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"我是"},"logprobs":null,"finish_reason":null}]}
'

# 测试stop参数
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-instruct",
    "messages": [
      {
        "role": "user",
        "content": "重复1234#END#5678"
      }
    ],
    "stop": ["#END#"]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-2",
 "object": "chat.completion",
 "created": 1727433345,
 "model": "qwen2.5-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": "1234#END#"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 41,
  "completion_tokens": 7,
  "total_tokens": 48
 }
}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！我是Qwen，由阿里云开发的人工智能模型。我被设计用来提供信息、回答问题和进行各种对话任务。有什么我可以帮助你的吗？', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733895, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=36, prompt_tokens=34, total_tokens=70, completion_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "你好，你是谁？" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='你好', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-10', choices=[Choice(delta=ChoiceDelta(content='我是', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1726733914, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

# 输入32k长文本小说验证长文本的支持
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "上面这篇小说作者是谁？" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-11', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这篇小说的作者是弦三千。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733931, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=8, prompt_tokens=31615, total_tokens=31623, completion_tokens_details=None))
'

# 输入32k长文本小说进行总结
python3 client/openai_txt_cli.py 127.0.0.1:9997 ./data/32k_novel.txt "简述一下上面这篇小说的前几章内容。" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-12', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='以下是《拜托，只想干饭的北极熊超酷的！》前几章的主要内容概述：\n\n1. **第一章**：楚云霁意外穿越成了一只北极熊，他发现了一群科考队，并用鱼与他们交流。楚云霁在暴风雪中艰难生存，通过抓鱼和捕猎海豹来获取食物。\n\n2. **第二章**：楚云霁在暴风雪后继续捕猎，遇到了一只北极白狼。白狼似乎对楚云霁很友好，甚至带他去捕猎海豹。楚云霁吃了一顿饱饭后，与白狼一起回到白狼的洞穴休息。\n\n3. **第三章**：楚云霁在白狼的洞穴中休息，醒来后发现白狼已经离开。他继续捕猎，遇到了一群海豹，但海豹很快被一只成年北极熊吓跑。楚云霁在冰面上发现了一群生蚝，但白狼对生蚝不感兴趣，楚云霁只好自己吃了。\n\n4. **第四章**：楚云霁在捕猎时遇到了一只成年北极熊，成年北极熊似乎在挑衅他。楚云霁和白狼一起捕猎了一只驯鹿，分享了食物。直播设备记录下了这一幕，引起了观众的热议。\n\n5. **第五章**：楚云霁和白狼一起捕猎了一只驯鹿，分享了食物。楚云霁在捕猎时遇到了一只北极狐，但北极狐被北极熊吓跑。楚云霁还遇到了一只海鸟，海鸟试图抢食，但被白狼赶走。楚云霁和白狼一起处理了一只驯鹿，白狼还帮助楚云霁取下了鹿角。\n\n6. **第六章**：楚云霁和白狼一起捕猎，楚云霁在冰面上睡觉时被冰面漂走。醒来后，楚云霁发现白狼还在身边，感到非常高兴。他们一起捕猎了一只海象，但海象偷走了鱼竿。楚云霁和白狼一起追捕海象，最终成功捕获了海象。\n\n7. **第七章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n8. **第八章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n9. **第九章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n10. **第十章**：楚云霁和白狼一起捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉。\n\n11. **第十一章**：楚云霁在白狼的洞穴中发现了一个背包，背包里装满了各种食物和补给品。楚云霁和白狼一起分享了这些食物，包括罐头和海带。楚云霁还和白狼一起出去捕猎，但没有成功。\n\n12. **第十二章**：楚云霁和白狼一起出去捕猎，楚云霁发现了一根鱼竿。他们一起用鱼竿钓鱼，但鱼竿被海象带走。楚云霁和白狼一起追捕海象，最终成功捕获了海象。楚云霁和白狼一起分享了海象肉，并一起出去探索周围的环境。楚云霁还发现了一个背包，背包里装满了各种食物和补给品。楚云霁和白狼一起分享了这些食物，包括罐头和海带。楚云霁还和白狼一起出去捕猎，但没有成功。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1726733966, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=959, prompt_tokens=31621, total_tokens=32580, completion_tokens_details=None))
'

# openai_func_call.py进行function call模拟
python3 client/openai_func_call.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the weather like in Boston today? ...
Server response: thought: None, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result(59.0) ...
Final server response: The current temperature in Boston today is 59°F.
'

# openai_func_call2.py进行一次两个函数的function call模拟
python3 client/openai_func_call2.py 127.0.0.1:9997
# 返回如下：
: '
Query server with question: What's the postcode of Boston and what's the weather like in Boston today? ...
Server response: thought: None, call local function(get_postcode) with arguments: location=Boston, MA
Server response: thought: None, call local function(get_current_weather) with arguments: location=Boston, MA, unit=fahrenheit
Send the result back to the server with function result ...
Final server response: The postcode for Boston, MA is 02138. The current temperature in Boston today is 59.0°F.
'

# llama-index ai agent模拟
pip install llama_index llama_index.llms.openai_like
python3 client/llamaindex_ai_agent.py 127.0.0.1:9997
# 返回如下：
: '
Query: What is the weather in Boston today?
Added user message to memory: What is the weather in Boston today?
=== Calling Function ===
Calling function: get_weather with args: {"location":"Boston, MA","unit":"fahrenheit"}
Got output: 59.0
========================

Response: The current temperature in Boston is 59.0 degrees Fahrenheit.
'
```

### 指标观测

通过访问```http://ip:9997/``` 可以查看服务的指标信息。如下指标：

![metrics_0.png](docs/metrics_0.png)<br>
![metrics_1.png](docs/metrics_1.png)

### 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```

## TODO

* 当前基于```tensorrt-llm v0.10.0```之后的版本进行的实现，最新支持到```v0.16.0```
  （主分支），具体见仓库的分支信息。由于人力受限，一些bug不能及时在每一个分支修复，请尽量使用最新版本分支。
* 由于不同家族系的```LLM```的```chat```和```function call```
  的```prompt```构建以及结果解析风格不同，所以需要实现不同```LLM```家族的```styler```，见```src/llm_styler.cc/.h```
  ，用户可以自行扩展。拓展后需要修改```conf/inference.yml```的```llm_style```为对应的家族名。
  不同家族的```styler```持续开发中...。
* 不同多模态模型的```vit```实现不同，见```src/vit```，用户可以自行扩展。拓展后需要修改```conf/inference.yml```
  的```vit_type```为对应的类型名。
  不同多模态模型的```vit```持续开发中...。
* 书写用户自定义拓展```llm_styler```与```vit```开发文档。
