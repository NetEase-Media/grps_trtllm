# grps-trtllm

Qwen2.5-Coder模型的部署示例，以Qwen2.5-Coder-7B-Instruct为例。

## 开发环境

见[快速开始](../README.md#快速开始)的拉取代码和创建容器部分。

## 构建trtllm引擎

```bash
# 下载Qwen2.5-Coder-7B-Instruct模型
apt update && apt install git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct /tmp/Qwen2.5-Coder-7B-Instruct

# 进入TensorRT-LLM/examples/qwen目录，参考README进行构建trtllm引擎。
cd third_party/TensorRT-LLM/examples/qwen
# 转换ckpt
rm -rf /tmp/Qwen2.5-Coder-7B-Instruct/tllm_checkpoint/
python3 convert_checkpoint.py --model_dir /tmp/Qwen2.5-Coder-7B-Instruct \
--output_dir /tmp/Qwen2.5-Coder-7B-Instruct/tllm_checkpoint/ --dtype bfloat16 --load_model_on_cpu
# 构建引擎
rm -rf /tmp/Qwen2.5-Coder-7B-Instruct/trt_engines/
trtllm-build --checkpoint_dir /tmp/Qwen2.5-Coder-7B-Instruct/tllm_checkpoint/ \
--output_dir /tmp/Qwen2.5-Coder-7B-Instruct/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 16 --paged_kv_cache enable --use_paged_context_fmha enable \
--max_input_len 32256 --max_seq_len 32768 --max_num_tokens 32256
# 运行测试
python3 ../run.py --input_text "你好，你是谁？" --max_output_len=50 \
--tokenizer_dir /tmp/Qwen2.5-Coder-7B-Instruct/ \
--engine_dir=/tmp/Qwen2.5-Coder-7B-Instruct/trt_engines/
# 回到工程根目录
cd ../../../../
```

## 构建与部署

```bash
# 构建
grpst archive .

# 部署，
# 通过--inference_conf参数指定模型对应的inference.yml配置文件启动服务。
# 如需修改服务端口，并发限制等，可以修改conf/server.yml文件，然后启动时指定--server_conf参数指定新的server.yml文件。
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
grpst start ./server.mar --inference_conf=conf/inference_qwen2.5-coder.yml

# 查看服务状态
grpst ps
# 如下输出
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH         
9997                my_grps             65322               /home/appops/.grps/my_grps
```

## 模拟请求

```bash
# curl命令非stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-instruct",
    "messages": [
      {
        "role": "user",
        "content": "请给一个快速排序的代码实例："
      }
    ]
  }'
# 返回如下：
: '
{
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "created": 1727683829,
 "model": "qwen2.5-coder-instruct",
 "system_fingerprint": "grps-trtllm-server",
 "choices": [
  {
   "index": 0,
   "message": {
    "role": "assistant",
    "content": " 当然！以下是一个用Python实现的快速排序算法的代码实例：\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[len(arr) // 2]\n        left = [x for x in arr if x < pivot]\n        middle = [x for x in arr if x == pivot]\n        right = [x for x in arr if x > pivot]\n        return quicksort(left) + middle + quicksort(right)\n\n# 示例用法\nif __name__ == \"__main__\":\n    example_array = [3, 6, 8, 10, 1, 2, 1]\n    print(\"原始数组:\", example_array)\n    sorted_array = quicksort(example_array)\n    print(\"排序后的数组:\", sorted_array)\n```\n\n这个代码定义了一个`quicksort`函数，它使用递归的方式对数组进行排序。具体步骤如下：\n\n1. 选择一个基准值（pivot），这里选择数组的中间元素。\n2. 将数组分成三个部分：小于基准值的元素、等于基准值的元素和大于基准值的元素。\n3. 递归地对小于和大于基准值的部分进行排序。\n4. 将排序后的部分和等于基准值的部分合并，得到最终的排序结果。\n\n你可以运行这个代码来查看快速排序的效果。"
   },
   "logprobs": null,
   "finish_reason": "stop"
  }
 ],
 "usage": {
  "prompt_tokens": 38,
  "completion_tokens": 290,
  "total_tokens": 328
 }
}
'

# curl命令stream请求
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-instruct",
    "messages": [
      {
        "role": "user",
        "content": "请给一个快速排序的代码实例："
      }
    ],
    "stream": true
  }'
# 返回如下：
: '
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1727683879,"model":"qwen2.5-coder-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"role":"assistant","content":" 当"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1727683879,"model":"qwen2.5-coder-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"然"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1727683879,"model":"qwen2.5-coder-instruct","system_fingerprint":"grps-trtllm-server","choices":[{"index":0,"delta":{"content":"！"},"logprobs":null,"finish_reason":null}]}
'

# openai_cli.py 非stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "请给一个快速排序的代码实例：" false
# 返回如下：
: '
ChatCompletion(id='chatcmpl-3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' 当然！以下是一个用Python实现的快速排序算法的代码实例：\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[len(arr) // 2]\n        left = [x for x in arr if x < pivot]\n        middle = [x for x in arr if x == pivot]\n        right = [x for x in arr if x > pivot]\n        return quicksort(left) + middle + quicksort(right)\n\n# 示例用法\nif __name__ == "__main__":\n    example_array = [3, 6, 8, 10, 1, 2, 1]\n    print("原始数组:", example_array)\n    sorted_array = quicksort(example_array)\n    print("排序后的数组:", sorted_array)\n```\n\n这个代码定义了一个`quicksort`函数，它使用递归的方式对数组进行排序。具体步骤如下：\n\n1. 选择一个基准值（pivot），这里选择数组的中间元素。\n2. 将数组分成三个部分：小于基准值的元素、等于基准值的元素和大于基准值的元素。\n3. 递归地对小于和大于基准值的部分进行排序。\n4. 将排序后的部分和等于基准值的部分合并，得到最终的排序结果。\n\n你可以运行这个代码来查看快速排序的效果。', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1727684015, model='', object='chat.completion', service_tier=None, system_fingerprint='grps-trtllm-server', usage=CompletionUsage(completion_tokens=290, prompt_tokens=38, total_tokens=328, completion_tokens_details=None))
'

# openai_cli.py stream请求
python3 client/openai_cli.py 127.0.0.1:9997 "请给一个快速排序的代码实例：" true
# 返回如下：
: '
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content=' 当', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1727684064, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content='然', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1727684064, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
ChatCompletionChunk(id='chatcmpl-4', choices=[Choice(delta=ChoiceDelta(content='！', function_call=None, refusal=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1727684064, model='', object='chat.completion.chunk', service_tier=None, system_fingerprint='grps-trtllm-server', usage=None)
'

```

## 关闭服务

```bash
# 关闭服务
grpst stop my_grps
```