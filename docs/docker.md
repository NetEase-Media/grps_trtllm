# docker部署

```bash
# 更新conf/inference.yml软链接为具体的inference*.yml配置文件
rm -f conf/inference.yml
ln -s conf/inference_qwen2.5.yml conf/inference.yml
# 构建自定义工程docker镜像
docker build -t grps_trtllm_server:1.0.0 -f docker/Dockerfile .

# 使用上面构建好的镜像启动docker容器
# 注意挂载/tmp目录，因为构建的trtllm引擎文件在/tmp目录下
# 映射默认服务端口9997
# 注意如果使用多卡推理，需要使用mpi方式启动，--mpi_np参数为并行推理的GPU数量。
docker run -itd --runtime=nvidia --name="grps_trtllm_server" --shm-size=2g --ulimit memlock=-1 \
--ulimit stack=67108864 -v /tmp:/tmp -p 9997:9997 \
grps_trtllm_server:1.0.0 grpst start server.mar

# 使用docker logs可以跟踪服务日志
docker logs -f grps_trtllm_server

# 模拟请求
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

# 关闭容器
docker rm -f grps_trtllm_server
```