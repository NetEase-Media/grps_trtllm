# 采样参数配置

通过请求参数可以指定采样参数，例如：

```json
{
  "model": "qwen2.5-instruct",
  "messages": [
    {
      "role": "user",
      "content": "你好，你是谁？"
    }
  ],
  "top_k": 50,
  "top_p": 1.0
}
```

同时也支持在部署模型服务时通过配置的方式设置模型默认的采样参数，见```inference.yml```文件中的```sampling```字段，这有助于设置一些
```OpenAI```
协议不支持的采样参数。例如：

```yaml
sampling:
  top_k: 50
  top_p: 1.0
```

注意，当请求参数中指定了采样参数时，会覆盖默认的采样参数。
具体支持的采样参数见[src/constants.h](../src/constants.h)文件中的```SamplingConfig```。