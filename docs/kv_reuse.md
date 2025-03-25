# 前缀cache重用

通过`tensorrt-llm` [kv cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
功能实现prompt前缀cache重复利用功能。通过共享和重复使用以相同prompt开头的请求的
`kv cache page`，可以大大降低首字延迟，即生成第一个输出token所花费的时间。对于多轮对话和相同系统提示等场景性能提升很大。

## 如何打开该功能？

在默认样例中都是默认打开该功能的，主要涉及到如下几点：

### 构建参数

`trtllm-build`构建时涉及到如下参数：

* 需要打开`--use_paged_context_fmha enable`参数。
* `--tokens_per_block`参数可以调整`kv cache reuse block`
  的大小，详细介绍见[Situations that can prevent kv cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html#situations-that-can-prevent-kv-cache-reuse)。

### 服务部署参数

`inference_*.yml`涉及到如下参数：

* 需要打开`enable_kv_cache_reuse: true`参数。
* `kv_cache_host_memory_bytes`参数配置后可以允许`kv cache`从`gpu`卸载到`host`
  内存，通过该参数可以配置用于主机内存用于缓存的大小。详细介绍见[Offloading to host
  memor](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html#offloading-to-host-memory)。

## 其他

* `kv cache reuse`遵循`lru`策略，当缓存不够时会清除最久未使用的缓存。
* 多模态大模型通过将图片`hash`成虚拟`token`实现了对图片`kv cache`的重用。