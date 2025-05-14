# Lookahead解码优化

## 介绍

Lookahead解码优化主要通过“并行推理多个token” + “n-gram缓存” +
“一次性验证”来实现解码阶段一次推理多个token，充分利用GPU的并行计算能力，提升LLM解码阶段的吞吐量。具体介绍可以参考：

* [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
* [TensorRT-LLM lookahead](../third_party/TensorRT-LLM/examples/lookahead/README.md)

<img src="lookahead-decoding.gif" alt="glookahead-decoding.gif" height="400"/>

## 如何打开该功能？

### 构建参数

`trtllm-build`构建时涉及到如下参数：

* 需要打开`--speculative_decoding_mode lookahead_decoding`参数。
* 通过`--max_draft_len`参数配置`lookahead-decoding`的最大草稿长度，该参数通过如下公式计算而来：

```python
def max_draft_len(windows_size, ngram_size, verification_set_size):
    return (0 if (ngram_size == 1) else ngram_size - 2)
    + (windows_size - 1 + verification_set_size) * (ngram_size - 1)
```

### 服务部署参数

`inference_*.yml`涉及到如下参数：

```
lookahead_decoding:
  window_size: 4
  ngram_size: 2
  verification_set_size: 4
```

## 压测记录

### 环境配置

```
GPU：A10 * 1
CUDA：12.6
TensorRT-LLM：0.16.0
LLM：Qwen3-8B
Lookahead参数：window_size（4），ngram_size（2），verification_set_size（4）
Max Tokens：50
```

### 压测结果

#### 首字延迟

| 服务 \ 首字延迟(ms) \ 并发 | 1     | 2      | 4      | 6     | 8      | 10      | 16      |
|--------------------|-------|--------|--------|-------|--------|---------|---------|
| autoregressive     | 40.12 | 71.69  | 76.88  | 77.32 | 87.47  | 96.88   | 142.59  |
| lookahead-decoding | 40.47 | 69.80  | 76.11  | 81.72 | 83.15  | 85.29   | 108.16  |
| 同比                 | 0.87% | -2.64% | -1.00% | 5.69% | -4.94% | -11.96% | -24.15% |

#### 解码吞吐

| 服务 \ 解码阶段吞吐(tokens/s) \ 并发 | 1       | 2       | 4       | 6       | 8       | 10      | 16      |
|----------------------------|---------|---------|---------|---------|---------|---------|---------|
| autoregressive             | 31.10   | 61.71   | 122.17  | 169.61  | 210.61  | 262.98  | 417.92  |
| lookahead-decoding         | 39.72   | 69.59   | 136.67  | 200.83  | 258.66  | 315.89  | 475.54  |
| 同比                         | +27.72% | +12.77% | +11.87% | +18.41% | +22.81% | +20.12% | +13.79% |