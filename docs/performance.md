# 性能

## 文本LLM

通过Qwen2.5-7B-Instruct模型对比与```xinference-vllm```
服务的性能差异。grps-trtllm打开[kv cache ruse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
，xinference-vllm打开类似功能[automatic prefix caching](https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html)。

```
GPU: A10 * 1
CUDA: cuda_12.6
Trtllm: 0.16.0
xinference: 1.1.1
vLLM: 0.6.6
CPU: Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz
Mem：128G
LLM: Qwen2.5-7B-Instruct
```

短输入输出：
固定输入（华盛顿是谁？），120 tokens左右。

| 服务 \ 吞吐(tokens/s) \ 并发 | 1       | 2       | 4       | 6       | 8       | 10      | 16      |
|------------------------|---------|---------|---------|---------|---------|---------|---------|
| xinference-vllm        | 34.28   | 66.68   | 132.96  | 193.38  | 259.02  | 315.72  | 494.42  |
| grps-trtllm            | 41.36   | 81.49   | 161.45  | 225.40  | 287.35  | 355.39  | 557.89  |
| 同比                     | +20.65% | +22.21% | +21.43% | +15.56% | +10.94% | +12.56% | +12.84% |

长输入输出：
固定输入为1.2k左右tokens数量的文章，输出为190左右token数量的总结。

| 服务 \ 吞吐(tokens/s) \ 并发 | 1       | 2       | 4       | 6       | 8       | 10      | 16      |
|------------------------|---------|---------|---------|---------|---------|---------|---------|
| xinference-vllm        | 217.51  | 426.68  | 835.96  | 1225.67 | 1616.11 | 1944.35 | 3009.80 |
| grps-trtllm            | 250.77  | 519.31  | 1016.36 | 1383.31 | 1672.20 | 2064.29 | 3297.62 |
| 同比                     | +15.29% | +21.70% | +21.58% | +12.86% | +3.47%  | +6.17%  | +9.56%  |

## 多模态LLM

通过InternVL2.5-4B模型对比与```vllm```服务的性能差异。

```
GPU: A10 * 1
CUDA: cuda_12.6
Trtllm: 0.16.0
vLLM: 0.6.6
CPU: Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz
Mem：128G
LLM: InternVL2.5-4B
```

输入一张[image1.jpg](../data/image1.jpg)图片，输入prompt：“`<image>`\n简述一下这张图片的内容。”，由于输出长度可能会变化，这里限制输出50个tokens数量。

| 服务 \ 吞吐(tokens/s) \ 并发 | 1       | 2       | 3       | 4       | 
|------------------------|---------|---------|---------|---------|
| vllm                   | 2221.40 | 3372.00 | 3818.08 | 4256.31 | 
| grps-trtllm            | 2681.85 | 3872.88 | 4530.22 | 4945.15 | 
| 同比                     | +20.72% | +14.85% | +18.65% | +16.18% | 