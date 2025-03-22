# 重要更新

* 2025-03-22
    * 支持MiniCPM-V-2_6。

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
