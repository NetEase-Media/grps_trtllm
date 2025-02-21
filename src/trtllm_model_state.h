// State associated with a trtllm model that is using this backend.
// Refer to:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/model_state.h

#pragma once

#include <yaml-cpp/yaml.h>

#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

namespace netease::grps {

class TrtLlmModelState {
public:
  template <typename T>
  T GetParameter(std::string const& name) {
    assert(false);
    auto dummy = T();
    return dummy;
  }

  virtual ~TrtLlmModelState() = default;

  const YAML::Node& GetModelConfig();
  std::string GetExecutorWorkerPath();

  std::optional<std::vector<int32_t>> GetDeviceIds() { return gpu_device_ids_; }

private:
  YAML::Node model_config_;
  std::shared_ptr<nvinfer1::ILogger> trt_logger_{};

  // model parameters
  std::optional<std::vector<int32_t>> gpu_device_ids_;

  void LoadParameters();

public:
  template <typename T>
  TrtLlmModelState(T&& model_config) : model_config_(std::forward<T>(model_config)) {
    trt_logger_ = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
    initTrtLlmPlugins(trt_logger_.get());
    LoadParameters();
  }
};

template <>
std::string TrtLlmModelState::GetParameter<std::string>(std::string const& name);

template <>
int32_t TrtLlmModelState::GetParameter<int32_t>(std::string const& name);

template <>
uint32_t TrtLlmModelState::GetParameter<uint32_t>(std::string const& name);

template <>
int64_t TrtLlmModelState::GetParameter<int64_t>(std::string const& name);

template <>
uint64_t TrtLlmModelState::GetParameter<uint64_t>(std::string const& name);

template <>
float TrtLlmModelState::GetParameter<float>(std::string const& name);

template <>
bool TrtLlmModelState::GetParameter<bool>(std::string const& name);

template <>
std::vector<int32_t> TrtLlmModelState::GetParameter<std::vector<int32_t>>(std::string const& name);

template <>
std::vector<std::string> TrtLlmModelState::GetParameter<std::vector<std::string>>(std::string const& name);

template <>
std::vector<std::vector<int32_t>> TrtLlmModelState::GetParameter<std::vector<std::vector<int32_t>>>(
  std::string const& name);

template <>
std::unordered_map<std::string, int32_t> TrtLlmModelState::GetParameter<std::unordered_map<std::string, int32_t>>(
  std::string const& name);

template <>
tensorrt_llm::executor::SamplingConfig TrtLlmModelState::GetParameter<tensorrt_llm::executor::SamplingConfig>(
  std::string const& name);

} // namespace netease::grps
