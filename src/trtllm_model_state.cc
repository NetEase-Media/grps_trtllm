// State associated with a trtllm model that is using this backend.
// Porting from:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/model_state.cc

#include "trtllm_model_state.h"

#include <algorithm>

#include "logger/logger.h"

namespace netease::grps {

#define YAML_TRY_EXTRACT(yaml_node, conf_name, DataType, output)                   \
  try {                                                                            \
    output = yaml_node[conf_name].as<DataType>();                                  \
  } catch (const std::exception& e) {                                              \
    std::string err = std::string("Failed to parse conf, conf_name: " #conf_name); \
    err += ", data_type: " #DataType;                                              \
    err += ", err: ";                                                              \
    err += e.what();                                                               \
    throw std::invalid_argument(err);                                              \
  }

#define YAML_TRY_EXTRACT_VEC(yaml_node, conf_name, DataType, output)                             \
  auto seq = yaml_node[conf_name];                                                               \
  if (seq && seq.IsSequence()) {                                                                 \
    for (const auto& id : seq) {                                                                 \
      try {                                                                                      \
        output.emplace_back(id.as<DataType>());                                                  \
      } catch (const std::exception& e) {                                                        \
        std::string err = std::string("Failed to parse conf, conf_name: " #conf_name);           \
        err += ", data_type: " #DataType;                                                        \
        err += ", err: ";                                                                        \
        err += e.what();                                                                         \
        throw std::invalid_argument(err);                                                        \
      }                                                                                          \
    }                                                                                            \
  } else {                                                                                       \
    throw std::invalid_argument("Failed to parse conf, not a sequence, conf_name: " #conf_name); \
  }

#define YAML_TRY_EXTRACT_VEC_VEC(yaml_node, conf_name, DataType, output)                           \
  auto seq = yaml_node[conf_name];                                                                 \
  if (seq && seq.IsSequence()) {                                                                   \
    for (const auto& id : seq) {                                                                   \
      std::vector<DataType> inner_vec;                                                             \
      if (id && id.IsSequence()) {                                                                 \
        for (const auto& inner_id : id) {                                                          \
          try {                                                                                    \
            inner_vec.emplace_back(inner_id.as<DataType>());                                       \
          } catch (const std::exception& e) {                                                      \
            std::string err = std::string("Failed to parse conf, conf_name: " #conf_name);         \
            err += ", data_type: " #DataType;                                                      \
            err += ", err: ";                                                                      \
            err += e.what();                                                                       \
            throw std::invalid_argument(err);                                                      \
          }                                                                                        \
        }                                                                                          \
        output.emplace_back(inner_vec);                                                            \
      } else {                                                                                     \
        throw std::invalid_argument("Failed to parse conf, not a 2D vec, conf_name: " #conf_name); \
      }                                                                                            \
    }                                                                                              \
  } else {                                                                                         \
    throw std::invalid_argument("Failed to parse conf, not a 2D vec, conf_name: " #conf_name);     \
  }

void TrtLlmModelState::LoadParameters() {
  try {
    gpu_device_ids_ = GetParameter<std::vector<int32_t>>("gpu_device_ids");
  } catch (const std::exception& e) {
    CLOG4(WARN, "gpu_device_ids is not specified, will be automatically set");
  }

  if (gpu_device_ids_) {
    std::string device_id_info("Using GPU device ids: ");
    for (auto const& device_id : *gpu_device_ids_) {
      device_id_info += std::to_string(device_id) + " ";
    }
    CLOG4(INFO, device_id_info);
  }
}

const YAML::Node& TrtLlmModelState::GetModelConfig() {
  return model_config_;
}

std::string TrtLlmModelState::GetExecutorWorkerPath() {
  std::string executorWorkerPath;
  try {
    executorWorkerPath = GetParameter<std::string>("executor_worker_path");
  } catch (std::exception const& e) {
    CLOG4(WARN, "executor_worker_path is not specified, will use default value");
  }
  return executorWorkerPath.empty() ? "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"
                                    : executorWorkerPath;
}

template <>
std::string TrtLlmModelState::GetParameter<std::string>(std::string const& name) {
  std::string str_value;
  YAML_TRY_EXTRACT(model_config_, name, std::string, str_value);
  return str_value;
}

template <>
int32_t TrtLlmModelState::GetParameter<int32_t>(std::string const& name) {
  int32_t int_value;
  YAML_TRY_EXTRACT(model_config_, name, int32_t, int_value);
  return int_value;
}

template <>
std::vector<int32_t> TrtLlmModelState::GetParameter<std::vector<int32_t>>(std::string const& name) {
  std::vector<int32_t> vec;
  YAML_TRY_EXTRACT_VEC(model_config_, name, int32_t, vec);
  return vec;
}

template <>
std::vector<std::string> TrtLlmModelState::GetParameter<std::vector<std::string>>(std::string const& name) {
  std::vector<std::string> vec;
  YAML_TRY_EXTRACT_VEC(model_config_, name, std::string, vec);
  return vec;
}

template <>
uint32_t TrtLlmModelState::GetParameter<uint32_t>(std::string const& name) {
  uint32_t uint_value;
  YAML_TRY_EXTRACT(model_config_, name, uint32_t, uint_value);
  return uint_value;
}

template <>
int64_t TrtLlmModelState::GetParameter<int64_t>(std::string const& name) {
  int64_t int64_value;
  YAML_TRY_EXTRACT(model_config_, name, int64_t, int64_value);
  return int64_value;
}

template <>
uint64_t TrtLlmModelState::GetParameter<uint64_t>(std::string const& name) {
  uint64_t uint64_value;
  YAML_TRY_EXTRACT(model_config_, name, uint64_t, uint64_value);
  return uint64_value;
}

template <>
float TrtLlmModelState::GetParameter<float>(std::string const& name) {
  float float_value;
  YAML_TRY_EXTRACT(model_config_, name, float, float_value);
  return float_value;
}

template <>
bool TrtLlmModelState::GetParameter<bool>(std::string const& name) {
  bool bool_value;
  YAML_TRY_EXTRACT(model_config_, name, bool, bool_value);
  return bool_value;
}

template <>
std::vector<std::vector<int32_t>> TrtLlmModelState::GetParameter<std::vector<std::vector<int32_t>>>(
  std::string const& name) {
  std::vector<std::vector<int32_t>> vec;
  YAML_TRY_EXTRACT_VEC_VEC(model_config_, name, int, vec);
  return vec;
}

template <>
std::unordered_map<std::string, int32_t> TrtLlmModelState::GetParameter<std::unordered_map<std::string, int32_t>>(
  std::string const& name) {
  std::unordered_map<std::string, int32_t> str2int;
  auto map = model_config_[name];
  if (map && map.IsMap()) {
    for (const auto& item : map) {
      try {
        str2int[item.first.as<std::string>()] = item.second.as<int32_t>();
      } catch (const std::exception& e) {
        std::string err = std::string("Failed to parse conf, conf_name: ") + name;
        err += ", data_type: std::unordered_map<std::string, int32_t>";
        err += ", err: ";
        err += e.what();
        throw std::invalid_argument(err);
      }
    }
  } else {
    throw std::invalid_argument("Failed to parse conf, not a map, conf_name: " + name);
  }
  return str2int;
}

} // namespace netease::grps
