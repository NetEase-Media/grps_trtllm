// State associated with a trtllm model that is using this backend.
// Refer to:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/model_state.cc

#include "trtllm_model_state.h"

#include <algorithm>

#include "logger/logger.h"
#include "src/constants.h"

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

template <>
tensorrt_llm::executor::SamplingConfig TrtLlmModelState::GetParameter<tensorrt_llm::executor::SamplingConfig>(
  std::string const& name) {
  int32_t beam_width = 1;
  std::optional<int32_t> top_k{std::nullopt};
  std::optional<float> top_p{std::nullopt};
  std::optional<float> top_p_min{std::nullopt};
  std::optional<float> top_p_decay{std::nullopt};
  std::optional<int32_t> top_p_reset_ids{std::nullopt};
  std::optional<float> temperature{std::nullopt};
  std::optional<int32_t> early_stopping{std::nullopt};
  std::optional<int32_t> min_length{std::nullopt};
  std::optional<float> beam_search_diversity_rate{std::nullopt};
  std::optional<float> length_penalty{std::nullopt};
  std::optional<float> repetition_penalty{std::nullopt};
  std::optional<float> presence_penalty{std::nullopt};
  std::optional<float> frequency_penalty{std::nullopt};
  std::optional<uint64_t> random_seed{std::nullopt};
  std::optional<int32_t> no_repeat_ngram_size(std::nullopt);

  if (model_config_[name] && model_config_[name].IsMap()) {
    if (model_config_[name][InputFieldsNames::kBeamWidth]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kBeamWidth, int32_t, beam_width);
    }
    if (model_config_[name][InputFieldsNames::kTopK]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTopK, int32_t, top_k);
    }
    if (model_config_[name][InputFieldsNames::kTopP]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTopP, float, top_p);
    }
    if (top_p.has_value() && top_p.value() <= 0.F) {
      top_p.reset();
    }
    if (model_config_[name][InputFieldsNames::kTopPMin]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTopPMin, float, top_p_min);
    }
    if (model_config_[name][InputFieldsNames::kTopPDecay]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTopPDecay, float, top_p_decay);
    }
    if (model_config_[name][InputFieldsNames::kTopPResetIds]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTopPResetIds, int32_t, top_p_reset_ids);
    }
    if (model_config_[name][InputFieldsNames::kTemperature]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kTemperature, float, temperature);
    }
    if (model_config_[name][InputFieldsNames::kEarlyStopping]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kEarlyStopping, int32_t, early_stopping);
    }
    if (model_config_[name][InputFieldsNames::kMinLength]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kMinLength, int32_t, min_length);
    }
    if (model_config_[name][InputFieldsNames::kBeamSearchDiversityRate]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kBeamSearchDiversityRate, float,
                       beam_search_diversity_rate);
    }
    if (model_config_[name][InputFieldsNames::kLengthPenalty]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kLengthPenalty, float, length_penalty);
    }
    if (model_config_[name][InputFieldsNames::kRepetitionPenalty]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kRepetitionPenalty, float, repetition_penalty);
    }
    if (model_config_[name][InputFieldsNames::kPresencePenalty]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kPresencePenalty, float, presence_penalty);
    }
    if (model_config_[name][InputFieldsNames::kFrequencyPenalty]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kFrequencyPenalty, float, frequency_penalty);
    }
    if (model_config_[name][InputFieldsNames::kRandomSeed]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kRandomSeed, uint64_t, random_seed);
    }
    if (model_config_[name][InputFieldsNames::kNoRepeatNgramSize]) {
      YAML_TRY_EXTRACT(model_config_[name], InputFieldsNames::kNoRepeatNgramSize, int32_t, no_repeat_ngram_size);
    }
  }

  CLOG4(INFO,
        "Loaded default sampling config: "
          << "beam_width: " << beam_width << ", top_k: " << (top_k.has_value() ? std::to_string(top_k.value()) : "None")
          << ", top_p: " << (top_p.has_value() ? std::to_string(top_p.value()) : "None")
          << ", top_p_min: " << (top_p_min.has_value() ? std::to_string(top_p_min.value()) : "None")
          << ", top_p_decay: " << (top_p_decay.has_value() ? std::to_string(top_p_decay.value()) : "None")
          << ", top_p_reset_ids: " << (top_p_reset_ids.has_value() ? std::to_string(top_p_reset_ids.value()) : "None")
          << ", temperature: " << (temperature.has_value() ? std::to_string(temperature.value()) : "None")
          << ", min_length: " << (min_length.has_value() ? std::to_string(min_length.value()) : "None")
          << ", beam_search_diversity_rate: "
          << (beam_search_diversity_rate.has_value() ? std::to_string(beam_search_diversity_rate.value()) : "None")
          << ", repetition_penalty: "
          << (repetition_penalty.has_value() ? std::to_string(repetition_penalty.value()) : "None")
          << ", presence_penalty: "
          << (presence_penalty.has_value() ? std::to_string(presence_penalty.value()) : "None")
          << ", frequency_penalty: "
          << (frequency_penalty.has_value() ? std::to_string(frequency_penalty.value()) : "None")
          << ", length_penalty: " << (length_penalty.has_value() ? std::to_string(length_penalty.value()) : "None")
          << ", early_stopping: " << (early_stopping.has_value() ? std::to_string(early_stopping.value()) : "None")
          << ", no_repeat_ngram_size: "
          << (no_repeat_ngram_size.has_value() ? std::to_string(no_repeat_ngram_size.value()) : "None"));

  return tensorrt_llm::executor::SamplingConfig(beam_width, top_k, top_p, top_p_min, top_p_reset_ids, top_p_decay,
                                                random_seed, temperature, min_length, beam_search_diversity_rate,
                                                repetition_penalty, presence_penalty, frequency_penalty, length_penalty,
                                                early_stopping, no_repeat_ngram_size);
}

} // namespace netease::grps
