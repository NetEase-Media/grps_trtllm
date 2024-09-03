// Trt llm model instance.
// Porting from:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/model_instance_state.h

#pragma once

#include <map>
#include <queue>
#include <unordered_map>

#include "context/context.h"
#include "model_infer/tensor_wrapper.h"
#include "src/llm_styler.h"
#include "src/tokenizer.h"
#include "src/trtllm_model_state.h"
#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/executor/types.h"

using namespace tensorrt_llm;
using namespace tensorrt_llm::batch_manager;

namespace netease::grps {

/// @brief Struct to hold configs that is will be used later when creating the executor requests
struct InstanceSpecificConfig {
  bool exclude_input_from_output;
  int cancellation_check_period_ms;
  int stats_check_period_ms;
};

/// @brief Timestamps for each request, used to report Triton metrics
struct Timestamps {
  uint64_t exec_start_ns = 0;
  uint64_t compute_start_ns = 0;
  uint64_t compute_end_ns = 0;
  uint64_t exec_end_ns = 0;

  void Reset() {
    exec_start_ns = 0;
    compute_start_ns = 0;
    compute_end_ns = 0;
    exec_end_ns = 0;
  }
};

/// @brief Per-request data stored for handling requests
struct RequestData {
  std::string model;
  uint64_t trtllm_req_id;
  uint64_t created_timestamp;
  size_t input_tokens_size;
  bool func_call; // If request is function call.
  GrpsContext* grps_ctx;
  std::promise<void>* promise;
  std::vector<int32_t> history_tokens; // For fix garbled test case.
  size_t last_tokens;                  // last tokens count for streaming mode.
};

// TrtLlmModelInstance
class TrtLlmModelInstance {
  using InferenceRequest = tensorrt_llm::batch_manager::InferenceRequest;
  using NamedTensor = tensorrt_llm::batch_manager::NamedTensor;
  using TrtGptModelType = tensorrt_llm::batch_manager::TrtGptModelType;

public:
  // number of cpu workers used to move weights host cache to gpu cache
  static constexpr executor::SizeType32 kPeftCacheNumEnsureWorkers = 4;
  // number of cuda streams used for H2D copies of peft cache pages
  static constexpr executor::SizeType32 kPeftCacheNumCopyStreams = 4;
  // number of cpu workers used to load weight into host cache
  static constexpr executor::SizeType32 kPeftCacheNumPutWorkers = 4;

  /// @brief Constructor
  explicit TrtLlmModelInstance(TrtLlmModelState* model_state, LLMStyler* llm_styler, MultiInstanceTokenizer* tokenizer);

  virtual ~TrtLlmModelInstance() {
    stop_wait_for_response_ = true;
    wait_for_response_thread_.join();

    stop_wait_for_stats_ = true;
    wait_for_stats_thread_.join();
  }

  // Get the state of the model that corresponds to this instance.
  TrtLlmModelState* StateForModel() const { return model_state_; }

  /// @brief Add the request to the executor
  void EnqueueAndWait(GrpsContext& grps_ctx, const std::string& http_body);

private:
  /// @brief Get batching type
  executor::BatchingType GetBatchingTypeFromParams();

  /// @brief Get kv cache config
  executor::KvCacheConfig GetKvCacheConfigFromParams();

  /// @brief Get scheduler config
  executor::SchedulerConfig GetSchedulerConfigFromParams(bool enable_chunked_context);

  /// @brief Get peft config
  executor::PeftCacheConfig GetPeftCacheConfigFromParams();

  /// @brief Get parallel config
  executor::ParallelConfig GetParallelConfigFromParams();

  /// @brief Get executor config
  executor::ExecutorConfig GetExecutorConfigFromParams();

  /// @brief Fill in response based on executor response
  std::tuple<std::string, std::vector<int32_t>, bool, std::string> ParseTrtllmResponse(
    executor::Response const& response, RequestData& request_data);

  /// @brief Retrieve responses from the executor
  void WaitForResponse();

  /// @brief Retrieve stats from the executor
  void WaitForStats();

  TrtLlmModelState* model_state_;

  LLMStyler* llm_styler_;

  MultiInstanceTokenizer* tokenizer_;

  std::unordered_set<std::string> stop_words_;
  std::unordered_set<std::string> bad_words_;

  executor::ModelType model_type_;

  size_t max_batch_size_;
  size_t max_input_len_;
  size_t max_output_len_;

  /// @brief TRT-LLM Executor that handles requests
  std::unique_ptr<executor::Executor> executor_{};
  /// @brief Config to be used when sending requests to executor
  InstanceSpecificConfig instance_specific_config{};

  /// @brief The thread for WaitForResponse() to run
  std::thread wait_for_response_thread_;
  /// @brief Flag to stop the WaitForResponse thread when the model instance is being destroyed
  bool stop_wait_for_response_;

  /// @brief The thread for WaitForStats() to run
  std::thread wait_for_stats_thread_;
  /// @brief Flag to stop the WaitForStats thread when the model instance is being destroyed
  bool stop_wait_for_stats_;

  std::unordered_map<executor::IdType, RequestData> trtllm_request_id_to_request_data_{};
  std::mutex trtllm_request_id_to_request_data_mutex_;
};

} // namespace netease::grps
