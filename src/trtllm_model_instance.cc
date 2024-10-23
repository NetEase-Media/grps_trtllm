// State associated with a model instance.
// Porting from:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/model_instance_state.cc

#include "trtllm_model_instance.h"

#include <rapidjson/document.h>

#include <memory>

#include "logger/logger.h"
#include "monitor/monitor.h"
#include "utils.h"

namespace netease::grps {

executor::BatchingType TrtLlmModelInstance::GetBatchingTypeFromParams() {
  executor::BatchingType batching_type;
  auto gpt_model_type = model_state_->GetParameter<std::string>("gpt_model_type");

  if (gpt_model_type == "V1" || gpt_model_type == "v1") {
    batching_type = executor::BatchingType::kSTATIC;
  } else if (gpt_model_type == "inflight_batching" || gpt_model_type == "inflight_fused_batching") {
    batching_type = executor::BatchingType::kINFLIGHT;
  } else {
    throw std::runtime_error(
      "Invalid gpt_model_type. Must be "
      "v1/inflight_batching/inflight_fused_batching.");
  }
  return batching_type;
}

executor::KvCacheConfig TrtLlmModelInstance::GetKvCacheConfigFromParams() {
  std::optional<int32_t> maxTokensInPagedKvCache = std::nullopt;
  try {
    maxTokensInPagedKvCache = model_state_->GetParameter<int32_t>("max_tokens_in_paged_kv_cache");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN,
          "max_tokens_in_paged_kv_cache is not specified, will "
          "use default value");
  }

  std::optional<float> kvCacheFreeGpuMemFraction = std::nullopt;
  try {
    kvCacheFreeGpuMemFraction = model_state_->GetParameter<float>("kv_cache_free_gpu_mem_fraction");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN,
          "kv_cache_free_gpu_mem_fraction is not specified, will use default value of 0.9 or "
          "max_tokens_in_paged_kv_cache");
  }

  std::optional<size_t> kvCacheHostCacheSize = std::nullopt;
  try {
    kvCacheHostCacheSize = model_state_->GetParameter<size_t>("kv_cache_host_memory_bytes");
  } catch (std::exception const& e) {
    CLOG4(WARN, "kv_cache_host_memory_bytes not set, defaulting to 0");
  }

  bool kvCacheOnboardBlocks = true;
  try {
    kvCacheOnboardBlocks = model_state_->GetParameter<bool>("kv_cache_onboard_blocks");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "kv_cache_onboard_blocks not set, defaulting to true");
  }

  std::optional<int32_t> maxAttentionWindow = std::nullopt;
  try {
    maxAttentionWindow = model_state_->GetParameter<int32_t>("max_attention_window_size");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN,
          "max_attention_window_size is not specified, will "
          "use default value (i.e. max_sequence_length)");
  }

  std::optional<int32_t> sinkTokenLength = std::nullopt;
  try {
    sinkTokenLength = model_state_->GetParameter<int32_t>("sink_token_length");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN,
          "sink_token_length is not specified, will "
          "use default value");
  }

  bool enableKVCacheReuse = false;
  try {
    enableKVCacheReuse = model_state_->GetParameter<bool>("enable_kv_cache_reuse");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "enable_kv_cache_reuse is not specified, will be set to false");
  }

  std::optional<SizeType32> maxAttentionWindowSizeType = std::nullopt;
  if (maxAttentionWindow.has_value()) {
    maxAttentionWindowSizeType = static_cast<SizeType32>(maxAttentionWindow.value());
  }

  return executor::KvCacheConfig(enableKVCacheReuse, maxTokensInPagedKvCache, maxAttentionWindowSizeType,
                                 sinkTokenLength, kvCacheFreeGpuMemFraction, kvCacheHostCacheSize,
                                 kvCacheOnboardBlocks);
}

executor::ParallelConfig TrtLlmModelInstance::GetParallelConfigFromParams() {
  executor::ParallelConfig parallel_config;
  auto const gpu_device_ids = model_state_->GetDeviceIds();
  if (gpu_device_ids.has_value()) {
    parallel_config.setDeviceIds(gpu_device_ids.value());
  }

  char const* str = std::getenv("TRTLLM_ORCHESTRATOR");
  if (str && std::atoi(str) != 0) {
    parallel_config.setCommunicationMode(executor::CommunicationMode::kORCHESTRATOR);
    auto worker_executable_path = model_state_->GetExecutorWorkerPath();
    auto orchestrator_config = executor::OrchestratorConfig(true, worker_executable_path);
    parallel_config.setOrchestratorConfig(orchestrator_config);
  }
  return parallel_config;
}

executor::PeftCacheConfig TrtLlmModelInstance::GetPeftCacheConfigFromParams() {
  // parse LoRA / Peft cache parameters
  // lora_cache_max_adapter_size
  // lora_cache_optimal_adapter_size
  // lora_cache_gpu_memory_fraction
  // lora_cache_host_memory_bytes

  SizeType32 max_adapter_size = 64;
  SizeType32 optimal_adapter_size = 8;
  std::optional<size_t> host_cache_size = std::nullopt;
  std::optional<float> device_cache_percent = std::nullopt;

  std::string field_name = "lora_cache_max_adapter_size";
  try {
    max_adapter_size = model_state_->GetParameter<SizeType32>(field_name);
  } catch (std::exception const& e) {
    CLOG4(WARN, field_name + " not set, defaulting to 64");
  }

  field_name = "lora_cache_optimal_adapter_size";
  try {
    optimal_adapter_size = model_state_->GetParameter<SizeType32>(field_name);
  } catch (std::exception const& e) {
    CLOG4(WARN, field_name + " not set, defaulting to 8");
  }
  field_name = "lora_cache_gpu_memory_fraction";
  try {
    device_cache_percent = model_state_->GetParameter<float>(field_name);
  } catch (std::exception const& e) {
    CLOG4(WARN, field_name + " not set, defaulting to 0.05");
  }
  field_name = "lora_cache_host_memory_bytes";
  try {
    host_cache_size = model_state_->GetParameter<size_t>(field_name);
  } catch (std::exception const& e) {
    CLOG4(WARN, field_name + " not set, defaulting to 1GB");
  }

  return executor::PeftCacheConfig(
    0, 0, optimal_adapter_size, max_adapter_size, TrtLlmModelInstance::kPeftCacheNumPutWorkers,
    TrtLlmModelInstance::kPeftCacheNumEnsureWorkers, TrtLlmModelInstance::kPeftCacheNumCopyStreams, 24, 8,
    device_cache_percent, host_cache_size);
}

executor::SchedulerConfig TrtLlmModelInstance::GetSchedulerConfigFromParams(bool enable_chunked_context) {
  using executor::CapacitySchedulerPolicy;
  auto scheduler_policy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
  try {
    std::string scheduler_policy_str = model_state_->GetParameter<std::string>("batch_scheduler_policy");
    if (scheduler_policy_str == "max_utilization") {
      scheduler_policy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
    } else if (scheduler_policy_str == "guaranteed_no_evict") {
      scheduler_policy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    } else {
      throw std::runtime_error(
        "batch_scheduler_policy parameter was not found or is invalid "
        "(must be max_utilization or guaranteed_no_evict)");
    }
  } catch (std::exception const& e) {
    CLOG4(WARN, e.what() + std::string(", batch_scheduler_policy will use guaranteed_no_evict for default."));
  }

  if (scheduler_policy != CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT) {
    if (!enable_chunked_context) {
      CLOG4(WARN,
            "Batch scheduler policy other than guaranteed_no_evict "
            "requires building the model with use_paged_context_fmha and setting "
            "enable_chunked_context to true. "
            "The batch scheduler policy will be set to guaranteed_no_evict "
            "since enable_chunked_context is false.");
      scheduler_policy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    }
  }
  return executor::SchedulerConfig(scheduler_policy);
}

executor::ExecutorConfig TrtLlmModelInstance::GetExecutorConfigFromParams() {
  auto batching_type = GetBatchingTypeFromParams();

  int32_t max_beam_width = 1;
  /* Only support max beam width of 1 for now
  try {
    max_beam_width = model_state_->GetParameter<int32_t>("max_beam_width");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "max_beam_width is not specified, will use default value of 1");
  }
  */

  int32_t iter_stats_max_iterations = executor::kDefaultIterStatsMaxIterations;
  try {
    iter_stats_max_iterations = model_state_->GetParameter<int32_t>("iter_stats_max_iterations");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "iter_stats_max_iterations is not specified, will use default value of " +
                  std::to_string(iter_stats_max_iterations));
  }

  int32_t request_stats_max_iterations = executor::kDefaultRequestStatsMaxIterations;
  try {
    request_stats_max_iterations = model_state_->GetParameter<int32_t>("request_stats_max_iterations");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "request_stats_max_iterations is not specified, will use default value of " +
                  std::to_string(request_stats_max_iterations));
  }

  // try {
  //   model_state_->GetParameter<bool>("enable_trt_overlap");
  //   CLOG4(WARN, "enable_trt_overlap is deprecated and will be ignored");
  // } catch (std::exception const& e) {
  // }

  bool normalize_log_probs = true;
  try {
    normalize_log_probs = model_state_->GetParameter<bool>("normalize_log_probs");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "normalize_log_probs is not specified, will be set to true");
  }

  executor::ExecutorConfig executor_config;

  auto kv_cache_config = GetKvCacheConfigFromParams();

  bool enable_chunked_context = false;
  try {
    enable_chunked_context = model_state_->GetParameter<bool>("enable_chunked_context");
    if (enable_chunked_context) {
      CLOG4(WARN,
            "enable_chunked_context is set to true, will use context chunking "
            "(requires building the model with use_paged_context_fmha).");
    }
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "enable_chunked_context is not specified, will be set to false.");
  }

  auto scheduler_config = GetSchedulerConfigFromParams(enable_chunked_context);

  auto peft_cache_config = GetPeftCacheConfigFromParams();

  auto parallel_config = GetParallelConfigFromParams();

  std::optional<executor::DecodingMode> decoding_mode = std::nullopt;
  try {
    std::string decoding_mode_str = model_state_->GetParameter<std::string>("decoding_mode");
    if (decoding_mode_str == "top_k") {
      decoding_mode = executor::DecodingMode::TopK();
    } else if (decoding_mode_str == "top_p") {
      decoding_mode = executor::DecodingMode::TopP();
    } else if (decoding_mode_str == "top_k_top_p") {
      decoding_mode = executor::DecodingMode::TopKTopP();
    } else if (decoding_mode_str == "beam_search") {
      decoding_mode = executor::DecodingMode::BeamSearch();
    } else if (decoding_mode_str == "medusa") {
      decoding_mode = executor::DecodingMode::Medusa();
    } else {
      throw std::runtime_error("");
    }
  } catch (std::exception const& e) {
    CLOG4(WARN,
          "decoding_mode parameter is invalid or not specified"
          "(must be one of the {top_k, top_p, top_k_top_p, beam_search, medusa}). "
          "Using default: top_k_top_p if max_beam_width == 1, beam_search otherwise");
  }

  executor::DecodingConfig decoding_config(decoding_mode);
  try {
    auto medusa_choices = model_state_->GetParameter<executor::MedusaChoices>("medusa_choices");
    decoding_config.setMedusaChoices(medusa_choices);
  } catch (std::exception const& e) {
    if (decoding_mode && decoding_mode->isMedusa()) {
      CLOG4(WARN,
            "medusa_choices parameter is not specified. "
            "Will be using default mc_sim_7b_63 choices instead");
    }
  }

  float gpu_weights_percent = 1.0f;
  try {
    gpu_weights_percent = model_state_->GetParameter<float>("gpu_weights_percent");
  } catch (std::exception const& e) {
    CLOG4(WARN, "gpu_weights_percent parameter is not specified, will use default value of 1.0");
  }

  return executor::ExecutorConfig(max_beam_width, scheduler_config, kv_cache_config, enable_chunked_context,
                                  normalize_log_probs, iter_stats_max_iterations, request_stats_max_iterations,
                                  batching_type, std::nullopt, std::nullopt, parallel_config, peft_cache_config,
                                  std::nullopt, std::nullopt, decoding_config, gpu_weights_percent);
}

TrtLlmModelInstance::TrtLlmModelInstance(TrtLlmModelState* model_state,
                                         LLMStyler* llm_styler,
                                         MultiInstanceTokenizer* tokenizer,
                                         VIT* vit)
    : model_state_(model_state), llm_styler_(llm_styler), tokenizer_(tokenizer), vit_(vit) {
  std::string decoder_model_path;
  try {
    decoder_model_path = model_state_->GetParameter<std::string>("gpt_model_path");
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(decoder_model_path), "Decoder (GPT) model path at %s does not exist.",
                         decoder_model_path.c_str());
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "gpt_model_path is not specified, will be left empty");
    decoder_model_path = "";
  }

  auto model_conf_bytes = utils::LoadBytesFromFile<std::string>(decoder_model_path + "config.json");
  rapidjson::Document model_conf_doc;
  model_conf_doc.Parse(model_conf_bytes.data(), model_conf_bytes.size());
  if (model_conf_doc.HasParseError()) {
    throw std::runtime_error("Failed to parse model config file: " + decoder_model_path + "config.json");
  }
  if (!model_conf_doc.HasMember("build_config") || !model_conf_doc["build_config"].IsObject()) {
    throw std::runtime_error("Model config file does not contain valid build_config field: " + decoder_model_path +
                             "config.json");
  }
  auto build_config = model_conf_doc["build_config"].GetObject();
  if (!build_config.HasMember("max_batch_size") || !build_config["max_batch_size"].IsInt()) {
    throw std::runtime_error("Model config file does not contain valid max_batch_size field: " + decoder_model_path +
                             "config.json");
  }
  max_batch_size_ = build_config["max_batch_size"].GetInt();
  if (!build_config.HasMember("max_input_len") || !build_config["max_input_len"].IsInt()) {
    throw std::runtime_error("Model config file does not contain valid max_input_len field: " + decoder_model_path +
                             "config.json");
  }
  max_input_len_ = build_config["max_input_len"].GetInt();
  if (!build_config.HasMember("max_seq_len") || !build_config["max_seq_len"].IsInt()) {
    throw std::runtime_error("Model config file does not contain valid max_seq_len field: " + decoder_model_path +
                             "config.json");
  }
  max_output_len_ = build_config["max_seq_len"].GetInt() - max_input_len_;

  // Parse default sampling config
  def_sampling_config_ = model_state_->GetParameter<executor::SamplingConfig>("sampling");

  std::string encoder_model_path;
  try {
    encoder_model_path = model_state_->GetParameter<std::string>("encoder_model_path");
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(encoder_model_path), "Encoder model path at %s does not exist.",
                         encoder_model_path.c_str());
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "encoder_model_path is not specified, will be left empty");
    encoder_model_path = "";
  }

  TLLM_CHECK_WITH_INFO(!decoder_model_path.empty() || !encoder_model_path.empty(),
                       "Both encoder and decoder model paths are empty");

  auto executor_config = GetExecutorConfigFromParams();

  if (!decoder_model_path.empty()) {
    // Encoder-decoder model
    if (!encoder_model_path.empty()) {
      model_type_ = executor::ModelType::kENCODER_DECODER;
      executor_ =
        std::make_unique<executor::Executor>(encoder_model_path, decoder_model_path, model_type_, executor_config);
    }
    // Decoder only model
    else {
      model_type_ = executor::ModelType::kDECODER_ONLY;
      executor_ = std::make_unique<executor::Executor>(decoder_model_path, model_type_, executor_config);
    }
  }
  // Encoder only
  else {
    model_type_ = executor::ModelType::kENCODER_ONLY;
    executor_ = std::make_unique<executor::Executor>(encoder_model_path, model_type_, executor_config);
  }

  bool exclude_input_in_output = false;
  try {
    exclude_input_in_output = model_state_->GetParameter<bool>("exclude_input_in_output");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "exclude_input_in_output is not specified, will be set to false");
  }
  instance_specific_config.exclude_input_from_output = exclude_input_in_output;

  int cancellation_check_period_ms = 100;
  try {
    cancellation_check_period_ms = model_state_->GetParameter<int>("cancellation_check_period_ms");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "cancellation_check_period_ms is not specified, will be set to 100 (ms)");
  }
  instance_specific_config.cancellation_check_period_ms = cancellation_check_period_ms;

  int stats_check_period_ms = 100;
  try {
    stats_check_period_ms = model_state_->GetParameter<int>("stats_check_period_ms");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "stats_check_period_ms is not specified, will be set to 100 (ms)");
  }
  instance_specific_config.stats_check_period_ms = stats_check_period_ms;

  std::vector<std::string> stop_words;
  try {
    stop_words = model_state_->GetParameter<std::vector<std::string>>("stop_words");
    stop_words_.insert(stop_words.begin(), stop_words.end());
  } catch (const std::exception& e) {
    CLOG4(WARN, "stop_words is not specified, will use default value of empty.");
  }

  std::vector<std::string> bad_words;
  try {
    bad_words = model_state_->GetParameter<std::vector<std::string>>("bad_words");
    bad_words_.insert(bad_words.begin(), bad_words.end());
  } catch (const std::exception& e) {
    CLOG4(WARN, "bad_words is not specified, will use default value of empty.");
  }

  if (executor_->canEnqueueRequests()) {
    stop_wait_for_response_ = false;
    wait_for_response_thread_ = std::thread(&TrtLlmModelInstance::WaitForResponse, this);

    stop_wait_for_stats_ = false;
    wait_for_stats_thread_ = std::thread(&TrtLlmModelInstance::WaitForStats, this);
  } else {
    // Shutdown the worker ranks which will cause them to wait for leader/orchestrator to terminate
    executor_->shutdown();
  }
}

void TrtLlmModelInstance::EnqueueAndWait(GrpsContext& grps_ctx, const std::string& http_body) {
  auto [func_call, model, executor_request] = utils::CreateRequestFromOpenAiHttpBody(
    http_body, instance_specific_config.exclude_input_from_output, grps_ctx.IfStreaming(), llm_styler_, tokenizer_,
    vit_, stop_words_, bad_words_, max_output_len_, model_type_, def_sampling_config_);
  size_t input_tokens_size = executor_request.getInputTokenIds().size();
  if (input_tokens_size > max_input_len_) {
    std::string err = "Input tokens size " + std::to_string(input_tokens_size) + " exceeds max input length " +
                      std::to_string(max_input_len_);
    CLOG4(ERROR, err);
    throw std::runtime_error(err);
  }

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  {
    std::lock_guard<std::mutex> lock(trtllm_request_id_to_request_data_mutex_);
    uint64_t compute_start_ns{0};
    SET_TIMESTAMP(compute_start_ns);
    auto trtllm_request_id = executor_->enqueueRequest(executor_request);
    if (trtllm_request_id_to_request_data_.count(trtllm_request_id)) {
      std::string err =
        "Executor returns a request ID that already exists. This shouldn't happen unless there is "
        "something wrong in TRT-LLM runtime.";
      CLOG4(ERROR, err);
      throw std::runtime_error(err);
    }

    uint64_t begin_time_us = GET_SYS_TME_US();
    trtllm_request_id_to_request_data_.emplace(trtllm_request_id, RequestData{model,
                                                                              trtllm_request_id,
                                                                              begin_time_us / 1000000,
                                                                              input_tokens_size,
                                                                              0,
                                                                              func_call,
                                                                              &grps_ctx,
                                                                              &promise,
                                                                              {},
                                                                              0,
                                                                              begin_time_us});
  }

  future.wait();
}

std::tuple<std::string, std::vector<int32_t>, bool, std::string> TrtLlmModelInstance::ParseTrtllmResponse(
  executor::Response const& response, RequestData& request_data) {
  std::string res;
  std::vector<int32_t> tokens = std::move(request_data.history_tokens);
  std::string error;
  bool is_final = false;
  try {
    if (!response.hasError()) {
      auto const& result = response.getResult();
      is_final = result.isFinal;
      auto output_ids = result.outputTokenIds;
      if (!output_ids.empty()) { // Only support max beam width of 1 for now
        tokens.insert(tokens.end(), output_ids[0].begin(), output_ids[0].end());
        res.append(tokenizer_->Decode(tokens));
      }
    } else {
      is_final = true;
      error = "Executor failed process request_id " + std::to_string(response.getRequestId()) +
              " due to the following error: " + response.getErrorMsg();
      CLOG4(ERROR, error);
    }
  } catch (std::exception const& e) {
    // In case of error while processing response, return response with error
    is_final = true;
    error = "Error encountered while populating response: " + std::string(e.what());
    CLOG4(ERROR, error);
  }

  return {res, tokens, is_final, error};
}

void TrtLlmModelInstance::WaitForResponse() {
  while (!stop_wait_for_response_) {
    std::chrono::milliseconds wait_time(1);
    auto responses = executor_->awaitResponses(wait_time);

    for (auto const& response : responses) {
      auto trtllm_request_id = response.getRequestId();
      RequestData* request_data;
      {
        std::lock_guard<std::mutex> lock(trtllm_request_id_to_request_data_mutex_);
        if (!trtllm_request_id_to_request_data_.count(trtllm_request_id)) {
          CLOG4(WARN, "Unexpected response for a request ID that is not active");
          continue;
        }
        request_data = &trtllm_request_id_to_request_data_[trtllm_request_id];
      }

      auto* grps_ctx = request_data->grps_ctx;

      auto [res, tokens, is_final, error] = ParseTrtllmResponse(response, *request_data);
      // CLOG4(INFO, "res: " << res);
      auto generated_tokens_size = tokens.size();
      request_data->output_tokens_size += generated_tokens_size;

      if (grps_ctx->IfStreaming() && !is_final && !utils::IsValidUTF8(res)) {
        request_data->history_tokens = std::move(tokens);
        continue;
      }

      if (is_final) {
        if (grps_ctx->IfStreaming()) {
          if (!error.empty()) {
            utils::HttpStreamingRespondErrorWithOpenAi(*grps_ctx, error);
          } else {
            if (request_data->func_call) {
              utils::HttpStreamingRespondWithOpenAi(*grps_ctx, request_data->trtllm_req_id,
                                                    request_data->created_timestamp, request_data->model, "", true,
                                                    false, true, llm_styler_);
              utils::HttpStreamingRespondWithOpenAi(*grps_ctx, request_data->trtllm_req_id,
                                                    request_data->created_timestamp, request_data->model, res, false,
                                                    true, true, llm_styler_);
            } else {
              utils::HttpStreamingRespondWithOpenAi(
                *grps_ctx, request_data->trtllm_req_id, request_data->created_timestamp, request_data->model, res,
                request_data->last_tokens == 0, false, request_data->func_call, llm_styler_);
              utils::HttpStreamingRespondWithOpenAi(*grps_ctx, request_data->trtllm_req_id,
                                                    request_data->created_timestamp, request_data->model, "", false,
                                                    true, request_data->func_call, llm_styler_);
            }
            MONITOR_INC("tp(token/s)", generated_tokens_size);
          }
        } else {
          if (!error.empty()) {
            utils::HttpRespondErrorWithOpenAi(*grps_ctx, 500, error);
          } else {
            utils::HttpRespondWithOpenAi(*grps_ctx, request_data->trtllm_req_id, request_data->created_timestamp,
                                         request_data->model, res, request_data->input_tokens_size,
                                         generated_tokens_size, request_data->func_call, llm_styler_);
            MONITOR_INC("tp(token/s)", generated_tokens_size);
          }
        }

        uint64_t finish_time_us = GET_SYS_TME_US();
        CLOG4(INFO, "Finished request: " << request_data->trtllm_req_id << ", model: " << request_data->model
                                         << ", if func call: " << request_data->func_call
                                         << ", if streaming: " << grps_ctx->IfStreaming()
                                         << ", input tokens count: " << request_data->input_tokens_size
                                         << ", output tokens count: " << request_data->output_tokens_size
                                         << ", total tokens count: "
                                         << request_data->input_tokens_size + request_data->output_tokens_size
                                         << ", used time: " << (finish_time_us - request_data->begin_time_us) / 1000
                                         << "ms");

        // Clean up request data and notify the waiting thread
        {
          std::lock_guard<std::mutex> lock(trtllm_request_id_to_request_data_mutex_);
          trtllm_request_id_to_request_data_.erase(trtllm_request_id);
        }
        request_data->promise->set_value();
      } else {
        if (grps_ctx->IfStreaming()) {
          if (!error.empty()) {
            utils::HttpStreamingRespondErrorWithOpenAi(*grps_ctx, error);
          } else {
            if (!grps_ctx->IfDisconnected()) {
              utils::HttpStreamingRespondWithOpenAi(*grps_ctx, request_data->trtllm_req_id,
                                                    request_data->created_timestamp, request_data->model, res,
                                                    request_data->last_tokens == 0, false, false, llm_styler_);
              request_data->last_tokens = generated_tokens_size;
              MONITOR_INC("tp(token/s)", generated_tokens_size);
            } else {
              // If the client has disconnected, cancel the request and clean up request data and notify the waiting
              // thread.
              executor_->cancelRequest(trtllm_request_id);
              {
                std::lock_guard<std::mutex> lock(trtllm_request_id_to_request_data_mutex_);
                trtllm_request_id_to_request_data_.erase(trtllm_request_id);
              }
              request_data->promise->set_value();
            }
          }
        }
      }
    }
  }
}

void TrtLlmModelInstance::WaitForStats() {
  while (!stop_wait_for_stats_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(instance_specific_config.stats_check_period_ms));
    auto stats = executor_->getLatestIterationStats();
    for (auto const& stat : stats) {
      std::string stat_json = "{";
      stat_json.append("\"Active Request Count\":" + std::to_string(stat.numActiveRequests) + ",");
      stat_json.append("\"Iteration Counter\":" + std::to_string(stat.iter) + ",");
      stat_json.append("\"Max Request Count\":" + std::to_string(stat.maxNumActiveRequests) + ",");
      stat_json.append("\"Runtime CPU Memory Usage\":" + std::to_string(stat.cpuMemUsage) + ",");
      stat_json.append("\"Runtime GPU Memory Usage\":" + std::to_string(stat.gpuMemUsage) + ",");
      stat_json.append("\"Runtime Pinned Memory Usage\":" + std::to_string(stat.pinnedMemUsage) + ",");
      stat_json.append("\"Timestamp\":" + ("\"" + stat.timestamp + "\"") + ",");

      if (stat.inflightBatchingStats.has_value()) {
        auto const& model_stats = stat.inflightBatchingStats.value();
        stat_json.append("\"Context Requests\":" + std::to_string(model_stats.numContextRequests) + ",");
        stat_json.append("\"Generation Requests\":" + std::to_string(model_stats.numGenRequests) + ",");
        stat_json.append("\"MicroBatch ID\":" + std::to_string(model_stats.microBatchId) + ",");
        stat_json.append("\"Paused Requests\":" + std::to_string(model_stats.numPausedRequests) + ",");
        stat_json.append("\"Scheduled Requests\":" + std::to_string(model_stats.numScheduledRequests) + ",");
        stat_json.append("\"Total Context Tokens\":" + std::to_string(model_stats.numCtxTokens) + ",");
      } else if (stat.staticBatchingStats.has_value()) {
        auto const& model_stats = stat.staticBatchingStats.value();
        stat_json.append("\"Context Requests\":" + std::to_string(model_stats.numContextRequests) + ",");
        stat_json.append("\"Scheduled Requests\":" + std::to_string(model_stats.numScheduledRequests) + ",");
        stat_json.append("\"Total Context Tokens\":" + std::to_string(model_stats.numCtxTokens) + ",");
        stat_json.append("\"Total Generation Tokens\":" + std::to_string(model_stats.numGenTokens) + ",");
        stat_json.append("\"Empty Generation Slots\":" + std::to_string(model_stats.emptyGenSlots) + ",");
      } else {
        CLOG4(ERROR, "Missing stats");
        continue;
      }

      if (stat.kvCacheStats.has_value()) {
        auto const& kv_stats = stat.kvCacheStats.value();
        stat_json.append("\"Free KV cache blocks\":" + std::to_string(kv_stats.freeNumBlocks) + ",");
        stat_json.append("\"Max KV cache blocks\":" + std::to_string(kv_stats.maxNumBlocks) + ",");
        stat_json.append("\"Tokens per KV cache block\":" + std::to_string(kv_stats.tokensPerBlock) + ",");
        stat_json.append("\"Used KV cache blocks\":" + std::to_string(kv_stats.usedNumBlocks) + ",");
      }

      stat_json.back() = '}';

      // CLOG4(INFO, stat_json.c_str());
    }
  }
}
} // namespace netease::grps
