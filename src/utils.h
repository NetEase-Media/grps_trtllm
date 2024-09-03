// Utils.
// Porting from:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/utils.h

#pragma once

#include <jinja2cpp/template.h>
#include <rapidjson/document.h>

#include <chrono>
#include <map>
#include <string>
#include <unordered_set>

#include "context/context.h"
#include "model_infer/tensor_wrapper.h"
#include "src/constants.h"
#include "src/llm_styler.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tokenizer.h"

using namespace tensorrt_llm;

namespace netease::grps {

#define SET_TIMESTAMP(TS_NS)                                                                                          \
  {                                                                                                                   \
    TS_NS = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()) \
              .count();                                                                                               \
  }
#define DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;             \
  SET_TIMESTAMP(TS_NS);

namespace utils {
bool IsValidUTF8(const std::string& str);

std::string LoadBytesFromFile(const std::string& path);

jinja2::Value RapidJson2JinjaVal(const rapidjson::GenericValue<rapidjson::UTF8<>>& json_val);

void SetHttpResponse(GrpsContext& grps_ctx,
                     int status_code,
                     const std::string& content_type,
                     const std::string& content);

/// @brief Respond error http response with OpenAI format.
inline void HttpRespondErrorWithOpenAi(GrpsContext& grps_ctx, int status_code, const std::string& error_msg) {
  return SetHttpResponse(grps_ctx, status_code, "application/json", R"({"error": ")" + error_msg + "\"}");
}

/// @brief Streaming respond error http response with OpenAI format.
inline void HttpStreamingRespondErrorWithOpenAi(GrpsContext& grps_ctx, const std::string& error_msg) {
  std::string content = R"({"error": ")" + error_msg + "\"}";
  grps_ctx.CustomizedHttpStreamingRespond(content.c_str(), content.size(), true);
}

/// @brief Respond normal http response with OpenAI format.
void HttpRespondWithOpenAi(GrpsContext& grps_ctx,
                           uint64_t request_id,
                           uint64_t created_timestamp,
                           const std::string& model,
                           const std::string& generated_text,
                           size_t prompt_tokens_size,
                           size_t generated_tokens_size,
                           bool func_call,
                           LLMStyler* llm_styler);

/// @brief Streaming respond normal http response with OpenAI format.
void HttpStreamingRespondWithOpenAi(GrpsContext& grps_ctx,
                                    uint64_t request_id,
                                    uint64_t created_timestamp,
                                    const std::string& model,
                                    const std::string& generated_text,
                                    bool first,
                                    bool final,
                                    bool func_call,
                                    LLMStyler* llm_styler);

/// @brief Construct <if_func_call, model_id, executor::Request> from OpenAI format http body
std::tuple<bool, std::string, executor::Request> CreateRequestFromOpenAiHttpBody(
  const std::string& http_body,
  bool exclude_input_from_output,
  bool streaming,
  LLMStyler* llm_styler,
  MultiInstanceTokenizer* tokenizer,
  const std::unordered_set<std::string>& stop_words,
  const std::unordered_set<std::string>& bad_words,
  size_t max_output_len,
  executor::ModelType model_type);
} // namespace utils
} // namespace netease::grps
