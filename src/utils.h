// Utils.
// Refer to:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/utils.h

#pragma once

#include <rapidjson/document.h>

#include <chrono>
#include <map>
#include <string>
#include <unordered_set>

#include "context/context.h"
#include "src/constants.h"
#include "src/llm_styler.h"
#include "src/tokenizer.h"
#include "src/vit/vit.h"
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

#define GET_SYS_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

#define GET_SYS_TIME_MS() \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

namespace utils {
// --------------------------------------- String utils [BEGIN] ---------------------------------------
static inline bool StartsWith(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}
static inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
std::string Lstrip(const std::string& str);
std::string Rstrip(const std::string& str);
std::string Strip(const std::string& str);
bool IsValidUTF8(const std::string& str);
// Get word count in string.
size_t GetWordCount(const std::string& str, const std::string& word);
/**
 * Replace word to new word in string.
 * @param str: string to replace.
 * @param word: word to replace.
 * @param replace: new word.
 * @param beg_pos: begin position.
 * @param count: replace count.
 * @return final position.
 */
size_t ReplaceWorld(
  std::string& str, const std::string& word, const std::string& replace, size_t beg_pos = 0, size_t count = 1);

// --------------------------------------- Http utils [BEGIN] ---------------------------------------
template <typename T>
T DownloadFile(const std::string& url, int timeout_s = 10);
template <>
std::string DownloadFile(const std::string& url, int timeout_s);
template <>
std::vector<char> DownloadFile(const std::string& url, int timeout_s);

// --------------------------------------- File utils [BEGIN] ---------------------------------------
template <typename T>
T LoadBytesFromFile(const std::string& path);
template <>
std::string LoadBytesFromFile(const std::string& path);
template <>
std::vector<char> LoadBytesFromFile(const std::string& path);

// --------------------------------------- OpenAI http interface [BEGIN] ---------------------------------------
void SetHttpResponse(GrpsContext& grps_ctx,
                     int status_code,
                     const std::string& content_type,
                     const std::string& content);

/// @brief Respond error http response with OpenAI format.
inline void HttpRespondErrorWithOpenAi(GrpsContext& grps_ctx, int status_code, const std::string& error_msg) {
  grps_ctx.set_err_msg(error_msg);
  return SetHttpResponse(grps_ctx, status_code, "application/json", R"({"error": {"message": ")" + error_msg + "\"}}");
}

/// @brief Streaming respond error http response with OpenAI format.
inline void HttpStreamingRespondErrorWithOpenAi(GrpsContext& grps_ctx, const std::string& error_msg) {
  grps_ctx.set_err_msg(error_msg);

  std::string content;
  if (error_msg.length() > 512) {
    content = R"(data: {"error": {"message": ")" + error_msg.substr(0, 512) + "......\"}}\n\n";
  } else {
    content = R"(data: {"error": {"message": ")" + error_msg + "\"}}\n\n";
  }
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
  VIT* vit,
  const std::unordered_set<std::string>& stop_words,
  const std::unordered_set<std::string>& bad_words,
  size_t max_output_len,
  executor::ModelType model_type,
  const executor::SamplingConfig& def_sampling_config);
} // namespace utils
} // namespace netease::grps
