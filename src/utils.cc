// Utils.
// Refer to:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/utils.cc

#include "utils.h"

#include <NvInferRuntime.h>
#include <glog/logging.h>
// brpc include after glog
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "logger/logger.h"

using namespace tensorrt_llm::batch_manager;

namespace netease::grps::utils {

std::string Lstrip(const std::string& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  return str.substr(start);
}

std::string Rstrip(const std::string& str) {
  size_t end = str.find_last_not_of(" \t\n\r");
  if (end == std::string::npos) {
    return "";
  }
  return str.substr(0, end + 1);
}

std::string Strip(const std::string& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = str.find_last_not_of(" \t\n\r");
  return str.substr(start, end - start + 1);
}

bool IsValidUTF8(const std::string& str) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  try {
    std::wstring wide = converter.from_bytes(str);
    // check if have "\xef\xbf\xbd" in the string
    if (wide.find(L'\ufffd') != std::wstring::npos) {
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

size_t GetWordCount(const std::string& str, const std::string& word) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = str.find(word, pos)) != std::string::npos) {
    count++;
    pos += word.length();
  }
  return count;
}

size_t ReplaceWorld(
  std::string& str, const std::string& word, const std::string& replace, size_t beg_pos, size_t count) {
  size_t pos = beg_pos;
  while (count > 0 && (pos = str.find(word, pos)) != std::string::npos) {
    str.replace(pos, word.length(), replace);
    pos += replace.length();
    count--;
  }
  return pos;
}

template <>
std::string DownloadFile(const std::string& url, int timeout_s) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = timeout_s * 1000;
  options.connect_timeout_ms = 3000;

  if (channel.Init(url.c_str(), &options) != 0) {
    CLOG4(ERROR, "Fail to initialize channel to " + url);
    throw std::runtime_error("Fail to initialize channel to " + url);
  }

  brpc::Controller cntl;
  cntl.http_request().uri() = url;
  cntl.http_request().set_method(brpc::HTTP_METHOD_GET);

  channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
  if (cntl.Failed()) {
    CLOG4(ERROR, "Fail to call " + url + ", error: " + cntl.ErrorText());
    throw std::runtime_error("Fail to call " + url + ", error: " + cntl.ErrorText());
  }

  if (cntl.response_attachment().size() == 0) {
    CLOG4(ERROR, "Downloaded data is empty from " + url);
    throw std::runtime_error("Downloaded data is empty from " + url);
  }

  return cntl.response_attachment().to_string();
}

template <>
std::vector<char> DownloadFile(const std::string& url, int timeout_s) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = timeout_s * 1000;
  options.connect_timeout_ms = 3000;

  if (channel.Init(url.c_str(), &options) != 0) {
    CLOG4(ERROR, "Fail to initialize channel to " + url);
    throw std::runtime_error("Fail to initialize channel to " + url);
  }

  brpc::Controller cntl;
  cntl.http_request().uri() = url;
  cntl.http_request().set_method(brpc::HTTP_METHOD_GET);

  channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
  if (cntl.Failed()) {
    CLOG4(ERROR, "Fail to call " + url + ", error: " + cntl.ErrorText());
    throw std::runtime_error("Fail to call " + url + ", error: " + cntl.ErrorText());
  }

  std::vector<char> data(cntl.response_attachment().size());
  if (data.empty()) {
    CLOG4(ERROR, "Downloaded data is empty from " + url);
    throw std::runtime_error("Downloaded data is empty from " + url);
  }
  cntl.response_attachment().copy_to_cstr(data.data(), data.size());

  return data;
}

template <>
std::string LoadBytesFromFile(const std::string& path) {
  if (std::filesystem::exists(path) == false) {
    CLOG4(ERROR, "File not found: " + path);
    throw std::invalid_argument("File not found: " + path);
  }
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    CLOG4(ERROR, "Cannot open " + path);
    throw std::invalid_argument("Cannot open" + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());

  if (size == 0) {
    CLOG4(ERROR, "File is empty: " + path);
    throw std::invalid_argument("File is empty: " + path);
  }

  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), long(size));
  return data;
}

template <>
std::vector<char> LoadBytesFromFile(const std::string& path) {
  if (std::filesystem::exists(path) == false) {
    CLOG4(ERROR, "File not found: " + path);
    throw std::invalid_argument("File not found: " + path);
  }
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    CLOG4(ERROR, "Cannot open " + path);
    throw std::invalid_argument("Cannot open" + path);
  }
  std::vector<char> data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());

  if (size == 0) {
    CLOG4(ERROR, "File is empty: " + path);
    throw std::invalid_argument("File is empty: " + path);
  }

  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), long(size));
  return data;
}

void SetHttpResponse(GrpsContext& grps_ctx,
                     int status_code,
                     const std::string& content_type,
                     const std::string& content) {
  auto* http_controller = grps_ctx.http_controller();
  if (http_controller == nullptr) {
    return;
  }
  http_controller->http_response().set_status_code(status_code);
  http_controller->http_response().set_content_type(content_type);
  http_controller->response_attachment().append(content);
}

void HttpRespondWithOpenAi(GrpsContext& grps_ctx,
                           uint64_t request_id,
                           uint64_t created_timestamp,
                           const std::string& model,
                           const std::string& generated_text,
                           size_t prompt_tokens_size,
                           size_t generated_tokens_size,
                           bool func_call,
                           LLMStyler* llm_styler) {
  rapidjson::Document json_body(rapidjson::kObjectType);
  std::string id = "chatcmpl-" + std::to_string(request_id);
  json_body.AddMember("id", rapidjson::Value(id.c_str(), json_body.GetAllocator()), json_body.GetAllocator());
  json_body.AddMember("object", "chat.completion", json_body.GetAllocator());
  json_body.AddMember("created", created_timestamp, json_body.GetAllocator());
  json_body.AddMember("model", rapidjson::Value(model.c_str(), json_body.GetAllocator()), json_body.GetAllocator());
  json_body.AddMember("system_fingerprint", SYSTEM_FINGERPRINT, json_body.GetAllocator());

  json_body.AddMember("choices", rapidjson::Value(rapidjson::kArrayType), json_body.GetAllocator());
  auto& choices = json_body["choices"];
  choices.PushBack(rapidjson::Value(rapidjson::kObjectType), json_body.GetAllocator());
  auto& choice = choices[0];
  choice.AddMember("index", 0, json_body.GetAllocator());
  choice.AddMember("message", rapidjson::Value(rapidjson::kObjectType), json_body.GetAllocator());
  auto& message = choice["message"];
  message.AddMember("role", "assistant", json_body.GetAllocator());
  std::string finish_reason = "stop";
  if (func_call) {
    try {
      finish_reason = llm_styler->ParseFunctionCall(generated_text, request_id, message, json_body.GetAllocator());
    } catch (const std::exception& e) {
      CLOG4(ERROR, "Parse function call failed: " << e.what());
      HttpRespondErrorWithOpenAi(grps_ctx, 500, "Parse function call failed: " + std::string(e.what()));
    }
  } else {
    message.AddMember("content", rapidjson::Value(generated_text.c_str(), json_body.GetAllocator()),
                      json_body.GetAllocator());
  }
  choice.AddMember("logprobs", rapidjson::Value(), json_body.GetAllocator());
  choice.AddMember("finish_reason", rapidjson::Value(finish_reason.c_str(), json_body.GetAllocator()),
                   json_body.GetAllocator());

  json_body.AddMember("usage", rapidjson::Value(rapidjson::kObjectType), json_body.GetAllocator());
  auto& usage = json_body["usage"];
  usage.AddMember("prompt_tokens", prompt_tokens_size, json_body.GetAllocator());
  usage.AddMember("completion_tokens", generated_tokens_size, json_body.GetAllocator());
  usage.AddMember("total_tokens", prompt_tokens_size + generated_tokens_size, json_body.GetAllocator());

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  writer.SetIndent(' ', 1);
  json_body.Accept(writer);

  SetHttpResponse(grps_ctx, 200, "application/json", buffer.GetString());
}

void HttpStreamingRespondWithOpenAi(GrpsContext& grps_ctx,
                                    uint64_t request_id,
                                    uint64_t created_timestamp,
                                    const std::string& model,
                                    const std::string& generated_text,
                                    bool first,
                                    bool final,
                                    bool func_call,
                                    LLMStyler* llm_styler) {
  // CLOG4(INFO, "HttpStreamingRespondWithOpenAi, generated_text: " << generated_text << ", first: " << first
  //                                                                << ", final: " << final);
  rapidjson::Document json_body(rapidjson::kObjectType);
  std::string id = "chatcmpl-" + std::to_string(request_id);
  json_body.AddMember("id", rapidjson::Value(id.c_str(), json_body.GetAllocator()), json_body.GetAllocator());
  json_body.AddMember("object", "chat.completion.chunk", json_body.GetAllocator());
  json_body.AddMember("created", created_timestamp, json_body.GetAllocator());
  json_body.AddMember("model", rapidjson::Value(model.c_str(), json_body.GetAllocator()), json_body.GetAllocator());
  json_body.AddMember("system_fingerprint", SYSTEM_FINGERPRINT, json_body.GetAllocator());

  json_body.AddMember("choices", rapidjson::Value(rapidjson::kArrayType), json_body.GetAllocator());
  auto& choices = json_body["choices"];
  choices.PushBack(rapidjson::Value(rapidjson::kObjectType), json_body.GetAllocator());
  auto& choice = choices[0];
  choice.AddMember("index", 0, json_body.GetAllocator());
  choice.AddMember("delta", rapidjson::Value(rapidjson::kObjectType), json_body.GetAllocator());
  if (func_call) {
    auto& message = choice["delta"];
    std::string finish_reason;
    if (first) {
      message.AddMember("role", "assistant", json_body.GetAllocator());
      message.AddMember("content", "", json_body.GetAllocator());
    } else {
      try {
        finish_reason = llm_styler->ParseFunctionCall(generated_text, request_id, message, json_body.GetAllocator());
      } catch (const std::exception& e) {
        CLOG4(ERROR, "Parse function call failed: " << e.what());
        HttpStreamingRespondErrorWithOpenAi(grps_ctx, "Parse function call failed: " + std::string(e.what()));
      }
    }
    choice.AddMember("logprobs", rapidjson::Value(), json_body.GetAllocator());
    if (finish_reason.empty()) {
      choice.AddMember("finish_reason", rapidjson::Value(), json_body.GetAllocator());
    } else {
      choice.AddMember("finish_reason", rapidjson::Value(finish_reason.c_str(), json_body.GetAllocator()),
                       json_body.GetAllocator());
    }
  } else {
    if (!final) {
      auto& message = choice["delta"];
      if (first) {
        message.AddMember("role", "assistant", json_body.GetAllocator());
      }
      message.AddMember("content", rapidjson::Value(generated_text.c_str(), json_body.GetAllocator()),
                        json_body.GetAllocator());
    }
    choice.AddMember("logprobs", rapidjson::Value(), json_body.GetAllocator());
    if (final) {
      choice.AddMember("finish_reason", "stop", json_body.GetAllocator());
    } else {
      choice.AddMember("finish_reason", rapidjson::Value(), json_body.GetAllocator());
    }
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json_body.Accept(writer);

  std::string content = "data: ";
  content.append(buffer.GetString());
  content.append("\n\n");

  grps_ctx.CustomizedHttpStreamingRespond(content.c_str(), content.size(), final);
}

static executor::OutputConfig GetOutputConfigFromJsonBody(const rapidjson::Document& json_body) {
  bool return_log_probs{false};
  if (json_body.HasMember(InputFieldsNames::kReturnLogProbs) && json_body[InputFieldsNames::kReturnLogProbs].IsBool()) {
    return_log_probs = json_body[InputFieldsNames::kReturnLogProbs].GetBool();
  }
  bool return_generation_logits = false;
  bool return_context_logits = false;
  // Note that currently excludeInputFromOutput is set from the backend parameters.
  return executor::OutputConfig(return_log_probs, return_generation_logits, return_context_logits);
}

static executor::SamplingConfig GetSamplingConfigFromJsonBody(const rapidjson::Document& json_body,
                                                              const std::string& model,
                                                              const executor::SamplingConfig& def_sampling_config) {
  int32_t beam_width = def_sampling_config.getBeamWidth();
  if (json_body.HasMember(InputFieldsNames::kBeamWidth)) {
    if (!json_body[InputFieldsNames::kBeamWidth].IsInt()) {
      throw std::invalid_argument("`beam_width` is not an integer");
    }
    beam_width = json_body[InputFieldsNames::kBeamWidth].GetInt();
  }

  std::optional<executor::SizeType32> top_k = def_sampling_config.getTopK();
  if (json_body.HasMember(InputFieldsNames::kTopK)) {
    if (!json_body[InputFieldsNames::kTopK].IsInt()) {
      throw std::invalid_argument("`top_k` is not an integer");
    }
    top_k = json_body[InputFieldsNames::kTopK].GetInt();
  }

  std::optional<float> top_p = def_sampling_config.getTopP();
  if (json_body.HasMember(InputFieldsNames::kTopP)) {
    if (!json_body[InputFieldsNames::kTopP].IsFloat()) {
      throw std::invalid_argument("`top_p` is not a float");
    }
    top_p = json_body[InputFieldsNames::kTopP].GetFloat();
  }
  if (top_p.has_value() && top_p.value() <= 0.F) {
    top_p.reset();
  }

  std::optional<float> top_p_min = def_sampling_config.getTopPMin();
  if (json_body.HasMember(InputFieldsNames::kTopPMin)) {
    if (!json_body[InputFieldsNames::kTopPMin].IsFloat()) {
      throw std::invalid_argument("`top_p_min` is not a float");
    }
    top_p_min = json_body[InputFieldsNames::kTopPMin].GetFloat();
  }

  std::optional<float> top_p_decay = def_sampling_config.getTopPDecay();
  if (json_body.HasMember(InputFieldsNames::kTopPDecay)) {
    if (!json_body[InputFieldsNames::kTopPDecay].IsFloat()) {
      throw std::invalid_argument("`top_p_decay` is not a float");
    }
    top_p_decay = json_body[InputFieldsNames::kTopPDecay].GetFloat();
  }

  std::optional<int32_t> top_p_reset_ids = def_sampling_config.getTopPResetIds();
  if (json_body.HasMember(InputFieldsNames::kTopPResetIds)) {
    if (!json_body[InputFieldsNames::kTopPResetIds].IsInt()) {
      throw std::invalid_argument("`top_p_reset_ids` is not an integer");
    }
    top_p_reset_ids = json_body[InputFieldsNames::kTopPResetIds].GetInt();
  }

  std::optional<float> temperature = def_sampling_config.getTemperature();
  if (json_body.HasMember(InputFieldsNames::kTemperature)) {
    if (!json_body[InputFieldsNames::kTemperature].IsFloat()) {
      throw std::invalid_argument("`temperature` is not a float");
    }
    temperature = json_body[InputFieldsNames::kTemperature].GetFloat();
  }

  std::optional<int32_t> early_stopping = def_sampling_config.getEarlyStopping();
  if (json_body.HasMember(InputFieldsNames::kEarlyStopping)) {
    if (!json_body[InputFieldsNames::kEarlyStopping].IsInt()) {
      throw std::invalid_argument("`early_stopping` is not an integer");
    }
    early_stopping = json_body[InputFieldsNames::kEarlyStopping].GetInt();
  }

  std::optional<int32_t> min_length = def_sampling_config.getMinTokens();
  if (json_body.HasMember(InputFieldsNames::kMinLength)) {
    if (!json_body[InputFieldsNames::kMinLength].IsInt()) {
      throw std::invalid_argument("`min_length` is not an integer");
    }
    min_length = json_body[InputFieldsNames::kMinLength].GetInt();
  }

  std::optional<float> beam_search_diversity_rate = def_sampling_config.getBeamSearchDiversityRate();
  if (json_body.HasMember(InputFieldsNames::kBeamSearchDiversityRate)) {
    if (!json_body[InputFieldsNames::kBeamSearchDiversityRate].IsFloat()) {
      throw std::invalid_argument("`beam_search_diversity_rate` is not a float");
    }
    beam_search_diversity_rate = json_body[InputFieldsNames::kBeamSearchDiversityRate].GetFloat();
  }

  std::optional<float> length_penalty = def_sampling_config.getLengthPenalty();
  if (json_body.HasMember(InputFieldsNames::kLengthPenalty)) {
    if (!json_body[InputFieldsNames::kLengthPenalty].IsFloat()) {
      throw std::invalid_argument("`length_penalty` is not a float");
    }
    length_penalty = json_body[InputFieldsNames::kLengthPenalty].GetFloat();
  }

  std::optional<float> repetition_penalty = def_sampling_config.getRepetitionPenalty();
  if (json_body.HasMember(InputFieldsNames::kRepetitionPenalty)) {
    if (!json_body[InputFieldsNames::kRepetitionPenalty].IsFloat()) {
      throw std::invalid_argument("`repetition_penalty` is not a float");
    }
    repetition_penalty = json_body[InputFieldsNames::kRepetitionPenalty].GetFloat();
  }

  std::optional<float> presence_penalty = def_sampling_config.getPresencePenalty();
  if (json_body.HasMember(InputFieldsNames::kPresencePenalty)) {
    if (!json_body[InputFieldsNames::kPresencePenalty].IsFloat()) {
      throw std::invalid_argument("`presence_penalty` is not a float");
    }
    presence_penalty = json_body[InputFieldsNames::kPresencePenalty].GetFloat();
  }

  std::optional<float> frequency_penalty = def_sampling_config.getFrequencyPenalty();
  if (json_body.HasMember(InputFieldsNames::kFrequencyPenalty)) {
    if (!json_body[InputFieldsNames::kFrequencyPenalty].IsFloat()) {
      throw std::invalid_argument("`frequency_penalty` is not a float");
    }
    frequency_penalty = json_body[InputFieldsNames::kFrequencyPenalty].GetFloat();
  }

  std::optional<uint64_t> random_seed = def_sampling_config.getSeed();
  if (json_body.HasMember(InputFieldsNames::kRandomSeed)) {
    if (!json_body[InputFieldsNames::kRandomSeed].IsUint64()) {
      throw std::invalid_argument("`seed` is not an unsigned integer");
    }
    random_seed = json_body[InputFieldsNames::kRandomSeed].GetUint64();
  }

  std::optional<int32_t> no_repeat_ngram_size = def_sampling_config.getNoRepeatNgramSize();
  if (json_body.HasMember(InputFieldsNames::kNoRepeatNgramSize)) {
    if (!json_body[InputFieldsNames::kNoRepeatNgramSize].IsUint64()) {
      throw std::invalid_argument("`no_repeat_ngram_size` is not an unsigned integer");
    }
    no_repeat_ngram_size = json_body[InputFieldsNames::kNoRepeatNgramSize].GetInt();
  }

  // CLOG4(
  //   INFO,
  //   "SamplingConfig: "
  //     << "beam_width: " << beam_width << ", top_k: " << (top_k.has_value() ? std::to_string(top_k.value()) : "None")
  //     << ", top_p: " << (top_p.has_value() ? std::to_string(top_p.value()) : "None")
  //     << ", top_p_min: " << (top_p_min.has_value() ? std::to_string(top_p_min.value()) : "None")
  //     << ", top_p_decay: " << (top_p_decay.has_value() ? std::to_string(top_p_decay.value()) : "None")
  //     << ", top_p_reset_ids: " << (top_p_reset_ids.has_value() ? std::to_string(top_p_reset_ids.value()) : "None")
  //     << ", temperature: " << (temperature.has_value() ? std::to_string(temperature.value()) : "None")
  //     << ", min_length: " << (min_length.has_value() ? std::to_string(min_length.value()) : "None")
  //     << ", beam_search_diversity_rate: "
  //     << (beam_search_diversity_rate.has_value() ? std::to_string(beam_search_diversity_rate.value()) : "None")
  //     << ", repetition_penalty: "
  //     << (repetition_penalty.has_value() ? std::to_string(repetition_penalty.value()) : "None")
  //     << ", presence_penalty: " << (presence_penalty.has_value() ? std::to_string(presence_penalty.value()) : "None")
  //     << ", frequency_penalty: " << (frequency_penalty.has_value() ? std::to_string(frequency_penalty.value()) :
  //     "None")
  //     << ", length_penalty: " << (length_penalty.has_value() ? std::to_string(length_penalty.value()) : "None")
  //     << ", early_stopping: " << (early_stopping.has_value() ? std::to_string(early_stopping.value()) : "None")
  //     << ", no_repeat_ngram_size: "
  //     << (no_repeat_ngram_size.has_value() ? std::to_string(no_repeat_ngram_size.value()) : "None"));

  return executor::SamplingConfig(beam_width, top_k, top_p, top_p_min, top_p_reset_ids, top_p_decay, random_seed,
                                  temperature, min_length, beam_search_diversity_rate, repetition_penalty,
                                  presence_penalty, frequency_penalty, length_penalty, early_stopping,
                                  no_repeat_ngram_size);
}

static std::tuple<PtuningEmbeddingTableType, MropeConfType> BuildPromptTuningForImages(
  const std::vector<std::string>& img_urls, VIT* vit, std::string& prompt, executor::VecTokens& token_ids) {
  if (!img_urls.empty() && vit == nullptr) {
    throw std::invalid_argument("There is no vit model.");
  }
  if (vit == nullptr) {
    return {std::nullopt, std::nullopt};
  }
  return vit->Encode(img_urls, prompt, token_ids);
}

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
  const executor::SamplingConfig& def_sampling_config) {
  rapidjson::Document json_body;
  json_body.Parse(http_body.c_str());
  if (json_body.HasParseError()) {
    CLOG4(ERROR, "Parse http json body failed: " << json_body.GetParseError() << ", body: " << http_body);
    throw std::invalid_argument("Parse http json body failed.");
  }

  if (tokenizer == nullptr) {
    throw std::invalid_argument("tokenizer is nullptr");
  }

  // Model id.
  if (!json_body.HasMember(InputFieldsNames::kModelName) || !json_body[InputFieldsNames::kModelName].IsString()) {
    throw std::invalid_argument("`model` field is not present or not a string");
  }
  std::string model = json_body[InputFieldsNames::kModelName].GetString();

  // Prompt input tokens.
  if (json_body.HasMember("tools") && !llm_styler->support_func_call()) {
    throw std::invalid_argument("Function call is not supported for this llm.");
  }
  auto [func_call, prompt, image_urls] = llm_styler->BuildPrompt(json_body);

  // Output config.
  executor::OutputConfig out_config = utils::GetOutputConfigFromJsonBody(json_body);
  out_config.excludeInputFromOutput = exclude_input_from_output;

  // Max tokens.
  executor::SizeType32 max_new_tokens;
  if (json_body.HasMember(InputFieldsNames::kMaxNewTokens) && json_body[InputFieldsNames::kMaxNewTokens].IsInt()) {
    max_new_tokens = json_body[InputFieldsNames::kMaxNewTokens].GetInt();
  } else {
    max_new_tokens = executor::SizeType32(max_output_len);
  }
  if (max_new_tokens <= 0) {
    throw std::invalid_argument("`max_new_tokens` must > 0");
  }

  // End and pad id.
  std::optional<executor::SizeType32> end_id = tokenizer->end_token_id();
  std::optional<executor::SizeType32> pad_id = tokenizer->pad_token_id();

  // Sampling config.
  auto sampling_config = utils::GetSamplingConfigFromJsonBody(json_body, model, def_sampling_config);

  // Bad words.
  std::list<executor::VecTokens> bad_words_list;
  if (json_body.HasMember(InputFieldsNames::kBadWords)) {
    if (json_body[InputFieldsNames::kBadWords].IsString()) {
      std::string bad_words_str = json_body[InputFieldsNames::kBadWords].GetString();
      auto words_id = tokenizer->Encode(bad_words_str, false, false);
      bad_words_list.emplace_back(words_id);
    } else if (json_body[InputFieldsNames::kBadWords].IsArray()) {
      for (auto& bad : json_body[InputFieldsNames::kBadWords].GetArray()) {
        if (bad.IsString()) {
          auto bad_words_str = bad.GetString();
          auto words_id = tokenizer->Encode(bad_words_str, false, false);
          bad_words_list.emplace_back(words_id);
        } else {
          throw std::invalid_argument("`bad_words` is not a string or an array of strings");
        }
      }
    } else {
      throw std::invalid_argument("`bad_words` is not a string or an array of strings");
    }
  }
  for (auto& bad_word : bad_words) {
    bad_words_list.emplace_back(tokenizer->Encode(bad_word, false, false));
  }

  // Stop words.
  std::list<executor::VecTokens> stop_words_list;
  if (json_body.HasMember(InputFieldsNames::kStopWords)) {
    if (json_body[InputFieldsNames::kStopWords].IsString()) {
      std::string stop_words_str = json_body[InputFieldsNames::kStopWords].GetString();
      if (!stop_words_str.empty()) {
        auto words_id = tokenizer->Encode(stop_words_str, false, false);
        stop_words_list.emplace_back(words_id);
      }
    } else if (json_body[InputFieldsNames::kStopWords].IsArray()) {
      for (auto& stop : json_body[InputFieldsNames::kStopWords].GetArray()) {
        if (stop.IsString()) {
          std::string stop_words_str = stop.GetString();
          if (!stop_words_str.empty()) {
            auto words_id = tokenizer->Encode(stop_words_str, false, false);
            stop_words_list.emplace_back(words_id);
          }
        } else {
          throw std::invalid_argument("`stop` is not a string or an array of strings");
        }
      }
    } else {
      throw std::invalid_argument("`stop` is not a string or an array of strings");
    }
  }
  for (auto& stop_word : stop_words) {
    stop_words_list.emplace_back(tokenizer->Encode(stop_word, false, false));
  }
  if (func_call &&
      !llm_styler->func_call_observation_words().empty()) { // Add function call observation words to early stop.
    stop_words_list.emplace_back(tokenizer->Encode(llm_styler->func_call_observation_words(), false, false));
  }

  executor::VecTokens input_tokens;

  std::optional<executor::Tensor> embedding_bias{std::nullopt};
  auto [p_tuning_config, rope_config] = BuildPromptTuningForImages(image_urls, vit, prompt, input_tokens);
  if (input_tokens.empty()) {
    input_tokens = tokenizer->Encode(prompt);
  }

  std::optional<executor::LoraConfig> lora_config{std::nullopt};
  std::optional<executor::ExternalDraftTokensConfig> external_draft_tokens_config{std::nullopt};

  std::optional<executor::VecTokens> encoder_input_tokens{std::nullopt};

  if (model_type == executor::ModelType::kENCODER_ONLY || model_type == executor::ModelType::kENCODER_DECODER) {
    encoder_input_tokens = input_tokens;
    if (!pad_id) {
      input_tokens = {pad_id.value()};
    }
  }

  std::optional<std::vector<executor::SizeType32>> position_ids{std::nullopt};
  std::optional<executor::LookaheadDecodingConfig> lookahead_config{std::nullopt};
  std::optional<executor::KvCacheRetentionConfig> kv_cache_retention_config{std::nullopt};
  std::optional<std::string> logits_post_processor_name{std::nullopt};
  std::optional<executor::IdType> client_id{std::nullopt};
  bool return_all_generated_tokens = false;
  executor::PriorityType priority = executor::Request::kDefaultPriority;
  executor::RequestType type = executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION;
  std::optional<executor::ContextPhaseParams> context_phase_params{std::nullopt};
  std::optional<executor::Tensor> encoder_input_features{std::nullopt};
  std::optional<executor::SizeType32> encoder_output_length{std::nullopt};
  std::optional<executor::Tensor> cross_attention_mask{std::nullopt};
  executor::SizeType32 num_return_sequences = 1;
  std::optional<executor::EagleConfig> eagle_config{std::nullopt};
  std::optional<executor::Tensor> skip_cross_attn_blocks{std::nullopt};

  return {func_call, std::move(model),
          executor::Request{input_tokens,
                            max_new_tokens,
                            !func_call && streaming,
                            sampling_config,
                            out_config,
                            end_id,
                            pad_id,
                            position_ids,
                            bad_words_list.empty() ? std::nullopt : std::make_optional(bad_words_list),
                            stop_words_list.empty() ? std::nullopt : std::make_optional(stop_words_list),
                            embedding_bias,
                            external_draft_tokens_config,
                            p_tuning_config,
                            rope_config,
                            lora_config,
                            lookahead_config,
                            kv_cache_retention_config,
                            logits_post_processor_name,
                            encoder_input_tokens,
                            client_id,
                            return_all_generated_tokens,
                            priority,
                            type,
                            context_phase_params,
                            encoder_input_features,
                            encoder_output_length,
                            cross_attention_mask,
                            num_return_sequences,
                            eagle_config,
                            skip_cross_attn_blocks}};
}

} // namespace netease::grps::utils
