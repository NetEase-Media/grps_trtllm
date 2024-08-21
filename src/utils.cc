// Utils.
// Porting from:
// https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/src/utils.cc

#include "utils.h"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "logger/logger.h"

using namespace tensorrt_llm::batch_manager;

namespace netease::grps::utils {

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

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    CLOG4(FATAL, "Cannot open" + path);
    throw std::invalid_argument("Cannot open" + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
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
    finish_reason = llm_styler->ParseFunctionCall(generated_text, request_id, message, json_body.GetAllocator());
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
      finish_reason = llm_styler->ParseFunctionCall(generated_text, request_id, message, json_body.GetAllocator());
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
                                                              const std::string& model) {
  int32_t beam_width = 1;

  std::optional<executor::SizeType32> top_k{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kTopK)) {
    if (!json_body[InputFieldsNames::kTopK].IsInt()) {
      throw std::invalid_argument("`top_k` is not an integer");
    }
    top_k = json_body[InputFieldsNames::kTopK].GetInt();
  }

  std::optional<float> top_p{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kTopP)) {
    if (!json_body[InputFieldsNames::kTopP].IsFloat()) {
      throw std::invalid_argument("`top_p` is not a float");
    }
    top_p = json_body[InputFieldsNames::kTopP].GetFloat();
  }
  if (top_p.has_value() && top_p.value() <= 0.F) {
    top_p.reset();
  }

  std::optional<float> top_p_min{std::nullopt};
  std::optional<float> top_p_decay{std::nullopt};
  std::optional<int32_t> top_p_reset_ids{std::nullopt};

  std::optional<float> temperature{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kTemperature)) {
    if (!json_body[InputFieldsNames::kTemperature].IsFloat()) {
      throw std::invalid_argument("`temperature` is not a float");
    }
    temperature = json_body[InputFieldsNames::kTemperature].GetFloat();
  }

  std::optional<float> length_penalty = 1.0f;
  std::optional<int32_t> early_stopping{std::nullopt};
  std::optional<float> repetition_penalty = 1.0f;
  std::optional<int32_t> min_length{std::nullopt};
  std::optional<float> beam_search_diversity_rate{std::nullopt};

  std::optional<float> presence_penalty{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kPresencePenalty)) {
    if (!json_body[InputFieldsNames::kPresencePenalty].IsFloat()) {
      throw std::invalid_argument("`presence_penalty` is not a float");
    }
    presence_penalty = json_body[InputFieldsNames::kPresencePenalty].GetFloat();
  }

  std::optional<float> frequency_penalty{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kFrequencyPenalty)) {
    if (!json_body[InputFieldsNames::kFrequencyPenalty].IsFloat()) {
      throw std::invalid_argument("`frequency_penalty` is not a float");
    }
    frequency_penalty = json_body[InputFieldsNames::kFrequencyPenalty].GetFloat();
  }

  std::optional<uint64_t> random_seed{std::nullopt};
  if (json_body.HasMember(InputFieldsNames::kRandomSeed)) {
    if (!json_body[InputFieldsNames::kRandomSeed].IsUint64()) {
      throw std::invalid_argument("`seed` is not an unsigned integer");
    }
    random_seed = json_body[InputFieldsNames::kRandomSeed].GetUint64();
  }

  return executor::SamplingConfig(beam_width, top_k, top_p, top_p_min, top_p_reset_ids, top_p_decay, random_seed,
                                  temperature, min_length, beam_search_diversity_rate, repetition_penalty,
                                  presence_penalty, frequency_penalty, length_penalty, early_stopping);
}

std::tuple<bool, std::string, executor::Request> CreateRequestFromOpenAiHttpBody(const std::string& http_body,
                                                                                 bool exclude_input_from_output,
                                                                                 bool streaming,
                                                                                 LLMStyler* llm_styler,
                                                                                 MultiInstanceTokenizer* tokenizer,
                                                                                 size_t max_output_len) {
  rapidjson::Document json_body;
  json_body.Parse(http_body.c_str());
  if (json_body.HasParseError()) {
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
  auto [func_call, prompt] = llm_styler->BuildPrompt(json_body);
  // CLOG4(INFO, "Prompt: " << prompt);
  executor::VecTokens input_tokens = tokenizer->Encode(prompt);

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
  auto sampling_config = utils::GetSamplingConfigFromJsonBody(json_body, model);

  // Bad words.
  std::list<executor::VecTokens> bad_words_list;
  if (json_body.HasMember(InputFieldsNames::kBadWords)) {
    if (json_body[InputFieldsNames::kBadWords].IsString()) {
      std::string bad_words_str = json_body[InputFieldsNames::kBadWords].GetString();
      auto bad_words = tokenizer->Encode(bad_words_str);
      bad_words_list.emplace_back(bad_words);
    } else if (json_body[InputFieldsNames::kBadWords].IsArray()) {
      for (auto& bad : json_body[InputFieldsNames::kBadWords].GetArray()) {
        if (bad.IsString()) {
          auto bad_words_str = bad.GetString();
          auto bad_words = tokenizer->Encode(bad_words_str);
          bad_words_list.emplace_back(bad_words);
        } else {
          throw std::invalid_argument("`bad_words` is not a string or an array of strings");
        }
      }
    } else {
      throw std::invalid_argument("`bad_words` is not a string or an array of strings");
    }
  }
  for (auto& bad_words : tokenizer->bad_words()) {
    bad_words_list.emplace_back(tokenizer->Encode(bad_words));
  }

  // Stop words.
  std::list<executor::VecTokens> stop_words_list;
  if (json_body.HasMember(InputFieldsNames::kStopWords)) {
    if (json_body[InputFieldsNames::kStopWords].IsString()) {
      std::string stop_words_str = json_body[InputFieldsNames::kStopWords].GetString();
      auto stop_words = tokenizer->Encode(stop_words_str);
      stop_words_list.emplace_back(stop_words);
    } else if (json_body[InputFieldsNames::kStopWords].IsArray()) {
      for (auto& stop : json_body[InputFieldsNames::kStopWords].GetArray()) {
        if (stop.IsString()) {
          auto stop_words_str = stop.GetString();
          auto stop_words = tokenizer->Encode(stop_words_str);
          stop_words_list.emplace_back(stop_words);
        } else {
          throw std::invalid_argument("`stop` is not a string or an array of strings");
        }
      }
    } else {
      throw std::invalid_argument("`stop` is not a string or an array of strings");
    }
  }
  for (auto& stop_words : tokenizer->stop_words()) {
    stop_words_list.emplace_back(tokenizer->Encode(stop_words));
  }
  if (func_call) { // Add function call observation words to early stop.
    stop_words_list.emplace_back(tokenizer->Encode(llm_styler->func_call_observation_words()));
  }

  // [TODO]: Support embedding_bias, p_tuning_config, lora_config, speculative_decoding_config
  std::optional<executor::Tensor> embedding_bias{std::nullopt};
  std::optional<executor::PromptTuningConfig> p_tuning_config{std::nullopt};
  std::optional<executor::LoraConfig> lora_config{std::nullopt};
  std::optional<executor::SpeculativeDecodingConfig> speculative_decoding_config{std::nullopt};

  return {func_call, std::move(model),
          executor::Request{input_tokens, max_new_tokens, func_call ? false : streaming, sampling_config, out_config,
                            end_id, pad_id, bad_words_list.empty() ? std::nullopt : std::make_optional(bad_words_list),
                            stop_words_list.empty() ? std::nullopt : std::make_optional(stop_words_list),
                            embedding_bias, speculative_decoding_config, p_tuning_config, lora_config, std::nullopt}};
}

} // namespace netease::grps::utils
