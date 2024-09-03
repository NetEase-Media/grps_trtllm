// Customized deep learning model inferer. Including model load and model infer.

#include "customized_inferer.h"

#include <rapidjson/document.h>

#include "logger/logger.h"
#include "utils.h"

namespace netease::grps {
TrtllmInferer::TrtllmInferer() = default;
TrtllmInferer::~TrtllmInferer() = default;

void TrtllmInferer::Init(const std::string& path, const std::string& device, const YAML::Node& args) {
  ModelInferer::Init(path, device, args);
  model_state_ = std::make_unique<TrtLlmModelState>(args);
  CLOG4(INFO, "TrtllmInferer init success. args: " << args);
}

void TrtllmInferer::Load() {
  try {
    tokenizer_ = std::make_unique<MultiInstanceTokenizer>();

    std::string tokenizer_type = model_state_->GetParameter<std::string>("tokenizer_type");
    if (tokenizer_type.empty()) {
      throw std::invalid_argument("tokenizer_type is not specified.");
    }
    std::string tokenizer_path = model_state_->GetParameter<std::string>("tokenizer_path");
    if (tokenizer_path.empty()) {
      throw std::invalid_argument("tokenizer_path is not specified.");
    }

    int instance_num = 1;
    try {
      instance_num = model_state_->GetParameter<int32_t>("tokenizer_parallelism");
    } catch (const std::exception& e) {
      CLOG4(WARN, "tokenizer_parallelism is not specified, will use default value of 1.");
    }

    std::optional<int32_t> end_token_id = std::nullopt;
    try {
      end_token_id = model_state_->GetParameter<int32_t>("end_token_id");
    } catch (const std::exception& e) {
      CLOG4(WARN, "end_token_id is not specified, will use default value of null.");
    }

    std::optional<int32_t> pad_token_id = std::nullopt;
    try {
      pad_token_id = model_state_->GetParameter<int32_t>("pad_token_id");
    } catch (const std::exception& e) {
      CLOG4(WARN, "pad_token_id is not specified, will use default value of null.");
    }

    std::vector<int32_t> skip_special_tokens;
    try {
      skip_special_tokens = model_state_->GetParameter<std::vector<int32_t>>("skip_special_tokens");
    } catch (const std::exception& e) {
      CLOG4(WARN, "skip_special_tokens is not specified, will use default value of true.");
    }

    std::unordered_map<std::string, int32_t> force_tokens_dict;
    try {
      const auto& model_config = model_state_->GetModelConfig();
      if (model_config["force_tokens_dict"] && model_config["force_tokens_dict"].IsSequence()) {
        for (const auto& item : model_config["force_tokens_dict"]) {
          if (item.IsMap() && item["token"] && item["id"]) {
            force_tokens_dict[item["token"].as<std::string>()] = item["id"].as<int32_t>();
          } else {
            throw std::invalid_argument("Failed to parse force_tokens_dict, not a sequence of <token, id> map.");
          }
        }
      } else {
        throw std::invalid_argument("Failed to parse force_tokens_dict, not a sequence of <token, id> map.");
      }
    } catch (const std::exception& e) {
      CLOG4(WARN, "force_tokens_dict is not specified, will use default value of empty.");
    }

    std::vector<int32_t> prefix_tokens_id;
    try {
      prefix_tokens_id = model_state_->GetParameter<std::vector<int32_t>>("prefix_tokens_id");
    } catch (const std::exception& e) {
      CLOG4(WARN, "prefix_tokens_id is not specified, will use default value of empty.");
    }

    std::vector<int32_t> suffix_tokens_id;
    try {
      suffix_tokens_id = model_state_->GetParameter<std::vector<int32_t>>("suffix_tokens_id");
    } catch (const std::exception& e) {
      CLOG4(WARN, "suffix_tokens_id is not specified, will use default value of empty.");
    }

    tokenizer_->Load(tokenizer_type, tokenizer_path, instance_num, pad_token_id, end_token_id, skip_special_tokens,
                     force_tokens_dict, prefix_tokens_id, suffix_tokens_id);
  } catch (const std::exception& e) {
    throw InfererException("Load tokenizer failed: " + std::string(e.what()));
  }

  std::string llm_style = model_state_->GetParameter<std::string>("llm_style");
  llm_styler_ = LLMStylerFactory::CreateLLMStyler(llm_style);

  trtllm_instance_ = std::make_unique<TrtLlmModelInstance>(model_state_.get(), llm_styler_.get(), tokenizer_.get());
  CLOG4(INFO, "TrtllmInferer load success.");
}

void TrtllmInferer::Infer(const ::grps::protos::v1::GrpsMessage& input,
                          ::grps::protos::v1::GrpsMessage& output,
                          GrpsContext& ctx) {
  try {
    const auto& content_type = ctx.http_controller()->http_request().content_type();
    if (content_type != "application/json") {
      throw std::invalid_argument("Invalid content type: " + content_type + ". Expecting application/json.");
    }

    // Enqueue.
    trtllm_instance_->EnqueueAndWait(ctx, ctx.http_controller()->request_attachment().to_string());
  } catch (const std::exception& e) {
    utils::HttpRespondErrorWithOpenAi(ctx, 500, e.what());
  }
}
} // namespace netease::grps