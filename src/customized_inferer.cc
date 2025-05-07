// Customized deep learning model inferer. Including model load and model infer.

#include "customized_inferer.h"

#include <rapidjson/document.h>

#include "config/global_config.h"
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
  // 1. Load tokenizer.
  std::string tokenizer_path;
  try {
    tokenizer_ = std::make_unique<MultiInstanceTokenizer>();

    std::string tokenizer_type = model_state_->GetParameter<std::string>("tokenizer_type");
    if (tokenizer_type.empty()) {
      throw std::invalid_argument("tokenizer_type is not specified.");
    }
    tokenizer_path = model_state_->GetParameter<std::string>("tokenizer_path");
    if (tokenizer_path.empty()) {
      throw std::invalid_argument("tokenizer_path is not specified.");
    }

    int instance_num = 8;
    try {
      instance_num = model_state_->GetParameter<int32_t>("tokenizer_parallelism");
    } catch (const std::exception& e) {
      CLOG4(WARN, "tokenizer_parallelism is not specified, will use default value of 8.");
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

    std::string img_token;
    try {
      img_token = model_state_->GetParameter<std::string>("img_token");
    } catch (const std::exception& e) {
    }

    int32_t img_begin_token_id = 0;
    try {
      img_begin_token_id = model_state_->GetParameter<int32_t>("img_begin_token_id");
    } catch (const std::exception& e) {
    }

    tokenizer_->Load(tokenizer_type, tokenizer_path, instance_num, pad_token_id, end_token_id, skip_special_tokens,
                     force_tokens_dict, prefix_tokens_id, suffix_tokens_id, img_token, img_begin_token_id);
  } catch (const std::exception& e) {
    throw InfererException("Load tokenizer failed: " + std::string(e.what()));
  }

  // 2. Load llm styler.
  std::string chat_template;
  try {
    auto tokenizer_config_path = tokenizer_path + "/tokenizer_config.json";
    auto tokenizer_config_content = utils::LoadBytesFromFile<std::string>(tokenizer_config_path);
    rapidjson::Document tokenizer_config_doc;
    tokenizer_config_doc.Parse(tokenizer_config_content.c_str());
    if (tokenizer_config_doc.HasParseError()) {
      throw std::runtime_error("Parse tokenizer_config.json failed.");
    }
    if (!tokenizer_config_doc.HasMember("chat_template") || !tokenizer_config_doc["chat_template"].IsString()) {
      throw std::invalid_argument("chat_template not found or not a string.");
    }
    chat_template = tokenizer_config_doc["chat_template"].GetString();
  } catch (const std::exception& e) {
    CLOG4(WARN, "Failed to load chat_template from tokenizer_config.json, will use default value of empty, error: "
                  << e.what());
  }
  std::string llm_style = model_state_->GetParameter<std::string>("llm_style");
  llm_styler_ = LLMStylerFactory::CreateLLMStyler(llm_style, chat_template);

  // 3. Load vit if multi-modal model.
  std::string vit_type;
  try {
    vit_type = model_state_->GetParameter<std::string>("vit_type");
  } catch (const std::exception& e) {
  }
  std::string vit_path;
  try {
    vit_path = model_state_->GetParameter<std::string>("vit_path");
  } catch (const std::exception& e) {
  }
  int vit_worker_tp = 8;
  try {
    vit_worker_tp = model_state_->GetParameter<int>("vit_worker_tp");
  } catch (const std::exception& e) {
    CLOG4(WARN, "vit_worker_tp is not specified, will use default value of 8.");
  }
  YAML::Node vit_trt_args;
  try {
    vit_trt_args = model_state_->GetModelConfig()["vit_trt_args"];
  } catch (const std::exception& e) {
  }
  YAML::Node vit_processor_args;
  try {
    vit_processor_args = model_state_->GetModelConfig()["vit_processor_args"];
  } catch (const std::exception& e) {
  }
  bool enable_kv_cache_reuse = false;
  try {
    enable_kv_cache_reuse = model_state_->GetParameter<bool>("enable_kv_cache_reuse");
  } catch (std::exception const& e) {
    // If parameter is not specified, just ignore
    CLOG4(WARN, "enable_kv_cache_reuse is not specified, will be set to false");
  }
  if (!vit_type.empty()) {
    vit_ = VITFactory::CreateVIT(vit_type);
    if (vit_ == nullptr) {
      throw InfererException("Unsupported vit type: " + vit_type);
    }
    vit_->Init(vit_path, vit_worker_tp, "gpu:0", vit_trt_args, vit_processor_args, tokenizer_.get(),
               enable_kv_cache_reuse);
    if (GlobalConfig::Instance().mpi().world_rank == 0) { // Only rank 0 load vit when using MPI.
      vit_->Load();
    }
  }

  // 4. Load trtllm instance.
  trtllm_instance_ =
    std::make_unique<TrtLlmModelInstance>(model_state_.get(), llm_styler_.get(), tokenizer_.get(), vit_.get());
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
    // CLOG4(INFO, "Request: " << ctx.http_controller()->request_attachment().to_string());
    trtllm_instance_->EnqueueAndWait(ctx, ctx.http_controller()->request_attachment().to_string());
  } catch (const std::exception& e) {
    if (ctx.IfStreaming()) {
      utils::HttpStreamingRespondErrorWithOpenAi(ctx, e.what());
    } else {
      utils::HttpRespondErrorWithOpenAi(ctx, 500, e.what());
    }
  }
}
} // namespace netease::grps