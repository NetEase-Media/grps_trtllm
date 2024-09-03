// Huggingface tokenizer.

#include "tokenizer.h"

#include <sentencepiece_processor.h>

#include <fstream>

#include "logger/logger.h"
#include "src/utils.h"

#define TOKENIZER_DEBUG 0
namespace netease::grps {

void MultiInstanceTokenizer::Load(const std::string& type,
                                  const std::string& conf_path,
                                  int instance_num,
                                  std::optional<int32_t> pad_token_id,
                                  std::optional<int32_t> end_token_id,
                                  const std::vector<int32_t>& skip_special_tokens,
                                  const std::unordered_map<std::string, int32_t>& force_tokens_dict,
                                  const std::vector<int32_t>& prefix_tokens_id,
                                  const std::vector<int32_t>& suffix_tokens_id) {
  CLOG4(INFO, "Loading tokenizer from: " << conf_path << " with type: " << type << ", instance_num: " << instance_num);

  // Load tokenizer.
  for (int i = 0; i < instance_num; ++i) {
    if (type == "huggingface") {
      auto blob = utils::LoadBytesFromFile(conf_path + "/tokenizer.json");
      tokenizers_.emplace_back(tokenizers::Tokenizer::FromBlobJSON(blob));
    } else if (type == "sentencepiece") {
      auto blob = utils::LoadBytesFromFile(conf_path + "/tokenizer.model");
      tokenizers_.emplace_back(tokenizers::Tokenizer::FromBlobSentencePiece(blob));
    } else {
      throw std::runtime_error("Unsupported tokenizer type: " + type);
    }
    tokenizers_mtxs_.emplace_back(std::make_unique<std::mutex>());
  }
  CLOG4(INFO, "Load tokenizer success.");
  type_ = type;
  cur_idx_ = 0;

  // Load chat template.
  auto tokenizer_config_path = conf_path + "/tokenizer_config.json";
  if (std::filesystem::exists(tokenizer_config_path)) {
    auto blob = utils::LoadBytesFromFile(tokenizer_config_path);
    rapidjson::Document tokenizer_config_doc;
    tokenizer_config_doc.Parse(reinterpret_cast<const char*>(blob.data()), blob.size());
    if (tokenizer_config_doc.HasParseError()) {
      throw std::runtime_error("Parse tokenizer_config.json failed.");
    }
    chat_template_str_ = tokenizer_config_doc["chat_template"].GetString();
    chat_templater_ = std::make_unique<jinja2::Template>();
    chat_templater_->Load(chat_template_str_);
  }

  pad_token_id_ = pad_token_id;
  end_token_id_ = end_token_id;
  skip_special_tokens_.insert(skip_special_tokens.begin(), skip_special_tokens.end());

  for (auto& [k, v] : force_tokens_dict) {
    for (auto& [k1, v1] : force_token2id_) {
      if (k1.find(k) != std::string::npos || k.find(k1) != std::string::npos) {
        throw std::runtime_error("Force tokens can not be substring of each other: " + k + ", " + k1);
      }
    }
    force_token2id_[k] = v;
  }
  for (auto& [k, v] : force_token2id_) {
    if (force_id2token_.count(v)) {
      throw std::runtime_error("Duplicate force token id: " + std::to_string(v));
    }
    force_id2token_[v] = k;
  }

  prefix_tokens_id_ = prefix_tokens_id;
  suffix_tokens_id_ = suffix_tokens_id;
}

std::vector<int32_t> MultiInstanceTokenizer::Encode(const std::string& text, bool add_prefix, bool add_suffix) {
#if TOKENIZER_DEBUG
  DECL_TIMESTAMP(begin);
#endif

  std::vector<int32_t> ids;

  tokenizers::Tokenizer* tokenizer = GetTokenizer();
  if (!force_token2id_.empty()) {
    // Split text by force tokens.
    std::vector<std::pair<int, std::string>> splits; // <begin_idx, token>
    for (auto& [token, id] : force_token2id_) {
      size_t pos = 0;
      while ((pos = text.find(token, pos)) != std::string::npos) {
        splits.emplace_back(pos, token);
        pos += token.size();
      }
    }

    if (splits.empty()) {
      std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
      ids = tokenizer->Encode(text);
    } else {
      // Encode all sub texts split by force tokens.
      std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
      std::sort(splits.begin(), splits.end());
      size_t begin_idx = 0;
      for (auto& [idx, token] : splits) {
        if (idx > begin_idx) {
          auto sub_text = text.substr(begin_idx, idx - begin_idx);
          auto sub_ids = tokenizer->Encode(sub_text);
          ids.insert(ids.end(), sub_ids.begin(), sub_ids.end());
        }
        ids.push_back(force_token2id_[token]);
        begin_idx = idx + token.size();
      }
      if (begin_idx < text.size()) {
        auto sub_text = text.substr(begin_idx);
        auto sub_ids = tokenizer->Encode(sub_text);
        ids.insert(ids.end(), sub_ids.begin(), sub_ids.end());
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
    ids = tokenizer->Encode(text);
  }

  if (add_prefix && !prefix_tokens_id_.empty()) {
    ids.insert(ids.begin(), prefix_tokens_id_.begin(), prefix_tokens_id_.end());
  }

  if (add_suffix && !suffix_tokens_id_.empty()) {
    ids.insert(ids.end(), suffix_tokens_id_.begin(), suffix_tokens_id_.end());
  }

#if TOKENIZER_DEBUG
  DECL_TIMESTAMP(end);
  CLOG4(INFO, "Encode cost: " << (end - begin) / 1e3 << " us.");
  std::string ids_str;
  for (auto id : ids) {
    ids_str += std::to_string(id) + " ";
  }
  CLOG4(INFO, "Encode text: " << text << ", ids: " << ids_str);
#endif

  return ids;
}

std::string MultiInstanceTokenizer::Decode(std::vector<int32_t>& ids) {
#if TOKENIZER_DEBUG
  DECL_TIMESTAMP(begin);
#endif

  std::string text;
  tokenizers::Tokenizer* tokenizer = GetTokenizer();

  if (!skip_special_tokens_.empty() || !force_id2token_.empty()) {
    std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
    std::vector<int32_t> buffer;
    for (auto id : ids) {
      if (skip_special_tokens_.count(id) || force_id2token_.count(id)) {
        if (!buffer.empty()) {
          std::string buffer_text;
          buffer_text = tokenizer->Decode(buffer);
          text += buffer_text;
          buffer.clear();
        }
        if (!skip_special_tokens_.count(id) && force_id2token_.count(id)) {
          text += force_id2token_[id];
        }
      } else {
        buffer.push_back(id);
      }
    }
    if (!buffer.empty()) {
      std::string buffer_text;
      buffer_text = tokenizer->Decode(buffer);
      text += buffer_text;
    }
  } else {
    std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
    text = tokenizer->Decode(ids);
  }

#if TOKENIZER_DEBUG
  DECL_TIMESTAMP(end);
  CLOG4(INFO, "Decode cost: " << (end - begin) / 1e3 << " us.");
  std::string ids_str;
  for (auto id : ids) {
    ids_str += std::to_string(id) + " ";
  }
  CLOG4(INFO, "Decode ids: " << ids_str << ", text: " << text);
#endif

  return text;
}

tokenizers::Tokenizer* MultiInstanceTokenizer::GetTokenizer() {
  if (tokenizers_.empty()) {
    throw std::runtime_error("Tokenizers is empty. Should init firstly.");
  }

  // Get idx by cas.
  int idx = cur_idx_.load();
  while (!cur_idx_.compare_exchange_weak(idx, (idx + 1) % int(tokenizers_.size()))) {
    idx = cur_idx_.load();
  }
#if TOKENIZER_DEBUG
  CLOG4(INFO, "GetTokenizer idx: " << idx);
#endif
  return tokenizers_[idx].get();
}
} // namespace netease::grps