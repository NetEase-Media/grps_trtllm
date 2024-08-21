// Huggingface tokenizer.

#include "tokenizer.h"

#include <fstream>

#include "logger/logger.h"
#include "src/utils.h"

#define TOKENIZER_DEBUG 0
namespace netease::grps {

void MultiInstanceTokenizer::Load(const std::string& conf_path,
                                  int instance_num,
                                  std::optional<int32_t> pad_token_id,
                                  std::optional<int32_t> end_token_id,
                                  const std::vector<std::string>& stop_words,
                                  const std::vector<std::string>& bad_words,
                                  const std::vector<int32_t>& special_tokens,
                                  bool skip_special_tokens) {
  auto file_name = conf_path.substr(conf_path.find_last_of('/') + 1);
  auto blob = utils::LoadBytesFromFile(conf_path);
  // Load tokenizer.
  for (int i = 0; i < instance_num; ++i) {
    if (file_name == "tokenizer.json") {
      tokenizers_.emplace_back(tokenizers::Tokenizer::FromBlobJSON(blob));
    } else if (file_name == "tokenizer.model") {
      tokenizers_.emplace_back(tokenizers::Tokenizer::FromBlobSentencePiece(blob));
    } else if (file_name == "tokenizer_model") {
      tokenizers_.emplace_back(tokenizers::Tokenizer::FromBlobRWKVWorld(conf_path));
    } else {
      throw std::runtime_error("Unsupported tokenizer format: " + file_name);
    }
    tokenizers_mtxs_.emplace_back(std::make_unique<std::mutex>());
  }
  cur_idx_ = 0;

  pad_token_id_ = pad_token_id;
  end_token_id_ = end_token_id;
  stop_words_.insert(stop_words.begin(), stop_words.end());
  bad_words_.insert(bad_words.begin(), bad_words.end());
  special_tokens_.insert(special_tokens.begin(), special_tokens.end());
  skip_special_tokens_ = skip_special_tokens;
}

std::vector<int32_t> MultiInstanceTokenizer::Encode(const std::string& text) {
#if TOKENIZER_DEBUG
  DECL_TIMESTAMP(begin);
#endif

  std::vector<int32_t> ids;
  tokenizers::Tokenizer* tokenizer = GetTokenizer();
  {
    std::lock_guard<std::mutex> lock(*tokenizers_mtxs_[cur_idx_]);
    ids = tokenizer->Encode(text);
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

  if (skip_special_tokens_) {
    // skip special tokens.
    ids.erase(std::remove_if(ids.begin(), ids.end(), [this](int32_t id) { return special_tokens_.count(id); }),
              ids.end());
  }

  std::string text;
  tokenizers::Tokenizer* tokenizer = GetTokenizer();
  {
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