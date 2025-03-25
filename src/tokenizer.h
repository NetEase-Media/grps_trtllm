// Huggingface tokenizer.

#pragma once

#include <tokenizers_cpp.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace netease::grps {

class MultiInstanceTokenizer {
public:
  MultiInstanceTokenizer() = default;
  ~MultiInstanceTokenizer() = default;
  MultiInstanceTokenizer(const MultiInstanceTokenizer&) = delete;
  MultiInstanceTokenizer& operator=(const MultiInstanceTokenizer&) = delete;

  void Load(const std::string& type,
            const std::string& conf_path,
            int instance_num,
            std::optional<int32_t> pad_token_id,
            std::optional<int32_t> end_token_id,
            const std::vector<int32_t>& skip_special_tokens,
            const std::unordered_map<std::string, int32_t>& force_tokens_dict,
            const std::vector<int32_t>& prefix_tokens_id,
            const std::vector<int32_t>& suffix_tokens_id,
            const std::string& img_token,
            int32_t img_begin_token_id);

  std::vector<int32_t> Encode(const std::string& text, bool add_prefix = true, bool add_suffix = true);

  std::string Decode(std::vector<int32_t>& ids);

  [[nodiscard]] std::string type() const { return type_; }
  [[nodiscard]] std::optional<int32_t> pad_token_id() const { return pad_token_id_; }
  [[nodiscard]] std::optional<int32_t> end_token_id() const { return end_token_id_; }
  [[nodiscard]] int32_t img_begin_token_id() const { return img_begin_token_id_; }
  [[nodiscard]] std::unordered_set<int32_t> skip_special_tokens() const { return skip_special_tokens_; }
  [[nodiscard]] const std::unordered_map<std::string, int32_t>& force_token2id_map() const { return force_token2id_; }
  [[nodiscard]] const std::unordered_map<int32_t, std::string>& force_id2token_map() const { return force_id2token_; }
  [[nodiscard]] const std::vector<int32_t>& prefix_tokens_id_vec() const { return prefix_tokens_id_; }
  [[nodiscard]] const std::vector<int32_t>& suffix_tokens_id_vec() const { return suffix_tokens_id_; }

private:
  tokenizers::Tokenizer* GetTokenizer();

  std::string type_;
  std::vector<std::unique_ptr<tokenizers::Tokenizer>> tokenizers_{};
  std::vector<std::unique_ptr<std::mutex>> tokenizers_mtxs_{};
  std::atomic<int> cur_idx_{};

  std::optional<int32_t> pad_token_id_ = std::nullopt;
  std::optional<int32_t> end_token_id_ = std::nullopt;
  std::unordered_set<int32_t> skip_special_tokens_{};

  // will be used to force map tokens to ids when encode and decode instead of using tokenizer. Empty if not set.
  std::unordered_map<std::string, int32_t> force_token2id_;
  std::unordered_map<int32_t, std::string> force_id2token_;

  std::vector<int32_t>
    prefix_tokens_id_; // prefix tokens id will be added to the beginning of the input ids. Empty if not set.
  std::vector<int32_t>
    suffix_tokens_id_; // suffix tokens id will be added to the end of the input ids. Empty if not set.

  std::string img_token_; // image token text.
  // the beginning token id used to mark the image tokens. Multi img_tokens will be mapped to consecutive token ids
  // with the beginning token id. The beginning token id must be the (max-token-id + 1), that is the true vocab size.
  // Such as: "<IMG_TOKEN><IMG_TOKEN><IMG_TOKEN>" will be mapped to
  // [img_begin_token_id, img_begin_token_id + 1, img_begin_token_id + 2].
  int32_t img_begin_token_id_;
};

} // namespace netease::grps