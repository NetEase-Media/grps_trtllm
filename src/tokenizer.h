// Huggingface tokenizer.

#pragma once

#include <tokenizers_cpp.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace netease::grps {

class MultiInstanceTokenizer {
public:
  MultiInstanceTokenizer() = default;
  ~MultiInstanceTokenizer() = default;
  MultiInstanceTokenizer(const MultiInstanceTokenizer&) = delete;
  MultiInstanceTokenizer& operator=(const MultiInstanceTokenizer&) = delete;

  void Load(const std::string& conf_path,
            int instance_num,
            std::optional<int32_t> pad_token_id,
            std::optional<int32_t> end_token_id,
            const std::vector<std::string>& stop_words,
            const std::vector<std::string>& bad_words,
            const std::vector<int32_t>& special_tokens,
            bool skip_special_tokens);

  std::vector<int32_t> Encode(const std::string& text);

  std::string Decode(std::vector<int32_t>& ids);

  [[nodiscard]] std::optional<int32_t> pad_token_id() const { return pad_token_id_; }
  [[nodiscard]] std::optional<int32_t> end_token_id() const { return end_token_id_; }
  [[nodiscard]] const std::unordered_set<std::string>& stop_words() const { return stop_words_; }
  [[nodiscard]] const std::unordered_set<std::string>& bad_words() const { return bad_words_; }
  [[nodiscard]] const std::unordered_set<int32_t>& special_tokens() const { return special_tokens_; }
  [[nodiscard]] bool skip_special_tokens() const { return skip_special_tokens_; }

private:
  tokenizers::Tokenizer* GetTokenizer();

  std::vector<std::unique_ptr<tokenizers::Tokenizer>> tokenizers_{};
  std::vector<std::unique_ptr<std::mutex>> tokenizers_mtxs_{};
  std::atomic<int> cur_idx_{};

  std::optional<int32_t> pad_token_id_ = std::nullopt;
  std::optional<int32_t> end_token_id_ = std::nullopt;
  std::unordered_set<std::string> stop_words_{};
  std::unordered_set<std::string> bad_words_{};
  std::unordered_set<int32_t> special_tokens_{};
  bool skip_special_tokens_{false};
};

} // namespace netease::grps