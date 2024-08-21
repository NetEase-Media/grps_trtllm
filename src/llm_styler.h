// Prompt builder and llm generated txt parser for different style of llm model family.

#pragma once

#include <rapidjson/document.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace netease::grps {

class LLMStyler {
public:
  /**
   * @brief Construct a new LLMStyler object
   * @param style_name: Style name.
   * @param system_prompt: System prompt.
   * @param roles: Roles name.
   * @param intra_message_sep: Intra message separator.
   * @param func_call_observation_words: Function call observation words, used to early stop.
   */
  LLMStyler(std::string style_name,
            std::string system_prompt,
            const std::vector<std::string>& roles,
            std::string intra_message_sep,
            std::string func_call_observation_words)
      : style_name_(std::move(style_name))
      , system_prompt_(std::move(system_prompt))
      , roles_(roles)
      , intra_message_sep_(std::move(intra_message_sep))
      , func_call_observation_words_(std::move(func_call_observation_words)) {}
  virtual ~LLMStyler() = default;

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
   * @param json_body: Json body from client.
   * @return <if_function_call, prompt>: if_function_call is true if the prompt contains function call.
   */
  virtual std::tuple<bool, std::string> BuildPrompt(const rapidjson::Document& json_body) = 0;

  /**
   * @brief Parse function call response from generated text and build content and tool_calls array of message
   * member of OpenAI interface response.
   * @param gen_txt: Generated text.
   * @param req_id: Request id.
   * @param message: Message member of OpenAI interface response format.
   * @param allocator: Json allocator.
   * @return stop reason.
   */
  virtual std::string ParseFunctionCall(const std::string& gen_txt,
                                        int64_t req_id,
                                        rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                        rapidjson::MemoryPoolAllocator<>& allocator) = 0;

  const std::string& GetRole(const std::string& role_name) {
    if (role_name == "user") {
      return roles_[0];
    } else if (role_name == "assistant") {
      return roles_[1];
    } else {
      return role_name;
    }
  }

  [[nodiscard]] const std::string& style_name() const { return style_name_; }
  [[nodiscard]] const std::string& system_prompt() const { return system_prompt_; }
  [[nodiscard]] const std::vector<std::string>& roles() const { return roles_; }
  [[nodiscard]] const std::string& intra_message_sep() const { return intra_message_sep_; }
  [[nodiscard]] const std::string& func_call_observation_words() const { return func_call_observation_words_; }

private:
  std::string style_name_;
  std::string system_prompt_;
  std::vector<std::string> roles_;
  std::string intra_message_sep_;
  std::string func_call_observation_words_; // Function call observation words. Used to early stop.
};

class QwenStyler : public LLMStyler {
public:
  QwenStyler() : LLMStyler("qwen", "You are a helpful assistant.", {"user", "assistant"}, "", "Observation:") {}
  ~QwenStyler() override = default;

  /**
   * @brief Build prompt for model input from OpenAI interface request.
   * @param json_body: Json body from client.
   * @return <if_function_call, prompt>: if_function_call is true if the prompt contains function call.
   */
  std::tuple<bool, std::string> BuildPrompt(const rapidjson::Document& json_body) override;

  /**
   * @brief Parse function call response from generated text and build content and tool_calls array of message
   * member of OpenAI interface response.
   * @param gen_txt: Generated text.
   * @param req_id: Request id.
   * @param message: Message member of OpenAI interface response format.
   * @param allocator: Json allocator.
   * @return stop reason.
   */
  std::string ParseFunctionCall(const std::string& gen_txt,
                                int64_t req_id,
                                rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                rapidjson::MemoryPoolAllocator<>& allocator) override;
};

class LLMStylerFactory {
public:
  LLMStylerFactory() = default;
  virtual ~LLMStylerFactory() = default;
  static std::unique_ptr<LLMStyler> CreateLLMStyler(const std::string& llm_style);
};

} // namespace netease::grps
