// Prompt builder and llm generated txt parser for different style of llm model family.

#pragma once

#include <jinja2cpp/template.h>
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
   * @param system_prompt: System role prompt.
   * @param roles: Roles name. [`system_name`, `user_name`, `assistant_name`]
   * @param support_func_call: If support function call.
   * @param func_call_observation_words: Function call observation words, used to early stop.
   * @param use_default_chat_template: If true, will use default jinja2 templater loaded from `chat_template` in
   * `tokenizer_config.json` to build  prompt. The `chat_template` must be compatible with openai request format.
   * User will needn't implement `BuildPrompt` function.
   * @param add_generation_prompt: If true, will add generation prompt in the end of prompt for default chat template.
   */
  LLMStyler(std::string style_name,
            std::string system_prompt,
            const std::vector<std::string>& roles,
            bool support_func_call,
            std::string func_call_observation_words,
            bool use_default_chat_template = false,
            bool add_generation_prompt = false)
      : style_name_(std::move(style_name))
      , system_prompt_(std::move(system_prompt))
      , roles_(roles)
      , support_func_call_(support_func_call)
      , func_call_observation_words_(std::move(func_call_observation_words))
      , use_default_chat_template_(use_default_chat_template)
      , add_generation_prompt_(add_generation_prompt) {}
  virtual ~LLMStyler() = default;

  inline std::tuple<bool, std::string> BuildPromptWrap(const rapidjson::Document& json_body, jinja2::Template* tpl) {
    if (use_default_chat_template_) {
      return BuildPromptWithChatTemplate(json_body, tpl);
    } else {
      return BuildPrompt(json_body);
    }
  }

  /**
   * @brief Build prompt for model input from OpenAI interface json body request with `chat_template` in
   * `tokenizer_config.json`.
   * @param json_body: Json body from client.
   * @param tpl: Jinja2 template loaded by `chat_template` in `tokenizer_config.json`.
   * @return <if_function_call, prompt>: if_function_call is true if the prompt contains function call.
   */
  std::tuple<bool, std::string> BuildPromptWithChatTemplate(const rapidjson::Document& json_body,
                                                            jinja2::Template* tpl);

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
   * @param json_body: Json body from client.
   * @return <if_function_call, prompt>: if_function_call is true if the prompt contains function call.
   */
  virtual std::tuple<bool, std::string> BuildPrompt(const rapidjson::Document& json_body);

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
                                        rapidjson::MemoryPoolAllocator<>& allocator);

  const std::string& GetRole(const std::string& role_name) {
    if (role_name == "system") {
      return roles_[0];
    } else if (role_name == "user") {
      return roles_[1];
    } else if (role_name == "assistant") {
      return roles_[2];
    } else {
      return role_name;
    }
  }

  [[nodiscard]] const std::string& style_name() const { return style_name_; }
  [[nodiscard]] const std::string& system_prompt() const { return system_prompt_; }
  [[nodiscard]] const std::vector<std::string>& roles() const { return roles_; }
  [[nodiscard]] bool support_func_call() const { return support_func_call_; }
  [[nodiscard]] const std::string& func_call_observation_words() const { return func_call_observation_words_; }
  [[nodiscard]] bool use_default_chat_template() const { return use_default_chat_template_; }

private:
  std::string style_name_;                  // Style name.
  std::string system_prompt_;               // System role prompt.
  std::vector<std::string> roles_;          // Roles name. [`system_name`, `user_name`, `assistant_name`]
  bool support_func_call_ = false;          // If support function call.
  std::string func_call_observation_words_; // Function call observation words. Used to early stop.

  // If true, will use default jinja2 templater loaded from `chat_template` in `tokenizer_config.json` to build prompt.
  // The `chat_template` must be compatible with openai request format. User will needn't implement `BuildPrompt`
  // function.
  bool use_default_chat_template_ = false;
  bool add_generation_prompt_ = false;
};

class QwenStyler : public LLMStyler {
public:
  QwenStyler()
      : LLMStyler(
          "qwen", "You are a helpful assistant.", {"system", "user", "assistant"}, true, "Observation:", false) {}
  ~QwenStyler() override = default;

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
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

class ChatGlm3Styler : public LLMStyler {
public:
  ChatGlm3Styler()
      : LLMStyler("chatglm3", "", {"<|system|>", "<|user|>", "<|assistant|>"}, false, "", false) {
    chat_templater_.Load(
      "{% for item in messages %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n {{ "
      "item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}");
  }
  ~ChatGlm3Styler() override = default;

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
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

private:
  jinja2::Template chat_templater_;
};

class Glm4Styler : public LLMStyler {
public:
  Glm4Styler()
      : LLMStyler("glm4",
                  "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 "
                  "模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。",
                  {"<|system|>", "<|user|>", "<|assistant|>"},
                  false,
                  "",
                  false) {
    chat_templater_.Load(
      "[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM "
      "的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 "
      "模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] "
      "%}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ "
      "tool['function'] | tojson(indent=4) }}\n在调用上述函数时，请使用 Json 格式表示调用的参数。{% elif tool['type'] "
      "== 'python' %}\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 "
      "Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` "
      "将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API "
      "调用，这些在线内容的访问将不会成功。{% elif tool['type'] == 'simple_browser' %}\n\n## "
      "simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: "
      "int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: "
      "list[int])`：获取一系列指定 id "
      "的页面内容。每次调用时，须选择3-"
      "10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能"
      "有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` "
      "来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. "
      "根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` "
      "直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。{% elif tool['type'] == "
      "'cogview' %}\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` "
      "来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` "
      "的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 "
      "英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- "
      "如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。{% endif %}{% endfor %}{% endif %}{% if "
      "item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% "
      "if add_generation_prompt %}<|assistant|>\n{% endif %}");
  }
  ~Glm4Styler() override = default;

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
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

private:
  jinja2::Template chat_templater_;
};

class LLMStylerFactory {
public:
  LLMStylerFactory() = default;
  virtual ~LLMStylerFactory() = default;
  static std::unique_ptr<LLMStyler> CreateLLMStyler(const std::string& llm_style);
};

} // namespace netease::grps
