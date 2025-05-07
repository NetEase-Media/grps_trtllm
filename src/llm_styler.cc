// Prompt builder and llm generated txt parser for different llm model family.

#include "llm_styler.h"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <regex>
#include <unordered_set>
#include <vector>

#include "logger/logger.h"
#include "src/utils.h"

namespace netease::grps {

std::tuple<bool, std::string, std::vector<std::string>> LLMStyler::BuildPrompt(const rapidjson::Document& json_body) {
  throw std::runtime_error("BuildPrompt not implemented.");
}

std::string LLMStyler::ParseFunctionCall(const std::string& gen_txt,
                                         int64_t req_id,
                                         rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                         rapidjson::MemoryPoolAllocator<>& allocator) {
  throw std::runtime_error("ParseFunctionCall not implemented.");
}

std::tuple<bool, std::string, std::vector<std::string>> QwenStyler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse tools.
  bool has_tools = false;
  std::string tool_system;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    std::vector<std::string> tools_text;
    std::vector<std::string> tools_name_text;

    for (auto& tool : json_body["tools"].GetArray()) {
      if (tool.HasMember("function") && tool["function"].IsObject()) {
        auto& function = tool["function"];

        auto& fp = function["parameters"];
        std::string parameters;
        if (fp.IsObject()) {
          std::unordered_set<std::string> required_parameters;
          if (fp.HasMember("required") && fp["required"].IsArray()) {
            for (auto& required : fp["required"].GetArray()) {
              if (!required.IsString()) {
                throw std::invalid_argument("function `required` parameter is not a string arr");
              }
              required_parameters.insert(required.GetString());
            }
          }

          // Build parameters array json.
          rapidjson::Document parameters_doc(rapidjson::kArrayType);
          for (auto& [name, p] : fp["properties"].GetObject()) {
            auto param_doc = rapidjson::Value(rapidjson::kObjectType);
            param_doc.CopyFrom(p, parameters_doc.GetAllocator());
            param_doc.AddMember("name", rapidjson::Value(name.GetString(), parameters_doc.GetAllocator()),
                                parameters_doc.GetAllocator());
            if (required_parameters.find(name.GetString()) != required_parameters.end()) {
              param_doc.AddMember("required", true, parameters_doc.GetAllocator());
            }
            parameters_doc.PushBack(param_doc, parameters_doc.GetAllocator());
          }

          // Parameters json to string
          rapidjson::StringBuffer buffer;
          rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
          parameters_doc.Accept(writer);
          parameters = buffer.GetString();
        }

        if (!function.HasMember("name") || !function["name"].IsString()) {
          throw std::invalid_argument("function `name` not found or not a string");
        }
        if (!function.HasMember("description") || !function["description"].IsString()) {
          throw std::invalid_argument("function `description` not found or not a string");
        }
        const auto& name = function["name"].GetString();
        const auto& desc = function["description"].GetString();
        std::string tool_string = name;
        tool_string.append(": Call this tool to interact with the ");
        tool_string.append(name);
        tool_string.append(" API. What is the ");
        tool_string.append(name);
        tool_string.append(" API useful for? ");
        tool_string.append(desc);
        tool_string.append(" Parameters: ");
        tool_string.append(parameters);
        tool_string.append(" Format the arguments as a JSON object.");
        tools_text.push_back(tool_string);
        tools_name_text.emplace_back(name);
      }
    }

    std::string tools_text_string;
    for (size_t i = 0; i < tools_text.size(); ++i) {
      tools_text_string.append(tools_text[i]);
      if (i != tools_text.size() - 1) {
        tools_text_string.append("\n\n");
      }
    }
    std::string tools_name_text_string;
    for (size_t i = 0; i < tools_name_text.size(); ++i) {
      tools_name_text_string.append(tools_name_text[i]);
      if (i != tools_name_text.size() - 1) {
        tools_name_text_string.append(", ");
      }
    }
    tool_system.append("Answer the following questions as best you can. You have access to the following APIs:\n\n");
    tool_system.append(tools_text_string);
    tool_system.append("\n\nUse the following format:\n\n");
    tool_system.append("Question: the input question you must answer\n");
    tool_system.append("Thought: you should always think about what to do\n");
    tool_system.append("Action: the action to take, should be one of [");
    tool_system.append(tools_name_text_string);
    tool_system.append("]\n");
    tool_system.append("Action Input: the input to the action\n");
    tool_system.append("Observation: the result of the action\n");
    tool_system.append("... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n");
    tool_system.append("Thought: I now know the final answer\n");
    tool_system.append("Final Answer: the final answer to the original input question\n\n");
    tool_system.append("Begin!");
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string ori_content = message["content"].GetString();

    prompt.append("");

    std::string content;
    if (has_tools) {
      if (role == "user") {
        if (!tool_system.empty()) {
          content = tool_system;
          tool_system.clear();
          content.append("\n\nQuestion: ");
          content.append(ori_content);
        } else {
          content.append("Question: ");
          content.append(ori_content);
        }
      } else if (role == "assistant") {
        if (message.HasMember("tool_calls") && message["tool_calls"].IsArray()) {
          auto& tool_calls = message["tool_calls"];
          if (!tool_calls.Empty()) {
            if (!tool_calls[0].HasMember("function") || !tool_calls[0]["function"].IsObject()) {
              throw std::invalid_argument("`function` not found in `tool_calls` or not an object");
            }
            auto& func_call = tool_calls[0]["function"];
            if (!func_call.HasMember("name") || !func_call["name"].IsString()) {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            auto& f_name = func_call["name"];
            if (!func_call.HasMember("arguments") || !func_call["arguments"].IsString()) {
              throw std::invalid_argument("`arguments` not found in `function` or not a string");
            }
            auto& f_args = func_call["arguments"];
            content.append("Thought: I can use ");
            content.append(f_name.GetString());
            content.append(".\nAction: ");
            content.append(f_name.GetString());
            content.append("\nAction Input: ");
            content.append(f_args.GetString());
          } else if (!ori_content.empty()) {
            content.append("Thought: I now know the final answer.\nFinal answer: ");
            content.append(ori_content);
          }
        }
      } else if (role == "tool") {
        role = "function";
        content.append("Observation: ");
        content.append(ori_content);
      } else {
        throw std::invalid_argument("Unsupported message role: " + role);
      }
    } else {
      content = std::move(ori_content);
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {has_tools, prompt, {}};
}

std::string QwenStyler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  std::string thought, func_name, func_args;
  size_t i = gen_txt.rfind("\nAction:");
  size_t j = gen_txt.rfind("\nAction Input:");
  size_t k = gen_txt.rfind("\nObservation:");

  if (i != std::string::npos && j != std::string::npos) {
    // Found `Thought:` before `Action:`.
    std::string sub_str = gen_txt.substr(0, i);
    size_t h1 = sub_str.rfind("\nThought:");
    size_t h2 = sub_str.rfind("Thought:");
    size_t h = std::string::npos;
    if (h1 != std::string::npos && h2 != std::string::npos) {
      h = std::max(h1, h2);
    } else if (h1 != std::string::npos) {
      h = h1;
    } else if (h2 != std::string::npos) {
      h = h2;
    }

    if (h != std::string::npos && h < i && i < j) { // If the text has `Action` and `Action input`,
      if (k == std::string::npos) {                 // but does not contain `Observation`,
        // then it is likely that `Observation` is omitted by the LLM,
        // because the output text may have discarded the stop word.
        // gen_txt.append("\nObservation:");
        // k = gen_txt.rfind("\nObservation:");
        k = gen_txt.size();
      }
      thought = gen_txt.substr(h + 9, i - h - 9);
      thought = thought.substr(thought.find_first_not_of(' ')); // strip left spaces
      func_name = gen_txt.substr(i + 8, j - i - 8);
      func_args = gen_txt.substr(j + 14, k - j - 14);
      // strip all spaces
      func_name.erase(std::remove_if(func_name.begin(), func_name.end(), ::isspace), func_name.end());
      func_args.erase(std::remove_if(func_args.begin(), func_args.end(), ::isspace), func_args.end());
    }
  }
  if (!func_name.empty()) {
    message.AddMember("content", rapidjson::Value(thought.c_str(), allocator), allocator);
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls = message["tool_calls"];
    tool_calls.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& tool_call = tool_calls[0];
    std::string call_id = std::string("call_") + std::to_string(req_id);
    tool_call.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
    tool_call.AddMember("type", rapidjson::Value("function", allocator), allocator);
    tool_call.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& function = tool_call["function"];
    function.AddMember("name", rapidjson::Value(func_name.c_str(), allocator), allocator);
    function.AddMember("arguments", rapidjson::Value(func_args.c_str(), allocator), allocator);
    return "tool_calls";
  } else {
    size_t y1 = gen_txt.rfind("\nThought:");
    size_t y2 = gen_txt.rfind("Thought:");
    size_t y = std::string::npos;
    if (y1 != std::string::npos && y2 != std::string::npos) {
      y = std::max(y1, y2);
    } else if (y1 != std::string::npos) {
      y = y1;
    } else if (y2 != std::string::npos) {
      y = y2;
    }

    size_t z1 = gen_txt.rfind("\nFinal Answer:");
    size_t z2 = gen_txt.rfind("Final Answer:");
    size_t z = std::string::npos;
    if (z1 != std::string::npos && z2 != std::string::npos) {
      z = std::max(z1, z2);
    } else if (z1 != std::string::npos) {
      z = z1;
    } else if (z2 != std::string::npos) {
      z = z2;
    }

    if (z != std::string::npos) {
      std::string final_answer = gen_txt.substr(z + 14);
      final_answer = final_answer.substr(final_answer.find_first_not_of(' ')); // strip left spaces
      message.AddMember("content", rapidjson::Value(final_answer.c_str(), allocator), allocator);
    } else if (y != std::string::npos) { // Only have thought.
      std::string thought_answer = gen_txt.substr(y + 9);
      thought_answer = thought_answer.substr(thought_answer.find_first_not_of(' ')); // strip left spaces
      message.AddMember("content", rapidjson::Value(thought_answer.c_str(), allocator), allocator);
    }
    return "stop";
  }
}

void Qwen25Styler::ApplyChatTemplate(const std::string& chat_template) {
  LLMStyler::ApplyChatTemplate(chat_template);

  // Parse system prompt from chat_template.
  if (chat_template.empty()) {
    CLOG4(WARN, "chat_template is empty, will use default system prompt: `" << system_prompt_ << "`.");
    return;
  }
  auto pre = std::string(
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        "
    "{{- messages[0]['content'] }}\n    {%- else %}\n        {{- '");
  if (chat_template.find(pre) != 0) {
    CLOG4(WARN, "Failed to parse system prompt from chat_template, will use default system prompt: `" << system_prompt_
                                                                                                      << "`.");
    return;
  }
  auto sub_str = chat_template.substr(pre.size());
  auto post = std::string("' }}\n    {%- endif %}\n");
  auto end = sub_str.find(post);
  if (end == std::string::npos) {
    CLOG4(WARN, "Failed to parse system prompt from chat_template, will use default system prompt: `" << system_prompt_
                                                                                                      << "`.");
    return;
  }
  system_prompt_ = sub_str.substr(0, end);
  CLOG4(INFO, "Parse system prompt from chat_template success: `" << system_prompt_ << "`.");
}

std::tuple<bool, std::string, std::vector<std::string>> Qwen25Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse tools.
  bool has_tools = false;
  std::string tool_system;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    tool_system = tool_prompt_pre_;
    for (auto& tool : json_body["tools"].GetArray()) {
      tool_system.append("\n");
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      tool.Accept(writer);
      tool_system.append(buffer.GetString());
    }
    tool_system.append(tool_prompt_post_);
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    if (has_tools) {
      prompt.append(tool_system);
    }
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    if (has_tools) {
      prompt.append(tool_system);
    }
    prompt.append("<|im_end|>\n");
  }

  // Following messages.
  size_t cur_idx = 0;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      cur_idx++;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());

    if (role == "user" || role == "system" || (role == "assistant" && !message.HasMember("tool_calls"))) {
      prompt.append("<|im_start|>");
      prompt.append(role);
      if (message.HasMember("content") && message["content"].IsString()) {
        prompt.append("\n");
        prompt.append(message["content"].GetString());
      }
      prompt.append("<|im_end|>\n");
    } else if (role == "assistant") {
      prompt.append("<|im_start|>");
      prompt.append(role);
      if (message.HasMember("content") && message["content"].IsString()) {
        prompt.append("\n");
        prompt.append(message["content"].GetString());
      }
      if (message.HasMember("tool_calls") && message["tool_calls"].IsArray()) {
        for (auto& tool_call : message["tool_calls"].GetArray()) {
          if (tool_call.HasMember("function") && tool_call["function"].IsObject()) {
            auto& func_call = tool_call["function"];
            if (!func_call.HasMember("name") || !func_call["name"].IsString()) {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            if (!func_call.HasMember("arguments") || !func_call["arguments"].IsString()) {
              throw std::invalid_argument("`arguments` not found in `function` or not a string");
            }
            prompt.append("\n<tool_call>\n{\"name\": \"");
            prompt.append(func_call["name"].GetString());
            prompt.append("\", \"arguments\": ");
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            func_call["arguments"].Accept(writer);
            prompt.append(buffer.GetString());
            prompt.append("}\n</tool_call>");
          }
        }
        prompt.append("<|im_end|>\n");
      }
    } else if (role == "tool") {
      if (message.HasMember("content") && message["content"].IsString()) {
        if (cur_idx == 0 || std::string(json_body["messages"][cur_idx - 1]["role"].GetString()) != "tool") {
          prompt.append("<|im_start|>user");
        }
        prompt.append("\n<tool_response>\n");
        prompt.append(message["content"].GetString());
        prompt.append("\n</tool_response>");
        if (cur_idx == json_body["messages"].Size() - 1 ||
            std::string(json_body["messages"][cur_idx + 1]["role"].GetString()) != "tool") {
          prompt.append("<|im_end|>\n");
        }
      }
      if (cur_idx == json_body["messages"].Size() - 1) { // Last message is tool response. Will not func call.
        has_tools = false;
      }
    } else {
      throw std::invalid_argument("Unsupported message role: " + role);
    }

    cur_idx++;
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {has_tools, prompt, {}};
}

std::string Qwen25Styler::ParseFunctionCall(const std::string& gen_txt,
                                            int64_t req_id,
                                            rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                            rapidjson::MemoryPoolAllocator<>& allocator) {
  // <tool_call>
  // {{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}}
  // </tool_call>
  // <tool_call>
  // {{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}}
  // </tool_call>

  std::vector<std::string> tool_calls;
  size_t start = 0;
  while (true) {
    size_t tool_call_start = gen_txt.find("<tool_call>", start);
    if (tool_call_start == std::string::npos) {
      break;
    }
    size_t tool_call_end = gen_txt.find("</tool_call>", tool_call_start);
    if (tool_call_end == std::string::npos) {
      break;
    }
    auto tool_call = gen_txt.substr(tool_call_start + 11, tool_call_end - tool_call_start - 11);
    // lstrip \n
    tool_call = tool_call.substr(tool_call.find_first_not_of('\n'));
    // rstrip \n
    tool_call.erase(std::find_if(tool_call.rbegin(), tool_call.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    tool_call.end());
    if (tool_call.size() < 4) {
      throw std::invalid_argument("Invalid tool call: " + tool_call);
    }
    if (utils::StartsWith(tool_call, "{{") && utils::EndsWith(tool_call, "}}")) {
      tool_calls.emplace_back(tool_call.substr(1, tool_call.size() - 2)); // strip {}
    } else {
      tool_calls.emplace_back(tool_call);
    }
    start = tool_call_end + 12;
  }

  if (!tool_calls.empty()) {
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls_array = message["tool_calls"];
    for (const auto& tool_call : tool_calls) {
      tool_calls_array.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
      auto& tool_call_json = tool_calls_array[tool_calls_array.Size() - 1];
      std::string call_id = std::string("call_") + std::to_string(req_id);
      tool_call_json.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
      tool_call_json.AddMember("type", rapidjson::Value("function", allocator), allocator);
      tool_call_json.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);

      // parse function from tool_call
      // CLOG4(INFO, "tool_call: " << tool_call);
      auto func_doc = rapidjson::Value(rapidjson::kObjectType);
      rapidjson::Document tool_call_doc;
      tool_call_doc.Parse(tool_call.c_str());
      if (tool_call_doc.HasParseError()) {
        // throw std::invalid_argument("Parse tool call failed, tool_call: " + tool_call);
        CLOG4(ERROR, "Parse tool call failed, tool_call: " << tool_call);
        continue;
      }
      if (!tool_call_doc.HasMember("name") || !tool_call_doc["name"].IsString()) {
        throw std::invalid_argument("`name` not found in `tool_call` or not a string");
      }
      func_doc.AddMember("name", rapidjson::Value(tool_call_doc["name"].GetString(), allocator), allocator);
      if (!tool_call_doc.HasMember("arguments")) {
        throw std::invalid_argument("`arguments` not found in `tool_call` or not an json object or an string");
      }
      if (tool_call_doc["arguments"].IsString()) {
        func_doc.AddMember("arguments", rapidjson::Value(tool_call_doc["arguments"].GetString(), allocator), allocator);
      } else {
        // arguments to json str.
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        tool_call_doc["arguments"].Accept(writer);
        func_doc.AddMember("arguments", rapidjson::Value(buffer.GetString(), allocator), allocator);
      }

      tool_call_json.AddMember("function", func_doc, allocator);
    }
    return "tool_calls";
  } else {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    return "stop";
  }
}

void QwQStyler::ApplyChatTemplate(const std::string& chat_template) {
  LLMStyler::ApplyChatTemplate(chat_template);
}

std::tuple<bool, std::string, std::vector<std::string>> QwQStyler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse tools.
  bool has_tools = false;
  std::string tool_system;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    tool_system = tool_prompt_pre_;
    for (auto& tool : json_body["tools"].GetArray()) {
      tool_system.append("\n");
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      tool.Accept(writer);
      tool_system.append(buffer.GetString());
    }
    tool_system.append(tool_prompt_post_);
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    if (has_tools) {
      prompt.append(tool_system);
    }
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else if (has_tools) {
    // If the first message is not a system message but have tool, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append(tool_system);
    prompt.append("<|im_end|>\n");
  }

  // Following messages.
  size_t cur_idx = 0;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      cur_idx++;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());

    if (role == "user" || role == "system") {
      prompt.append("<|im_start|>");
      prompt.append(role);
      if (message.HasMember("content") && message["content"].IsString()) {
        prompt.append("\n");
        prompt.append(message["content"].GetString());
      }
      prompt.append("<|im_end|>\n");
    } else if (role == "assistant" && !message.HasMember("tool_calls")) {
      prompt.append("<|im_start|>");
      prompt.append(role);
      std::string content;
      if (message.HasMember("content") && message["content"].IsString()) {
        content = message["content"].GetString();
        // content = message.content.split('</think>')[-1]
        size_t pos = content.find("</think>");
        if (pos != std::string::npos) {
          content = content.substr(pos + 8);
        }
        // content = content.lstrip("\n")
        content = content.substr(content.find_first_not_of('\n'));
      }
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else if (role == "assistant") {
      prompt.append("<|im_start|>");
      prompt.append(role);
      std::string content;
      if (message.HasMember("content") && message["content"].IsString()) {
        content = message["content"].GetString();
        // content = message.content.split('</think>')[-1]
        size_t pos = content.find("</think>");
        if (pos != std::string::npos) {
          content = content.substr(pos + 8);
        }
        // content = content.lstrip("\n")
        content = content.substr(content.find_first_not_of('\n'));
      }
      prompt.append("\n");
      prompt.append(content);
      if (message.HasMember("tool_calls") && message["tool_calls"].IsArray()) {
        for (auto& tool_call : message["tool_calls"].GetArray()) {
          if (tool_call.HasMember("function") && tool_call["function"].IsObject()) {
            auto& func_call = tool_call["function"];
            if (!func_call.HasMember("name") || !func_call["name"].IsString()) {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            if (!func_call.HasMember("arguments") || !func_call["arguments"].IsString()) {
              throw std::invalid_argument("`arguments` not found in `function` or not a string");
            }
            prompt.append("\n<tool_call>\n{\"name\": \"");
            prompt.append(func_call["name"].GetString());
            prompt.append("\", \"arguments\": ");
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            func_call["arguments"].Accept(writer);
            prompt.append(buffer.GetString());
            prompt.append("}\n</tool_call>");
          }
        }
        prompt.append("<|im_end|>\n");
      }
    } else if (role == "tool") {
      if (message.HasMember("content") && message["content"].IsString()) {
        if (cur_idx == 0 || std::string(json_body["messages"][cur_idx - 1]["role"].GetString()) != "tool") {
          prompt.append("<|im_start|>user");
        }
        prompt.append("\n<tool_response>\n");
        prompt.append(message["content"].GetString());
        prompt.append("\n</tool_response>");
        if (cur_idx == json_body["messages"].Size() - 1 ||
            std::string(json_body["messages"][cur_idx + 1]["role"].GetString()) != "tool") {
          prompt.append("<|im_end|>\n");
        }
      }
      if (cur_idx == json_body["messages"].Size() - 1) { // Last message is tool response. Will not func call.
        has_tools = false;
      }
    } else {
      throw std::invalid_argument("Unsupported message role: " + role);
    }

    cur_idx++;
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n<think>\n");
  }
  return {has_tools, prompt, {}};
}

std::tuple<bool, std::string, std::vector<std::string>> ChatGlm3Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse tools.
  bool has_tools = false;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    auto& tools = json_body["tools"];
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetIndent(' ', 4);
    tools.Accept(writer);
    prompt += GetRole("system");
    prompt += "\n ";
    prompt += buffer.GetString();
    has_tools = true;
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = message["role"].GetString();
    if (role == "tool") {
      role = "<|observation|>";
    }
    if (role != "system" && role != "user" && role != "assistant" && role != "<|observation|>") {
      throw std::invalid_argument("Unsupported message role: " + role);
    }
    role = GetRole(role);
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string content = message["content"].GetString();
    prompt += role;
    prompt += "\n ";
    prompt += content;
  }

  if (add_generation_prompt()) {
    prompt += GetRole("assistant");
    prompt += "\n";
  }

  return {has_tools, prompt, {}};
}

std::string ChatGlm3Styler::ParseFunctionCall(const std::string& gen_txt,
                                              int64_t req_id,
                                              rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                              rapidjson::MemoryPoolAllocator<>& allocator) {
  // The user wants to know the weather in Boston today. The function 'get_current_weather' can be used to retrieve the
  // weather information.get_current_weather
  //  ```python
  // tool_call(location='Boston', unit='celsius')
  // ```

  std::string func_name, func_args;
  size_t j = gen_txt.rfind("\n ```python");
  if (j != std::string::npos) {
    std::string sub_str = gen_txt.substr(0, j);
    size_t i = sub_str.rfind(".");
    if (i != std::string::npos) {
      func_name = sub_str.substr(i + 1);
      // strip all spaces
      func_name.erase(std::remove_if(func_name.begin(), func_name.end(), ::isspace), func_name.end());
    }

    size_t k = gen_txt.rfind(")");
    size_t l = gen_txt.rfind("(");
    if (k != std::string::npos && l != std::string::npos && l < k) {
      func_args = gen_txt.substr(l, k - l + 1);

      // convert str to json string
      // (location='Boston', unit='celsius') -> {"location": "Boston", "unit": "celsius"}
      // strip all spaces.
      func_args.erase(std::remove_if(func_args.begin(), func_args.end(), ::isspace), func_args.end());
      // replace '(' -> '{', ')' -> '}'
      func_args[0] = '{';
      func_args[func_args.size() - 1] = '}';
      // replace ' -> "
      func_args = std::regex_replace(func_args, std::regex(R"(')"), "\"");
      // replace ' = ' -> ': '
      func_args = std::regex_replace(func_args, std::regex(R"(\s*=\s*)"), ":");
      // replace param_name -> "param_name"
      func_args = std::regex_replace(func_args, std::regex(R"((\w+):)"), R"("$1":)");
    }
  }

  if (!func_name.empty()) {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls = message["tool_calls"];
    tool_calls.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& tool_call = tool_calls[0];
    std::string call_id = std::string("call_") + std::to_string(req_id);
    tool_call.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
    tool_call.AddMember("type", rapidjson::Value("function", allocator), allocator);
    tool_call.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& function = tool_call["function"];
    function.AddMember("name", rapidjson::Value(func_name.c_str(), allocator), allocator);
    function.AddMember("arguments", rapidjson::Value(func_args.c_str(), allocator), allocator);
    return "tool_calls";
  } else {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    return "stop";
  }
}

std::tuple<bool, std::string, std::vector<std::string>> Glm4Styler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt = "[gMASK]<sop>";

  // Parse tools.
  bool has_tools = false;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    auto& tools = json_body["tools"];
    std::string content =
      "你是一个名为 GhatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 "
      "模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。";
    content += "\n\n# 可用工具";
    for (auto& tool : tools.GetArray()) {
      if (!tool.HasMember("type") || !tool["type"].IsString()) {
        throw std::invalid_argument("`type` not found or not a string");
      }
      std::string type = tool["type"].GetString();
      if (type == "function") {
        if (!tool.HasMember("function") || !tool["function"].IsObject()) {
          throw std::invalid_argument("tool `function` not found or not an object");
        }
        auto& function = tool["function"];
        if (!function.HasMember("name") || !function["name"].IsString()) {
          throw std::invalid_argument("tool function `name` not found or not a string");
        }
        const auto& name = function["name"].GetString();
        content += "\n\n## ";
        content += name;
        content += "\n\n";
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        writer.SetIndent(' ', 4);
        function.Accept(writer);
        content += buffer.GetString();
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。";
      } else {
        throw std::invalid_argument("Unsupported tool type: " + type + ", only support `function` now.");
      }
    }
    prompt += GetRole("system");
    prompt += "\n ";
    prompt += content;
    has_tools = true;
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }
  for (auto& message : json_body["messages"].GetArray()) {
    if (message.HasMember("content") && message["content"].IsString()) {
      if (!message.HasMember("role") || !message["role"].IsString()) {
        throw std::invalid_argument("`role` not found or not a string");
      }
      std::string role = message["role"].GetString();
      if (role == "tool") {
        role = "<|observation|>";
      }
      if (role != "system" && role != "user" && role != "assistant" && role != "<|observation|>") {
        throw std::invalid_argument("Unsupported message role: " + role);
      }
      role = GetRole(role);
      std::string content = message["content"].GetString();
      prompt += GetRole(role);
      prompt += "\n ";
      prompt += content;
    }
  }

  if (add_generation_prompt()) {
    prompt += GetRole("assistant");
    prompt += "\n";
  }

  return {has_tools, prompt, {}};
}

std::string Glm4Styler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  // I can help you with that. I will use the 'get_current_weather_for_location' function to get the current weather in
  // Boston.get_current_weather
  // {"location": "Boston, MA"}

  std::string func_name, func_args;
  size_t j = gen_txt.rfind("\n{");
  if (j != std::string::npos) {
    func_args = gen_txt.substr(j + 1);
    std::string sub_str = gen_txt.substr(0, j);
    size_t i = sub_str.rfind(".");
    if (i != std::string::npos) {
      func_name = sub_str.substr(i + 1);
      // strip all spaces
      func_name.erase(std::remove_if(func_name.begin(), func_name.end(), ::isspace), func_name.end());
    }
  }

  if (!func_name.empty()) {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls = message["tool_calls"];
    tool_calls.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& tool_call = tool_calls[0];
    std::string call_id = std::string("call_") + std::to_string(req_id);
    tool_call.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
    tool_call.AddMember("type", rapidjson::Value("function", allocator), allocator);
    tool_call.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& function = tool_call["function"];
    function.AddMember("name", rapidjson::Value(func_name.c_str(), allocator), allocator);
    function.AddMember("arguments", rapidjson::Value(func_args.c_str(), allocator), allocator);
    return "tool_calls";
  } else {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    return "stop";
  }
}

std::tuple<bool, std::string, std::vector<std::string>> Llama3Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  std::string prompt = "<|begin_of_text|>";

  bool first = true;
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = message["role"].GetString();
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }

    if (first) {
      first = false;
      if (role != "system") {
        prompt += "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>";
      }
    }

    std::string content = "<|start_header_id|>";
    content += role;
    content += "<|end_header_id|>\n\n";
    content += message["content"].GetString();
    // ltrim and rtrim
    content = content.substr(content.find_first_not_of(' '));
    content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                  content.end());
    content += "<|eot_id|>";
    prompt += content;
  }

  if (add_generation_prompt()) {
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  }

  return {false, prompt, {}};
}

std::string Llama3Styler::ParseFunctionCall(const std::string& gen_txt,
                                            int64_t req_id,
                                            rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                            rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Internlm2Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse tools.
  bool has_tools = false;
  std::string tool_system;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    std::vector<std::string> tools_text;
    std::vector<std::string> tools_name_text;

    for (auto& tool : json_body["tools"].GetArray()) {
      if (tool.HasMember("function") && tool["function"].IsObject()) {
        auto& function = tool["function"];

        auto& fp = function["parameters"];
        std::string parameters;
        if (fp.IsObject()) {
          std::unordered_set<std::string> required_parameters;
          if (fp.HasMember("required") && fp["required"].IsArray()) {
            for (auto& required : fp["required"].GetArray()) {
              if (!required.IsString()) {
                throw std::invalid_argument("function `required` parameter is not a string arr");
              }
              required_parameters.insert(required.GetString());
            }
          }

          // Build parameters array json.
          rapidjson::Document parameters_doc(rapidjson::kArrayType);
          for (auto& [name, p] : fp["properties"].GetObject()) {
            auto param_doc = rapidjson::Value(rapidjson::kObjectType);
            param_doc.CopyFrom(p, parameters_doc.GetAllocator());
            param_doc.AddMember("name", rapidjson::Value(name.GetString(), parameters_doc.GetAllocator()),
                                parameters_doc.GetAllocator());
            if (required_parameters.find(name.GetString()) != required_parameters.end()) {
              param_doc.AddMember("required", true, parameters_doc.GetAllocator());
            }
            parameters_doc.PushBack(param_doc, parameters_doc.GetAllocator());
          }

          // Parameters json to string
          rapidjson::StringBuffer buffer;
          rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
          parameters_doc.Accept(writer);
          parameters = buffer.GetString();
        }

        if (!function.HasMember("name") || !function["name"].IsString()) {
          throw std::invalid_argument("function `name` not found or not a string");
        }
        if (!function.HasMember("description") || !function["description"].IsString()) {
          throw std::invalid_argument("function `description` not found or not a string");
        }
        const auto& name = function["name"].GetString();
        const auto& desc = function["description"].GetString();
        std::string tool_string = name;
        tool_string.append(": Call this tool to interact with the ");
        tool_string.append(name);
        tool_string.append(" API. What is the ");
        tool_string.append(name);
        tool_string.append(" API useful for? ");
        tool_string.append(desc);
        tool_string.append(" Parameters: ");
        tool_string.append(parameters);
        tool_string.append(" Format the arguments as a JSON object.");
        tools_text.push_back(tool_string);
        tools_name_text.emplace_back(name);
      }
    }

    std::string tools_text_string;
    for (size_t i = 0; i < tools_text.size(); ++i) {
      tools_text_string.append(tools_text[i]);
      if (i != tools_text.size() - 1) {
        tools_text_string.append("\n\n");
      }
    }
    std::string tools_name_text_string;
    for (size_t i = 0; i < tools_name_text.size(); ++i) {
      tools_name_text_string.append(tools_name_text[i]);
      if (i != tools_name_text.size() - 1) {
        tools_name_text_string.append(", ");
      }
    }
    tool_system.append("Answer the following questions as best you can. You have access to the following APIs:\n\n");
    tool_system.append(tools_text_string);
    tool_system.append("\n\nUse the following format:\n\n");
    tool_system.append("Question: the input question you must answer\n");
    tool_system.append("Thought: you should always think about what to do\n");
    tool_system.append("Action: the action to take, should be one of [");
    tool_system.append(tools_name_text_string);
    tool_system.append("]\n");
    tool_system.append("Action Input: the input to the action\n");
    tool_system.append("Observation: the result of the action\n");
    tool_system.append("... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n");
    tool_system.append("Thought: I now know the final answer\n");
    tool_system.append("Final Answer: the final answer to the original input question\n\n");
    tool_system.append("Begin!");
  }

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<s><|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<s><|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string ori_content = message["content"].GetString();

    prompt.append("");

    std::string content;
    if (has_tools) {
      if (role == "user") {
        if (!tool_system.empty()) {
          content = tool_system;
          tool_system.clear();
          content.append("\n\nQuestion: ");
          content.append(ori_content);
        } else {
          content.append("Question: ");
          content.append(ori_content);
        }
      } else if (role == "assistant") {
        if (message.HasMember("tool_calls") && message["tool_calls"].IsArray()) {
          auto& tool_calls = message["tool_calls"];
          if (!tool_calls.Empty()) {
            if (!tool_calls[0].HasMember("function") || !tool_calls[0]["function"].IsObject()) {
              throw std::invalid_argument("`function` not found in `tool_calls` or not an object");
            }
            auto& func_call = tool_calls[0]["function"];
            if (!func_call.HasMember("name") || !func_call["name"].IsString()) {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            auto& f_name = func_call["name"];
            if (!func_call.HasMember("arguments") || !func_call["arguments"].IsString()) {
              throw std::invalid_argument("`arguments` not found in `function` or not a string");
            }
            auto& f_args = func_call["arguments"];
            content.append("Thought: I can use ");
            content.append(f_name.GetString());
            content.append(".\nAction: ");
            content.append(f_name.GetString());
            content.append("\nAction Input: ");
            content.append(f_args.GetString());
          } else if (!ori_content.empty()) {
            content.append("Thought: I now know the final answer.\nFinal answer: ");
            content.append(ori_content);
          }
        }
      } else if (role == "tool") {
        role = "function";
        content.append("Observation: ");
        content.append(ori_content);
      } else {
        throw std::invalid_argument("Unsupported message role: " + role);
      }
    } else {
      content = std::move(ori_content);
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {has_tools, prompt, {}};
}

std::string Internlm2Styler::ParseFunctionCall(const std::string& gen_txt,
                                               int64_t req_id,
                                               rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                               rapidjson::MemoryPoolAllocator<>& allocator) {
  std::string thought, func_name, func_args;
  size_t i = gen_txt.rfind("\nAction:");
  size_t j = gen_txt.rfind("\nAction Input:");
  size_t k = gen_txt.rfind("\nObservation:");

  if (i != std::string::npos && j != std::string::npos) {
    // Found `Thought:` before `Action:`.
    std::string sub_str = gen_txt.substr(0, i);
    size_t h1 = sub_str.rfind("\nThought:");
    size_t h2 = sub_str.rfind("Thought:");
    size_t h = std::string::npos;
    if (h1 != std::string::npos && h2 != std::string::npos) {
      h = std::max(h1, h2);
    } else if (h1 != std::string::npos) {
      h = h1;
    } else if (h2 != std::string::npos) {
      h = h2;
    }

    if (h != std::string::npos && h < i && i < j) { // If the text has `Action` and `Action input`,
      if (k == std::string::npos) {                 // but does not contain `Observation`,
        // then it is likely that `Observation` is omitted by the LLM,
        // because the output text may have discarded the stop word.
        // gen_txt.append("\nObservation:");
        // k = gen_txt.rfind("\nObservation:");
        k = gen_txt.size();
      }
      thought = gen_txt.substr(h + 9, i - h - 9);
      thought = thought.substr(thought.find_first_not_of(' ')); // strip left spaces
      func_name = gen_txt.substr(i + 8, j - i - 8);
      func_args = gen_txt.substr(j + 14, k - j - 14);
      // strip all spaces
      func_name.erase(std::remove_if(func_name.begin(), func_name.end(), ::isspace), func_name.end());
      func_args.erase(std::remove_if(func_args.begin(), func_args.end(), ::isspace), func_args.end());
    }
  }
  if (!func_name.empty()) {
    message.AddMember("content", rapidjson::Value(thought.c_str(), allocator), allocator);
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls = message["tool_calls"];
    tool_calls.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& tool_call = tool_calls[0];
    std::string call_id = std::string("call_") + std::to_string(req_id);
    tool_call.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
    tool_call.AddMember("type", rapidjson::Value("function", allocator), allocator);
    tool_call.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);
    auto& function = tool_call["function"];
    function.AddMember("name", rapidjson::Value(func_name.c_str(), allocator), allocator);
    function.AddMember("arguments", rapidjson::Value(func_args.c_str(), allocator), allocator);
    return "tool_calls";
  } else {
    size_t y1 = gen_txt.rfind("\nThought:");
    size_t y2 = gen_txt.rfind("Thought:");
    size_t y = std::string::npos;
    if (y1 != std::string::npos && y2 != std::string::npos) {
      y = std::max(y1, y2);
    } else if (y1 != std::string::npos) {
      y = y1;
    } else if (y2 != std::string::npos) {
      y = y2;
    }

    size_t z1 = gen_txt.rfind("\nFinal Answer:");
    size_t z2 = gen_txt.rfind("Final Answer:");
    size_t z = std::string::npos;
    if (z1 != std::string::npos && z2 != std::string::npos) {
      z = std::max(z1, z2);
    } else if (z1 != std::string::npos) {
      z = z1;
    } else if (z2 != std::string::npos) {
      z = z2;
    }

    if (z != std::string::npos) {
      std::string final_answer = gen_txt.substr(z + 14);
      final_answer = final_answer.substr(final_answer.find_first_not_of(' ')); // strip left spaces
      message.AddMember("content", rapidjson::Value(final_answer.c_str(), allocator), allocator);
    } else if (y != std::string::npos) { // Only have thought.
      std::string thought_answer = gen_txt.substr(y + 9);
      thought_answer = thought_answer.substr(thought_answer.find_first_not_of(' ')); // strip left spaces
      message.AddMember("content", rapidjson::Value(thought_answer.c_str(), allocator), allocator);
    }
    return "stop";
  }
}

std::tuple<bool, std::string, std::vector<std::string>> Internvl2Internlm2Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      size_t img_count = utils::GetWordCount(content, "<image>");
      if (img_count != cur_img_urls.size()) {
        throw std::invalid_argument("The number of <image> tokens does not match the number of image urls");
      }

      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {false, prompt, image_urls};
}

std::string Internvl2Internlm2Styler::ParseFunctionCall(const std::string& gen_txt,
                                                        int64_t req_id,
                                                        rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                                        rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Internvl2Phi3Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt.append(GetRole("system"));
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|end|>");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt.append(GetRole("system"));
    prompt.append(system_prompt());
    prompt.append("<|end|>");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      size_t img_count = utils::GetWordCount(content, "<image>");
      if (img_count != cur_img_urls.size()) {
        throw std::invalid_argument("The number of <image> tokens does not match the number of image urls");
      }

      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append(role);
      prompt.append(content);
      prompt.append("<|end|>");
    } else {
      prompt.append(role);
      prompt.append("<|end|>");
    }
  }

  if (add_generation_prompt()) {
    prompt.append(GetRole("assistant"));
  }
  return {false, prompt, image_urls};
}

std::string Internvl2Phi3Styler::ParseFunctionCall(const std::string& gen_txt,
                                                   int64_t req_id,
                                                   rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                                   rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Internvl25Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      size_t img_count = utils::GetWordCount(content, "<image>");
      if (img_count != cur_img_urls.size()) {
        throw std::invalid_argument("The number of <image> tokens does not match the number of image urls");
      }

      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {false, prompt, image_urls};
}

std::string Internvl25Styler::ParseFunctionCall(const std::string& gen_txt,
                                                int64_t req_id,
                                                rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                                rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> InternVideo25Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> video_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "What'\''s in this image?"
      //           },
      //           {
      //             "type": "video_url",
      //             "video_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_video_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "video_url") {
          if (!item.HasMember("video_url") || !item["video_url"].IsObject()) {
            throw std::invalid_argument("`video_url` not found or not an object");
          }
          auto& video_url = item["video_url"];
          if (!video_url.HasMember("url") || !video_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_video_urls.emplace_back(video_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      if (cur_video_urls.size() > 1) {
        throw std::invalid_argument("Current only support one video url");
      }
      if (!cur_video_urls.empty()) {
        content = "<VIDEO_CONTEXT>" + content; // Insert video context placeholder.
      }

      video_urls.insert(video_urls.end(), cur_video_urls.begin(), cur_video_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {false, prompt, video_urls};
}

std::string InternVideo25Styler::ParseFunctionCall(const std::string& gen_txt,
                                                   int64_t req_id,
                                                   rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                                   rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> QwenvlStyler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
          content.append("Picture ");
          content.append(std::to_string(cur_img_urls.size()));
          content.append(": ");
          content.append(img_ctx_replace_str_);
          content.append("\n");
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {false, prompt, image_urls};
}

std::string QwenvlStyler::ParseFunctionCall(const std::string& gen_txt,
                                            int64_t req_id,
                                            rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                            rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Qwen2vlStyler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt = "<|im_start|>system\n";
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|im_end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt = "<|im_start|>system\n";
    prompt.append(system_prompt());
    prompt.append("<|im_end|>\n");
  }
  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
          // content.append("Picture ");
          // content.append(std::to_string(cur_img_urls.size()));
          // content.append(": ");
          content.append("<|vision_start|><|image_pad|><|vision_end|>");
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.lstrip("\n")
      content = content.substr(content.find_first_not_of('\n'));
      // content.rstrip()
      content.erase(std::find_if(content.rbegin(), content.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    content.end());
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
      prompt.append(content);
      prompt.append("<|im_end|>\n");
    } else {
      prompt.append("<|im_start|>");
      prompt.append(role);
      prompt.append("\n");
    }
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>assistant\n");
  }
  return {false, prompt, image_urls};
}

std::string Qwen2vlStyler::ParseFunctionCall(const std::string& gen_txt,
                                             int64_t req_id,
                                             rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                             rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Phi3Styler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    prompt.append(GetRole("system"));
    prompt.append(json_body["messages"][0]["content"].GetString());
    prompt.append("<|end|>\n");
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    prompt.append(GetRole("system"));
    prompt.append(system_prompt());
    prompt.append("<|end|>\n");
  }
  // Following messages.
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string content = message["content"].GetString();

    prompt.append(role);
    prompt.append(content);
    prompt.append("<|end|>\n");
  }

  if (add_generation_prompt()) {
    prompt.append(GetRole("assistant"));
  }
  return {false, prompt, {}};
}

std::string Phi3Styler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Phi4Styler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool skip_first = false;
  std::string sys_prompt;
  // System message.
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // If the first message is a system message, use it as the system prompt.
    if (!json_body["messages"][0].HasMember("content") || !json_body["messages"][0]["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    sys_prompt = json_body["messages"][0]["content"].GetString();
    skip_first = true;
  } else {
    // If the first message is not a system message, add a system message.
    sys_prompt = system_prompt();
  }
  prompt.append("<|im_start|>");
  prompt.append(GetRole("system"));
  prompt.append("<|im_sep|>");
  prompt.append(sys_prompt);
  prompt.append("<|im_end|>");

  // Following messages.
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string content = message["content"].GetString();

    prompt.append("<|im_start|>");
    prompt.append(role);
    prompt.append("<|im_sep|>");
    prompt.append(content);
    prompt.append("<|im_end|>");
  }

  if (add_generation_prompt()) {
    prompt.append("<|im_start|>");
    prompt.append(GetRole("assistant"));
    prompt.append("<|im_sep|>");
  }
  return {false, prompt, {}};
}

std::string Phi4Styler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> DeepSeekR1Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt = "<｜begin▁of▁sentence｜>";

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  // Only use the last system message.
  std::string system_content;
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    if (std::string(message["role"].GetString()) != "system") {
      continue;
    }

    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    system_content = message["content"].GetString();
  }
  prompt.append(system_content);

  // Following messages.
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    if (std::string(message["role"].GetString()) == "system") { // Skip system message.
      continue;
    }
    std::string role = GetRole(message["role"].GetString());

    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    std::string content = message["content"].GetString();

    prompt.append(role);

    if (role == "<｜Assistant｜>") {
      auto think_end = content.find("</think>");
      if (think_end != std::string::npos) {
        content = content.substr(think_end + 8);
      }
      prompt.append(content);
      prompt.append("<｜end▁of▁sentence｜>");
    } else {
      prompt.append(content);
    }
  }

  if (add_generation_prompt()) {
    prompt.append(GetRole("assistant"));
    prompt.append("<think>\n");
  }
  return {false, prompt, {}};
}

std::string DeepSeekR1Styler::ParseFunctionCall(const std::string& gen_txt,
                                                int64_t req_id,
                                                rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                                rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> JanusProStyler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt = "";
  prompt.append(GetRole("system"));
  prompt.append(system_prompt());
  prompt.append("\n\n");

  // Following messages.
  // Parse images urls.
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = GetRole(message["role"].GetString());
    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    if (message["content"].IsString()) {
      content = message["content"].GetString();
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "<image_placeholder>\nWhat'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(item["text"].GetString());
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      size_t img_count = utils::GetWordCount(content, "<image_placeholder>");
      if (img_count != cur_img_urls.size()) {
        throw std::invalid_argument("The number of <image_placeholder> tokens does not match the number of image urls");
      }

      // Replace <image_placeholder> with (<begin_of_image><image_placeholder>*576<end_of_image>)
      // 576: token count for every image.
      if (img_count > 0) {
        utils::ReplaceWorld(content, "<image_placeholder>", img_ctx_replace_str_, 0, img_count);
      }

      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    }

    if (!content.empty()) {
      // content.strip()
      content = utils::Strip(content);
      prompt.append(role);
      prompt.append(": ");
      prompt.append(content);
      if (role == GetRole("assistant")) {
        prompt.append("<｜end▁of▁sentence｜>");
      } else {
        prompt.append("\n\n");
      }
    } else {
      prompt.append(role);
      prompt.append(":");
    }
  }

  if (add_generation_prompt()) {
    prompt.append(GetRole("assistant"));
    prompt.append(":");
  }
  return {false, prompt, image_urls};
}

std::string JanusProStyler::ParseFunctionCall(const std::string& gen_txt,
                                              int64_t req_id,
                                              rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                              rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Gemma3Styler::BuildPrompt(
  const rapidjson::Document& json_body) {
  std::string prompt = "<bos>";

  // System message.
  std::string first_user_prefix;
  bool skip_first = false;
  if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
      std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    if (json_body["messages"][0].HasMember("content") && json_body["messages"][0]["content"].IsString()) {
      first_user_prefix = json_body["messages"][0]["content"].GetString();
      first_user_prefix.append("\n\n");
    } else if (json_body["messages"][0].HasMember("content") && json_body["messages"][0]["content"].IsArray()) {
      if (json_body["messages"][0]["content"][0].HasMember("text") &&
          json_body["messages"][0]["content"][0]["text"].IsString()) {
        first_user_prefix = json_body["messages"][0]["content"][0]["text"].GetString();
        first_user_prefix.append("\n\n");
      }
    }
    skip_first = true;
  }

  // Parse following messages.
  int idx = 0;
  std::vector<std::string> image_urls;
  for (auto& message : json_body["messages"].GetArray()) {
    if (skip_first) {
      skip_first = false;
      continue;
    }
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }

    std::string role = GetRole(message["role"].GetString());
    if (role == "user" && idx % 2 == 1) {
      throw std::invalid_argument("Conversation roles must alternate user/assistant/user/assistant/...");
    } else if (role == "model" && idx % 2 == 0) {
      throw std::invalid_argument("Conversation roles must alternate user/assistant/user/assistant/...");
    }

    if (!message.HasMember("content")) {
      throw std::invalid_argument("`content` not found");
    }

    std::string content;
    std::string image_pre;
    if (message["content"].IsString()) {
      content = utils::Strip(message["content"].GetString());
    } else if (message["content"].IsArray()) {
      //  "content": [
      //           {
      //             "type": "text",
      //             "text": "What'\''s in this image?"
      //           },
      //           {
      //             "type": "image_url",
      //             "image_url": {
      //               "url": "https://**"
      //             }
      //           }
      //         ]
      std::vector<std::string> cur_img_urls;
      for (auto& item : message["content"].GetArray()) {
        if (!item.HasMember("type") || !item["type"].IsString()) {
          throw std::invalid_argument("`type` not found or not a string");
        }
        std::string type = item["type"].GetString();
        if (type == "text") {
          if (!item.HasMember("text") || !item["text"].IsString()) {
            throw std::invalid_argument("`text` not found or not a string");
          }
          content.append(utils::Strip(item["text"].GetString()));
        } else if (type == "image_url") {
          if (!item.HasMember("image_url") || !item["image_url"].IsObject()) {
            throw std::invalid_argument("`image_url` not found or not an object");
          }
          auto& image_url = item["image_url"];
          if (!image_url.HasMember("url") || !image_url["url"].IsString()) {
            throw std::invalid_argument("`url` not found or not a string");
          }
          cur_img_urls.emplace_back(image_url["url"].GetString());
        } else {
          throw std::invalid_argument("Unsupported content type: " + type);
        }
      }
      for (size_t i = 0; i < cur_img_urls.size(); ++i) {
        image_pre.append(img_ctx_replace_str_);
      }
      image_urls.insert(image_urls.end(), cur_img_urls.begin(), cur_img_urls.end());
    } else {
      throw std::invalid_argument("Unsupported content type");
    }

    prompt.append("<start_of_turn>");
    prompt.append(role);
    prompt.append("\n");
    if (idx == 0) {
      prompt.append(first_user_prefix);
    }
    if (!content.empty()) {
      // content.strip()
      content = utils::Strip(content);
      prompt.append(image_pre);
      prompt.append(content);
    }
    prompt.append("<end_of_turn>\n");

    idx++;
  }

  if (add_generation_prompt()) {
    prompt.append("<start_of_turn>");
    prompt.append(GetRole("assistant"));
    prompt.append("\n");
  }

  return {false, prompt, image_urls};
}

std::string Gemma3Styler::ParseFunctionCall(const std::string& gen_txt,
                                            int64_t req_id,
                                            rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                            rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> MiniCPMVStyler::BuildPrompt(
  const rapidjson::Document& json_body) {
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }

  // json body convert to string directly, will parse by vit.
  std::string prompt;
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json_body["messages"].Accept(writer);
  prompt.append(buffer.GetString());
  return {false, prompt, {}};
}

std::string MiniCPMVStyler::ParseFunctionCall(const std::string& gen_txt,
                                              int64_t req_id,
                                              rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                              rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string, std::vector<std::string>> Qwen3Styler::BuildPrompt(const rapidjson::Document& json_body) {
  std::string prompt;

  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool func_call = false;
  bool skip_first = false;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    // Parse tools.
    prompt += "<|im_start|>system\n";
    if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
        std::string(json_body["messages"][0]["role"].GetString()) == "system") {
      if (json_body["messages"][0].HasMember("content") && json_body["messages"][0]["content"].IsString()) {
        prompt += json_body["messages"][0]["content"].GetString();
      }
      prompt += "\n\n";
      skip_first = true;
    }

    prompt += tool_prompt_pre_;
    for (auto& tool : json_body["tools"].GetArray()) { // parse per tool to json.
      prompt += "\n";
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      tool.Accept(writer);
      // Add space after colon and comma to match python json.
      prompt += utils::JsonAddSpaceAfterColonAndComma(buffer.GetString());
    }
    prompt += tool_prompt_post_;

    prompt += "<|im_end|>\n";

    func_call = true;
  } else if (json_body["messages"][0].HasMember("role") && json_body["messages"][0]["role"].IsString() &&
             std::string(json_body["messages"][0]["role"].GetString()) == "system") {
    // Parse system message.
    prompt += "<|im_start|>system\n";
    if (json_body["messages"][0].HasMember("content") && json_body["messages"][0]["content"].IsString()) {
      prompt += json_body["messages"][0]["content"].GetString();
    }
    prompt += "<|im_end|>\n";
    skip_first = true;
  }

  bool multi_step_tool = true;
  size_t last_query_index = json_body["messages"].Size() - 1; // last user query index
  for (int i = last_query_index; i >= 0; i--) {
    if (!json_body["messages"][i].HasMember("role") || !json_body["messages"][i]["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }

    auto role = std::string(json_body["messages"][i]["role"].GetString());
    std::string content;
    if (json_body["messages"][i].HasMember("content") && json_body["messages"][i]["content"].IsString()) {
      content = json_body["messages"][i]["content"].GetString();
    }
    if (multi_step_tool && role == "user" && !utils::StartsWith(content, "<tool_response>") &&
        !utils::EndsWith(content, "</tool_response>")) {
      multi_step_tool = false;
      last_query_index = i;
    }
  }

  for (size_t i = 0; i < json_body["messages"].Size(); i++) {
    if (skip_first) {
      skip_first = false;
      continue;
    }
    auto& message = json_body["messages"][i];

    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    auto role = std::string(message["role"].GetString());

    if (role == "user" || role == "system") {
      std::string content;
      if (message.HasMember("content") && message["content"].IsString()) {
        content = json_body["messages"][i]["content"].GetString();
      }

      prompt += "<|im_start|>";
      prompt += GetRole(role);
      prompt += "\n";
      prompt += message["content"].GetString();
      prompt += "<|im_end|>\n";
    } else if (role == "assistant") {
      std::string content;
      if (message.HasMember("content") && message["content"].IsString()) {
        content = json_body["messages"][i]["content"].GetString();
      }
      std::string reasoning_content;

      if (content.find("</think>") != std::string::npos) {
        // {%- set reasoning_content =
        // message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
        reasoning_content = content.substr(0, content.find("</think>"));
        reasoning_content = utils::Rstrip(reasoning_content, "\n");
        reasoning_content = reasoning_content.substr(reasoning_content.find("<think>") + 7);
        reasoning_content = utils::Lstrip(reasoning_content, "\n");

        // {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}
        content = content.substr(content.find("</think>") + 8);
        content = utils::Lstrip(content, "\n");
      }

      if (i > last_query_index) {
        if (i == json_body["messages"].Size() - 1 ||
            (i != json_body["messages"].Size() - 1 && !reasoning_content.empty())) {
          prompt += "<|im_start|>";
          prompt += GetRole(role);
          prompt += "\n<think>\n";
          prompt += reasoning_content;
          prompt += "\n</think>\n\n";
          prompt += content;
        } else {
          prompt += "<|im_start|>";
          prompt += GetRole(role);
          prompt += "\n";
          prompt += content;
        }
      } else {
        prompt += "<|im_start|>";
        prompt += GetRole(role);
        prompt += "\n";
        prompt += content;
      }

      if (message.HasMember("tool_calls") && message["tool_calls"].IsArray()) {
        for (size_t j = 0; j < message["tool_calls"].Size(); j++) {
          if ((j == 0 && !content.empty()) || j != 0) {
            prompt += "\n";
          }
          auto& tool_call = message["tool_calls"][j];
          if (tool_call.HasMember("function") && tool_call["function"].IsObject()) {
            auto& function = tool_call["function"];
            prompt += "<tool_call>\n{\"name\": \"";
            if (function.HasMember("name") && function["name"].IsString()) {
              prompt += function["name"].GetString();
            } else {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            prompt += "\", \"arguments\": ";
            if (function.HasMember("arguments")) {
              if (function["arguments"].IsString()) {
                prompt += function["arguments"].GetString();
              } else if (function["arguments"].IsObject()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                function["arguments"].Accept(writer);
                prompt += utils::JsonAddSpaceAfterColonAndComma(buffer.GetString());
              } else {
                throw std::invalid_argument("`arguments` not found in `function` or not a string");
              }
            } else {
              throw std::invalid_argument("`arguments` not found in `function`");
            }
            prompt += "}\n</tool_call>";
          } else {
            throw std::invalid_argument("`function` not found in `tool_call` or not an object");
          }
        }
      }
      prompt += "<|im_end|>\n";
    } else if (role == "tool") {
      if (message.HasMember("content") && message["content"].IsString()) {
        if (i == 0 || std::string(json_body["messages"][i - 1]["role"].GetString()) != "tool") {
          prompt += "<|im_start|>user";
        }
        prompt += "\n<tool_response>\n";
        prompt += message["content"].GetString();
        prompt += "\n</tool_response>";
        if (i == json_body["messages"].Size() - 1 ||
            std::string(json_body["messages"][i + 1]["role"].GetString()) != "tool") {
          prompt += "<|im_end|>\n";
        }
      }
      if (i == json_body["messages"].Size() - 1) {
        func_call = false;
      }
    } else {
      throw std::invalid_argument("Unsupported role: " + role);
    }
  }

  if (add_generation_prompt()) {
    prompt += "<|im_start|>";
    prompt += GetRole("assistant");
    prompt += "\n";
  }

  if (json_body.HasMember("enable_thinking") && json_body["enable_thinking"].IsBool() &&
      !json_body["enable_thinking"].GetBool()) {
    prompt += "<think>\n\n</think>\n\n";
  }

  return {func_call, prompt, {}};
}

std::string Qwen3Styler::ParseFunctionCall(const std::string& gen_txt,
                                           int64_t req_id,
                                           rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                           rapidjson::MemoryPoolAllocator<>& allocator) {
  // <tool_call>
  // {{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}}
  // </tool_call>
  // <tool_call>
  // {{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}}
  // </tool_call>

  std::vector<std::string> tool_calls;
  size_t start = 0;
  while (true) {
    size_t tool_call_start = gen_txt.find("<tool_call>", start);
    if (tool_call_start == std::string::npos) {
      break;
    }
    size_t tool_call_end = gen_txt.find("</tool_call>", tool_call_start);
    if (tool_call_end == std::string::npos) {
      break;
    }
    auto tool_call = gen_txt.substr(tool_call_start + 11, tool_call_end - tool_call_start - 11);
    // lstrip \n
    tool_call = tool_call.substr(tool_call.find_first_not_of('\n'));
    // rstrip \n
    tool_call.erase(std::find_if(tool_call.rbegin(), tool_call.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    tool_call.end());
    if (tool_call.size() < 4) {
      throw std::invalid_argument("Invalid tool call: " + tool_call);
    }
    if (utils::StartsWith(tool_call, "{{") && utils::EndsWith(tool_call, "}}")) {
      tool_calls.emplace_back(tool_call.substr(1, tool_call.size() - 2)); // strip {}
    } else {
      tool_calls.emplace_back(tool_call);
    }
    start = tool_call_end + 12;
  }

  if (!tool_calls.empty()) {
    message.AddMember("tool_calls", rapidjson::Value(rapidjson::kArrayType), allocator);
    auto& tool_calls_array = message["tool_calls"];
    for (const auto& tool_call : tool_calls) {
      tool_calls_array.PushBack(rapidjson::Value(rapidjson::kObjectType), allocator);
      auto& tool_call_json = tool_calls_array[tool_calls_array.Size() - 1];
      std::string call_id = std::string("call_") + std::to_string(req_id);
      tool_call_json.AddMember("id", rapidjson::Value(call_id.c_str(), allocator), allocator);
      tool_call_json.AddMember("type", rapidjson::Value("function", allocator), allocator);
      tool_call_json.AddMember("function", rapidjson::Value(rapidjson::kObjectType), allocator);

      // parse function from tool_call
      // CLOG4(INFO, "tool_call: " << tool_call);
      auto func_doc = rapidjson::Value(rapidjson::kObjectType);
      rapidjson::Document tool_call_doc;
      tool_call_doc.Parse(tool_call.c_str());
      if (tool_call_doc.HasParseError()) {
        // throw std::invalid_argument("Parse tool call failed, tool_call: " + tool_call);
        CLOG4(ERROR, "Parse tool call failed, tool_call: " << tool_call);
        continue;
      }
      if (!tool_call_doc.HasMember("name") || !tool_call_doc["name"].IsString()) {
        throw std::invalid_argument("`name` not found in `tool_call` or not a string");
      }
      func_doc.AddMember("name", rapidjson::Value(tool_call_doc["name"].GetString(), allocator), allocator);
      if (!tool_call_doc.HasMember("arguments")) {
        throw std::invalid_argument("`arguments` not found in `tool_call` or not an json object or an string");
      }
      if (tool_call_doc["arguments"].IsString()) {
        func_doc.AddMember("arguments", rapidjson::Value(tool_call_doc["arguments"].GetString(), allocator), allocator);
      } else {
        // arguments to json str.
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        tool_call_doc["arguments"].Accept(writer);
        func_doc.AddMember("arguments", rapidjson::Value(buffer.GetString(), allocator), allocator);
      }

      tool_call_json.AddMember("function", func_doc, allocator);
    }
    return "tool_calls";
  } else {
    message.AddMember("content", rapidjson::Value(gen_txt.c_str(), allocator), allocator);
    return "stop";
  }
}

std::unique_ptr<LLMStyler> LLMStylerFactory::CreateLLMStyler(const std::string& llm_style,
                                                             const std::string& chat_template) {
  std::unique_ptr<LLMStyler> llm_styler;
  if (llm_style == "qwen") {
    llm_styler = std::make_unique<QwenStyler>();
  } else if (llm_style == "qwenvl") {
    llm_styler = std::make_unique<QwenvlStyler>();
  } else if (llm_style == "qwen2.5") {
    llm_styler = std::make_unique<Qwen25Styler>();
  } else if (llm_style == "qwen2vl") {
    llm_styler = std::make_unique<Qwen2vlStyler>();
  } else if (llm_style == "qwq") {
    llm_styler = std::make_unique<QwQStyler>();
  } else if (llm_style == "qwq-preview") {
    llm_styler = std::make_unique<QwQPreviewStyler>();
  } else if (llm_style == "qwen3") {
    llm_styler = std::make_unique<Qwen3Styler>();
  } else if (llm_style == "chatglm3") {
    llm_styler = std::make_unique<ChatGlm3Styler>();
  } else if (llm_style == "glm4") {
    llm_styler = std::make_unique<Glm4Styler>();
  } else if (llm_style == "llama3") {
    llm_styler = std::make_unique<Llama3Styler>();
  } else if (llm_style == "internlm2") {
    llm_styler = std::make_unique<Internlm2Styler>();
  } else if (llm_style == "internvl2-internlm2") {
    llm_styler = std::make_unique<Internvl2Internlm2Styler>();
  } else if (llm_style == "internvl2-phi3") {
    llm_styler = std::make_unique<Internvl2Phi3Styler>();
  } else if (llm_style == "internvl2-qwen2") {
    llm_styler = std::make_unique<Internvl2Qwen2Styler>();
  } else if (llm_style == "internvl2.5") {
    llm_styler = std::make_unique<Internvl25Styler>();
  } else if (llm_style == "intern-video2.5") {
    llm_styler = std::make_unique<InternVideo25Styler>();
  } else if (llm_style == "internvl3") {
    llm_styler = std::make_unique<Internvl3Styler>();
  } else if (llm_style == "phi3") {
    llm_styler = std::make_unique<Phi3Styler>();
  } else if (llm_style == "phi4") {
    llm_styler = std::make_unique<Phi4Styler>();
  } else if (llm_style == "deepseek-r1") {
    llm_styler = std::make_unique<DeepSeekR1Styler>();
  } else if (llm_style == "janus-pro") {
    llm_styler = std::make_unique<JanusProStyler>();
  } else if (llm_style == "gemma3") {
    llm_styler = std::make_unique<Gemma3Styler>();
  } else if (llm_style == "minicpmv") {
    llm_styler = std::make_unique<MiniCPMVStyler>();
  } else {
    throw std::runtime_error("LLM style " + llm_style + " not supported now.");
  }

  llm_styler->ApplyChatTemplate(chat_template);
  return llm_styler;
}
} // namespace netease::grps