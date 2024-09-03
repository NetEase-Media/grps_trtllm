// Prompt builder and llm generated txt parser for different llm model family.

#include "llm_styler.h"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "logger/logger.h"
#include "src/utils.h"

namespace netease::grps {

std::tuple<bool, std::string> LLMStyler::BuildPromptWithChatTemplate(const rapidjson::Document& json_body,
                                                                     jinja2::Template* tpl) {
  if (!tpl) {
    throw std::invalid_argument("Jinja2 chat templater is null.");
  }

  jinja2::ValuesMap params;

  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  params["messages"] = utils::RapidJson2JinjaVal(json_body["messages"]);

  // Parse tools.
  bool has_tools = false;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    params["tools"] = utils::RapidJson2JinjaVal(json_body["tools"]);
  }

  if (add_generation_prompt_) {
    params["add_generation_prompt"] = true;
  }

  return {has_tools, tpl->RenderAsString(params).value()};
}

std::tuple<bool, std::string> LLMStyler::BuildPrompt(const rapidjson::Document& json_body) {
  throw std::runtime_error("BuildPrompt not implemented.");
}

std::string LLMStyler::ParseFunctionCall(const std::string& gen_txt,
                                         int64_t req_id,
                                         rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                         rapidjson::MemoryPoolAllocator<>& allocator) {
  throw std::runtime_error("ParseFunctionCall not implemented.");
}

std::tuple<bool, std::string> QwenStyler::BuildPrompt(const rapidjson::Document& json_body) {
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

  prompt.append("<|im_start|>assistant\n");
  return {has_tools, prompt};
}

std::string QwenStyler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  std::string thought, func_name, func_args;
  size_t h = gen_txt.rfind("Thought:");
  size_t i = gen_txt.rfind("\nAction:");
  size_t j = gen_txt.rfind("\nAction Input:");
  size_t k = gen_txt.rfind("\nObservation:");
  if (h != std::string::npos && i != std::string::npos && j != std::string::npos) {
    if (h < i && i < j) {           // If the text has `Action` and `Action input`,
      if (k == std::string::npos) { // but does not contain `Observation`,
        // then it is likely that `Observation` is omitted by the LLM,
        // because the output text may have discarded the stop word.
        // gen_txt.append("\nObservation:");
        // k = gen_txt.rfind("\nObservation:");
        k = gen_txt.size();
      }
      thought = gen_txt.substr(h + 8, i - h - 8);
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
    size_t z = gen_txt.rfind("\nFinal Answer: ");
    if (z != std::string::npos) {
      message.AddMember("content", rapidjson::Value(gen_txt.substr(z + 15).c_str(), allocator), allocator);
    }
    return "stop";
  }
}

std::tuple<bool, std::string> ChatGlm3Styler::BuildPrompt(const rapidjson::Document& json_body) {
  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }
  jinja2::ValuesMap params;
  params["messages"] = jinja2::ValuesList();
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = message["role"].GetString();
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    params["messages"].asList().emplace_back(jinja2::ValuesMap{
      {"role", role},
      {"content", message["content"].GetString()},
    });
  }

  // Parse tools.
  std::string tools_str;
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    // To string.
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetIndent(' ', 4);
    json_body["tools"].Accept(writer);
    tools_str = buffer.GetString();
  }
  if (!tools_str.empty()) {
    params["messages"].asList().front().asMap()["content"].asString().append("\n" + tools_str);
  }

  params["add_generation_prompt"] = true;

  // jinja render to string.
  return {!tools_str.empty(), chat_templater_.RenderAsString(params).value()};
}

std::string ChatGlm3Styler::ParseFunctionCall(const std::string& gen_txt,
                                              int64_t req_id,
                                              rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                              rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::tuple<bool, std::string> Glm4Styler::BuildPrompt(const rapidjson::Document& json_body) {
  // Parse messages.
  if (!json_body.HasMember("messages") || !json_body["messages"].IsArray()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].Empty()) {
    throw std::invalid_argument("`messages` is empty");
  }
  jinja2::ValuesMap params;
  params["messages"] = jinja2::ValuesList();
  for (auto& message : json_body["messages"].GetArray()) {
    if (!message.HasMember("role") || !message["role"].IsString()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    std::string role = message["role"].GetString();
    if (!message.HasMember("content") || !message["content"].IsString()) {
      throw std::invalid_argument("`content` not found or not a string");
    }
    params["messages"].asList().emplace_back(jinja2::ValuesMap{
      {"role", role},
      {"content", message["content"].GetString()},
    });
  }

  bool has_tools;
  // Parse tools, if there is tools, will insert to first `message`.
  if (json_body.HasMember("tools") && json_body["tools"].IsArray()) {
    has_tools = true;
    auto& first_message = params["messages"].asList().front().asMap();
    first_message["tools"] = jinja2::ValuesList();
    for (auto& tool : json_body["tools"].GetArray()) {
      if (tool.HasMember("type") && tool["type"].IsString()) {
        if (std::string(tool["type"].GetString()) == "function") {
          if (!tool.HasMember("function") || !tool["function"].IsObject()) {
            throw std::invalid_argument("`function` not found or not an object");
          }
          auto& function = tool["function"];
          if (!function.HasMember("name") || !function["name"].IsString()) {
            throw std::invalid_argument("function `name` not found or not a string");
          }

          auto function_map = utils::RapidJson2JinjaVal(function);

          jinja2::ValuesMap tool_map;
          tool_map["type"] = "function";
          tool_map["function"] = function_map;
          first_message["tools"].asList().emplace_back(tool_map);
        } else {
          throw std::invalid_argument("Unsupported tool type: " + std::string(tool["type"].GetString()));
        }
      }
    }
  }

  params["add_generation_prompt"] = true;

  // jinja render to string.
  return {has_tools, chat_templater_.RenderAsString(params).value()};
}

std::string Glm4Styler::ParseFunctionCall(const std::string& gen_txt,
                                          int64_t req_id,
                                          rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                          rapidjson::MemoryPoolAllocator<>& allocator) {
  return "";
}

std::unique_ptr<LLMStyler> LLMStylerFactory::CreateLLMStyler(const std::string& llm_style) {
  if (llm_style == "qwen") {
    return std::make_unique<QwenStyler>();
  } else if (llm_style == "chatglm3") {
    return std::make_unique<ChatGlm3Styler>();
  } else if (llm_style == "glm4") {
    return std::make_unique<Glm4Styler>();
  } else {
    throw std::runtime_error("LLM style " + llm_style + " not supported now.");
  }
}
} // namespace netease::grps