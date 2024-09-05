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

std::tuple<bool, std::string> ChatGlm3Styler::BuildPrompt(const rapidjson::Document& json_body) {
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
    if (role != "system" && role != "user" && role != "assistant" && role != "observation") {
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

  return {has_tools, prompt};
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
      std::regex re(R"(\w+)");
      std::smatch match;
      std::string json_str = "{";
      while (std::regex_search(func_args, match, re)) {
        json_str += "\"" + match.str() + "\": ";
        func_args = match.suffix();
        std::cout << func_args << std::endl;
        if (std::regex_search(func_args, match, re)) {
          json_str += "\"" + match.str() + "\", ";
          func_args = match.suffix();
        }
      }
      json_str.pop_back();
      json_str.pop_back();
      json_str += "}";

      func_args = std::move(json_str);
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

std::tuple<bool, std::string> Glm4Styler::BuildPrompt(const rapidjson::Document& json_body) {
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
      if (role != "system" && role != "user" && role != "assistant" && role != "observation") {
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

  return {has_tools, prompt};
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