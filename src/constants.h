// Constants.

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace netease::grps {

#define SYSTEM_FINGERPRINT "grps-trtllm-server"

/// @brief Names of input fields
struct InputFieldsNames {
  static constexpr char const* kModelName = "model";
  static constexpr char const* kInputId = "input_id";
  static constexpr char const* kInputTokens = "input_ids";
  static constexpr char const* kMaxNewTokens = "max_tokens";
  static constexpr char const* kEndId = "end_id";
  static constexpr char const* kPadId = "pad_id";
  static constexpr char const* kBadWords = "bad";
  static constexpr char const* kStopWords = "stop";
  static constexpr char const* kEmbeddingBias = "embedding_bias";

  // OutputConfig
  static constexpr char const* kReturnLogProbs = "logprobs";
  static constexpr char const* kReturnGenerationLogits = "return_generation_logits";
  static constexpr char const* kReturnContextLogits = "return_context_logits";

  // SamplingConfig
  static constexpr char const* kBeamWidth = "beam_width";
  static constexpr char const* kTopK = "top_k";
  static constexpr char const* kTopP = "top_p";
  static constexpr char const* kTopPMin = "top_k_min";
  static constexpr char const* kTopPDecay = "top_p_decay";
  static constexpr char const* kTopPResetIds = "top_p_reset_ids";
  static constexpr char const* kTemperature = "temperature";
  static constexpr char const* kLengthPenalty = "length_penalty";
  static constexpr char const* kEarlyStopping = "early_stopping";
  static constexpr char const* kRepetitionPenalty = "repetition_penalty";
  static constexpr char const* kMinLength = "min_length";
  static constexpr char const* kBeamSearchDiversityRate = "beam_search_diversity_rate";
  static constexpr char const* kPresencePenalty = "presence_penalty";
  static constexpr char const* kFrequencyPenalty = "frequency_penalty";
  static constexpr char const* kRandomSeed = "seed";

  // PromptTuningConfig
  static constexpr char const* kPromptEmbeddingTable = "prompt_embedding_table";

  // LoraConfig
  static constexpr char const* kLoraTaskId = "lora_task_id";
  static constexpr char const* kLoraWeights = "lora_weights";
  static constexpr char const* kLoraConfig = "lora_config";

  // SpeculativeDecodingConfig
  static constexpr char const* kDraftInputs = "draft_input_ids";
  static constexpr char const* kDraftLogits = "draft_logits";
  static constexpr char const* kDraftAcceptanceThreshold = "draft_acceptance_threshold";
};

/// @brief Names of output fields
struct OutputFieldsNames {
  static constexpr char const* kOutputIds = "output_ids";
  static constexpr char const* kSequenceLength = "sequence_length";
  static constexpr char const* kContextLogits = "context_logits";
  static constexpr char const* kGenerationLogits = "generation_logits";
  static constexpr char const* kOutputLogProbs = "output_log_probs";
  static constexpr char const* kCumLogProbs = "cum_log_probs";
};

} // namespace netease::grps