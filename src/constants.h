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
  /// @brief The beam width. Default is 1 which disables beam search.
  static constexpr char const* kBeamWidth = "beam_width";
  /// @brief Controls number of logits to sample from. Default is 0 (all logits).
  static constexpr char const* kTopK = "top_k";
  /// @brief Controls the top-P probability to sample from. Default is 0.f
  static constexpr char const* kTopP = "top_p";
  /// @brief Controls decay in the top-P algorithm. topPMin is lower-bound. Default is 1.e-6.
  static constexpr char const* kTopPMin = "top_p_min";
  /// @brief Controls decay in the top-P algorithm. The decay value. Default is 1.f
  static constexpr char const* kTopPDecay = "top_p_decay";
  /// @brief Controls decay in the top-P algorithm. Indicates where to reset the decay. Default is 1.
  static constexpr char const* kTopPResetIds = "top_p_reset_ids";
  /// @brief Controls the modulation of logits when sampling new tokens. It can have values > 0.f. Default is 1.0f
  static constexpr char const* kTemperature = "temperature";
  /// @brief Controls how to penalize longer sequences in beam search. Default is 0.f
  static constexpr char const* kLengthPenalty = "length_penalty";
  /// @brief Controls whether the generation process finishes once beamWidth sentences are generated (ends with
  /// end_token)
  static constexpr char const* kEarlyStopping = "early_stopping";
  /// @brief Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f.
  /// Values < 1.f encourages repetition, values > 1.f discourages it. Default is 1.f.
  static constexpr char const* kRepetitionPenalty = "repetition_penalty";
  /// @brief Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1.
  static constexpr char const* kMinLength = "min_length";
  /// @brief Controls the diversity in beam search.
  static constexpr char const* kBeamSearchDiversityRate = "beam_search_diversity_rate";
  /// @brief Used to penalize tokens already present in the sequence (irrespective of the number of appearances). It
  /// can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f.
  static constexpr char const* kPresencePenalty = "presence_penalty";
  /// @brief Used to penalize tokens already present in the sequence (dependent on the number of appearances). It can
  /// have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f.
  static constexpr char const* kFrequencyPenalty = "frequency_penalty";
  /// @brief Controls the random seed used by the random number generator in sampling
  static constexpr char const* kRandomSeed = "seed";
  /// @brief Controls how many repeat ngram size are acceptable. Default is 1 << 30.
  static constexpr char const* kNoRepeatNgramSize = "no_repeat_ngram_size";

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

  // LookaheadDecodingConfig
  static constexpr char const* kLookaheadWindowSize = "window_size";
  static constexpr char const* kLookaheadNGramSize = "ngram_size";
  static constexpr char const* kLookaheadVerificationSetSize = "verification_set_size";
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