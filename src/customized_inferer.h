// Customized deep learning model inferer. Including model load and model infer.

#pragma once

#include "model_infer/inferer.h"
#include "src/tokenizer.h"
#include "src/trtllm_model_instance.h"
#include "src/trtllm_model_state.h"
#include "src/vit/vit.h"

namespace netease::grps {
class TrtllmInferer : public ModelInferer {
public:
  TrtllmInferer();
  ~TrtllmInferer() override;

  // Clone inferer for duplicated use. Don't edit this function.
  ModelInferer* Clone() override { return new TrtllmInferer(); }

  /**
   * @brief Init model inferer.
   * @param path: Model path, it can be a file path or a directory path.
   * @param device: Device to run model.
   * @param args: More args.
   * @throw InfererException: If init failed, can throw InfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Init(const std::string& path, const std::string& device, const YAML::Node& args) override;

  /**
   * @brief Load model.
   * @throw InfererException: If load failed, can throw InfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Load() override;

  /**
   * Used when in `no converter mode`. Input and output are directly GrpsMessage.
   * @brief Infer model.
   * @param input: Input.
   * @param output: Output.
   * @param ctx: Context of current request.
   * @throw InfererException: If infer failed, can throw InfererException and will be caught by server and return error
   * message to client.
   */
  virtual void Infer(const ::grps::protos::v1::GrpsMessage& input,
                     ::grps::protos::v1::GrpsMessage& output,
                     GrpsContext& ctx) override;

private:
  std::unique_ptr<TrtLlmModelState> model_state_ = nullptr;
  std::unique_ptr<TrtLlmModelInstance> trtllm_instance_ = nullptr;
  std::unique_ptr<LLMStyler> llm_styler_ = nullptr;
  std::unique_ptr<VIT> vit_ = nullptr;
  std::unique_ptr<MultiInstanceTokenizer> tokenizer_ = nullptr;
};

REGISTER_INFERER(TrtllmInferer, trtllm_inferer); // Register your inferer.

// Define other inferer class after here.

} // namespace netease::grps