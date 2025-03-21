// Gemma3 VIT(Vision transformer).

#pragma once

#ifdef PILLOW_RESIZE_ENABLE
#include <PillowResize/PillowResize.hpp>
#endif
#include <opencv2/opencv.hpp>

#include "src/vit/vit.h"

namespace netease::grps {

class Gemma3VIT : public VIT {
public:
  Gemma3VIT() : VIT("gemma3") {}
  ~Gemma3VIT() override = default;

  /**
   * @brief Preprocess images, and will be used as the input to the ViT model.
   * @param images: images bytes will be preprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @param token_ids: token ids may generated when vit.
   * @return processed image data with trt tensor format, will be used as the input to the ViT model.
   */
  VitModelInputType Preprocess(const std::vector<std::vector<char>>& images_bytes,
                               std::string& prompt,
                               tensorrt_llm::executor::VecTokens& token_ids) override;

  /**
   * @brief Postprocess output of vit trt model, and will be used as trtllm ptuning embedding table.
   * @param model_out: output of vit trt model will be postprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @param token_ids: token ids may generated when vit.
   * @return vit embeddings that will be used as trtllm ptuning embedding table, and mrope config if need or nullopt if
   * not.
   */
  std::tuple<PtuningEmbeddingTableType, MropeConfType> Postprocess(
    VitModelOutputType& model_out, std::string& prompt, tensorrt_llm::executor::VecTokens& token_ids) override;

private:
  void Normalize(cv::Mat& img);
  void LoadImage(const std::vector<std::vector<char>>& images_bytes,
                 std::vector<std::vector<cv::Mat>>& out,
                 size_t idx,
                 int input_size = 896);

  // IMAGENET mean and std for normalization
  cv::Scalar imagenet_mean_ = {0.5, 0.5, 0.5};
  cv::Scalar imagenet_std_ = {0.5, 0.5, 0.5};
};

} // namespace netease::grps