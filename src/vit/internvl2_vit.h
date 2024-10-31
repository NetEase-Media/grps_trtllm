// InternVL2 VIT(Vision transformer).

#pragma once

#ifdef PILLOW_RESIZE_ENABLE
#include <PillowResize/PillowResize.hpp>
#endif
#include <opencv2/opencv.hpp>

#include "src/vit/vit.h"

namespace netease::grps {

class Internvl2VIT : public VIT {
public:
  Internvl2VIT() : VIT("internvl2") {}
  ~Internvl2VIT() override = default;

  /**
   * @brief Preprocess images, and will be used as the input to the ViT model.
   * @param images: images bytes will be preprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @return processed image data with trt tensor format, will be used as the input to the ViT model.
   */
  VitModelInputType Preprocess(const std::vector<std::vector<char>>& images_bytes, std::string& prompt) override;

  /**
   * @brief Postprocess output of vit trt model, and will be used as trtllm ptuning embedding table.
   * @param model_out: output of vit trt model will be postprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @return vit embeddings will be used as trtllm ptuning embedding table.
   */
  PtuningEmbeddingTableType Postprocess(VitModelOutputType& model_out, std::string& prompt) override;

private:
  static std::pair<int, int> FindClosestAspectRatio(
    float aspect_ratio, const std::vector<std::pair<int, int>>& target_ratios, int width, int height, int image_size);

  static std::vector<cv::Mat> DynamicPreprocess(
    cv::Mat& image, int min_num = 1, int max_num = 12, int image_size = 448, bool use_thumbnail = false);

  void Normalize(cv::Mat& img);

  void LoadImage(const std::vector<std::vector<char>>& images_bytes,
                 std::vector<std::vector<cv::Mat>>& out,
                 size_t idx,
                 int input_size = 448,
                 int max_num = 12);

  // IMAGENET mean and std for normalization
  cv::Scalar imagenet_mean_ = {0.485, 0.456, 0.406};
  cv::Scalar imagenet_std_ = {0.229, 0.224, 0.225};
};

} // namespace netease::grps