// Qwen2VL VIT(Vision transformer).

#pragma once

#ifdef PILLOW_RESIZE_ENABLE
#include <PillowResize/PillowResize.hpp>
#endif
#include <cuda_bf16.h>

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <xtensor/xarray.hpp>

#include "src/vit/vit.h"

namespace netease::grps {

class Qwen2vlVIT : public VIT {
public:
  Qwen2vlVIT() : VIT("qwen2vl") {}
  ~Qwen2vlVIT() override = default;

  /**
   * @brief Initialize vit.
   * @param path: path to the vit trt engine path.
   * @param worker_tp: worker thread pool size for load image, preprocessing, postprocessing...
   * @param device: device to run the vit trt model. Will be ignored now and run in gpu:0 currently.
   * @param trt_args: more arguments for vit trt model.
   * @param processor_args: more arguments for vit processor.
   */
  void Init(const std::string& path,
            int worker_tp,
            const std::string& device,
            const YAML::Node& trt_args,
            const YAML::Node& processor_args,
            MultiInstanceTokenizer* tokenizer) override;

  /**
   * @brief Encode image to embeddings.
   * @param img_url: image url. Can be a local file path or a http url(will be downloaded).
   * @param prompt: prompt may be changed when vit encoding.
   * @param token_ids: token ids may generated when vit.
   * @return vit embeddings that will be used as trtllm ptuning embedding table, and mrope config if need or nullopt if
   * not.
   */
  virtual std::tuple<PtuningEmbeddingTableType, MropeConfType> Encode(const std::vector<std::string>& img_urls,
                                                                      std::string& prompt,
                                                                      tensorrt_llm::executor::VecTokens& token_ids);

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
  // Resize parameters.
  int min_pixels_ = 4 * 28 * 28;
  int max_pixels_ = 16384 * 28 * 28;
  int image_factor_ = 28;
  int max_ratio_ = 200;
  int spatial_merge_size_ = 2;

  // For normalization
  cv::Scalar imagenet_mean_ = {0.48145466, 0.4578275, 0.40821073};
  cv::Scalar imagenet_std_ = {0.26862954, 0.26130258, 0.27577711};

  // For transpose with t h w.
  int temporal_patch_size_ = 2;
  int patch_size_ = 14;
  int merge_size_ = 2;

  std::unique_ptr<boost::asio::thread_pool> worker_tp2_; // Avoid deadlock if nested using of worker_tp_.

  std::unique_ptr<TrtModelInferer> mrope_inferer_; // mrope only inferer.

  std::tuple<int, int> SmartResize(int height, int width, int factor, int min_pixels, int max_pixels) const;

  void Normalize(cv::Mat& img);

  void LoadImage(const std::vector<std::vector<char>>& images_bytes,
                 std::vector<std::vector<cv::Mat>>& out,
                 size_t idx);

  void GetThw(std::vector<std::vector<cv::Mat>>& images,
              Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw,
              int& thw_sum);

  // Implementation of
  // https://github.com/huggingface/transformers/blob/329f5dbf97a5cb2473914c88c05aa3dcb242e19a/src/transformers
  // /models/qwen2_vl/image_processing_qwen2_vl.py#L291 ~ L315
  void CvtToThwPatches(std::vector<std::vector<cv::Mat>>& images,
                       Eigen::Tensor<float, 2, Eigen::RowMajor>& pixel_values,
                       int thw_sum,
                       Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw);
  // Use xtensor to optimize pixel_values calculation.
  void CvtToThwPatchesXtensor(std::vector<std::vector<cv::Mat>>& images,
                              std::vector<xt::xarray<nv_bfloat16>>& pixel_values,
                              int thw_sum,
                              Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw);

  void TokenizeEncodeAndComputeAttentionMaskVit(std::string& prompt,
                                                tensorrt_llm::executor::VecTokens& token_ids,
                                                Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw,
                                                Eigen::Tensor<bool, 3>& attention_mask_vit);

  // Implementation of
  // https://github.com/huggingface/transformers/blob/329f5dbf97a5cb2473914c88c05aa3dcb242e19a/src/transformers
  // /models/qwen2_vl/modeling_qwen2_vl.py#L1446 ~ L1593
  void GetRopeIndex(const tensorrt_llm::executor::VecTokens& input_ids,
                    const Eigen::Tensor<int64_t, 2, Eigen::RowMajor>* image_grid_thw,
                    Eigen::Tensor<int32_t, 3, Eigen::RowMajor>& position_ids,
                    Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& mrope_position_deltas);

  // Implementation of
  // https://github.com/huggingface/transformers/blob/329f5dbf97a5cb2473914c88c05aa3dcb242e19a/src/transformers
  // /models/qwen2_vl/modeling_qwen2_vl.py#L994 ~ L1017
  void ComputeRotaryPosIds(const Eigen::Tensor<int64_t, 2, Eigen::RowMajor>* image_grid_thw,
                           Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& position_ids);
};

} // namespace netease::grps