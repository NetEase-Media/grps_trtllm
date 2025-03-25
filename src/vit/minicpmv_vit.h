// MiniCPM-V VIT(Vision transformer).

#pragma once

#include "src/vit/vit.h"

namespace netease::grps {
class GrpsCli;

class MiniCPMVVIT : public VIT {
public:
  MiniCPMVVIT();
  ~MiniCPMVVIT() override;

  /**
   * @brief Initialize vit.
   * @param path: path to the vit trt engine path.
   * @param worker_tp: worker thread pool size for load image, preprocessing, postprocessing...
   * @param device: device to run the vit trt model. Will be ignored now and run in gpu:0 currently.
   * @param trt_args: more arguments for vit trt model.
   * @param processor_args: more arguments for vit processor.
   * @param tokenizer: tokenizer used to generate token ids.
   */
  void Init(const std::string& path,
            int worker_tp,
            const std::string& device,
            const YAML::Node& trt_args,
            const YAML::Node& processor_args,
            MultiInstanceTokenizer* tokenizer,
            bool kv_cache_reuse = false) override;

  // Load the vit trt model.
  void Load() override;

  /**
   * @brief Encode image to embeddings.
   * @param image_urls: image url. Can be a local file path or a http url(will be downloaded). Only support one now.
   * @param prompt: prompt may be changed when vit encoding.
   * @param token_ids: token ids may generated when vit.
   * @return vit embeddings that will be used as trtllm ptuning embedding table, and mrope config if need or nullopt if
   * not.
   */
  std::tuple<PtuningEmbeddingTableType, MropeConfType> Encode(const std::vector<std::string>& image_urls,
                                                              std::string& prompt,
                                                              tensorrt_llm::executor::VecTokens& token_ids) override;

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
   * @param img_hash: hash of images.
   * @return vit embeddings that will be used as trtllm ptuning embedding table, and mrope config if need or nullopt if
   * not.
   */
  virtual std::tuple<PtuningEmbeddingTableType, MropeConfType> Postprocess(VitModelOutputType& model_out,
                                                                           std::string& prompt,
                                                                           tensorrt_llm::executor::VecTokens& token_ids,
                                                                           uint64_t img_hash) override;

private:
  std::unique_ptr<GrpsCli> grps_cli_; // grps client to access remote vit.
  int shm_size_ = 512 * 1024 * 1024;  // Shared memory size for per shm used for images embeddings transfer.
  int image_token_id_ = 128244;       // The image token id.
  // The beginning token id used to mark the image tokens. The beginning token id must be the (max-token-id + 1),
  // that is the true vocab size.
  int img_begin_token_id_ = 151666;
  nvinfer1::DataType dtype_ = nvinfer1::DataType::kBF16; // Data type for embeddings.
};

} // namespace netease::grps