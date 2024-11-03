// VIT(Vision transformer) used for multi-modal image encoding to embeddings.

#pragma once

#include <boost/asio/thread_pool.hpp>
#include <memory>
#include <utility>

#include "src/tensorrt/trt_inferer.h"

#define VIT_DBG 0

namespace netease::grps {

using PtuningEmbeddingTableType = std::shared_ptr<tensorrt_llm::batch_manager::NamedTensor>;
using VitModelInputType = std::vector<std::pair<std::string, std::shared_ptr<TrtHostBinding>>>;
using VitModelOutputType =
  std::vector<std::pair<std::string, std::shared_ptr<tensorrt_llm::batch_manager::NamedTensor>>>;

class VitException : public std::exception {
public:
  explicit VitException(std::string message) : message_(std::move(message)) {}
  ~VitException() override = default;
  [[nodiscard]] const char* what() const noexcept override {
    static std::string err_message;
    err_message = "[VitException] " + message_;
    return err_message.c_str();
  }

private:
  std::string message_;
};

class VIT {
public:
  explicit VIT(std::string type_name) : type_name_(std::move(type_name)) {}
  virtual ~VIT() = default;

  /**
   * @brief Initialize vit.
   * @param path: path to the vit trt engine path.
   * @param worker_tp: worker thread pool size for load image, preprocessing, postprocessing...
   * @param device: device to run the vit trt model. Will be ignored now and run in gpu:0 currently.
   * @param args: more arguments for vit trt model.
   */
  virtual void Init(const std::string& path, int worker_tp, const std::string& device, const YAML::Node& trt_args);

  // Load the vit trt model.
  virtual void Load();

  /**
   * @brief Get images from url.
   * @param img_url: can be a local file path or a http url(will be downloaded).
   * @return images bytes.
   */
  virtual std::vector<std::vector<char>> GetImages(const std::vector<std::string>& img_urls);

  /**
   * @brief Encode image to embeddings.
   * @param img_url: image url. Can be a local file path or a http url(will be downloaded).
   * @param prompt: prompt may be changed when vit encoding.
   * @return vit embeddings will be used as trtllm ptuning embedding table.
   */
  virtual PtuningEmbeddingTableType Encode(const std::vector<std::string>& img_urls, std::string& prompt);

  /**
   * @brief Preprocess images, and will be used as the input to the ViT model.
   * @param images: images bytes will be preprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @return processed image data with trt tensor format, will be used as the input to the ViT model.
   */
  virtual VitModelInputType Preprocess(const std::vector<std::vector<char>>& images_bytes, std::string& prompt) = 0;

  /**
   * @brief Postprocess output of vit trt model, and will be used as trtllm ptuning embedding table.
   * @param model_out: output of vit trt model will be postprocessed.
   * @param prompt: prompt may be changed when vit encoding.
   * @return vit embeddings will be used as trtllm ptuning embedding table.
   */
  virtual PtuningEmbeddingTableType Postprocess(VitModelOutputType& model_out, std::string& prompt) = 0;

protected:
  std::string type_name_;
  std::unique_ptr<TrtModelInferer> inferer_;
  std::unique_ptr<boost::asio::thread_pool> worker_tp_; // TP used to get image, preprocessing, postprocessing...
};

class VITFactory {
public:
  VITFactory() = default;
  virtual ~VITFactory() = default;
  static std::unique_ptr<VIT> CreateVIT(const std::string& type_name);
};
} // namespace netease::grps
