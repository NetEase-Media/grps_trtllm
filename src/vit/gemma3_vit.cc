// Gemma3 VIT(Vision transformer).

#include "gemma3_vit.h"

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>

#include "src/utils.h"

namespace netease::grps {

void Gemma3VIT::Normalize(cv::Mat& img) {
  img.convertTo(img, CV_32F, 1.0 / 255);        // convert to float
  img = (img - imagenet_mean_) / imagenet_std_; // normalize
}

void Gemma3VIT::LoadImage(const std::vector<std::vector<char>>& images_bytes,
                          std::vector<std::vector<cv::Mat>>& out,
                          size_t idx,
                          int input_size) {
  auto begin = GET_SYS_TIME_US();
  // CLOG4(INFO, "Load image, image size: " << images_bytes[idx].size());
  // Load image
  cv::Mat image = cv::imdecode(images_bytes[idx], cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not open or find the image!");
  }
  auto imread_end = GET_SYS_TIME_US();

  // Convert to RGB.
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  auto cvt_end = GET_SYS_TIME_US();

  // Resize.
#ifdef PILLOW_RESIZE_ENABLE
  image = PillowResize::resize(image, cv::Size(input_size, input_size),
                               PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR);
#else
  cv::resize(image, image, cv::Size(input_size, input_size), cv::INTER_LINEAR);
#endif
  auto resize_end = GET_SYS_TIME_US();

  // Normalize each image
  Normalize(image);
  out[idx].emplace_back(image);
  auto normal_end = GET_SYS_TIME_US();

#if VIT_DBG
  CLOG4(INFO, "Load image success, image index: "
                << idx << ", image size: " << images_bytes[idx].size() << ", imread cost: " << imread_end - begin
                << " us, cvt cost: " << cvt_end - imread_end << " us, resize cost: " << resize_end - cvt_end
                << " us, normalize cost: " << normal_end - resize_end << "us");
#endif
}

VitModelInputType Gemma3VIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
                                        std::string& prompt,
                                        tensorrt_llm::executor::VecTokens& token_ids) {
  auto begin = GET_SYS_TIME_US();
  std::vector<std::vector<cv::Mat>> images_mats;
  boost::latch done(images_bytes.size());
  for (size_t i = 0; i < images_bytes.size(); i++) {
    images_mats.emplace_back();
    boost::asio::post(*worker_tp_, [&images_bytes, &images_mats, &done, this, i] {
      try {
        LoadImage(images_bytes, images_mats, i);
      } catch (const std::exception& e) {
        CLOG4(ERROR, "Load image failed, error: " << e.what());
      }
      done.count_down();
    });
  }
  done.wait();
  auto load_end = GET_SYS_TIME_US();

  size_t batch_size = 0;
  for (size_t i = 0; i < images_mats.size(); ++i) {
    if (images_mats[i].empty()) {
      std::string err = "Load image failed, image index: " + std::to_string(i);
      throw std::runtime_error(err);
    }
    batch_size += images_mats[i].size();
    // CLOG4(INFO, "Load image success, image index: " << i << ", patches count: " << images_mats[i].size());
  }

  auto dtype = inferer_->binding_type().at("input");
  auto inp_tensor = std::make_shared<TrtHostBinding>("input", nvinfer1::Dims4(int64_t(batch_size), 3, 896, 896), dtype);

  // Copy data from images_mat to inp_tensor(batch_size, channels, height, width).
  auto buffer = inp_tensor->buffer().Get();
  size_t buff_idx = 0;

  for (auto& images_mat : images_mats) { // Images count.
    for (auto img : images_mat) {        // Per image patches count.
      for (int c = 0; c < img.channels(); ++c) {
        for (int h = 0; h < img.rows; ++h) {
          for (int w = 0; w < img.cols; ++w) {
            if (dtype == nvinfer1::DataType::kBF16) {
              reinterpret_cast<nv_bfloat16*>(buffer)[buff_idx++] = static_cast<nv_bfloat16>(img.at<cv::Vec3f>(h, w)[c]);
            } else if (dtype == nvinfer1::DataType::kHALF) {
              reinterpret_cast<half*>(buffer)[buff_idx++] = static_cast<half>(img.at<cv::Vec3f>(h, w)[c]);
            } else if (dtype == nvinfer1::DataType::kFLOAT) {
              reinterpret_cast<float*>(buffer)[buff_idx++] = img.at<cv::Vec3f>(h, w)[c];
            }
          }
        }
      }
    }
  }
  auto memcpy_end = GET_SYS_TIME_US();

  // CLOG4(INFO, "Preprocess images success, input tensor: " << inp_tensor->DebugString());
  VitModelInputType inputs = {{"input", inp_tensor}};

#if VIT_DBG
  CLOG4(INFO, "Preprocess images success, images count: " << images_bytes.size() << ", patches count: " << batch_size
                                                          << ", load cost: " << load_end - begin
                                                          << " us, memcpy cost: " << memcpy_end - load_end << " us");
#endif
  return inputs;
}

std::tuple<PtuningEmbeddingTableType, MropeConfType> Gemma3VIT::Postprocess(
  VitModelOutputType& model_out, std::string& prompt, tensorrt_llm::executor::VecTokens& token_ids, uint64_t img_hash) {
  auto out = model_out[0].second;

  // reshape.
  auto& tensor = out->tensor;
  auto cur_shape = tensor->getShape();
  nvinfer1::Dims true_shape;
  true_shape.nbDims = 2;
  true_shape.d[0] = cur_shape.d[0] * cur_shape.d[1];
  true_shape.d[1] = cur_shape.d[2];
  tensor->reshape(true_shape);

#if VIT_DBG
  CLOG4(INFO, "Postprocess success, output tensor shape: [" << true_shape.d[0] << ", " << true_shape.d[1]
                                                            << "], dtype: " << out->tensor->getDataTypeName());
#endif

  auto e_tensor = executor::detail::ofITensor(out->tensor);
  if (token_ids.empty()) {
    token_ids = tokenizer_->Encode(prompt);
  }

  if (kv_cache_reuse_) {
    return {executor::PromptTuningConfig(std::move(e_tensor), PrepareExtraIds(img_hash, token_ids)), std::nullopt};
  } else {
    return {executor::PromptTuningConfig(std::move(e_tensor), std::nullopt), std::nullopt};
  }
}
} // namespace netease::grps