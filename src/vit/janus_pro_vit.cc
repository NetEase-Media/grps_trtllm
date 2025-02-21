// InternVL2 VIT(Vision transformer).

#include "janus_pro_vit.h"

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>

#include "src/utils.h"

namespace netease::grps {

cv::Mat JanusProVIT::Resize(cv::Mat& image) {
  int width = image.cols;
  int height = image.rows;

  float max_size = std::max(width, height);
  std::pair<int, int> size = {std::max(int(float(height) / max_size * image_size_), min_size_),
                              std::max(int(float(width) / max_size * image_size_), min_size_)};
  if (width <= 0 || height <= 0 || size.first <= 0 || size.second <= 0) {
    std::string err = "Invalid size, orig size: " + std::to_string(height) + "x" + std::to_string(width) +
                      ", new size: " + std::to_string(size.first) + "x" + std::to_string(size.second);
    throw std::invalid_argument(err);
  }

  cv::resize(image, image, cv::Size(size.second, size.first), cv::INTER_CUBIC);

  width = image.cols;
  height = image.rows;
  if (width == height) {
    return image;
  } else if (width > height) {
    cv::Mat result = cv::Mat(width, width, image.type(), background_color_);
    image.copyTo(result(cv::Rect(0, (width - height) / 2, image.cols, image.rows)));
    return result;
  } else {
    cv::Mat result = cv::Mat(height, height, image.type(), background_color_);
    image.copyTo(result(cv::Rect((height - width) / 2, 0, image.cols, image.rows)));
    return result;
  }
}

void JanusProVIT::Normalize(cv::Mat& img) {
  img.convertTo(img, CV_32F, 1.0 / 255);        // convert to float
  img = (img - imagenet_mean_) / imagenet_std_; // normalize
}

void JanusProVIT::LoadImage(const std::vector<std::vector<char>>& images_bytes,
                            std::vector<std::vector<cv::Mat>>& out,
                            size_t idx) {
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

  // Resize image.
  image = Resize(image);
  auto resize_end = GET_SYS_TIME_US();

  // Normalize image.
  Normalize(image);
  auto normal_end = GET_SYS_TIME_US();

  out[idx].emplace_back(image);
#if VIT_DBG
  CLOG4(INFO, "Load image success, image index: "
                << idx << ", image size: " << images_bytes[idx].size() << ", imread cost: " << imread_end - begin
                << " us, cvt cost: " << cvt_end - imread_end << " us, resize cost: " << resize_end - cvt_end
                << " us, normalize cost: " << normal_end - resize_end << "us");
#endif
}

VitModelInputType JanusProVIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
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
  auto inp_tensor =
    std::make_shared<TrtHostBinding>("input", nvinfer1::Dims4(int64_t(batch_size), 3, image_size_, image_size_), dtype);

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

std::tuple<PtuningEmbeddingTableType, MropeConfType> JanusProVIT::Postprocess(
  VitModelOutputType& model_out, std::string& prompt, tensorrt_llm::executor::VecTokens& token_ids) {
  auto out = model_out[0].second;

  // reshape.
  auto& tensor = out->tensor;
  auto cur_shape = tensor->getShape();
  nvinfer1::Dims true_shape;
  true_shape.nbDims = 2;
  true_shape.d[0] = cur_shape.d[0] * cur_shape.d[1];
  true_shape.d[1] = cur_shape.d[2];
  tensor->reshape(true_shape);
  auto e_tensor = executor::detail::ofITensor(out->tensor);
  return {executor::PromptTuningConfig(std::move(e_tensor)), std::nullopt};
}
} // namespace netease::grps