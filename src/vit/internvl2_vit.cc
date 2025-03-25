// InternVL2 VIT(Vision transformer).

#include "internvl2_vit.h"

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>

#include "src/utils.h"

namespace netease::grps {

std::pair<int, int> Internvl2VIT::FindClosestAspectRatio(
  float aspect_ratio, const std::vector<std::pair<int, int>>& target_ratios, int width, int height, int image_size) {
  float best_ratio_diff = std::numeric_limits<float>::infinity();
  std::pair<int, int> best_ratio(1, 1);
  float area = float(width) * float(height);

  for (const auto& ratio : target_ratios) {
    float target_aspect_ratio = static_cast<float>(ratio.first) / static_cast<float>(ratio.second);
    float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
    if (ratio_diff < best_ratio_diff) {
      best_ratio_diff = ratio_diff;
      best_ratio = ratio;
    } else if (ratio_diff == best_ratio_diff) {
      if (area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
        best_ratio = ratio;
      }
    }
  }
  return best_ratio;
}

std::vector<cv::Mat> Internvl2VIT::DynamicPreprocess(
  cv::Mat& image, int min_num, int max_num, int image_size, bool use_thumbnail) {
  int orig_width = image.cols;
  int orig_height = image.rows;
  float aspect_ratio = static_cast<float>(orig_width) / static_cast<float>(orig_height);

  // Generate target ratios
  std::vector<std::pair<int, int>> target_ratios;
  for (int n = min_num; n <= max_num; n++) {
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) {
        if (i * j <= max_num && i * j >= min_num) {
          target_ratios.emplace_back(i, j);
        }
      }
    }
  }

  // Find the closest aspect ratio
  std::pair<int, int> target_aspect_ratio =
    FindClosestAspectRatio(aspect_ratio, target_ratios, orig_width, orig_height, image_size);

  // Calculate target width and height
  int target_width = image_size * target_aspect_ratio.first;
  int target_height = image_size * target_aspect_ratio.second;
  int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

  // Resize the image
#ifdef PILLOW_RESIZE_ENABLE
  cv::Mat resized_img = PillowResize::resize(image, cv::Size(target_width, target_height),
                                             PillowResize::InterpolationMethods::INTERPOLATION_BICUBIC);
#else
  cv::Mat resized_img;
  cv::resize(image, resized_img, cv::Size(target_width, target_height), cv::INTER_CUBIC);
#endif

  std::vector<cv::Mat> processed_images;
  for (int i = 0; i < blocks; i++) {
    int x = (i % (target_width / image_size)) * image_size;
    int y = (i / (target_width / image_size)) * image_size;
    cv::Rect box(x, y, image_size, image_size);
    processed_images.push_back(resized_img(box)); // Split the image
  }

  if (use_thumbnail && blocks != 1) {
#ifdef PILLOW_RESIZE_ENABLE
    cv::Mat thumbnail_img = PillowResize::resize(image, cv::Size(image_size, image_size),
                                                 PillowResize::InterpolationMethods::INTERPOLATION_BICUBIC);
#else
    cv::Mat thumbnail_img;
    cv::resize(image, thumbnail_img, cv::Size(image_size, image_size), cv::INTER_CUBIC);
#endif
    processed_images.push_back(thumbnail_img);
  }

  return processed_images;
}

void Internvl2VIT::Normalize(cv::Mat& img) {
  img.convertTo(img, CV_32F, 1.0 / 255);        // convert to float
  img = (img - imagenet_mean_) / imagenet_std_; // normalize
}

void Internvl2VIT::LoadImage(const std::vector<std::vector<char>>& images_bytes,
                             std::vector<std::vector<cv::Mat>>& out,
                             size_t idx,
                             int input_size,
                             int max_num) {
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

  // Process image
  std::vector<cv::Mat> images = DynamicPreprocess(image, 1, max_num, input_size, true);
  auto dynamic_process_end = GET_SYS_TIME_US();
  // CLOG4(INFO, "Dynamic preprocess image success, patches count: " << images.size());

  // Normalize each image
  for (auto& img : images) {
    Normalize(img);
    out[idx].emplace_back(img);
  }
  auto normal_end = GET_SYS_TIME_US();

#if VIT_DBG
  CLOG4(INFO, "Load image success, image index: "
                << idx << ", image size: " << images_bytes[idx].size() << ", imread cost: " << imread_end - begin
                << " us, cvt cost: " << cvt_end - imread_end
                << " us, dynamic process cost: " << dynamic_process_end - cvt_end
                << " us, normalize cost: " << normal_end - dynamic_process_end << "us");
#endif
}

VitModelInputType Internvl2VIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
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
  auto inp_tensor = std::make_shared<TrtHostBinding>("input", nvinfer1::Dims4(int64_t(batch_size), 3, 448, 448), dtype);

  // Replace `<image>` with `<img><IMG_CONTEXT>*patches*256</img>` in prompt.
  // Copy data from images_mat to inp_tensor(batch_size, channels, height, width).
  auto buffer = inp_tensor->buffer().Get();
  size_t pos = 0;
  size_t buff_idx = 0;

  for (auto& images_mat : images_mats) { // Images count.
    std::string replace = "<img>";
    for (auto img : images_mat) { // Per image patches count.
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

      replace.append(img_ctx_replace_str_);
    }
    replace.append("</img>");
    pos = utils::ReplaceWorld(prompt, "<image>", replace, pos, 1);
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

std::tuple<PtuningEmbeddingTableType, MropeConfType> Internvl2VIT::Postprocess(
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