// Qwen2vl VIT(Vision transformer).

#include "qwen2vl_vit.h"

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "src/utils.h"

namespace netease::grps {

void Qwen2vlVIT::Init(const std::string& path,
                      int worker_tp,
                      const std::string& device,
                      const YAML::Node& trt_args,
                      const YAML::Node& processor_args,
                      MultiInstanceTokenizer* tokenizer) {
  VIT::Init(path, worker_tp, device, trt_args, processor_args, tokenizer);
  if (processor_args["min_pixels"] && !processor_args["min_pixels"].IsNull() &&
      processor_args["min_pixels"].IsScalar()) {
    min_pixels_ = processor_args["min_pixels"].as<int32_t>();
  }
  if (processor_args["max_pixels"] && !processor_args["max_pixels"].IsNull() &&
      processor_args["max_pixels"].IsScalar()) {
    max_pixels_ = processor_args["max_pixels"].as<int32_t>();
  }
  if (processor_args["mrope_only_path"] && !processor_args["mrope_only_path"].IsNull() &&
      processor_args["mrope_only_path"].IsScalar()) {
    mrope_inferer_ = std::make_unique<TrtModelInferer>();
    mrope_inferer_->Init(processor_args["mrope_only_path"].as<std::string>(), device, trt_args);
    mrope_inferer_->Load();
  } else {
    throw VitException("mrope_only_path is required.");
  }
  worker_tp2_ = std::make_unique<boost::asio::thread_pool>(worker_tp);
}

// Returns the closest integer to 'number' that is divisible by 'factor'.
// Four->floor, Six->ceil, Five->round to the nearest even
static int PyRoundByFactor(int number, int factor) {
  // return int(round(double(number) / double(factor))) * factor;
  double value = double(number) / double(factor);
  double integer_part, fractional_part;
  fractional_part = modf(value, &integer_part);
  if (fractional_part > 0.5) {
    return int(ceil(value)) * factor;
  } else if (fractional_part < 0.5) {
    return int(floor(value)) * factor;
  } else {
    if (static_cast<long long>(integer_part) % 2 == 0) {
      return int(floor(value)) * factor;
    } else {
      return int(ceil(value)) * factor;
    }
  }
}

// Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'.
static int CeilByFactor(int number, int factor) {
  return int(ceil(double(number) / double(factor))) * factor;
}

// Returns the largest integer less than or equal to 'number' that is divisible by 'factor'.
static int FloorByFactor(int number, int factor) {
  return int(floor(double(number) / double(factor))) * factor;
}

// Rescales the image so that the following conditions are met:
// 1. Both dimensions (height and width) are divisible by 'factor'.
// 2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
// 3. The aspect ratio of the image is maintained as closely as possible.
std::tuple<int, int> Qwen2vlVIT::SmartResize(int height, int width, int factor, int min_pixels, int max_pixels) const {
  if (std::max(height, width) / std::min(height, width) > max_ratio_) {
    throw std::invalid_argument("absolute aspect ratio must be smaller than " + std::to_string(max_ratio_) + ", got " +
                                std::to_string(double(std::max(height, width)) / double(std::min(height, width))));
  }
  int h_bar = std::max(factor, PyRoundByFactor(height, factor));
  int w_bar = std::max(factor, PyRoundByFactor(width, factor));
  if (h_bar * w_bar > max_pixels) {
    double beta = sqrt(double(height * width) / double(max_pixels));
    h_bar = FloorByFactor(int(height / beta), factor);
    w_bar = FloorByFactor(int(width / beta), factor);
  } else if (h_bar * w_bar < min_pixels) {
    double beta = sqrt(double(min_pixels) / double(height * width));
    h_bar = CeilByFactor(int(height * beta), factor);
    w_bar = CeilByFactor(int(width * beta), factor);
  }
  return {h_bar, w_bar};
}

void Qwen2vlVIT::Normalize(cv::Mat& img) {
  img.convertTo(img, CV_32F, 1.0 / 255);        // convert to float
  img = (img - imagenet_mean_) / imagenet_std_; // normalize
}

void Qwen2vlVIT::LoadImage(const std::vector<std::vector<char>>& images_bytes,
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

  // Resize.
  int width = image.cols;
  int height = image.rows;
  auto [resized_height, resized_width] = SmartResize(height, width, image_factor_, min_pixels_, max_pixels_);
#ifdef PILLOW_RESIZE_ENABLE
  image = PillowResize::resize(image, cv::Size(resized_width, resized_height),
                               PillowResize::InterpolationMethods::INTERPOLATION_BICUBIC);
#else
  cv::resize(image, image, cv::Size(resized_width, resized_height), cv::INTER_CUBIC);
#endif
  auto resize_end = GET_SYS_TIME_US();

  // Normalize each image
  Normalize(image);
  out[idx].emplace_back(image);
  auto normal_end = GET_SYS_TIME_US();

#if VIT_DBG
  CLOG4(INFO, "Load image success, image index: "
                << idx << ", image size: " << images_bytes[idx].size() << ", origin shape: [" << height << ", " << width
                << "], resized shape: [" << resized_height << ", " << resized_width
                << "], imread cost: " << imread_end - begin << " us, cvt cost: " << cvt_end - imread_end
                << " us, resize cost: " << resize_end - cvt_end << "us, "
                << "normalize cost: " << normal_end - resize_end << "us");
#endif
}

void Qwen2vlVIT::GetThw(std::vector<std::vector<cv::Mat>>& images,
                        Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw,
                        int& thw_sum) {
  auto begin = GET_SYS_TIME_US();
  image_grid_thw = Eigen::Tensor<int64_t, 2, Eigen::RowMajor>(int(images.size()), 3);
  thw_sum = 0;
  for (int i = 0; i < int(images.size()); ++i) {
    if (images[i].empty()) {
      std::string err = "Load image failed, index: " + std::to_string(i);
      throw std::runtime_error(err);
    }
    int batch_size = int(images[i].size()) == 1 ? 2 : int(images[i].size());
    int width = images[i][0].cols;
    int height = images[i][0].rows;
    int grid_t = batch_size / temporal_patch_size_;
    int grid_h = height / patch_size_;
    int grid_w = width / patch_size_;
    image_grid_thw(i, 0) = grid_t;
    image_grid_thw(i, 1) = grid_h;
    image_grid_thw(i, 2) = grid_w;
    thw_sum += grid_t * grid_h * grid_w;
  }
  auto end = GET_SYS_TIME_US();
#if VIT_DBG
  CLOG4(INFO, "GetThw success, image_grid_thw shape: ["
                << image_grid_thw.dimension(0) << ", " << image_grid_thw.dimension(1) << "] "
                << ", thw_sum: " << thw_sum << ", get thw cost: " << end - begin << " us");
  std::string image_grid_thw_str = "[";
  for (int i = 0; i < image_grid_thw.dimension(0); i++) {
    image_grid_thw_str += "[";
    for (int j = 0; j < image_grid_thw.dimension(1); j++) {
      image_grid_thw_str += std::to_string(image_grid_thw(i, j)) + ", ";
    }
    image_grid_thw_str += "], ";
  }
  image_grid_thw_str += "]";
  CLOG4(INFO, "image_grid_thw: " << image_grid_thw_str);
#endif
}

void Qwen2vlVIT::CvtToThwPatches(std::vector<std::vector<cv::Mat>>& images,
                                 Eigen::Tensor<float, 2, Eigen::RowMajor>& pixel_values,
                                 int thw_sum,
                                 Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw) {
  auto begin = GET_SYS_TIME_US();
  int patch_pixels = 3 * temporal_patch_size_ * patch_size_ * patch_size_;
  pixel_values = Eigen::Tensor<float, 2, Eigen::RowMajor>(thw_sum, patch_pixels);
  boost::latch done(images.size());

  int idx = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].empty()) {
      std::string err = "Load image failed, index: " + std::to_string(i);
      throw std::runtime_error(err);
    }

    int grid_t = int(image_grid_thw(i, 0));
    int grid_h = int(image_grid_thw(i, 1));
    int grid_w = int(image_grid_thw(i, 2));
    int cur_thw = grid_t * grid_h * grid_w;

    boost::asio::post(
      *worker_tp2_, [this, i, &images, &pixel_values, grid_t, grid_h, grid_w, cur_thw, idx, patch_pixels, &done]() {
        auto begin = GET_SYS_TIME_US();
        int batch_size = int(images[i].size());
        int width = images[i][0].cols;
        int height = images[i][0].rows;

        std::shared_ptr<Eigen::Tensor<float, 4, Eigen::RowMajor>> patches_p;
        if (batch_size == 1) {
          patches_p = std::make_shared<Eigen::Tensor<float, 4, Eigen::RowMajor>>(2, 3, height, width);
          auto& img = images[i][0];
          for (int c = 0; c < img.channels(); ++c) {
            for (int h = 0; h < img.rows; ++h) {
              for (int w = 0; w < img.cols; ++w) {
                (*patches_p)(0, c, h, w) = static_cast<float>(img.at<cv::Vec3f>(h, w)[c]);
                (*patches_p)(1, c, h, w) = static_cast<float>(img.at<cv::Vec3f>(h, w)[c]);
              }
            }
          }
          batch_size = 2;
        } else {
          patches_p = std::make_shared<Eigen::Tensor<float, 4, Eigen::RowMajor>>(batch_size, 3, height, width);
          for (size_t j = 0; j < images[i].size(); ++j) {
            auto& img = images[i][j];
            for (int c = 0; c < img.channels(); ++c) {
              for (int h = 0; h < img.rows; ++h) {
                for (int w = 0; w < img.cols; ++w) {
                  (*patches_p)(j, c, h, w) = static_cast<float>(img.at<cv::Vec3f>(h, w)[c]);
                }
              }
            }
          }
        }
        auto load_end = GET_SYS_TIME_US();

        // Transpose and reshape.
        int channel = int(patches_p->dimension(1));
        auto patches_reshape = patches_p->reshape(
          Eigen::array<Eigen::Index, 9>{grid_t, temporal_patch_size_, channel, grid_h / merge_size_, merge_size_,
                                        patch_size_, grid_w / merge_size_, merge_size_, patch_size_});
        auto reshape_end = GET_SYS_TIME_US();
        auto patches_reshape_sf = patches_reshape.shuffle(Eigen::array<Eigen::Index, 9>{0, 3, 6, 4, 7, 2, 1, 5, 8});
        auto shuffle_end = GET_SYS_TIME_US();
        auto patches_final = patches_reshape_sf.reshape(Eigen::array<Eigen::Index, 2>{cur_thw, patch_pixels});
        auto reshape2_end = GET_SYS_TIME_US();

        // CLOG4(INFO, "idx: " << idx << ", cur_thw: " << cur_thw << ", patch_pixels: " << patch_pixels
        //                     << ", patches_final shape: [" << patches_final.dimensions()[0] << ", "
        //                     << patches_final.dimensions()[1] << "]");
        pixel_values.slice(Eigen::array<Eigen::Index, 2>{idx, 0},
                           Eigen::array<Eigen::Index, 2>{cur_thw, patch_pixels}) = patches_final;

        auto end = GET_SYS_TIME_US();

#if VIT_DBG
        CLOG4(INFO, "CvtToThwPatches, image index: " << i << ", load cost: " << load_end - begin
                                                     << " us, reshape cost: " << reshape_end - load_end
                                                     << " us, shuffle cost: " << shuffle_end - reshape_end
                                                     << " us, reshape2 cost: " << reshape2_end - shuffle_end
                                                     << " us, total cost: " << end - begin << " us");
#endif

        done.count_down();
      });
    idx += cur_thw;
  }
  done.wait();
  auto cvt_thw_end = GET_SYS_TIME_US();
#if VIT_DBG
  CLOG4(INFO, "CvtToThwPatches, merge success, pixel_values shape: [" << pixel_values.dimension(0) << ", "
                                                                      << pixel_values.dimension(1) << "] "
                                                                      << "cost: " << cvt_thw_end - begin << " us");
#endif
}

void Qwen2vlVIT::CvtToThwPatchesXtensor(std::vector<std::vector<cv::Mat>>& images,
                                        std::vector<xt::xarray<nv_bfloat16>>& pixel_values,
                                        int thw_sum,
                                        Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw) {
  auto begin = GET_SYS_TIME_US();
  boost::latch done(images.size());

  for (int i = 0; i < int(images.size()); ++i) {
    if (images[i].empty()) {
      std::string err = "Load image failed, index: " + std::to_string(i);
      throw std::runtime_error(err);
    }

    int grid_t = int(image_grid_thw(i, 0));
    int grid_h = int(image_grid_thw(i, 1));
    int grid_w = int(image_grid_thw(i, 2));
    int cur_thw = grid_t * grid_h * grid_w;

    pixel_values.emplace_back();

    boost::asio::post(*worker_tp2_, [this, i, &images, &pixel_values, grid_t, grid_h, grid_w, &done]() {
      auto begin = GET_SYS_TIME_US();
      int batch_size = static_cast<int>(images[i].size());
      int width = images[i][0].cols;
      int height = images[i][0].rows;

      xt::xarray<nv_bfloat16> patches;

      if (batch_size == 1) {
        patches = xt::xarray<nv_bfloat16>::from_shape({2, 3, height, width});
        const auto& img = images[i][0];
        for (int c = 0; c < img.channels(); ++c) {
          for (int h = 0; h < img.rows; ++h) {
            for (int w = 0; w < img.cols; ++w) {
              nv_bfloat16 pixel_value = static_cast<nv_bfloat16>(img.at<cv::Vec3f>(h, w)[c]);
              patches(0, c, h, w) = pixel_value;
              patches(1, c, h, w) = pixel_value;
            }
          }
        }
        batch_size = 2;
      } else {
        patches = xt::xarray<nv_bfloat16>::from_shape({batch_size, 3, height, width});
        for (size_t j = 0; j < images[i].size(); ++j) {
          const auto& img = images[i][j];
          for (int c = 0; c < img.channels(); ++c) {
            for (int h = 0; h < img.rows; ++h) {
              for (int w = 0; w < img.cols; ++w) {
                patches(j, c, h, w) = static_cast<nv_bfloat16>(img.at<cv::Vec3f>(h, w)[c]);
              }
            }
          }
        }
      }
      auto load_end = GET_SYS_TIME_US();

      // Reshape and transpose
      auto patches_reshape =
        xt::reshape_view(patches, {grid_t, temporal_patch_size_, 3, grid_h / merge_size_, merge_size_, patch_size_,
                                   grid_w / merge_size_, merge_size_, patch_size_});
      // xt::eval(patches_reshape);
      auto reshape_end = GET_SYS_TIME_US();

      auto patches_transposed = xt::transpose(patches_reshape, {0, 3, 6, 4, 7, 2, 1, 5, 8});
      // xt::eval(patches_transposed);

      pixel_values[i] = std::move(patches_transposed);

      auto end = GET_SYS_TIME_US();

#if VIT_DBG
      CLOG4(INFO, "CvtToThwPatchesXtensor, image index: " << i << ", load cost: " << load_end - begin
                                                          << " us, reshape cost: " << reshape_end - load_end
                                                          << " us, transpose cost: " << end - reshape_end << " us");
#endif

      done.count_down();
    });
  }
  done.wait();
  auto cvt_thw_end = GET_SYS_TIME_US();
#if VIT_DBG
  CLOG4(INFO, "CvtToThwPatchesXtensor success, pixel_values size: " << pixel_values.size()
                                                                    << ", cost: " << cvt_thw_end - begin << " us");
#endif
}

void Qwen2vlVIT::TokenizeEncodeAndComputeAttentionMaskVit(std::string& prompt,
                                                          tensorrt_llm::executor::VecTokens& token_ids,
                                                          Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& image_grid_thw,
                                                          Eigen::Tensor<bool, 3>& attention_mask_vit) {
  // Replace `<|image_pad|>` with `<|image_pad|>` * thw / (merge_size_ * merge_size_) in prompt.
  size_t pos = 0;
  for (int i = 0; i < image_grid_thw.dimension(0); i++) {
    std::string replace;
    int cnt = 1;
    for (int j = 0; j < image_grid_thw.dimension(1); j++) {
      cnt *= image_grid_thw(i, j);
    }
    cnt /= (merge_size_ * merge_size_);
    for (int j = 0; j < cnt; j++) {
      replace += "<|image_pad|>";
    }
    pos = utils::ReplaceWorld(prompt, "<|image_pad|>", replace, pos, 1);
  }
  auto replace_end = GET_SYS_TIME_US();

  // Encode and get input ids and input ids mask.
  token_ids = tokenizer_->Encode(prompt);
  auto encode_end = GET_SYS_TIME_US();

  // Generate attention mask.
  std::vector<int64_t> cu_seqlens{0};
  int64_t sum = 0;
  for (int i = 0; i < image_grid_thw.dimension(0); i++) {
    int64_t cnt = image_grid_thw(i, 1) * image_grid_thw(i, 2);
    for (int j = 0; j < image_grid_thw(i, 0); j++) {
      sum += cnt;
      cu_seqlens.push_back(sum);
    }
  }
  boost::latch done2((cu_seqlens.size() - 1) * (cu_seqlens.size() - 1));
  for (size_t i = 1; i < cu_seqlens.size(); ++i) {
    int64_t start_h = cu_seqlens[i - 1];
    int64_t len_h = cu_seqlens[i] - start_h;
    int64_t start_w = 0;
    for (size_t j = 1; j < cu_seqlens.size(); ++j) {
      int64_t len_w = cu_seqlens[j] - cu_seqlens[j - 1];
      bool val = (i == j);
      // CLOG4(INFO, "i: " << i << ", j: " << j << ", start_h: " << start_h << ", start_w: " << start_w
      //                   << ", len_h: " << len_h << ", len_w: " << len_w << ", val: " << val);
      boost::asio::post(*worker_tp2_, [start_h, start_w, len_h, len_w, &attention_mask_vit, &done2, val]() {
        attention_mask_vit
          .slice(Eigen::array<Eigen::Index, 3>{0, start_h, start_w}, Eigen::array<Eigen::Index, 3>{1, len_h, len_w})
          .setConstant(val);
        done2.count_down();
      });
      start_w += len_w;
    }
  }
  done2.wait();
  auto gen_mask_vit_end = GET_SYS_TIME_US();
}

void Qwen2vlVIT::GetRopeIndex(const executor::VecTokens& input_ids,
                              const Eigen::Tensor<int64_t, 2, Eigen::RowMajor>* image_grid_thw,
                              Eigen::Tensor<int32_t, 3, Eigen::RowMajor>& position_ids,
                              Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& mrope_position_deltas) {
  int spatial_merge_size = 2;
  int vision_start_token_id = 151652;

  if (image_grid_thw == nullptr) {
    position_ids = Eigen::Tensor<int32_t, 3, Eigen::RowMajor>(3, 1, input_ids.size());
    for (size_t i = 0; i < input_ids.size(); ++i) {
      position_ids.slice(Eigen::array<Eigen::Index, 3>{0, 0, int(i)}, Eigen::array<Eigen::Index, 3>{3, 1, 1})
        .setConstant(i);
    }
    mrope_position_deltas = Eigen::Tensor<int64_t, 2, Eigen::RowMajor>(1, 1);
    mrope_position_deltas.setConstant(0);
  } else {
    int32_t image_index = 0;
    int32_t image_nums = image_grid_thw->dimension(0);
    std::vector<Eigen::Tensor<int32_t, 2, Eigen::RowMajor>> llm_pos_ids_list;
    int32_t st = 0;
    for (int32_t n = 0; n < image_nums; n++) {
      // ed_image = input_tokens.index(image_token_id, st)
      int32_t ed = -1;
      for (size_t j = st; j < input_ids.size(); j++) {
        if (input_ids[j] == vision_start_token_id) {
          ed = j + 1;
          break;
        }
      }

      // t, h, w = (
      //         image_grid_thw[image_index][0],
      //         image_grid_thw[image_index][1],
      //         image_grid_thw[image_index][2],
      //         )
      // llm_grid_t, llm_grid_h, llm_grid_w = (
      //                           t.item(),
      //                           h.item() // spatial_merge_size,
      //                           w.item() // spatial_merge_size,
      //                           )
      int32_t llm_grid_t = (*image_grid_thw)(image_index, 0);
      int32_t llm_grid_h = (*image_grid_thw)(image_index, 1) / spatial_merge_size;
      int32_t llm_grid_w = (*image_grid_thw)(image_index, 2) / spatial_merge_size;
      image_index++;
      int32_t text_len = ed - st;

      // st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
      // llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
      int32_t st_idx = 0;
      if (!llm_pos_ids_list.empty()) {
        auto& llm_pos_ids = llm_pos_ids_list.back();
        int32_t max = -1;
        for (int i = 0; i < 3; i++) {
          max = std::max(max, llm_pos_ids(i, llm_pos_ids.dimension(1) - 1));
        }
        st_idx = max + 1;
      }

      Eigen::Tensor<int32_t, 2, Eigen::RowMajor> llm_pos_ids(3, text_len);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < text_len; j++) {
          llm_pos_ids(i, j) = j + st_idx;
        }
      }
      llm_pos_ids_list.emplace_back(std::move(llm_pos_ids));

      int32_t thw = llm_grid_t * llm_grid_h * llm_grid_w;

      // t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
      Eigen::Tensor<int32_t, 1, Eigen::RowMajor> t_index(llm_grid_t);
      for (int i = 0; i < llm_grid_t; i++) {
        t_index(i) = i;
      }
      auto t_index_reshape = t_index.reshape(Eigen::array<Eigen::Index, 2>{llm_grid_t, 1});
      auto t_index_expand = t_index_reshape.broadcast(Eigen::array<Eigen::Index, 3>{1, llm_grid_h * llm_grid_w});
      auto t_index_flatten = t_index_expand.reshape(Eigen::array<Eigen::Index, 2>{1, thw});

      // h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
      Eigen::Tensor<int32_t, 1, Eigen::RowMajor> h_index(llm_grid_h);
      for (int i = 0; i < llm_grid_h; i++) {
        h_index(i) = i;
      }
      auto h_index_reshape = h_index.reshape(Eigen::array<Eigen::Index, 3>{1, llm_grid_h, 1});
      auto h_index_expand = h_index_reshape.broadcast(Eigen::array<Eigen::Index, 3>{llm_grid_t, 1, llm_grid_w});
      auto h_index_flatten = h_index_expand.reshape(Eigen::array<Eigen::Index, 2>{1, thw});

      // w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
      Eigen::Tensor<int32_t, 1, Eigen::RowMajor> w_index(llm_grid_w);
      for (int i = 0; i < llm_grid_w; i++) {
        w_index(i) = i;
      }
      auto w_index_reshape = w_index.reshape(Eigen::array<Eigen::Index, 3>{1, 1, llm_grid_w});
      auto w_index_expand = w_index_reshape.broadcast(Eigen::array<Eigen::Index, 3>{llm_grid_t, llm_grid_h, 1});
      auto w_index_flatten = w_index_expand.reshape(Eigen::array<Eigen::Index, 2>{1, thw});

      // llm_pos_ids = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
      Eigen::Tensor<int32_t, 2, Eigen::RowMajor> llm_pos_ids2(3, thw);
      llm_pos_ids2.slice(Eigen::array<Eigen::Index, 2>{0, 0}, Eigen::array<Eigen::Index, 2>{1, thw}) = t_index_flatten;
      llm_pos_ids2.slice(Eigen::array<Eigen::Index, 2>{1, 0}, Eigen::array<Eigen::Index, 2>{1, thw}) = h_index_flatten;
      llm_pos_ids2.slice(Eigen::array<Eigen::Index, 2>{2, 0}, Eigen::array<Eigen::Index, 2>{1, thw}) = w_index_flatten;
      llm_pos_ids2 = llm_pos_ids2 + text_len + st_idx;

      // llm_pos_ids_list.append(llm_pos_ids)
      // st = ed + llm_grid_t * llm_grid_h * llm_grid_w
      llm_pos_ids_list.emplace_back(std::move(llm_pos_ids2));
      st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
    }

    // if st < len(input_tokens):
    //   st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    //   text_len = len(input_tokens) - st
    //   llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    if (st < int32_t(input_ids.size())) {
      int32_t st_idx = 0;
      if (!llm_pos_ids_list.empty()) {
        auto& llm_pos_ids = llm_pos_ids_list.back();
        int32_t max = -1;
        for (int i = 0; i < 3; i++) {
          max = std::max(max, llm_pos_ids(i, llm_pos_ids.dimension(1) - 1));
        }
        st_idx = max + 1;
      }
      int32_t text_len = int32_t(input_ids.size()) - st;
      Eigen::Tensor<int32_t, 2, Eigen::RowMajor> llm_pos_ids(3, text_len);
      for (int j = 0; j < text_len; j++) {
        llm_pos_ids(0, j) = j + st_idx;
        llm_pos_ids(1, j) = j + st_idx;
        llm_pos_ids(2, j) = j + st_idx;
      }
      llm_pos_ids_list.emplace_back(std::move(llm_pos_ids));
    }

    // llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    // position_ids[..., 0, :] = llm_positions
    int32_t total_len = 0;
    for (const auto& llm_pos_ids : llm_pos_ids_list) {
      total_len += int32_t(llm_pos_ids.dimension(1));
    }
    position_ids = Eigen::Tensor<int32_t, 3, Eigen::RowMajor>(3, 1, total_len);
    int32_t idx = 0;
    for (const auto& llm_pos_ids : llm_pos_ids_list) {
      auto llm_pos_ids_reshape = llm_pos_ids.reshape(Eigen::array<Eigen::Index, 3>{3, 1, llm_pos_ids.dimension(1)});
      position_ids.slice(Eigen::array<Eigen::Index, 3>{0, 0, idx},
                         Eigen::array<Eigen::Index, 3>{3, 1, llm_pos_ids.dimension(1)}) = llm_pos_ids_reshape;
      idx += int32_t(llm_pos_ids.dimension(1));
    }

    // mrope_position_deltas = llm_positions.max() + 1 - len(input_ids)
    int32_t max = -1;
    for (int i = 0; i < 3; i++) {
      max = std::max(max, position_ids(i, 0, position_ids.dimension(2) - 1));
    }
    mrope_position_deltas = Eigen::Tensor<int64_t, 2, Eigen::RowMajor>(1, 1);
    mrope_position_deltas.setConstant(max + 1 - int64_t(input_ids.size()));
  }
}

void Qwen2vlVIT::ComputeRotaryPosIds(const Eigen::Tensor<int64_t, 2, Eigen::RowMajor>* image_grid_thw,
                                     Eigen::Tensor<int64_t, 2, Eigen::RowMajor>& position_ids) {
  std::vector<Eigen::Tensor<int64_t, 2, Eigen::RowMajor>> pos_ids_list;

  for (int i = 0; i < image_grid_thw->dimension(0); i++) {
    int64_t t = (*image_grid_thw)(i, 0);
    int64_t h = (*image_grid_thw)(i, 1);
    int64_t w = (*image_grid_thw)(i, 2);

    Eigen::Tensor<int64_t, 2, Eigen::RowMajor> hpos_ids(h, w);
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        hpos_ids(j, k) = j;
      }
    }
    auto hpos_ids_reshape = hpos_ids.reshape(Eigen::array<Eigen::Index, 4>{
      h / spatial_merge_size_, spatial_merge_size_, w / spatial_merge_size_, spatial_merge_size_});
    auto hpos_ids_permute = hpos_ids_reshape.shuffle(Eigen::array<Eigen::Index, 4>{0, 2, 1, 3});
    auto hpos_ids_flatten = hpos_ids_permute.reshape(Eigen::array<Eigen::Index, 1>{h * w});

    Eigen::Tensor<int64_t, 2, Eigen::RowMajor> wpos_ids(h, w);
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < h; k++) {
        wpos_ids(k, j) = j;
      }
    }
    auto wpos_ids_reshape = wpos_ids.reshape(Eigen::array<Eigen::Index, 4>{
      h / spatial_merge_size_, spatial_merge_size_, w / spatial_merge_size_, spatial_merge_size_});
    auto wpos_ids_permute = wpos_ids_reshape.shuffle(Eigen::array<Eigen::Index, 4>{0, 2, 1, 3});
    auto wpos_ids_flatten = wpos_ids_permute.reshape(Eigen::array<Eigen::Index, 1>{h * w});

    Eigen::Tensor<int64_t, 2, Eigen::RowMajor> pos_ids(h * w, 2);
    pos_ids.slice(Eigen::array<Eigen::Index, 2>{0, 0}, Eigen::array<Eigen::Index, 2>{h * w, 1}) = hpos_ids_flatten;
    pos_ids.slice(Eigen::array<Eigen::Index, 2>{0, 1}, Eigen::array<Eigen::Index, 2>{h * w, 1}) = wpos_ids_flatten;
    auto pos_ids_repeat = pos_ids.broadcast(Eigen::array<Eigen::Index, 2>{t, 1});

    pos_ids_list.emplace_back(std::move(pos_ids_repeat));
  }

  int64_t total_len = 0;
  for (const auto& pos_ids : pos_ids_list) {
    total_len += int64_t(pos_ids.dimension(0));
  }
  position_ids = Eigen::Tensor<int64_t, 2, Eigen::RowMajor>(total_len, 2);
  int64_t idx = 0;
  for (const auto& pos_ids : pos_ids_list) {
    position_ids.slice(Eigen::array<Eigen::Index, 2>{idx, 0}, Eigen::array<Eigen::Index, 2>{pos_ids.dimension(0), 2}) =
      pos_ids;
    idx += int64_t(pos_ids.dimension(0));
  }
}

VitModelInputType Qwen2vlVIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
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

  // Get thw.
  Eigen::Tensor<int64_t, 2, Eigen::RowMajor> image_grid_thw;
  int thw_sum = 0;
  GetThw(images_mats, image_grid_thw, thw_sum);
  auto get_thw_end = GET_SYS_TIME_US();

  boost::latch done2(3);
  // Eigen::Tensor<nv_bfloat16, 2, Eigen::RowMajor> pixel_values;
  std::vector<xt::xarray<nv_bfloat16>> pixel_values;
  Eigen::Tensor<bool, 3> attention_mask_vit(1, thw_sum, thw_sum);
  Eigen::Tensor<int32_t, 3, Eigen::RowMajor> mrope_position_ids;
  Eigen::Tensor<int64_t, 2, Eigen::RowMajor> mrope_position_deltas;
  Eigen::Tensor<int32_t, 3, Eigen::RowMajor> mrope_position_ids_padding(1, 3, 32768);
  Eigen::Tensor<int64_t, 2, Eigen::RowMajor> rotary_pos_ids;
  std::shared_ptr<TrtHostBinding> image_tensor;
  std::shared_ptr<TrtHostBinding> rotary_pos_ids_tensor;
  std::shared_ptr<TrtHostBinding> image_grid_thw_tensor;
  std::shared_ptr<TrtHostBinding> attention_mask_vit_tensor;
  std::shared_ptr<TrtHostBinding> mrope_position_ids_padding_tensor;
  std::shared_ptr<TrtHostBinding> mrope_position_deltas_tensor;

  // Async convert to thw patches.
  boost::asio::post(*worker_tp_, [&images_mats, &pixel_values, &images_bytes, &thw_sum, &image_grid_thw, &done2,
                                  &image_tensor, this]() {
    auto begin = GET_SYS_TIME_US();
    // Convert to thw patches.
    // CvtToThwPatches(images_mats, pixel_values, thw_sum, image_grid_thw);
    CvtToThwPatchesXtensor(images_mats, pixel_values, thw_sum, image_grid_thw);
    if (int(images_bytes.size()) != image_grid_thw.dimension(0)) {
      throw std::runtime_error("images count not equal to image_grid_thw dim 0.");
    }
    auto image_dtype = inferer_->binding_type().at("image");
    image_tensor = std::make_shared<TrtHostBinding>(
      "image", nvinfer1::Dims2(thw_sum, 3 * temporal_patch_size_ * patch_size_ * patch_size_), image_dtype);
    auto image_buffer = image_tensor->buffer().Get();
    int64_t buffer_idx = 0;
    for (auto& pixel_value : pixel_values) {
      std::memcpy((nv_bfloat16*)image_buffer + buffer_idx, pixel_value.data(),
                  pixel_value.size() * sizeof(nv_bfloat16));
      buffer_idx += pixel_value.size();
    }
    auto end = GET_SYS_TIME_US();
#if VIT_DBG
    CLOG4(INFO, "image dtype: " << int(image_dtype) << ", image size: [" << thw_sum << ", "
                                << 3 * temporal_patch_size_ * patch_size_ * patch_size_
                                << "], trt buffer size: " << image_tensor->buffer_size() << ", input image size: "
                                << (thw_sum * 3 * temporal_patch_size_ * patch_size_ * patch_size_)
                                << ", cost: " << end - begin << " us");
    CLOG4(INFO, "CvtToThwPatches cost: " << end - begin << " us");
#endif
    done2.count_down();
  });

  // Async compute attention mask and rope index.
  boost::asio::post(*worker_tp_, [&prompt, &token_ids, &image_grid_thw, &attention_mask_vit, &done2,
                                  &mrope_position_ids, &mrope_position_deltas, &mrope_position_ids_padding,
                                  &mrope_position_deltas_tensor, &mrope_position_ids_padding_tensor,
                                  &attention_mask_vit_tensor, this]() {
    auto begin = GET_SYS_TIME_US();
    TokenizeEncodeAndComputeAttentionMaskVit(prompt, token_ids, image_grid_thw, attention_mask_vit);
    // Get rope index.
    GetRopeIndex(token_ids, &image_grid_thw, mrope_position_ids, mrope_position_deltas);
    auto mrope_position_ids_trans = mrope_position_ids.shuffle(Eigen::array<Eigen::Index, 3>{1, 0, 2});
    mrope_position_ids_padding.setZero();
    mrope_position_ids_padding.slice(Eigen::array<Eigen::Index, 3>{0, 0, 0},
                                     Eigen::array<Eigen::Index, 3>{1, 3, mrope_position_ids.dimension(2)}) =
      mrope_position_ids_trans;

    auto mrope_position_deltas_dtype = inferer_->binding_type().at("mrope_position_deltas");
    mrope_position_deltas_tensor = std::make_shared<TrtHostBinding>(
      "mrope_position_deltas", nvinfer1::Dims2(mrope_position_deltas.dimension(0), mrope_position_deltas.dimension(1)),
      mrope_position_deltas_dtype);
    auto mrope_position_deltas_buffer = mrope_position_deltas_tensor->buffer().Get();
    std::memcpy(mrope_position_deltas_buffer, mrope_position_deltas.data(),
                mrope_position_deltas_tensor->buffer_size());

    auto mrope_position_ids_padding_dtype = inferer_->binding_type().at("mrope_position_ids_padding");
    mrope_position_ids_padding_tensor = std::make_shared<TrtHostBinding>(
      "mrope_position_ids_padding",
      nvinfer1::Dims3(mrope_position_ids_padding.dimension(0), mrope_position_ids_padding.dimension(1),
                      mrope_position_ids_padding.dimension(2)),
      mrope_position_ids_padding_dtype);
    auto mrope_position_ids_padding_buffer = mrope_position_ids_padding_tensor->buffer().Get();
    std::memcpy(mrope_position_ids_padding_buffer, mrope_position_ids_padding.data(),
                mrope_position_ids_padding_tensor->buffer_size());

    auto attention_mask_vit_dtype = inferer_->binding_type().at("attention_mask_vit");
    attention_mask_vit_tensor = std::make_shared<TrtHostBinding>(
      "attention_mask_vit", nvinfer1::Dims3(1, attention_mask_vit.dimension(1), attention_mask_vit.dimension(2)),
      attention_mask_vit_dtype);
    auto attention_mask_vit_buffer = attention_mask_vit_tensor->buffer().Get();
    std::memcpy(attention_mask_vit_buffer, attention_mask_vit.data(), attention_mask_vit_tensor->buffer_size());
    auto end = GET_SYS_TIME_US();
#if VIT_DBG
    CLOG4(INFO, "mrope_position_deltas dtype: "
                  << int(mrope_position_deltas_dtype) << ", mrope_position_deltas shape: ["
                  << mrope_position_deltas.dimension(0) << ", " << mrope_position_deltas.dimension(1)
                  << "], trt buffer size: " << mrope_position_deltas_tensor->buffer_size()
                  << ", input image size: " << (mrope_position_deltas.size() * sizeof(int64_t)));
    CLOG4(INFO, "mrope_position_ids_padding dtype: "
                  << int(mrope_position_ids_padding_dtype) << ", mrope_position_ids_padding shape: ["
                  << mrope_position_ids_padding.dimension(0) << ", " << mrope_position_ids_padding.dimension(1) << ", "
                  << mrope_position_ids_padding.dimension(2)
                  << "], trt buffer size: " << mrope_position_ids_padding_tensor->buffer_size()
                  << ", input image size: " << (mrope_position_ids_padding.size() * sizeof(int32_t)));
    CLOG4(INFO, "attention_mask_vit dtype: " << int(attention_mask_vit_dtype) << ", attention_mask_vit shape: [1, "
                                             << attention_mask_vit.dimension(1) << ", "
                                             << attention_mask_vit.dimension(2)
                                             << "], trt buffer size: " << attention_mask_vit_tensor->buffer_size()
                                             << ", input image size: " << (attention_mask_vit.size() * sizeof(bool)));
    CLOG4(INFO, "TokenizeEncodeAndComputeAttentionMaskVit and GetRopeIndex cost: " << end - begin << " us");
#endif
    done2.count_down();
  });

  // Sync compute rotary pos ids.
  boost::asio::post(*worker_tp_, [&image_grid_thw, &rotary_pos_ids, &done2, &rotary_pos_ids_tensor, this]() {
    auto begin = GET_SYS_TIME_US();
    ComputeRotaryPosIds(&image_grid_thw, rotary_pos_ids);
    auto rotary_pos_ids_dtype = inferer_->binding_type().at("rotary_pos_ids");
    rotary_pos_ids_tensor = std::make_shared<TrtHostBinding>(
      "rotary_pos_ids", nvinfer1::Dims2(rotary_pos_ids.dimension(0), rotary_pos_ids.dimension(1)),
      rotary_pos_ids_dtype);
    auto rotary_pos_ids_buffer = rotary_pos_ids_tensor->buffer().Get();
    std::memcpy(rotary_pos_ids_buffer, rotary_pos_ids.data(), rotary_pos_ids_tensor->buffer_size());
    auto end = GET_SYS_TIME_US();
#if VIT_DBG
    CLOG4(INFO, "rotary_pos_ids dtype: " << int(rotary_pos_ids_dtype) << ", rotary_pos_ids shape: ["
                                         << rotary_pos_ids.dimension(0) << ", " << rotary_pos_ids.dimension(1)
                                         << "], trt buffer size: " << rotary_pos_ids_tensor->buffer_size()
                                         << ", input image size: " << (rotary_pos_ids.size() * sizeof(int64_t)));
    CLOG4(INFO, "ComputeRotaryPosIds cost: " << end - begin << " us");
#endif
    done2.count_down();
  });

  auto image_grid_thw_dtype = inferer_->binding_type().at("image_grid_thw");
  image_grid_thw_tensor = std::make_shared<TrtHostBinding>(
    "image_grid_thw", nvinfer1::Dims2(image_grid_thw.dimension(0), image_grid_thw.dimension(1)), image_grid_thw_dtype);
  auto image_grid_thw_buffer = image_grid_thw_tensor->buffer().Get();
  std::memcpy(image_grid_thw_buffer, image_grid_thw.data(), image_grid_thw_tensor->buffer_size());

  done2.wait();
  auto async_wait_end = GET_SYS_TIME_US();
#if VIT_DBG
  CLOG4(INFO, "image_grid_thw dtype: " << int(image_grid_thw_dtype) << ", image_grid_thw shape: ["
                                       << image_grid_thw.dimension(0) << ", " << image_grid_thw.dimension(1)
                                       << "], trt buffer size: " << image_grid_thw_tensor->buffer_size()
                                       << ", input image size: " << (image_grid_thw.size() * sizeof(int64_t)));
  CLOG4(INFO, "Preprocess images success, images count: , load cost: "
                << load_end - begin << " us, get_thw cost: " << get_thw_end - load_end << " us, async_wait cost: "
                << async_wait_end - get_thw_end << " us, total cost: " << async_wait_end - begin << " us");
#endif
  return {{"image", image_tensor},
          {"rotary_pos_ids", rotary_pos_ids_tensor},
          {"image_grid_thw", image_grid_thw_tensor},
          {"attention_mask_vit", attention_mask_vit_tensor},
          {"mrope_position_ids_padding", mrope_position_ids_padding_tensor},
          {"mrope_position_deltas", mrope_position_deltas_tensor}};
}

std::tuple<PtuningEmbeddingTableType, std::optional<tensorrt_llm::executor::MropeConfig>> Qwen2vlVIT::Postprocess(
  VitModelOutputType& model_out, std::string& prompt, tensorrt_llm::executor::VecTokens& token_ids) {
  auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
  auto manager = tensorrt_llm::runtime::BufferManager{std::move(stream)};

  const auto& img_features = model_out[0].second->tensor;
  auto img_features_tensor = executor::detail::ofITensor(img_features);

  const auto& concat_cos_sin = model_out[1].second->tensor;
  auto concat_cos_sin_tensor = executor::Tensor::pooledPinned(executor::DataType::kFP32, {concat_cos_sin->getShape().d[1]});
  TLLM_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(concat_cos_sin_tensor.getData()), concat_cos_sin->data(),
                                  concat_cos_sin_tensor.getSizeInBytes(), cudaMemcpyDeviceToHost,
                                  manager.getStream().get()));

  const auto& mrope_position_deltas_out = model_out[2].second->tensor;
  int64_t mrope_position_deltas_out_cpu;
  TLLM_CUDA_CHECK(cudaMemcpyAsync(&mrope_position_deltas_out_cpu, mrope_position_deltas_out->data(), sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, manager.getStream().get()));
  manager.getStream().synchronize();
#if VIT_DBG
  CLOG4(INFO, "img_features_tensor shape: ["
                << img_features_tensor.getShape()[0] << ", " << img_features_tensor.getShape()[1] << "], dtype: "
                << int(img_features_tensor.getDataType()) << ", size: " << img_features_tensor.getSize());
  CLOG4(INFO, "concat_cos_sin_tensor shape: [" << concat_cos_sin_tensor.getShape()[0]
                                               << "], dtype: " << int(concat_cos_sin_tensor.getDataType())
                                               << ", size: " << concat_cos_sin_tensor.getSizeInBytes());
  CLOG4(INFO, "mrope_position_deltas_out_cpu: " << mrope_position_deltas_out_cpu);
#endif
  return {executor::PromptTuningConfig(std::move(img_features_tensor), std::nullopt),
          executor::MropeConfig(std::move(concat_cos_sin_tensor), int32_t(mrope_position_deltas_out_cpu))};
}

std::tuple<PtuningEmbeddingTableType, MropeConfType> Qwen2vlVIT::Encode(const std::vector<std::string>& img_urls,
                                                                        std::string& prompt,
                                                                        tensorrt_llm::executor::VecTokens& token_ids) {
  if (img_urls.empty()) { // No image, only calculate mrope.
    auto begin = GET_SYS_TIME_US();
    token_ids = tokenizer_->Encode(prompt);
    Eigen::Tensor<int32_t, 3, Eigen::RowMajor> mrope_position_ids;
    Eigen::Tensor<int64_t, 2, Eigen::RowMajor> mrope_position_deltas;
    Eigen::Tensor<int32_t, 3, Eigen::RowMajor> mrope_position_ids_padding(1, 3, 32768);
    GetRopeIndex(token_ids, nullptr, mrope_position_ids, mrope_position_deltas);
    auto mrope_position_ids_trans = mrope_position_ids.shuffle(Eigen::array<Eigen::Index, 3>{1, 0, 2});
    mrope_position_ids_padding.setZero();
    mrope_position_ids_padding.slice(Eigen::array<Eigen::Index, 3>{0, 0, 0},
                                     Eigen::array<Eigen::Index, 3>{1, 3, mrope_position_ids.dimension(2)}) =
      mrope_position_ids_trans;

    std::shared_ptr<TrtHostBinding> mrope_position_ids_padding_tensor;
    std::shared_ptr<TrtHostBinding> mrope_position_deltas_tensor;
    auto mrope_position_deltas_dtype = mrope_inferer_->binding_type().at("mrope_position_deltas");
    mrope_position_deltas_tensor = std::make_shared<TrtHostBinding>(
      "mrope_position_deltas", nvinfer1::Dims2(mrope_position_deltas.dimension(0), mrope_position_deltas.dimension(1)),
      mrope_position_deltas_dtype);
    auto mrope_position_deltas_buffer = mrope_position_deltas_tensor->buffer().Get();
    std::memcpy(mrope_position_deltas_buffer, mrope_position_deltas.data(),
                mrope_position_deltas_tensor->buffer_size());

    auto mrope_position_ids_padding_dtype = mrope_inferer_->binding_type().at("mrope_position_ids_padding");
    mrope_position_ids_padding_tensor = std::make_shared<TrtHostBinding>(
      "mrope_position_ids_padding",
      nvinfer1::Dims3(mrope_position_ids_padding.dimension(0), mrope_position_ids_padding.dimension(1),
                      mrope_position_ids_padding.dimension(2)),
      mrope_position_ids_padding_dtype);
    auto mrope_position_ids_padding_buffer = mrope_position_ids_padding_tensor->buffer().Get();
    std::memcpy(mrope_position_ids_padding_buffer, mrope_position_ids_padding.data(),
                mrope_position_ids_padding_tensor->buffer_size());

    VitModelInputType model_inp = {{"mrope_position_ids_padding", mrope_position_ids_padding_tensor},
                                   {"mrope_position_deltas", mrope_position_deltas_tensor}};

    auto preprocess_end = GET_SYS_TIME_US();

    VitModelOutputType outputs;
    mrope_inferer_->Infer(model_inp, outputs);

    auto infer_end = GET_SYS_TIME_US();

    auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    auto manager = tensorrt_llm::runtime::BufferManager{std::move(stream)};

    const auto& concat_cos_sin = outputs[0].second->tensor;
    auto concat_cos_sin_tensor = executor::Tensor::pooledPinned(executor::DataType::kFP32, {concat_cos_sin->getShape().d[1]});
    TLLM_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(concat_cos_sin_tensor.getData()), concat_cos_sin->data(),
                                    concat_cos_sin_tensor.getSizeInBytes(), cudaMemcpyDeviceToHost,
                                    manager.getStream().get()));

    const auto& mrope_position_deltas_out = outputs[1].second->tensor;
    int64_t mrope_position_deltas_out_cpu;
    TLLM_CUDA_CHECK(cudaMemcpyAsync(&mrope_position_deltas_out_cpu, mrope_position_deltas_out->data(), sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, manager.getStream().get()));
    manager.getStream().synchronize();

    auto postprocess_end = GET_SYS_TIME_US();

#if VIT_DBG
    CLOG4(INFO, "concat_cos_sin_tensor shape: [" << concat_cos_sin_tensor.getShape()[0]
                                                 << "], dtype: " << int(concat_cos_sin_tensor.getDataType())
                                                 << ", size: " << concat_cos_sin_tensor.getSizeInBytes());
    CLOG4(INFO, "VIT model encode success, type: " << type_name_ << ", img_urls size: " << img_urls.size()
                                                   << ", preprocess_time: " << preprocess_end - begin
                                                   << " us, infer_time: " << infer_end - preprocess_end
                                                   << " us, postprocess_time: " << postprocess_end - infer_end
                                                   << " us");
#endif

    return {std::nullopt,
            executor::MropeConfig(std::move(concat_cos_sin_tensor), int32_t(mrope_position_deltas_out_cpu))};
  } else {
    // 1. Get image data from urls.
    auto begin = GET_SYS_TIME_US();
    std::vector<std::vector<char>> images_bytes;
    if (!img_urls.empty()) {
      images_bytes = GetImages(img_urls);
    }
    auto get_images_end = GET_SYS_TIME_US();

    // 2. Preprocess image data to vit trt model input.
    VitModelInputType model_inp = Preprocess(images_bytes, prompt, token_ids);
    auto preprocess_end = GET_SYS_TIME_US();

    // 3. Vit model trt infer.
    VitModelOutputType outputs;
    inferer_->Infer(model_inp, outputs);
    auto infer_end = GET_SYS_TIME_US();

    // 4. Postprocess vit trt model output to trtllm ptuning embedding table.
    auto out = Postprocess(outputs, prompt, token_ids);
    auto postprocess_end = GET_SYS_TIME_US();

#if VIT_DBG
    CLOG4(INFO, "VIT model encode success, type: " << type_name_ << ", img_urls size: " << img_urls.size()
                                                   << ", get_images_time: " << get_images_end - begin
                                                   << " us, preprocess_time: " << preprocess_end - get_images_end
                                                   << " us, infer_time: " << infer_end - preprocess_end
                                                   << " us, postprocess_time: " << postprocess_end - infer_end
                                                   << " us");
#endif
    return out;
  }
}

} // namespace netease::grps