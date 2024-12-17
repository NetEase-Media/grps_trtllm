// VIT(Vision transformer) used for multi-modal image encoding to embeddings.

#include "vit.h"

#include <boost/asio/post.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <boost/thread/latch.hpp>
#include <fstream>
#include <regex>

#include "src/utils.h"
#include "src/vit/internvl2_vit.h"
#include "src/vit/qwen2vl_vit.h"
#include "src/vit/qwenvl_vit.h"

namespace netease::grps {

void VIT::Init(const std::string& path,
               int worker_tp,
               const std::string& device,
               const YAML::Node& trt_args,
               const YAML::Node& processor_args,
               MultiInstanceTokenizer* tokenizer) {
  inferer_ = std::make_unique<TrtModelInferer>();
  worker_tp_ = std::make_unique<boost::asio::thread_pool>(worker_tp);
  inferer_->Init(path, device, trt_args);
  processor_args_ = processor_args;
  tokenizer_ = tokenizer;
  CLOG4(INFO, "VIT model initialized, type: " << type_name_ << ", worker_tp" << worker_tp << ", path: " << path
                                              << ", trt_args: " << trt_args << ", processor_args: " << processor_args);
}

void VIT::Load() {
  inferer_->Load();
  CLOG4(INFO, "VIT model loaded, type: " << type_name_);
}

static void GetImageFn(const std::vector<std::string>& img_urls, std::vector<std::vector<char>>& imgs_bytes, size_t i) {
  const auto& img_url = img_urls[i];
  if (img_url.find("http://") == 0 || img_url.find("https://") == 0) {
    imgs_bytes[i] = utils::DownloadFile<std::vector<char>>(img_url);
  } else if (img_url.find("file://") == 0) {
    imgs_bytes[i] = utils::LoadBytesFromFile<std::vector<char>>(img_url.substr(7));
  } else { // base64 image
    if (img_url.find(',') == std::string::npos) {
      return;
    }
    const auto& prefix = img_url.substr(0, img_url.find(',') + 1);
    if (std::regex_match(prefix, std::regex("^data:image/[a-zA-Z]*;base64,$"))) {
      const auto& base64_str = img_url.substr(img_url.find(',') + 1);
      imgs_bytes[i].resize(boost::beast::detail::base64::decoded_size(base64_str.size()));
      auto res = boost::beast::detail::base64::decode(imgs_bytes[i].data(), base64_str.data(), base64_str.size());
      imgs_bytes[i].resize(res.first);
    }
  }
}

std::vector<std::vector<char>> VIT::GetImages(const std::vector<std::string>& img_urls) {
  std::vector<std::vector<char>> images_bytes;
  images_bytes.reserve(img_urls.size());

  boost::latch done(img_urls.size());
  for (size_t i = 0; i < img_urls.size(); i++) {
    images_bytes.emplace_back();
    boost::asio::post(*worker_tp_, [&img_urls, &images_bytes, &done, i] {
      try {
        GetImageFn(img_urls, images_bytes, i);
      } catch (const std::exception& e) {
        CLOG4(ERROR, "Get image failed from url: " << img_urls[i] << " failed, error: " << e.what());
      }
      done.count_down();
    });
  }
  done.wait();

  // Check images bytes.
  for (size_t i = 0; i < images_bytes.size(); ++i) {
    if (images_bytes[i].empty()) {
      throw VitException("Get image from url: " + img_urls[i] + " failed.");
    }
  }

  return images_bytes;
}

std::tuple<PtuningEmbeddingTableType, MropeConfType> VIT::Encode(const std::vector<std::string>& img_urls,
                                                                 std::string& prompt,
                                                                 tensorrt_llm::executor::VecTokens& token_ids) {
  if (img_urls.empty()) {
    return {std::nullopt, std::nullopt};
  }

  // 1. Get image data from urls.
  auto begin = GET_SYS_TIME_US();
  std::vector<std::vector<char>> images_bytes = GetImages(img_urls);
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
                                                 << " us, postprocess_time: " << postprocess_end - infer_end << " us");
#endif

  return out;
}

std::unique_ptr<VIT> VITFactory::CreateVIT(const std::string& type_name) {
  if (type_name == "internvl2") {
    return std::make_unique<Internvl2VIT>();
  } else if (type_name == "qwenvl") {
    return std::make_unique<QwenvlVIT>();
  } else if (type_name == "qwen2vl") {
    return std::make_unique<Qwen2vlVIT>();
  } else {
    return nullptr;
  }
}
} // namespace netease::grps