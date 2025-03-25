// InternVideo2.5 VIT(Vision transformer).

#include "intern_video_2_5_vit.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>
#include <cstring>

#include "grps_cli.h"
#include "src/utils.h"

namespace netease::grps {
InternVideo25VIT::InternVideo25VIT() : VIT("intern-video2.5") {}
InternVideo25VIT::~InternVideo25VIT() = default;

void InternVideo25VIT::Init(const std::string& path,
                            int worker_tp,
                            const std::string& device,
                            const YAML::Node& trt_args,
                            const YAML::Node& processor_args,
                            netease::grps::MultiInstanceTokenizer* tokenizer,
                            bool kv_cache_reuse) {
  grps_cli_ = std::make_unique<GrpsCli>("intern-video2.5");
  grps_cli_->Init(processor_args);

  if (processor_args["max_frames"] && !processor_args["max_frames"].IsNull() &&
      processor_args["max_frames"].IsScalar()) {
    max_frames_ = processor_args["max_frames"].as<int32_t>();
  } else {
    throw VitException("max_frames is required.");
  }
  if (processor_args["shm_size"] && !processor_args["shm_size"].IsNull() && processor_args["shm_size"].IsScalar()) {
    shm_size_ = processor_args["shm_size"].as<int32_t>();
  } else {
    throw VitException("shm_size is required.");
  }
  if (processor_args["dtype"] && !processor_args["dtype"].IsNull() && processor_args["dtype"].IsScalar()) {
    if (processor_args["dtype"].as<std::string>() == "float16") {
      dtype_ = nvinfer1::DataType::kHALF;
    } else if (processor_args["dtype"].as<std::string>() == "bfloat16") {
      dtype_ = nvinfer1::DataType::kBF16;
    } else {
      throw VitException("dtype should be float16 or bfloat16.");
    }
  } else {
    throw VitException("dtype is required.");
  }

  worker_tp_ = std::make_unique<boost::asio::thread_pool>(worker_tp);
  kv_cache_reuse_ = kv_cache_reuse;
  tokenizer_ = tokenizer;

  CLOG4(INFO, "InternVideo25VIT initialized, kv_cache_reuse: "
                << kv_cache_reuse_ << ", max_frames: " << max_frames_ << ", shm_size: " << shm_size_ << ", dtype: "
                << int(dtype_) << ", worker_tp: " << worker_tp << ", processor_args: " << processor_args);
}

void InternVideo25VIT::Load() {}

std::tuple<PtuningEmbeddingTableType, MropeConfType> InternVideo25VIT::Encode(
  const std::vector<std::string>& video_urls, std::string& prompt, tensorrt_llm::executor::VecTokens& token_ids) {
  if (video_urls.empty()) {
    return {std::nullopt, std::nullopt};
  }

#if VIT_DBG
  auto begin = GET_SYS_TIME_US();
#endif

  // 1. Call remote vit, predict video images embeddings.
  ::grps::protos::v1::GrpsMessage request;
  ::grps::protos::v1::GrpsMessage response;
  if (video_urls.size() != 1) {
    throw VitException("Only support one video now.");
  }
  request.mutable_gmap()->mutable_s_s()->insert({"video_url", video_urls[0]});
  request.mutable_gmap()->mutable_s_i32()->insert({"max_frames", max_frames_});
  request.mutable_gmap()->mutable_s_i32()->insert({"calc_hash", kv_cache_reuse_ ? 1 : 0});
  grps_cli_->Predict(request, response);
  if (response.status().status() != ::grps::protos::v1::Status::SUCCESS) {
    std::string err = utils::Rstrip(response.status().msg());
    auto pos = err.find_last_of("\n");
    if (pos != std::string::npos) {
      err = err.substr(pos + 1);
    }
    CLOG4(ERROR, "Predict failed: " << err);
    throw VitException("Predict failed, error: " + err);
  }

  std::vector<int64_t> tensor_shape;
  auto shape_gtensor = response.gtensors().tensors().Get(2).flat_int32();
  for (int i = 0; i < shape_gtensor.size(); ++i) {
    tensor_shape.push_back(shape_gtensor.Get(i));
  }

  // 2. Add prompt prefix for each frame.
  auto num_patches_gtensor = response.gtensors().tensors().Get(0).flat_int32();
  std::string prompt_replace;
  int num_frames = 1;
  for (int i = 0; i < num_patches_gtensor.size(); ++i) {
    for (int j = 0; j < num_patches_gtensor.Get(i); ++j) {
      prompt_replace += "Frame";
      prompt_replace += std::to_string(num_frames++);
      prompt_replace += ": <img>";
      for (int k = 0; k < tensor_shape[1]; ++k) {
        prompt_replace += "<IMG_CONTEXT>";
      }
      prompt_replace += "</img>\n";
    }
  }
  utils::ReplaceWorld(prompt, "<VIDEO_CONTEXT>", prompt_replace, 0, 1);

  // 3. Get embeddings from shared memory.
  std::string shm_path = response.gtensors().tensors().Get(1).flat_string().Get(0);
  int fd = shm_open(shm_path.c_str(), O_RDWR, 0666);
  if (fd == -1) {
    CLOG4(ERROR, "shm_open failed: " << strerror(errno) << ", path: " << shm_path);
    throw VitException("shm_open failed.");
  }

  char* char_ptr = (char*)mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (char_ptr == MAP_FAILED) {
    CLOG4(ERROR, "mmap failed: " << strerror(errno));
    throw VitException("mmap failed.");
  }
  void* ptr = (void*)(char_ptr + 1); // First byte is flag.

  // 4. Copy data to tensorrt-llm tensor.
  std::vector<int64_t> reshape = {tensor_shape[0] * tensor_shape[1], tensor_shape[2]};
  int64_t num_elements = reshape[0] * reshape[1];
  tensorrt_llm::batch_manager::NamedTensor named_tensor(dtype_, reshape, "output");
  auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
  auto manager = tensorrt_llm::runtime::BufferManager{std::move(stream)};
  TLLM_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(named_tensor.tensor->data()), ptr, num_elements * 2,
                                  cudaMemcpyHostToDevice, manager.getStream().get()));
  manager.getStream().synchronize();

#if VIT_DBG
  nv_bfloat16* bfloat16_ptr = reinterpret_cast<nv_bfloat16*>(ptr);
  CLOG4(INFO, "First 10 elements: ");
  for (int i = 0; i < 10; ++i) {
    CLOG4(INFO, float(bfloat16_ptr[i]));
  }
  CLOG4(INFO, "Last 10 elements: ");
  for (int i = num_elements - 10; i < num_elements; ++i) {
    CLOG4(INFO, float(bfloat16_ptr[i]));
  }
#endif

  // 5. Set flag to 0, stand for the data has been consumed.
  char_ptr[0] = 0;

  munmap(char_ptr, shm_size_);
  close(fd);
  auto e_tensor = executor::detail::ofITensor(named_tensor.tensor);

  // 6. Get image hash.
  uint64_t hash = response.gtensors().tensors().Get(3).flat_int64().Get(0);

  if (token_ids.empty()) {
    token_ids = tokenizer_->Encode(prompt);
  }

#if VIT_DBG
  auto end = GET_SYS_TIME_US();
  CLOG4(INFO, "shm_path: " << shm_path << ", tensor_shape: " << tensor_shape[0] << ", " << tensor_shape[1] << ", "
                           << tensor_shape[2] << ", hash: " << hash);
  CLOG4(INFO, "InternVideo25VIT encode success, type: " << type_name_ << ", video_urls size: " << video_urls.size()
                                                        << ", cost: " << end - begin << " us");
#endif

  if (kv_cache_reuse_) {
    return {executor::PromptTuningConfig(std::move(e_tensor), PrepareExtraIds(hash, token_ids)), std::nullopt};
  } else {
    return {executor::PromptTuningConfig(std::move(e_tensor), std::nullopt), std::nullopt};
  }
}

VitModelInputType InternVideo25VIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
                                               std::string& prompt,
                                               tensorrt_llm::executor::VecTokens& token_ids) {
  throw VitException("Not implemented.");
}

std::tuple<PtuningEmbeddingTableType, MropeConfType> InternVideo25VIT::Postprocess(
  netease::grps::VitModelOutputType& model_out,
  std::string& prompt,
  tensorrt_llm::executor::VecTokens& token_ids,
  uint64_t img_hash) {
  throw VitException("Not implemented.");
}

} // namespace netease::grps