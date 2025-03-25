// MiniCPM-V VIT(Vision transformer).

#include "minicpmv_vit.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/asio/post.hpp>
#include <boost/thread/latch.hpp>
#include <cstring>

#include "grps_cli.h"
#include "src/utils.h"

namespace netease::grps {
MiniCPMVVIT::MiniCPMVVIT() : VIT("minicpmv") {}
MiniCPMVVIT::~MiniCPMVVIT() = default;

void MiniCPMVVIT::Init(const std::string& path,
                       int worker_tp,
                       const std::string& device,
                       const YAML::Node& trt_args,
                       const YAML::Node& processor_args,
                       netease::grps::MultiInstanceTokenizer* tokenizer,
                       bool kv_cache_reuse) {
  grps_cli_ = std::make_unique<GrpsCli>("minicpmv");
  grps_cli_->Init(processor_args);

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

  if (processor_args["image_token_id"] && !processor_args["image_token_id"].IsNull() &&
      processor_args["image_token_id"].IsScalar()) {
    image_token_id_ = processor_args["image_token_id"].as<int32_t>();
  } else {
    throw VitException("image_token_id is required.");
  }

  if (processor_args["img_begin_token_id"] && !processor_args["img_begin_token_id"].IsNull() &&
      processor_args["img_begin_token_id"].IsScalar()) {
    img_begin_token_id_ = processor_args["img_begin_token_id"].as<int32_t>();
  } else {
    throw VitException("img_begin_token_id is required.");
  }

  worker_tp_ = std::make_unique<boost::asio::thread_pool>(worker_tp);
  kv_cache_reuse_ = kv_cache_reuse;
  tokenizer_ = tokenizer;

  CLOG4(INFO, "MiniCPMVVIT initialized, kv_cache_reuse: "
                << kv_cache_reuse_ << ", shm_size: " << shm_size_ << ", dtype: " << int(dtype_)
                << ", worker_tp: " << worker_tp << ", image_token_id: " << image_token_id_
                << ", img_begin_token_id: " << img_begin_token_id_ << ", processor_args: " << processor_args);
}

void MiniCPMVVIT::Load() {}

std::tuple<PtuningEmbeddingTableType, MropeConfType> MiniCPMVVIT::Encode(const std::vector<std::string>& image_urls,
                                                                         std::string& prompt,
                                                                         tensorrt_llm::executor::VecTokens& token_ids) {
#if VIT_DBG
  auto begin = GET_SYS_TIME_US();
#endif

  // 1. Call remote vit, predict image images embeddings.
  ::grps::protos::v1::GrpsMessage request;
  ::grps::protos::v1::GrpsMessage response;
  prompt = "{\"msgs\":" + prompt + ", \"calc_hash\":" + (kv_cache_reuse_ ? "true" : "false") + "}";
  *request.mutable_str_data() = std::move(prompt);
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

  // 2. Replace image token id.
  auto input_ids = response.gtensors().tensors().Get(0).flat_int32();
  token_ids.assign(input_ids.begin(), input_ids.end());
  int32_t cur_img_token_id = img_begin_token_id_;
  for (size_t i = 0; i < token_ids.size(); i++) {
    if (token_ids[i] == image_token_id_) {
      token_ids[i] = cur_img_token_id++;
    }
  }

  // 3. Get embeddings from shared memory.
  std::string shm_path = response.gtensors().tensors().Get(1).flat_string().Get(0);
  if (shm_path.empty()) { // No image input.
    return {std::nullopt, std::nullopt};
  }
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

  std::vector<int64_t> reshape = {tensor_shape[0] * tensor_shape[1], tensor_shape[2]};
  int64_t num_elements = reshape[0] * reshape[1];

  // 4. Copy data to tensorrt-llm tensor.
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

#if VIT_DBG
  auto end = GET_SYS_TIME_US();
  CLOG4(INFO, "shm_path: " << shm_path << ", tensor_shape: " << tensor_shape[0] << ", " << tensor_shape[1] << ", "
                           << tensor_shape[2]);
  CLOG4(INFO, "MiniCPMVVIT encode success, type: " << type_name_ << ", image_urls size: " << image_urls.size()
                                                   << ", cost: " << end - begin << " us");
#endif

  if (kv_cache_reuse_) {
    return {executor::PromptTuningConfig(std::move(e_tensor), PrepareExtraIds(hash, token_ids)), std::nullopt};
  } else {
    return {executor::PromptTuningConfig(std::move(e_tensor), std::nullopt), std::nullopt};
  }
}

VitModelInputType MiniCPMVVIT::Preprocess(const std::vector<std::vector<char>>& images_bytes,
                                          std::string& prompt,
                                          tensorrt_llm::executor::VecTokens& token_ids) {
  throw VitException("Not implemented.");
}

std::tuple<PtuningEmbeddingTableType, MropeConfType> MiniCPMVVIT::Postprocess(
  netease::grps::VitModelOutputType& model_out,
  std::string& prompt,
  tensorrt_llm::executor::VecTokens& token_ids,
  uint64_t img_hash) {
  throw VitException("Not implemented.");
}

} // namespace netease::grps