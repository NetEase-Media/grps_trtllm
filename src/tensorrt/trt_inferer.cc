/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2024/06/04
 * Brief  TensorRT model inferer implementation.
 */

#include "trt_inferer.h"

#include <NvInferRuntime.h>
#include <dlfcn.h>

#include <filesystem>
#include <fstream>
#include <regex>

#include "logger/logger.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#define TRT_INFERER_DEBUG 0

namespace netease::grps {
class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        CLOG4(ERROR, "[TRT-IE] " << msg);
        break;
      case Severity::kERROR:
        CLOG4(ERROR, "[TRT-E] " << msg);
        break;
      case Severity::kWARNING:
        CLOG4(WARN, "[TRT-W] " << msg);
        break;
      case Severity::kINFO:
        CLOG4(INFO, "[TRT-I] " << msg);
        break;
      case Severity::kVERBOSE:
        // CLOG4(INFO, "[TRT-V] " << msg);
        break;
    }
  }
};

static TrtLogger g_trt_logger;

void TrtDeviceBinding::Allocate(const nvinfer1::Dims& true_dims) {
  true_dims_ = true_dims;
  volume_ = Volume(true_dims, vec_dim_, comps_);
  size_t new_buffer_size = volume_ * data_type_size_;
  if (new_buffer_size > buffer_capacity_) {
    buffer_.Allocate(new_buffer_size);
    buffer_capacity_ = new_buffer_size;
  }
  buffer_size_ = new_buffer_size;
}

void TrtDeviceBinding::FromHost(TrtHostBinding& host_binding, CudaStream& stream) {
  // Check dims.
  if (dims_.nbDims != host_binding.dims().nbDims) {
    auto err_msg = std::string("Dims not match, binding: ") + name_ + ", dims: " + std::to_string(dims_.nbDims) +
                   ", host dims: " + std::to_string(host_binding.dims().nbDims);
    CLOG4(ERROR, err_msg);
    throw TrtInfererException(err_msg);
  }
  for (int i = 0; i < dims_.nbDims; ++i) {
    if (dims_.d[i] != -1 && dims_.d[i] != host_binding.dims().d[i]) {
      auto err_msg = std::string("Dims not match, binding: ") + name_ + ", binding dim[" + std::to_string(i) +
                     "]: " + std::to_string(dims_.d[i]) + ", host dim[" + std::to_string(i) +
                     "]: " + std::to_string(host_binding.dims().d[i]);
      CLOG4(ERROR, err_msg);
      throw TrtInfererException(err_msg);
    } else if (host_binding.dims().d[i] > max_dims_.d[i]) {
      auto err_msg = std::string("Dims not match, binding: ") + name_ + ", host dim[" + std::to_string(i) +
                     "]: " + std::to_string(host_binding.dims().d[i]) + ", binding max dim[" + std::to_string(i) +
                     "]: " + std::to_string(max_dims_.d[i]);
      CLOG4(ERROR, err_msg);
      throw TrtInfererException(err_msg);
    }
  }
  // Check dtype.
  if (data_type_ != host_binding.data_type()) {
    auto err_msg = std::string("Data type not match, binding: ") + name_ +
                   ", binding dtype: " + std::to_string(static_cast<int>(data_type_)) +
                   ", host dtype: " + std::to_string(static_cast<int>(host_binding.data_type()));
    CLOG4(ERROR, err_msg);
    throw TrtInfererException(err_msg);
  }

  // Allocate buffer if not enough.
  Allocate(host_binding.dims());
  if (host_binding.buffer_size() != buffer_size()) {
    auto err_msg = std::string("Dims not match, binding: ") + name_ +
                   ", buffer size: " + std::to_string(buffer_size()) +
                   ", host buffer size: " + std::to_string(host_binding.buffer_size());
    CLOG4(ERROR, err_msg);
    throw TrtInfererException(err_msg);
  }

  // Copy data from host.
  H2D(stream, host_binding.buffer(), buffer(), host_binding.buffer_size());
}

void TrtDeviceBinding::ToHost(CudaStream& stream, TrtHostBinding& host_binding) {
  D2H(stream, buffer(), host_binding.buffer(), buffer_size());
}

void TrtDeviceBinding::ToDevice(CudaStream& stream, TrtDeviceBinding& device_binding) {
  D2D(stream, buffer(), device_binding.buffer(), buffer_size());
}

TrtModelInferer::TrtModelInferer() = default;
TrtModelInferer::~TrtModelInferer() = default;

void TrtModelInferer::Init(const std::string& path, const std::string& device, const YAML::Node& args) {
  if (!std::filesystem::exists(path)) {
    CLOG4(ERROR, "Init tensorrt model inferer failed, file: " << path << " not exists.");
    throw TrtInfererException("File not exists: " + path);
  }
  path_ = path;

  if (args && !args.IsNull() && args.IsMap()) {
    if (args["dla_cores"] && args["dla_cores"].IsScalar()) {
      dla_cores_ = args["dla_cores"].as<int>();
      if (dla_cores_ < 0) {
        CLOG4(ERROR, "Init tensorrt model inferer failed, dla_cores: " << dla_cores_ << " should be >= 0.");
        throw TrtInfererException("dla_cores should be >= 0.");
      }
    }
    if (args["streams"] && args["streams"].IsScalar()) {
      streams_ = args["streams"].as<int>();
      if (streams_ <= 0) {
        CLOG4(ERROR, "Init tensorrt model inferer failed, streams: " << streams_ << " should be > 0.");
        throw TrtInfererException("streams should be > 0.");
      }
    }

    // Load customized op lib if exists.
    auto customized_op_paths = args["customized_op_paths"];
    if (customized_op_paths && !customized_op_paths.IsNull() && customized_op_paths.IsSequence()) {
      for (const auto& op_path : customized_op_paths) {
        if (op_path && !op_path.IsNull() && op_path.IsScalar()) {
          auto path_str = op_path.as<std::string>();
          if (!path_str.empty()) {
            auto* handler = dlopen(path_str.c_str(), RTLD_LAZY);
            if (handler == nullptr) {
              std::string err_str = "Load customized op lib failed, path: ";
              err_str.append(path_str).append(", error: ").append(dlerror());
              CLOG4(ERROR, err_str);
              throw TrtInfererException(err_str);
            }
            CLOG4(INFO, "Load customized op lib success, path: " << path_str);
          }
        }
      }
    }
  }
  cur_stream_ = 0;
}

void TrtModelInferer::Load() {
  // Load engine file.
  std::ifstream engine_file(path_, std::ios::binary);
  if (!engine_file) {
    CLOG4(ERROR, "Error loading engine file: " << path_);
    throw TrtInfererException("Error loading engine file: " + path_);
  }
  engine_file.seekg(0, std::ifstream::end);
  size_t fsize = static_cast<size_t>(engine_file.tellg());
  engine_file.seekg(0, std::ifstream::beg);
  std::vector<char> engine_data(fsize);
  engine_file.read(engine_data.data(), (long long)(fsize));
  if (!engine_file) {
    CLOG4(ERROR, "Error loading engine file: " << path_);
    throw TrtInfererException("Error loading engine file: " + path_);
  }

  // Create runtime.
  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(g_trt_logger)};
  if (dla_cores_ != -1) {
    runtime_->setDLACore(dla_cores_);
  }

  // Load all instances.
  for (int i = 0; i < streams_; i++) {
    auto instance = std::make_unique<Instance>();
    // Deserialize engine.
    instance->engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>{runtime_->deserializeCudaEngine(engine_data.data(), fsize)};
    if (!instance->engine_) {
      CLOG4(ERROR, "Error deserializing engine file: " << path_);
      throw TrtInfererException("Error deserializing engine file: " + path_);
    }

    // Load bindings.
    for (int j = 0; j < instance->engine_->getNbIOTensors(); ++j) {
      auto name = instance->engine_->getIOTensorName(j);
      auto dims = instance->engine_->getTensorShape(name);
      bool is_input_binding = instance->engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
      auto is_shape_binding = instance->engine_->isShapeInferenceIO(name);
      auto max_dims =
        is_input_binding ? instance->engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX) : dims;
      auto dtype = instance->engine_->getTensorDataType(name);
      auto vec_dim = instance->engine_->getTensorVectorizedDim(name);
      auto comps = instance->engine_->getTensorComponentsPerElement(name);
      instance->bindings_.emplace_back(name, dims, max_dims, dtype, vec_dim, comps, is_input_binding, is_shape_binding);

      if (is_shape_binding) {
        throw TrtInfererException("Shape binding is not supported.");
      }

      CLOG4(INFO, "Trt instance(" << i << ") add binding: " << instance->bindings_.back().DebugString());
    }

    // Create context.
    instance->trt_context_ = TrtUniquePtr<nvinfer1::IExecutionContext>{instance->engine_->createExecutionContext()};
    instances_.emplace_back(std::move(instance));
  }

  CLOG4(INFO, "Load tensorrt model success, path: " << path_ << ", streams: " << streams_);
}

void TrtModelInferer::Infer(
  const std::vector<std::pair<std::string, std::shared_ptr<TrtHostBinding>>>& inputs,
  std::vector<std::pair<std::string, std::shared_ptr<tensorrt_llm::batch_manager::NamedTensor>>>& outputs) {
  int idx = cur_stream_.load();
  while (!cur_stream_.compare_exchange_weak(idx, (idx + 1) % streams_)) {
    idx = cur_stream_.load();
  }
  auto& instance = instances_[idx];
  std::unique_lock<std::mutex> input_free_lock(instance->mutex_);

#if TRT_INFERER_DEBUG
  auto& inp_start_event = instance->multi_event_[static_cast<int>(EventType::kINPUT_START)];
  auto& inp_end_event = instance->multi_event_[static_cast<int>(EventType::kINPUT_END)];
  auto& compute_start_event = instance->multi_event_[static_cast<int>(EventType::kCOMPUTE_START)];
  auto& compute_end_event = instance->multi_event_[static_cast<int>(EventType::kCOMPUTE_END)];
  auto& out_start_event = instance->multi_event_[static_cast<int>(EventType::kOUTPUT_START)];
  auto& out_end_event = instance->multi_event_[static_cast<int>(EventType::kOUTPUT_END)];
#endif

  // 1. Prepare bindings.
#if TRT_INFERER_DEBUG
  inp_start_event.Record(instance->stream_);
#endif
  for (size_t i = 0; i < instance->bindings_.size(); i++) {
    auto& binding = instance->bindings_[i];
    if (binding.is_input_binding()) {
      auto it = inputs.end();
      if (inputs[i].first.empty()) { // Use default name and sequence.
        if (i >= inputs.size()) {
          auto err_msg = std::string("Input not found, binding: ") + binding.name();
          CLOG4(ERROR, err_msg);
          throw TrtInfererException(err_msg);
        }
        it = inputs.begin() + i;
      } else { // Find by name.
        it = std::find_if(inputs.begin(), inputs.end(),
                          [&binding](const auto& input) { return input.first == binding.name(); });
        if (it == inputs.end()) {
          auto err_msg = std::string("Input not found, binding: ") + binding.name();
          CLOG4(ERROR, err_msg);
          throw TrtInfererException(err_msg);
        }
      }
      auto& host_binding = *(it->second);

      binding.FromHost(host_binding, instance->stream_);
      instance->trt_context_->setInputShape(binding.name(), binding.true_dims());
      instance->trt_context_->setInputTensorAddress(binding.name(), binding.buffer().Get());
    } else {
      auto true_dims = instance->trt_context_->getTensorShape(binding.name());
      binding.Allocate(true_dims);
      instance->trt_context_->setOutputTensorAddress(binding.name(), binding.buffer().Get());
    }
  }
#if TRT_INFERER_DEBUG
  inp_end_event.Record(instance->stream_);
#endif

  // 2. Execute.
#if TRT_INFERER_DEBUG
  compute_start_event.Record(instance->stream_);
#endif
  instance->trt_context_->enqueueV3(instance->stream_.Get());
#if TRT_INFERER_DEBUG
  compute_end_event.Record(instance->stream_);
#endif

  // 3. Output to named tensor with device memory.
#if TRT_INFERER_DEBUG
  out_start_event.Record(instance->stream_);
#endif
  for (auto& binding : instance->bindings_) {
    if (!binding.is_input_binding()) {
      // Insert new NamedTensor.
      std::vector<int64_t> shape;
      for (int i = 0; i < binding.true_dims().nbDims; ++i) {
        shape.push_back(binding.true_dims().d[i]);
      }
      outputs.emplace_back(binding.name(), std::make_shared<tensorrt_llm::batch_manager::NamedTensor>(
                                             binding.data_type(), shape, binding.name()));
      auto& t = *outputs.back().second;

      // Copy data from trt inferer output device mem to NamedTensor device mem.
      auto manager = tensorrt_llm::runtime::BufferManager{
        std::make_shared<tensorrt_llm::runtime::CudaStream>(instance->stream_.Get())};
      t.tensor = manager.gpu(t.tensor->getShape(), binding.data_type());
      CudaCheck(cudaMemcpyAsync(static_cast<char*>(t.tensor->data()), binding.buffer().Get(),
                                t.tensor->getSizeInBytes(), cudaMemcpyDeviceToDevice, instance->stream_.Get()));
    }
  }
#if TRT_INFERER_DEBUG
  out_end_event.Record(instance->stream_);
#endif

  cudaStreamSynchronize(instance->stream_.Get());

#if TRT_INFERER_DEBUG
  CLOG4(INFO, "Trt instance: " << idx << ", H2D: " << (inp_end_event - inp_start_event) << " ms, "
                               << "Compute: " << (compute_end_event - compute_start_event) << " ms, "
                               << "D2D: " << (out_end_event - out_start_event) << " ms.");
#endif
}
} // namespace netease::grps