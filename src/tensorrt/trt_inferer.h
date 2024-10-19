/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2024/06/04
 * Brief  TensorRT model inferer implementation.
 */

#pragma once

#include <tensorrt_llm/batch_manager/namedTensor.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <memory>
#include <mutex>

#include "src/tensorrt/trt_cuda.h"

namespace netease::grps {
class TrtInfererException : public std::exception {
public:
  explicit TrtInfererException(std::string message) : message_(std::move(message)) {}
  ~TrtInfererException() override = default;
  [[nodiscard]] const char* what() const noexcept override {
    static std::string err_message;
    err_message = "[TrtInfererException] " + message_;
    return err_message.c_str();
  }

private:
  std::string message_;
};

// Tensorrt cpu tensor binding. (Use binding naming tensor because of tensorrt api before 9.0 version.)
// Data saved to cpu memory.
class TrtHostBinding {
public:
  /**
   * Constructor.
   * @param name: Binding name.
   * @param dims: Binding dimensions shape.
   * @param data_type: Data type.
   * @param vec_dim: Dimension index that the buffer is vectorized.
   * @param comps: The number of components included in one element.
   */
  explicit TrtHostBinding(
    const char* name, const nvinfer1::Dims& dims, const nvinfer1::DataType& data_type, int vec_dim = -1, int comps = -1)
      : name_(name), dims_(dims), data_type_(data_type), data_type_size_(TrtDtypeSize(data_type)) {
    volume_ = Volume(dims, vec_dim, comps);
    buffer_size_ = volume_ * data_type_size_;
    buffer_.Allocate(buffer_size_);
  }

  explicit TrtHostBinding(
    const char* name, nvinfer1::Dims&& dims, const nvinfer1::DataType& data_type, int vec_dim = -1, int comps = -1)
      : name_(name), dims_(dims), data_type_(data_type), data_type_size_(TrtDtypeSize(data_type)) {
    volume_ = Volume(dims, vec_dim, comps);
    buffer_size_ = volume_ * data_type_size_;
    buffer_.Allocate(buffer_size_);
  }

  TrtHostBinding(const TrtHostBinding&) = delete;
  TrtHostBinding& operator=(const TrtHostBinding&) = delete;
  TrtHostBinding(TrtHostBinding&&) = default;
  TrtHostBinding& operator=(TrtHostBinding&&) = default;

  [[nodiscard]] const char* name() const { return name_; }

  [[nodiscard]] const nvinfer1::Dims& dims() const { return dims_; }

  [[nodiscard]] nvinfer1::DataType data_type() const { return data_type_; }

  [[nodiscard]] uint8_t data_type_size() const { return data_type_size_; }

  [[nodiscard]] size_t volume() const { return volume_; }

  [[nodiscard]] size_t buffer_size() const { return buffer_size_; }

  [[nodiscard]] HostBuffer& buffer() { return buffer_; }

  [[nodiscard]] std::string DebugString() {
    std::string str = "TrtHostBinding: {";
    str += "name: " + std::string(name_) + ", ";
    str += "dims: [";
    for (int i = 0; i < dims_.nbDims; ++i) {
      str += std::to_string(dims_.d[i]);
      if (i != dims_.nbDims - 1) {
        str += ", ";
      }
    }
    str += "], ";
    str += "data_type: " + std::to_string(int(data_type_)) + ", ";
    str += "data_type_size: " + std::to_string(data_type_size_) + ", ";
    str += "volume: " + std::to_string(volume_) + ", ";
    str += "buffer_size: " + std::to_string(buffer_size_);
    str += "}";
    return str;
  }

private:
  const char* name_;             // Binding name.
  nvinfer1::Dims dims_;          // Binding dimensions shape.
  nvinfer1::DataType data_type_; // Binding data type.
  uint8_t data_type_size_;       // Size of this data type(bytes).
  size_t volume_;                // Data volume.
  HostBuffer buffer_;            // Binding host buffer.
  size_t buffer_size_;           // Buffer size.
};

// Tensorrt device tensor binding. (Use binding naming tensor because of tensorrt api before 9.0 version.)
// Data saved to device memory.
class TrtDeviceBinding {
public:
  /**
   * Constructor.
   * @param name: Binding name.
   * @param dims: Binding dimensions shape.
   * @param data_type: Data type.
   * @param vec_dim: Dimension index that the buffer is vectorized.
   * @param comps: The number of components included in one element.
   * @param is_input_binding: Whether this binding is an input binding.
   * @param is_shape_binding: Whether this binding is a shape binding.
   */
  explicit TrtDeviceBinding(const char* name,
                            const nvinfer1::Dims& dims,
                            const nvinfer1::Dims& max_dims,
                            const nvinfer1::DataType& data_type,
                            int vec_dim = -1,
                            int comps = -1,
                            bool is_input_binding = false,
                            bool is_shape_binding = false)
      : name_(name)
      , dims_(dims)
      , max_dims_(max_dims)
      , data_type_(data_type)
      , vec_dim_(vec_dim)
      , comps_(comps)
      , data_type_size_(TrtDtypeSize(data_type))
      , volume_(0)
      , buffer_size_(0)
      , buffer_capacity_(0)
      , is_input_binding_(is_input_binding)
      , is_shape_binding_(is_shape_binding) {}

  TrtDeviceBinding(const TrtDeviceBinding&) = delete;
  TrtDeviceBinding& operator=(const TrtDeviceBinding&) = delete;
  TrtDeviceBinding(TrtDeviceBinding&&) = default;
  TrtDeviceBinding& operator=(TrtDeviceBinding&&) = default;

  void Allocate(const nvinfer1::Dims& true_dims);

  void FromHost(TrtHostBinding& host_binding, CudaStream& stream);

  void ToHost(CudaStream& stream, TrtHostBinding& host_binding);

  void ToDevice(CudaStream& stream, TrtDeviceBinding& device_binding);

  [[nodiscard]] const char* name() const { return name_; }

  [[nodiscard]] const nvinfer1::Dims& dims() const { return dims_; }

  [[nodiscard]] const nvinfer1::Dims& max_dims() const { return max_dims_; }

  [[nodiscard]] const nvinfer1::Dims& true_dims() const { return true_dims_; }

  [[nodiscard]] int vec_dim() const { return vec_dim_; }

  [[nodiscard]] int comps() const { return comps_; }

  [[nodiscard]] nvinfer1::DataType data_type() const { return data_type_; }

  [[nodiscard]] uint8_t data_type_size() const { return data_type_size_; }

  [[nodiscard]] size_t volume() const { return volume_; }

  [[nodiscard]] size_t buffer_size() const { return buffer_size_; }

  [[nodiscard]] size_t buffer_capacity() const { return buffer_capacity_; }

  [[nodiscard]] DeviceBuffer& buffer() { return buffer_; }

  [[nodiscard]] bool is_input_binding() const { return is_input_binding_; }

  [[nodiscard]] bool is_shape_binding() const { return is_shape_binding_; }

  [[nodiscard]] std::string DebugString() {
    std::string str = "TrtDeviceBinding: {";
    str += "name: " + std::string(name_) + ", ";
    str += "dims: [";
    for (int i = 0; i < dims_.nbDims; ++i) {
      str += std::to_string(dims_.d[i]);
      if (i != dims_.nbDims - 1) {
        str += ", ";
      }
    }
    str += "], ";
    str += "max_dims: [";
    for (int i = 0; i < max_dims_.nbDims; ++i) {
      str += std::to_string(max_dims_.d[i]);
      if (i != max_dims_.nbDims - 1) {
        str += ", ";
      }
    }
    str += "], ";
    str += "true_dims: [";
    for (int i = 0; i < true_dims_.nbDims; ++i) {
      str += std::to_string(true_dims_.d[i]);
      if (i != true_dims_.nbDims - 1) {
        str += ", ";
      }
    }
    str += "], ";
    str += "data_type: " + std::to_string(int(data_type_)) + ", ";
    str += "data_type_size: " + std::to_string(data_type_size_) + ", ";
    str += "volume: " + std::to_string(volume_) + ", ";
    str += "buffer_size: " + std::to_string(buffer_size_) + ", ";
    str += "buffer_capacity: " + std::to_string(buffer_capacity_) + ", ";
    str += "is_input_binding: " + std::to_string(is_input_binding_) + ", ";
    str += "is_shape_binding: " + std::to_string(is_shape_binding_);
    str += "}";
    return str;
  }

private:
  const char* name_;             // Binding name.
  nvinfer1::Dims dims_;          // Binding dimensions shape that may have dynamic dim.
  nvinfer1::Dims max_dims_;      // Binding dimensions shape that dynamic dim is set to max.
  nvinfer1::Dims true_dims_{};   // Binding dimensions shape that dynamic dim is set to actual size.
  nvinfer1::DataType data_type_; // Binding data type.
  int vec_dim_;                  // Dimension index that the buffer is vectorized.
  int comps_;                    // The number of components included in one element.
  uint8_t data_type_size_;       // Size of this data type(bytes).
  size_t volume_;                // Data volume.
  DeviceBuffer buffer_;          // Binding device buffer.
  size_t buffer_size_;           // Buffer size.
  size_t buffer_capacity_;       // Buffer capacity.
  bool is_input_binding_;        // Whether this binding is an input binding.
  bool is_shape_binding_;        // Whether this binding is a shape binding.
};

class TrtModelInferer {
public:
  TrtModelInferer();
  ~TrtModelInferer();

  /**
   * @brief Init model inferer.
   * @param path: Model path, it can be a file path or a directory path.
   * @param device: Device name, currently will be ignored, and only support `gpu:0`.
   * @param args: More args.
   * @throw TrtInfererException: If init failed, throw TrtInfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Init(const std::string& path, const std::string& device, const YAML::Node& args);

  /**
   * @brief Load model.
   * @throw TrtInfererException: If load failed, throw TrtInfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Load();

  /**
   * @brief Infer model.
   * @param inputs: Input tensor(TrtHostBinding) of model.
   * @param outputs: Output tensor(tensorrt_llm::batch_manager::NamedTensor) of model.
   * @param ctx: Context of current request.
   * @throw TrtInfererException: If infer failed, can throw TrtInfererException and will be caught by server and return
   * error message to client.
   */
  void Infer(const std::vector<std::pair<std::string, std::shared_ptr<TrtHostBinding>>>& inputs,
             std::vector<std::pair<std::string, std::shared_ptr<tensorrt_llm::batch_manager::NamedTensor>>>& outputs);

  [[nodiscard]] const std::unordered_map<std::string, nvinfer1::DataType> binding_type() const { return binding_type_; }

private:
  struct Instance {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> trt_context_;
    CudaStream stream_;
    std::vector<TrtDeviceBinding> bindings_;
    MultiTrtEvent multi_event_;
    std::mutex mutex_;
  };

  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  std::vector<std::unique_ptr<Instance>> instances_;
  std::string path_;
  int dla_cores_ = -1;
  int streams_ = 1;
  std::atomic<int> cur_stream_ = 0; // Current stream index.
  std::unordered_map<std::string, nvinfer1::DataType> binding_type_;
};
} // namespace netease::grps