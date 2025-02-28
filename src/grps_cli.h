/*
 * copyright netease pctr team
 * @author zhaochaochao at corp netease dot com
 * @date   2025/02/27
 * @brief  Grps server proxy, used to access grps server.
 */

#pragma once
#include <grpcpp/grpcpp.h>
#include <grps_apis/grps.grpc.pb.h>
#include <yaml-cpp/yaml.h>

#include <memory>

namespace netease::grps {

class GrpsCliException : public std::exception {
public:
  explicit GrpsCliException(std::string message) : message_(std::move(message)) {}
  ~GrpsCliException() override = default;
  [[nodiscard]] const char* what() const noexcept override {
    static std::string err_message;
    err_message = "[GrpsCliException] " + message_;
    return err_message.c_str();
  }

private:
  std::string message_;
};

class GrpsCli {
public:
  ~GrpsCli();
  explicit GrpsCli(const std::string& name);

  void Init(const YAML::Node& cfg);

  void Predict(const ::grps::protos::v1::GrpsMessage& request, ::grps::protos::v1::GrpsMessage& response);

private:
  std::string name_;
  std::string server_addr_;
  int server_port_{};
  int timeout_ms_{};

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<::grps::protos::v1::GrpsService::Stub> stub_;
};
} // namespace netease::grps
