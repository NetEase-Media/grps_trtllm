/*
 * copyright netease pctr team
 * @author zhaochaochao at corp netease dot com
 * @date   2025/02/27
 * @brief  Grps server proxy, used to access grps server.
 */

#include "grps_cli.h"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include "logger/logger.h"
#include "utils.h"

#define GRPS_CLI_DBG 0

namespace netease::grps {
GrpsCli::~GrpsCli() = default;
GrpsCli::GrpsCli(const std::string& name) : name_(name) {}

void GrpsCli::Init(const YAML::Node& cfg) {
  if (cfg.IsNull()) {
    CLOG4(ERROR, "GrpsCli init failed, config is null.");
    throw GrpsCliException("GrpsCli init failed, config is null.");
  }

  try {
    server_addr_ = cfg["host"].as<std::string>();
    server_port_ = cfg["port"].as<int>();
    timeout_ms_ = cfg["timeout_ms"].as<int>();
  } catch (const std::exception& e) {
    CLOG4(ERROR, "GrpsCli init failed, exception: " << e.what());
    throw GrpsCliException("GrpsCli init failed, exception: " + std::string(e.what()));
  }

  grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(-1);
  channel_args.SetInt(GRPC_ARG_ENABLE_RETRIES, 1);
  channel_args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 5000);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, timeout_ms_);
  channel_ = grpc::CreateCustomChannel(server_addr_ + ":" + std::to_string(server_port_),
                                       grpc::InsecureChannelCredentials(), channel_args);
  if (!channel_) {
    CLOG4(ERROR, "GrpsCli init failed, create channel failed, server_addr: " << server_addr_
                                                                             << ", server_port: " << server_port_);
    throw GrpsCliException("GrpsCli init failed, create channel failed, server_addr: " + server_addr_ +
                           ", server_port: " + std::to_string(server_port_));
  }

  stub_ = ::grps::protos::v1::GrpsService::NewStub(channel_);
  if (!stub_) {
    CLOG4(ERROR, "GrpsCli init failed, create stub failed, server_addr: " << server_addr_
                                                                          << ", server_port: " << server_port_);
    throw GrpsCliException("GrpsCli init failed, create stub failed, server_addr: " + server_addr_ +
                           ", server_port: " + std::to_string(server_port_));
  }

  CLOG4(INFO, "GrpsCli init success, name: " << name_ << ", server_addr: " << server_addr_
                                             << ", server_port: " << server_port_ << ", timeout_ms: " << timeout_ms_);
}

void GrpsCli::Predict(const ::grps::protos::v1::GrpsMessage& request, ::grps::protos::v1::GrpsMessage& response) {
#if GRPS_CLI_DBG
  auto begin_us = GET_SYS_TIME_US();
#endif

  grpc::ClientContext context;
  grpc::Status status = stub_->Predict(&context, request, &response);
  if (!status.ok()) {
    CLOG4(ERROR, "GrpsCli predict failed, status: " << status.error_message());
    throw GrpsCliException("GrpsCli predict failed, status: " + status.error_message());
  }

#if GRPS_CLI_DBG
  auto end_us = GET_SYS_TIME_US();
  std::string res_str;
  ::google::protobuf::TextFormat::PrintToString(response, &res_str);
  CLOG4(INFO, "GrpsCli predict success, request: " << request.DebugString() << ", response: " << res_str
                                                   << ", cost: " << end_us - begin_us << " us");
#endif
}
} // namespace netease::grps
