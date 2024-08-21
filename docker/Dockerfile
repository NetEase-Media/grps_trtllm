# Build grps customized project image.

# --------Building stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.4_cudnn8.9_trtllm0.10.0_py3.10 AS build

# grps archive.
RUN mkdir -p /my_grps
ADD conf /my_grps/conf
ADD data /my_grps/data
ADD second_party /my_grps/second_party
ADD src /my_grps/src
ADD third_party /my_grps/third_party
ADD build.sh /my_grps/build.sh
ADD CMakeLists.txt /my_grps/CMakeLists.txt
ADD .config /my_grps/.config
RUN cd /my_grps && \
    grpst archive . --skip_unittest --output_path server.mar

# --------Release stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.4_cudnn8.9_trtllm0.10.0_py3.10

WORKDIR /my_grps
COPY --from=build /my_grps/server.mar /my_grps/server.mar
ENV LANG C.UTF-8