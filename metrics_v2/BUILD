load("@publisher_deps//:requirements.bzl", "requirement")

genrule(
  name = "metrics_proto_python",
  srcs = ["metrics.proto"],
  outs = ["metrics_pb2.py"],
  cmd = "$(execpath @protoc//:bin) --proto_path=external/protoc/include -I . --python_out=$(@D) $<",
  tools = [
    "@protoc//:bin",
    "@protoc//:includes",
  ],
)

py_library(
  name = "metrics_pb2",
  srcs = [":metrics_proto_python"],
  deps = [
    requirement("protobuf"),
  ],
  visibility = ["//visibility:public"],
)
