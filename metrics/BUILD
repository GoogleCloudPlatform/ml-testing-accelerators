load("@handler_deps//:requirements.bzl", handler_requirement="requirement")
load("@publisher_deps//:requirements.bzl", publisher_requirement="requirement")

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
    handler_requirement("protobuf"),
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "integration_test",
  srcs = ["integration_test.py"],
  deps = [
    ":metrics_pb2",
    "//handler:main_lib",
    "//handler:bigquery_client",
    "//publisher:event_publisher_lib",
    handler_requirement("absl-py"),
    handler_requirement("tensorflow-cpu"),
    publisher_requirement("kubernetes"),
  ]
)
