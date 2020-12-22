load("@publisher_deps//:requirements.bzl", "requirement")

'''
genrule(
  name = "metrics_proto_python",
  srcs = ["metrics.proto"],
  outs = ["metrics_pb2.py"],
  cmd = "$(execpath @protobuf//:protoc) --proto_path=external/protobuf/src -I . --python_out=$(@D) $<",
  tools = [
    "@protobuf//:protoc",
    "@protobuf//:well_known_protos",
  ],
)
'''

py_library(
  name = "metrics_pb2",
  srcs = ["metrics_pb2.py"],
  deps = [
    requirement("protobuf"),
  ],
  visibility = ["//visibility:public"],
)
