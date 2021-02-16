package(
  default_visibility = ["//handler:__subpackages__"]
)
load("@handler_deps//:requirements.bzl", "requirement")

py_library(
  name = "collectors",
  srcs = ["__init__.py"],
  deps = [
    ":literal_collector",
    ":perfzero_collector",
    ":tensorboard_collector",
  ]
)

py_library(
  name = "base",
  srcs = ["base.py"],
  deps = [
    "//:metrics_pb2",
    "//handler:utils",
    requirement("absl-py"),
    requirement("numpy"),
    requirement("protobuf"),
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "base_test",
  srcs = ["base_test.py"],
  deps = [
    ":base",
    "//handler:bigquery_client",
    "//handler:utils",
    requirement("absl-py"),
  ]
)

py_library(
  name = "literal_collector",
  srcs = ["literal_collector.py"],
  deps = [
    ":base",
  ]
)

py_test(
  name = "literal_collector_test",
  srcs = ["literal_collector_test.py"],
  deps = [
    ":base",
    ":literal_collector",
    "//handler:utils",
  ]
)

py_library(
  name = "perfzero_collector",
  srcs = ["perfzero_collector.py"],
  deps = [
    ":base",
    requirement("absl-py"),
    requirement("tensorflow-cpu"),
  ]
)

py_test(
  name = "perfzero_collector_test",
  srcs = ["perfzero_collector_test.py"],
  deps = [
    ":base",
    ":perfzero_collector",
    "//:metrics_pb2",
    "//handler:utils",
  ]
)

py_library(
  name = "tensorboard_collector",
  srcs = ["tensorboard_collector.py"],
  deps = [
    ":base",
    "//:metrics_pb2",
    requirement("absl-py"),
    requirement("numpy"),
    requirement("tensorflow-cpu"),
    requirement("tensorboard"),
  ]
)

py_test(
  name = "tensorboard_collector_test",
  srcs = ["tensorboard_collector_test.py"],
  deps = [
    ":base",
    ":tensorboard_collector",
    requirement("absl-py"),
    requirement("numpy"),
    requirement("tensorflow-cpu"),
  ]
)

# Export files used in Cloud Functions bundle
filegroup(
  name = "gcf_srcs",
  srcs = glob(["*.py"], exclude=["*_test.py"]),
)
