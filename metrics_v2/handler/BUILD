package(
  default_visibility = ["//:__subpackages__"],
)
load("@handler_deps//:requirements.bzl", "requirement")
load("@subpar//:subpar.bzl", "par_binary")

py_binary(
  name = "handler",
  main = "main.py",
  srcs = ["main.py"],
  deps = [
    ":main_lib",
  ]
)

py_library(
  name = "main_lib",
  srcs = ["main.py"],
  deps = [
    ":alerts",
    ":bigquery_client",
    ":utils",
    "//:metrics_pb2",
    "//handler/collectors",
    requirement("absl-py"),
    requirement("google-auth"),
    requirement("numpy"),
    requirement("sendgrid"),
  ]
)

py_library(
  name = "alerts",
  srcs = ["alerts.py"],
  deps = [
    requirement("jinja2"),
    requirement("sendgrid"),
    requirement("pytz"),
  ]
)

py_test(
  name = "alerts_test",
  srcs = ["alerts_test.py"],
  deps = [
    ":alerts",
    "//:metrics_pb2",
    requirement("absl-py"),
    requirement("pytz"),
  ],
)

py_library(
  name = "bigquery_client",
  srcs = ["bigquery_client.py"],
  deps = [
    requirement("absl-py"),
    requirement("google-cloud-bigquery"),
  ]
)

py_binary(
  name = "create_bq_tables",
  srcs = ["create_bq_tables.py"],
  deps = [
    ":bigquery_client",
    requirement("absl-py"),
  ]
)

py_library(
  name = "utils",
  srcs = ["utils.py"],
)

py_test(
  name = "utils_test",
  srcs = ["utils_test.py"],
  deps = [
    ":utils",
    requirement("absl-py"),
  ]
)
