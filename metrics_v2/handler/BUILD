package(
  default_visibility = ["//:__subpackages__"],
)
load("@handler_deps//:requirements.bzl", "requirement")
load("@subpar//:subpar.bzl", "par_binary")
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_push",
)
load("@rules_pkg//:pkg.bzl", "pkg_tar")

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

# Rules to build and deploy Google Cloud Function bundle.
filegroup(
  name = "gcf_srcs",
  srcs = glob(["*.py"], exclude=["*_test.py", "main.py"]),
  data = ["//handler/collectors:gcf_srcs"],
)

pkg_tar(
  name = "gcf_srcs_layer",
  srcs = [
    "//handler:gcf_srcs",
  ],
  # Keep the file structure of these sources.
  strip_prefix = "/",
  include_runfiles = True,
)

pkg_tar(
  name = "gcf_bundle",
  # These files will be included in the root directory.
  srcs = [
    ":main_lib",
    ":requirements.txt",
    "//:metrics_pb2",
  ],
  deps = [
    ":gcf_srcs_layer",
  ]
)

sh_binary(
  name = "deploy",
  srcs = ["deploy.sh"],
  data = ["gcf_bundle.tar"],
)
