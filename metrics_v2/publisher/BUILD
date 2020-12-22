load("@publisher_deps//:requirements.bzl", "requirement")
load("@subpar//:subpar.bzl", "par_binary")
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_push",
)

par_binary(
  name = "event_publisher_bin",
  main = "event_publisher.py",
  srcs = [
    "event_publisher.py"
  ],
  deps = [":event_publisher_lib"],
  zip_safe = False,
)

py_library(
  name = "event_publisher_lib",
  srcs = [
    "event_publisher.py"
  ],
  deps = [
    "//:metrics_pb2",
    requirement("absl-py"),
    requirement("google-cloud-pubsub"),
    requirement("kubernetes"),
    requirement("protobuf"),
  ],
  visibility = ["//:__subpackages__"],
)

py_test(
  name = "event_publisher_test",
  srcs = [
    "event_publisher_test.py",
  ],
  deps = [
    ":event_publisher_lib",
  ],
)

container_image(
  name = "event_publisher",
  base = "@python_base//image",
  files = [
    "event_publisher_bin.par",
  ],
  entrypoint = [
    "/event_publisher_bin.par",
  ],
  legacy_run_behavior = False,
)

container_push(
  name = "event_publisher_push",
  format = "Docker",
  image = ":event_publisher",
  registry = "gcr.io",
  repository = "$(project)/event-publisher",
  tag = "{BUILD_USER}",
)
