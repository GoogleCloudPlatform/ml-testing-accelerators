load("@publisher_deps//:requirements.bzl", "requirement")
load("@subpar//:subpar.bzl", "par_binary")
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_push",
)

alias(
  name = "publisher",
  actual = "event_publisher",
)

par_binary(
  name = "event_publisher",
  main = "event_publisher.py",
  srcs = [
    "event_publisher.py"
  ],
  deps = [":event_publisher_lib"],
  zip_safe = False,
)

container_image(
  name = "image",
  base = "@python_base//image",
  files = ["event_publisher.par"],
  entrypoint = ["/event_publisher.par"],
)

container_push(
  name = "image_push",
  format = "Docker",
  image = ":image",
  registry = "gcr.io",
  repository = "$(project)/event-publisher",
  tag = "{BUILD_USER}_{BUILD_TIMESTAMP}",
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
