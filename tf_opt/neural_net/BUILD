load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("//third_party/protobuf/bazel:proto_library.bzl", "proto_library")

package(
    default_applicable_licenses = ["//third_party/tf_opt:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

proto_library(
    name = "neural_net_proto",
    srcs = ["neural_net.proto"],
)

cc_proto_library(
    name = "neural_net_cc_proto",
    deps = [":neural_net_proto"],
)

py_proto_library(
    name = "neural_net_py_pb2",
    srcs = ["neural_net.proto"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "operation",
    srcs = ["operation.cc"],
    hdrs = ["operation.h"],
    deps = [
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_library(
    name = "operation_validator",
    srcs = ["operation_validator.cc"],
    hdrs = [
        "operation_validator.h",
    ],
    deps = [
        ":operation",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_test(
    name = "operation_validator_test",
    srcs = ["operation_validator_test.cc"],
    deps = [
        ":operation_validator",
        "//tf_opt/open_source:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_library(
    name = "operation_testing",
    testonly = 1,
    srcs = ["operation_testing.cc"],
    hdrs = [
        "operation_testing.h",
    ],
    deps = [
        ":operation",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_ortools//ortools/base:container_logging",
    ],
)

cc_library(
    name = "operation_visitor",
    hdrs = ["operation_visitor.h"],
    deps = [
        "//tf_opt/neural_net/ops:operation_types",
    ],
)

cc_library(
    name = "operation_evaluator",
    srcs = ["operation_evaluator.cc"],
    hdrs = [
        "operation_evaluator.h",
    ],
    deps = [
        ":operation",
        ":operation_visitor",
        "//tf_opt/neural_net/ops:all_operations",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_registry",
    srcs = ["op_registry.cc"],
    hdrs = ["op_registry.h"],
    deps = [
        ":operation",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/neural_net/ops:all_operations",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "op_registry_test",
    size = "small",
    srcs = ["op_registry_test.cc"],
    deps = [
        ":op_registry",
        ":operation",
        ":operation_testing",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
