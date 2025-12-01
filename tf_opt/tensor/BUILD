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

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        ":shape",
        ":tensor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    deps = [
        ":concat",
        ":shape",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

py_proto_library(
    name = "tensor_py_pb2",
    srcs = ["tensor.proto"],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        ":tensor_cc_proto",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":shape",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/open_source:protocol_buffer_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":shape",
        "//tf_opt/bounds",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_ortools//ortools/base",
    ],
)

cc_library(
    name = "tensor_testing",
    testonly = 1,
    srcs = ["tensor_testing.cc"],
    hdrs = ["tensor_testing.h"],
    deps = [
        ":tensor",
        "//tf_opt/bounds:bounds_testing",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tensor_testing_test",
    srcs = ["tensor_testing_test.cc"],
    deps = [
        ":tensor",
        ":tensor_testing",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":shape",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "batch_iterator",
    hdrs = ["batch_iterator.h"],
    deps = [
        ":tensor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "batch_iterator_test",
    srcs = ["batch_iterator_test.cc"],
    deps = [
        ":batch_iterator",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/open_source:status_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "embedding_lookup",
    srcs = ["embedding_lookup.cc"],
    hdrs = ["embedding_lookup.h"],
    deps = [
        ":shape",
        ":tensor",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "embedding_lookup_test",
    srcs = ["embedding_lookup_test.cc"],
    deps = [
        ":embedding_lookup",
        ":shape",
        ":tensor_testing",
        "//tf_opt/bounds",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convolve",
    srcs = ["convolve.cc"],
    hdrs = ["convolve.h"],
    deps = [
        ":shape",
        ":tensor",
        ":window",
        "//tf_opt/open_source:status_macros",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "convolve_test",
    srcs = ["convolve_test.cc"],
    deps = [
        ":convolve",
        ":shape",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/open_source:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "window",
    srcs = ["window.cc"],
    hdrs = ["window.h"],
    deps = [
        "//tf_opt/open_source:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "window_test",
    srcs = ["window_test.cc"],
    deps = [
        ":window",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "element_operations",
    hdrs = ["element_operations.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "element_operations_test",
    srcs = ["element_operations_test.cc"],
    deps = [
        ":element_operations",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "math_impl",
    srcs = ["math_impl.cc"],
    hdrs = ["math_impl.h"],
    deps = [
        ":element_operations",
        ":shape",
        ":tensor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_library(
    name = "math",
    srcs = ["math.cc"],
    hdrs = ["math.h"],
    deps = [
        ":element_operations",
        ":math_impl",
        ":shape",
        ":tensor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "math_test",
    srcs = ["math_test.cc"],
    deps = [
        ":math",
        ":shape",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reduce",
    srcs = ["reduce.cc"],
    hdrs = ["reduce.h"],
    deps = [
        ":element_operations",
        ":shape",
        ":tensor",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    deps = [
        ":reduce",
        ":shape",
        ":tensor",
        ":tensor_testing",
        "//tf_opt/bounds",
        "//tf_opt/open_source:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        ":element_operations",
        ":shape",
        ":tensor",
        ":window",
        "//tf_opt/open_source:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    deps = [
        ":pooling",
        ":shape",
        ":tensor",
        ":tensor_testing",
        ":window",
        "//tf_opt/open_source:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
