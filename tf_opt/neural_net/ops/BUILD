package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])

# A forwarding library for all the operations.
cc_library(
    name = "all_operations",
    hdrs = ["all_operations.h"],
    deps = [
        ":arithmetic_operations",
        ":clipped_relu_operation",
        ":concat_operation",
        ":constant_operation",
        ":conv1d_operation",
        ":conv2d_operation",
        ":embedding_lookup_operation",
        ":expand_dims_operation",
        ":matmul_operation",
        ":maxpool_operation",
        ":reduce_operations",
        ":relu_operation",
        ":reshape_operation",
        ":slice_operation",
        ":squeeze_operation",
        ":variable_operation",
    ],
)

cc_library(
    name = "operation_types",
    hdrs = ["operation_types.h"],
)

# Keep build targets ordered alphabetically!

cc_library(
    name = "arithmetic_operations",
    hdrs = ["arithmetic_operations.h"],
    deps = [
        ":operation_types",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:math",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "arithmetic_operations_test",
    srcs = ["arithmetic_operations_test.cc"],
    deps = [
        ":arithmetic_operations",
        ":constant_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "clipped_relu_operation",
    srcs = ["clipped_relu_operation.cc"],
    hdrs = ["clipped_relu_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/neural_net/neuron:clipped_relu_impl_type",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_test(
    name = "clipped_relu_operation_test",
    srcs = ["clipped_relu_operation_test.cc"],
    deps = [
        ":clipped_relu_operation",
        ":constant_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/neural_net/neuron:clipped_relu_impl_type",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "concat_operation",
    srcs = ["concat_operation.cc"],
    hdrs = ["concat_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:concat",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "concat_operation_test",
    srcs = ["concat_operation_test.cc"],
    deps = [
        ":concat_operation",
        ":constant_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "constant_operation",
    srcs = ["constant_operation.cc"],
    hdrs = ["constant_operation.h"],
    deps = [
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "constant_operation_test",
    srcs = ["constant_operation_test.cc"],
    deps = [
        ":constant_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor:shape",
        "//tf_opt/tensor:tensor_testing",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv1d_operation",
    srcs = ["conv1d_operation.cc"],
    hdrs = ["conv1d_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:convolve",
        "//tf_opt/tensor:shape",
        "//tf_opt/tensor:window",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "conv1d_operation_test",
    srcs = ["conv1d_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":conv1d_operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:convolve",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv2d_operation",
    srcs = ["conv2d_operation.cc"],
    hdrs = ["conv2d_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:convolve",
        "//tf_opt/tensor:shape",
        "//tf_opt/tensor:window",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "conv2d_operation_test",
    srcs = ["conv2d_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":conv2d_operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:convolve",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "embedding_lookup_operation",
    srcs = ["embedding_lookup_operation.cc"],
    hdrs = ["embedding_lookup_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:embedding_lookup",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "embedding_lookup_operation_test",
    srcs = ["embedding_lookup_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":embedding_lookup_operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "expand_dims_operation",
    srcs = ["expand_dims_operation.cc"],
    hdrs = ["expand_dims_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "expand_dims_operation_test",
    srcs = ["expand_dims_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":expand_dims_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matmul_operation",
    srcs = ["matmul_operation.cc"],
    hdrs = ["matmul_operation.h"],
    deps = [
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:math",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "matmul_operation_test",
    srcs = ["matmul_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":matmul_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "maxpool_operation",
    srcs = ["maxpool_operation.cc"],
    hdrs = ["maxpool_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/neural_net/neuron:maximum_impl_type",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:pooling",
        "//tf_opt/tensor:shape",
        "//tf_opt/tensor:window",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_test(
    name = "maxpool_operation_test",
    srcs = ["maxpool_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":maxpool_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "//tf_opt/tensor:window",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reduce_operations",
    srcs = ["reduce_operations.cc"],
    hdrs = ["reduce_operations.h"],
    deps = [
        ":operation_types",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/neural_net/neuron:maximum_impl_type",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:reduce",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "reduce_operations_test",
    srcs = ["reduce_operations_test.cc"],
    deps = [
        ":constant_operation",
        ":reduce_operations",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "relu_operation",
    srcs = ["relu_operation.cc"],
    hdrs = ["relu_operation.h"],
    deps = [
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/neural_net/neuron:relu_impl_type",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_test(
    name = "relu_operation_test",
    srcs = ["relu_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":relu_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/neural_net/neuron:relu_impl_type",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reshape_operation",
    srcs = ["reshape_operation.cc"],
    hdrs = ["reshape_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "reshape_operation_test",
    srcs = ["reshape_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":reshape_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "slice_operation",
    srcs = ["slice_operation.cc"],
    hdrs = ["slice_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "slice_operation_test",
    srcs = ["slice_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":slice_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "squeeze_operation",
    srcs = ["squeeze_operation.cc"],
    hdrs = ["squeeze_operation.h"],
    deps = [
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/base:map_util",
    ],
)

cc_test(
    name = "squeeze_operation_test",
    srcs = ["squeeze_operation_test.cc"],
    deps = [
        ":constant_operation",
        ":squeeze_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "variable_operation",
    srcs = ["variable_operation.cc"],
    hdrs = ["variable_operation.h"],
    deps = [
        "//tf_opt/neural_net:neural_net_cc_proto",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_validator",
        "//tf_opt/neural_net:operation_visitor",
        "//tf_opt/open_source:status_macros",
        "//tf_opt/tensor:shape",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/base",
    ],
)

cc_test(
    name = "variable_operation_test",
    srcs = ["variable_operation_test.cc"],
    deps = [
        ":variable_operation",
        "//tf_opt/neural_net:operation",
        "//tf_opt/neural_net:operation_testing",
        "//tf_opt/open_source:status_matchers",
        "//tf_opt/tensor:shape",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
