package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "relu_impl_type",
    srcs = ["relu_impl_type.cc"],
    hdrs = ["relu_impl_type.h"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "relu_impl_type_test",
    srcs = ["relu_impl_type_test.cc"],
    deps = [
        ":relu_impl_type",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "clipped_relu_impl_type",
    srcs = ["clipped_relu_impl_type.cc"],
    hdrs = ["clipped_relu_impl_type.h"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "clipped_relu_impl_type_test",
    srcs = ["clipped_relu_impl_type_test.cc"],
    deps = [
        ":clipped_relu_impl_type",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "maximum_impl_type",
    srcs = ["maximum_impl_type.cc"],
    hdrs = ["maximum_impl_type.h"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "maximum_impl_type_test",
    srcs = ["maximum_impl_type_test.cc"],
    deps = [
        ":maximum_impl_type",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
