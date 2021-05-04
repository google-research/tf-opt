package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "bounds",
    hdrs = ["bounds.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "bounds_test",
    srcs = ["bounds_test.cc"],
    deps = [
        ":bounds",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounds_testing",
    testonly = 1,
    srcs = ["bounds_testing.cc"],
    hdrs = ["bounds_testing.h"],
    deps = [
        ":bounds",
        "@com_github_glog_glog//:glog",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bounds_testing_test",
    srcs = ["bounds_testing_test.cc"],
    deps = [
        ":bounds",
        ":bounds_testing",
        "@com_google_googletest//:gtest_main",
    ],
)
