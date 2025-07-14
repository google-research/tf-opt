load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//third_party/tf_opt:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "inequality_checker",
    srcs = ["inequality_checker.cc"],
    hdrs = ["inequality_checker.h"],
    deps = [
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/linear_solver",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
    ],
)

cc_test(
    name = "inequality_checker_test",
    srcs = ["inequality_checker_test.cc"],
    deps = [
        ":inequality_checker",
        "@com_google_googletest//:gtest_main",
        "@com_google_ortools//ortools/linear_solver",
    ],
)
