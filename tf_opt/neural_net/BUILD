load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(
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
