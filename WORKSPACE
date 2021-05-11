load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# The sha256 hash is not obviously available on github. When upgrading a
# dependency to a newer version, you can regenerate the hash by downloading the
# new version and then running the command on the new file:
#   sha256sum [filename]
# e.g. for protobuf:
#   sha256sum v3.13.0.tar.gz

# Sept 2020, has Status and StatusOr
http_archive(
    name = "com_google_absl",
    sha256 = "b3744a4f7a249d5eaf2309daad597631ce77ea62e0fc6abffbab4b4c3dc0fc08",
    strip_prefix = "abseil-cpp-20200923",
    urls = [
        "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/20200923.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/20200923.tar.gz",
    ],
)

# ===== protobuf =====
# See https://bazel.build/blog/2017/02/27/protocol-buffers.html

http_archive(
    name = "com_google_protobuf",
    sha256 = "9b4ee22c250fe31b16f1a24d61467e40780a3fbb9b91c3b65be2a376ed913a1a",
    strip_prefix = "protobuf-3.13.0",
    urls = [
        "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# ===== googletest =====

http_archive(
    name = "com_google_googletest",
    sha256 = "774f5499dee0f9d2b583ce7fff62e575acce432b67a8396e86f3a87c90d2987e",
    strip_prefix = "googletest-604ba376c3a407c6a40e39fbd0d5055c545f9898",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/604ba376c3a407c6a40e39fbd0d5055c545f9898.tar.gz",
        "https://github.com/google/googletest/archive/604ba376c3a407c6a40e39fbd0d5055c545f9898.tar.gz",
    ],
)

# ========== re2 =======================
# This is a dependency of open_source/protocol_buffer_matchers.cc

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "0915741f524ad87debb9eb0429fe6016772a1569e21dc6d492039562308fcb0f",
    strip_prefix = "re2-2020-10-01",
    urls = ["https://github.com/google/re2/archive/2020-10-01.tar.gz"],
)


# ============== or-tools ==============

# April 2021
http_archive(
    name = "com_google_ortools",  # Apache 2.0
    sha256 = "fa7700b614ea2a5b2b6e37b76874bd2c3f04a80f03cbbf7871a2d2d5cd3a6091",
    strip_prefix = "or-tools-9.0",
    urls = [
        "https://mirror.bazel.build/github.com/google/or-tools/archive/v9.0.tar.gz",
        "https://github.com/google/or-tools/archive/v9.0.tar.gz",
    ],
)

http_archive(
    name = "bliss",
    build_file = "@com_google_ortools//bazel:bliss.BUILD",
    patches = ["@com_google_ortools//bazel:bliss-0.73.patch"],
    sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
    url = "http://www.tcs.hut.fi/Software/bliss/bliss-0.73.zip",
)

http_archive(
    name = "scip",
    build_file = "@com_google_ortools//bazel:scip.BUILD",
    patches = ["@com_google_ortools//bazel:scip.patch"],
    sha256 = "033bf240298d3a1c92e8ddb7b452190e0af15df2dad7d24d0572f10ae8eec5aa",
    url = "https://github.com/google/or-tools/releases/download/v7.7/scip-7.0.1.tgz",
)
