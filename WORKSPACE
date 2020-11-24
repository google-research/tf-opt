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

# ========== glog =========

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_glog_glog",
    sha256 = "62efeb57ff70db9ea2129a16d0f908941e355d09d6d83c9f7b18557c0a7ab59e",
    strip_prefix = "glog-d516278b1cd33cd148e8989aec488b6049a4ca0b",
    urls = ["https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip"],
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

# October 2020
http_archive(
    name = "com_google_ortools",  # Apache 2.0
    sha256 = "ac01d7ebde157daaeb0e21ce54923a48e4f1d21faebd0b08a54979f150f909ee",
    strip_prefix = "or-tools-8.0",
    urls = [
        "https://mirror.bazel.build/github.com/google/or-tools/archive/v8.0.tar.gz",
        "https://github.com/google/or-tools/archive/v8.0.tar.gz",
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
