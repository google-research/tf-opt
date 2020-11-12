// Copyright 2020 The tf.opt Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tf_opt/open_source/status_macros.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/open_source/status_matchers.h"

namespace tf_opt {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    ::absl::StatusCode::kInvalidArgument;

absl::Status MaybeError(bool is_error) {
  if (is_error) {
    return absl::InvalidArgumentError("Bad argument");
  }
  return absl::OkStatus();
}

absl::Status MaybePropagateError(bool is_error) {
  TFOPT_RETURN_IF_ERROR(MaybeError(is_error));
  return absl::OkStatus();
}

absl::Status MaybePropagateErrorAndAppend(bool is_error) {
  TFOPT_RETURN_IF_ERROR(MaybeError(is_error)) << " Error context";
  return absl::OkStatus();
}

TEST(ExpectOkTest, NoError) { TFOPT_EXPECT_OK(MaybePropagateError(false)); }

TEST(StatusIsTest, Error) {
  EXPECT_THAT(MaybePropagateError(true),
              StatusIs(kInvalidArgument, "Bad argument"));
}

TEST(ExpectOkTest, NoErrorAppend) {
  TFOPT_EXPECT_OK(MaybePropagateErrorAndAppend(false));
}

TEST(StatusIsTest, ErrorAppend) {
  EXPECT_THAT(MaybePropagateErrorAndAppend(true),
              StatusIs(kInvalidArgument, AllOf(HasSubstr("Bad argument"),
                                               HasSubstr("Error context"))));
}

TEST(CheckOkTest, NoError) { TFOPT_CHECK_OK(absl::OkStatus()); }
TEST(QCheckOkTest, NoError) { TFOPT_CHECK_OK(absl::OkStatus()); }
TEST(DCheckOkTest, NoError) { TFOPT_DCHECK_OK(absl::OkStatus()); }

TEST(CheckOkTestDeathTest, NotOk) {
  EXPECT_DEATH(TFOPT_CHECK_OK(absl::InvalidArgumentError("bad arg")),
               HasSubstr("bad arg"));
}
TEST(QCheckOkTestDeathTest, NotOk) {
  EXPECT_DEATH(TFOPT_CHECK_OK(absl::InvalidArgumentError("bad arg")),
               HasSubstr("bad arg"));
}
#ifndef NDEBUG
TEST(DCheckOkTestDeathTest, NotOk) {
  EXPECT_DEATH(TFOPT_DCHECK_OK(absl::InvalidArgumentError("bad arg")),
               HasSubstr("bad arg"));
}
#else
TEST(DCheckOkTest, NotOkNoCrash) {
  TFOPT_DCHECK_OK(absl::InvalidArgumentError("bad arg"));
}
#endif

}  // namespace
}  // namespace tf_opt
