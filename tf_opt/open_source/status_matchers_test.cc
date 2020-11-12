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

#include "tf_opt/open_source/status_matchers.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace tf_opt {
namespace {

using ::testing::Not;
using ::tf_opt::testing::IsOk;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

TEST(StatusMacrosTest, IsOk) {
  const absl::Status status = absl::InvalidArgumentError("bad arg");
  EXPECT_THAT(status, Not(IsOk()));
  EXPECT_THAT(absl::OkStatus(), IsOk());
}

TEST(StatusMacrosTest, ExpectOk) { TFOPT_EXPECT_OK(absl::OkStatus()); }

TEST(StatusMacrosTest, AssertOk) { TFOPT_ASSERT_OK(absl::OkStatus()); }

TEST(StatusMacrosTest, AssertOkAndAssign) {
  absl::StatusOr<int> maybe_int_ok = 7;
  TFOPT_ASSERT_OK_AND_ASSIGN(int seven, maybe_int_ok);
  EXPECT_EQ(seven, 7);
}

TEST(StatusMacrosTest, StatusIs) {
  const absl::Status status = absl::InvalidArgumentError("bad arg");
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status, Not(StatusIs(absl::StatusCode::kInternal)));
}

TEST(StatusMacrosTest, StatusIsWithMessage) {
  const absl::Status status = absl::InvalidArgumentError("bad arg");
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument, "bad arg"));
  EXPECT_THAT(status,
              Not(StatusIs(absl::StatusCode::kInvalidArgument, "the cat")));
  EXPECT_THAT(status, Not(StatusIs(absl::StatusCode::kInternal, "bad arg")));
}

TEST(StatusMacrosTest, IsOkAndHolds) {
  absl::StatusOr<int> maybe_int_ok = 7;
  absl::StatusOr<int> maybe_int_bad = absl::InvalidArgumentError("bad arg");
  EXPECT_THAT(maybe_int_ok, IsOkAndHolds(7));
  EXPECT_THAT(maybe_int_bad, Not(IsOkAndHolds(7)));
}

}  // namespace
}  // namespace tf_opt
