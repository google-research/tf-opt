// Copyright 2025 The tf.opt Authors.
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

#include "tf_opt/open_source/status_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "tf_opt/open_source/status_matchers.h"

namespace tf_opt {
namespace {

using ::tf_opt::testing::StatusIs;

TEST(StatusBuilderTest, All) {
  const absl::Status status = absl::InvalidArgumentError("bad arg");
  ASSERT_EQ(status.message(), "bad arg");
  ::tf_opt::StatusBuilder builder(status);
  const absl::Status result = builder << "testing " << 1 << 2 << 3;
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               "bad arg; testing 123"));
}

}  // namespace
}  // namespace tf_opt
