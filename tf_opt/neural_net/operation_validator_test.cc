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

#include "tf_opt/neural_net/operation_validator.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "tf_opt/open_source/status_matchers.h"
#include "ortools/base/map_util.h"

namespace tf_opt {
namespace {

using ::testing::HasSubstr;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

TEST(OperationValidatorTest, OperationValidationError) {
  OperationValidator validator("OpType", "TestOp");
  EXPECT_THAT(
      validator.OperationValidationError("Message"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Failed to validate operation TestOp of type OpType: Message"));
}

TEST(OperationValidatorTest, OperationValidationDoubleOption) {
  OperationValidator validator("OpType", "TestOp");
  Operation::Options options;
  ::gtl::InsertOrDie(&options.double_options, "OptionName", 10.0);
  EXPECT_THAT(validator.DoubleOption(options, "OptionName"),
              IsOkAndHolds(10.0));
  EXPECT_THAT(
      validator.DoubleOption(options, "InvalidOption"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Required double option not found: InvalidOption")));
}

TEST(OperationValidatorTest, OperationValidationIntegerOption) {
  OperationValidator validator("OpType", "TestOp");
  Operation::Options options;
  ::gtl::InsertOrDie(&options.integer_options, "OptionName", 8);
  EXPECT_THAT(validator.IntegerOption(options, "OptionName"), IsOkAndHolds(8));
  EXPECT_THAT(
      validator.IntegerOption(options, "InvalidOption"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Required integer option not found: InvalidOption")));
}

TEST(OperationValidatorTest, OperationValidationStringOption) {
  OperationValidator validator("OpType", "TestOp");
  Operation::Options options;
  ::gtl::InsertOrDie(&options.string_options, "OptionName", "Value");
  EXPECT_THAT(validator.StringOption(options, "OptionName"),
              IsOkAndHolds("Value"));
  EXPECT_THAT(
      validator.StringOption(options, "InvalidOption"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Required string option not found: InvalidOption")));
}

TEST(OperationValidatorTest, OperationValidationIntegerListOption) {
  OperationValidator validator("OpType", "TestOp");
  Operation::Options options;
  std::vector<int64_t> integer_list({1, 2, 3});
  ::gtl::InsertOrDie(&options.integer_list_options, "OptionName", integer_list);
  EXPECT_THAT(validator.IntegerListOption(options, "OptionName"),
              IsOkAndHolds(integer_list));
  EXPECT_THAT(
      validator.IntegerListOption(options, "InvalidOption"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Required integer list option not found: InvalidOption")));
}

TEST(OperationValidatorTest, OperationValidationExpectOptionsSizeAtMost) {
  OperationValidator validator("OpType", "TestOp");
  TFOPT_EXPECT_OK(validator.ExpectOptionsSizeAtMost(1, 2));
  EXPECT_THAT(
      validator.ExpectOptionsSizeAtMost(2, 1),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected number of options at most 1, found: 2")));
}

TEST(OperationValidatorTest, OperationValidationExpectInputSizeAtMost) {
  OperationValidator validator("OpType", "TestOp");
  TFOPT_EXPECT_OK(validator.ExpectInputSizeAtMost(1, 2));
  EXPECT_THAT(
      validator.ExpectInputSizeAtMost(2, 1),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected number of inputs at most 1, found: 2")));
}

TEST(OperationValidatorTest, OperationValidationExpectInputSizeAtLeast) {
  OperationValidator validator("OpType", "TestOp");
  TFOPT_EXPECT_OK(validator.ExpectInputSizeAtLeast(2, 1));
  EXPECT_THAT(
      validator.ExpectInputSizeAtLeast(1, 2),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected number of inputs at least 2, found: 1")));
}

TEST(OperationValidatorTest, OperationValidationExpectInputSizeEquals) {
  OperationValidator validator("OpType", "TestOp");
  TFOPT_EXPECT_OK(validator.ExpectInputSizeEquals(2, 2));
  EXPECT_THAT(
      validator.ExpectInputSizeEquals(1, 2),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected number of inputs equals to 2, found: 1")));
}

TEST(OperationValidatorTest, OperationValidationExpectOutputShapeEquals) {
  OperationValidator validator("OpType", "TestOp");
  TFOPT_EXPECT_OK(
      validator.ExpectOutputShapeEquals(Shape({1, 2, 3}), Shape({1, 2, 3})));
  EXPECT_THAT(validator.ExpectOutputShapeEquals(Shape({1}), Shape({1, 2})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected output shape: 1,2, found: 1")));
}

}  // namespace
}  // namespace tf_opt
