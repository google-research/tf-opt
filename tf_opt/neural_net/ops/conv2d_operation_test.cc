// Copyright 2024 The tf.opt Authors.
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

#include "tf_opt/neural_net/ops/conv2d_operation.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/convolve.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(Conv2dOperationTest, SimpleCreate) {
  const Shape input({1, 4, 4, 1});
  const Shape filter({2, 2, 1, 10});
  const Position2D stride(1, 1);
  const PaddingType padding = PaddingType::SAME;
  const Shape expected_result({1, 4, 4, 10});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      Conv2dOperation::Create("conv2d", input, filter, stride, padding));
  EXPECT_THAT(op, OperationArgsAre("conv2d", {input, filter}, expected_result));
  EXPECT_EQ(op.input_value(), input);
  EXPECT_EQ(op.filter(), filter);
  EXPECT_EQ(op.stride(), stride);
  EXPECT_EQ(op.padding(), padding);
}

TEST(Conv2dOperationTest, SimpleCreateBadShape) {
  const Shape input({1, 4, 4, 1});
  const Shape filter({2, 2, /*bad, should be #input channels=1*/ 6, 10});
  const Position2D stride(1, 1);
  const PaddingType padding = PaddingType::SAME;
  EXPECT_THAT(Conv2dOperation::Create("conv2d", input, filter, stride, padding),
              StatusIs(kInvalidArgument));
}

namespace {

Operation::Options MakeOptions(const Position2D stride,
                               const PaddingType padding_type) {
  Operation::Options result;
  result.integer_options[Conv2dOperation::kOptionsStrideRowKey] =
      static_cast<int>(stride.row);
  result.integer_options[Conv2dOperation::kOptionsStrideColKey] =
      static_cast<int>(stride.col);
  result.string_options[Conv2dOperation::kOptionsPaddingKey] =
      ToString(padding_type);
  return result;
}

}  // namespace

TEST(Conv2dOperationTest, GenericCreate) {
  const Shape input({1, 4, 4, 1});
  const Shape filter({2, 2, 1, 10});
  const Position2D stride(1, 1);
  const PaddingType padding = PaddingType::SAME;
  const Shape result({1, 4, 4, 10});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      Conv2dOperation::GenericCreate("conv2d", {input, filter}, result,
                                     MakeOptions(stride, padding)));
  EXPECT_THAT(op, OperationArgsAre("conv2d", {input, filter}, result));
  EXPECT_EQ(op.input_value(), input);
  EXPECT_EQ(op.filter(), filter);
  EXPECT_EQ(op.stride(), stride);
  EXPECT_EQ(op.padding(), padding);
}

TEST(Conv2dOperationTest, GenericCreateWrongNumberInputs) {
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}) /* Missing filter*/},
                  Shape({1, 4, 4, 10}),
                  MakeOptions(Position2D(1, 1), PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateBadOption) {
  Operation::Options options = MakeOptions(Position2D(1, 1), PaddingType::SAME);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 1, 10})},
                  Shape({1, 4, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateInvalidPaddingString) {
  Operation::Options options = MakeOptions(Position2D(1, 1), PaddingType::SAME);
  options.string_options[Conv2dOperation::kOptionsPaddingKey] =
      "bad_padding_value";
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 1, 10})},
                  Shape({1, 4, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateNoStrideRow) {
  Operation::Options options;
  options.integer_options["not_stride_row_key"] = 1;
  options.integer_options[Conv2dOperation::kOptionsStrideColKey] = 1;
  options.string_options[Conv2dOperation::kOptionsPaddingKey] =
      ToString(PaddingType::SAME);
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 1, 10})},
                  Shape({1, 4, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateNoStrideCol) {
  Operation::Options options;
  options.integer_options["not_stride_col_key"] = 1;
  options.integer_options[Conv2dOperation::kOptionsStrideRowKey] = 1;
  options.string_options[Conv2dOperation::kOptionsPaddingKey] =
      ToString(PaddingType::SAME);
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 1, 10})},
                  Shape({1, 4, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateIncompatibleShapes) {
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 2, 2})},
                  Shape({1, 4, 4, 10}),
                  MakeOptions(Position2D(1, 1), PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

TEST(Conv2dOperationTest, GenericCreateBadOutputShape) {
  EXPECT_THAT(Conv2dOperation::GenericCreate(
                  "conv2d", {Shape({1, 4, 4, 1}), Shape({2, 2, 1, 10})},
                  Shape({1, 4, 4, 5}),
                  MakeOptions(Position2D(1, 1), PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
