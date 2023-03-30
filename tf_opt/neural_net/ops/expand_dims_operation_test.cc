// Copyright 2023 The tf.opt Authors.
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

#include "tf_opt/neural_net/ops/expand_dims_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(ExpandDimsOperationTest, SimpleCreate) {
  const Shape input_shape({2, 4});
  const int axis = 1;
  const Shape output_shape({2, 1, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ExpandDimsOperation::Create("e1", input_shape, axis));
  EXPECT_THAT(op, OperationArgsAre("e1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axis(), axis);
}

TEST(ExpandDimsOperationTest, SimpleCreateBadAxis) {
  const Shape input_shape({2, 4});
  const int axis = 4;
  EXPECT_THAT(ExpandDimsOperation::Create("e1", input_shape, axis),
              StatusIs(kInvalidArgument));
}

Operation::Options MakeOptions(const int axis) {
  Operation::Options options;
  options.integer_options[ExpandDimsOperation::kOptionsAxisKey] = axis;
  return options;
}

TEST(ExpandDimsOperationTest, GenericCreate) {
  const Shape input_shape({2, 4});
  const int axis = 1;
  const Shape output_shape({2, 1, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ExpandDimsOperation::GenericCreate(
                         "e1", {input_shape}, output_shape, MakeOptions(axis)));
  EXPECT_THAT(op, OperationArgsAre("e1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axis(), axis);
}

TEST(ExpandDimsOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({2, 4});
  const int axis = 1;
  const Shape output_shape({2, 1, 4});
  EXPECT_THAT(
      ExpandDimsOperation::GenericCreate("e1", {input_shape, input_shape},
                                         output_shape, MakeOptions(axis)),
      StatusIs(kInvalidArgument));
}

TEST(ExpandDimsOperationTest, GenericCreateBadOption) {
  const Shape input_shape({2, 4});
  const int axis = 1;
  const Shape output_shape({2, 1, 4});
  auto options = MakeOptions(axis);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(ExpandDimsOperation::GenericCreate(
                  "e1", {input_shape, input_shape}, output_shape, options),
              StatusIs(kInvalidArgument));
}

TEST(ExpandDimsOperationTest, GenericCreateMissingAxis) {
  const Shape input_shape({2, 4});

  const Shape output_shape({2, 1, 4});
  EXPECT_THAT(ExpandDimsOperation::GenericCreate(
                  "e1", {input_shape}, output_shape, Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(ExpandDimsOperationTest, GenericCreateBadAxis) {
  const Shape input_shape({2, 4});
  const int axis = 4;
  const Shape output_shape({2, 1, 4});
  EXPECT_THAT(ExpandDimsOperation::GenericCreate(
                  "e1", {input_shape}, output_shape, MakeOptions(axis)),
              StatusIs(kInvalidArgument));
}

TEST(ExpandDimsOperationTest, GenericCreateBadOutputShape) {
  const Shape input_shape({2, 4});
  const int axis = 1;
  const Shape output_shape({1, 2, 4});
  EXPECT_THAT(ExpandDimsOperation::GenericCreate(
                  "e1", {input_shape}, output_shape, MakeOptions(axis)),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
