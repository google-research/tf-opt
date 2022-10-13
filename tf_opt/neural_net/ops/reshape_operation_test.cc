// Copyright 2022 The tf.opt Authors.
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

#include "tf_opt/neural_net/ops/reshape_operation.h"

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

TEST(ReshapeOperationTest, SimpleCreate) {
  const Shape input_shape({3, 3});
  const Shape output_shape({1, 9, 1});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ReshapeOperation::Create("r1", input_shape, output_shape));
  EXPECT_THAT(op, OperationArgsAre("r1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
}

TEST(ReshapeOperationTest, SimpleCreateInvalidOutputShape) {
  const Shape input_shape({3, 3});
  const Shape output_shape({12});
  EXPECT_THAT(ReshapeOperation::Create("r1", input_shape, output_shape),
              StatusIs(kInvalidArgument));
}

TEST(ReshapeOperationTest, GenericCreate) {
  const Shape input_shape({3, 3});
  const Shape output_shape({1, 9, 1});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      ReshapeOperation::GenericCreate("r1", {input_shape}, output_shape,
                                      Operation::Options()));
  EXPECT_THAT(op, OperationArgsAre("r1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
}

TEST(ReshapeOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({3, 3});
  const Shape output_shape({1, 9, 1});
  EXPECT_THAT(
      ReshapeOperation::GenericCreate("r1", {input_shape, input_shape},
                                      output_shape, Operation::Options()),
      StatusIs(kInvalidArgument));
}

TEST(ReshapeOperationTest, GenericCreateBadOption) {
  const Shape input_shape({3, 3});
  const Shape output_shape({1, 9, 1});
  auto options = Operation::Options();
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(ReshapeOperation::GenericCreate("r1", {input_shape}, output_shape,
                                              options),
              StatusIs(kInvalidArgument));
}

TEST(ReshapeOperationTest, GenericCreateBadOutputShape) {
  const Shape input_shape({3, 3});
  const Shape output_shape({1, 12, 1});
  EXPECT_THAT(ReshapeOperation::GenericCreate("r1", {input_shape}, output_shape,
                                              Operation::Options()),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
