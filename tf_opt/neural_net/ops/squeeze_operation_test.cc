// Copyright 2021 The tf.opt Authors.
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

#include "tf_opt/neural_net/ops/squeeze_operation.h"

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

TEST(SqueezeOperationTest, SimpleCreateWithAxes) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const std::vector<int> axes({1, 2});
  const Shape output_shape({2, 4, 1});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op,
                             SqueezeOperation::Create("s1", input_shape, axes));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axes(), axes);
}

TEST(SqueezeOperationTest, SimpleCreateWithoutAxes) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const std::vector<int> axes;
  const Shape output_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op,
                             SqueezeOperation::Create("s1", input_shape, axes));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axes(), axes);
}

TEST(SqueezeOperationTest, SimpleCreateBadAxis) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const std::vector<int> axes({1, 2, 9});
  EXPECT_THAT(SqueezeOperation::Create("s1", input_shape, axes),
              StatusIs(kInvalidArgument));
}

Operation::Options MakeOptions(const std::vector<int64_t>& axes) {
  Operation::Options options;
  options.integer_list_options[SqueezeOperation::kOptionsAxesKey] = axes;
  return options;
}

TEST(SqueezeOperationTest, GenericCreate) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({2, 4, 1});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      SqueezeOperation::GenericCreate("s1", {input_shape}, output_shape,
                                      MakeOptions({1, 2})));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axes(), std::vector<int>({1, 2}));
}

TEST(SqueezeOperationTest, GenericCreateNoAxesSet) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      SqueezeOperation::GenericCreate("s1", {input_shape}, output_shape,
                                      Operation::Options()));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.axes(), std::vector<int>({}));
}

TEST(SqueezeOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({2, 4, 1});
  EXPECT_THAT(
      SqueezeOperation::GenericCreate("s1", {input_shape, input_shape},
                                      output_shape, MakeOptions({1, 2})),
      StatusIs(kInvalidArgument));
}

TEST(SqueezeOperationTest, GenericCreateBadOption) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({2, 4, 1});
  auto options = MakeOptions({1, 1});
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(SqueezeOperation::GenericCreate("s1", {input_shape}, output_shape,
                                              options),
              StatusIs(kInvalidArgument));
}

TEST(SqueezeOperationTest, GenericCreateBadAxis) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({2, 1, 4});
  EXPECT_THAT(SqueezeOperation::GenericCreate("s1", {input_shape}, output_shape,
                                              MakeOptions({1, 10})),
              StatusIs(kInvalidArgument));
}

TEST(SqueezeOperationTest, GenericCreateBadOutputShape) {
  const Shape input_shape({2, 1, 1, 4, 1});
  const Shape output_shape({1, 2, 4});
  EXPECT_THAT(SqueezeOperation::GenericCreate("s1", {input_shape}, output_shape,
                                              MakeOptions({1, 2})),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
