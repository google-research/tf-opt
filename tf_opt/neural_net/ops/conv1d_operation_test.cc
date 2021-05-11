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

#include "tf_opt/neural_net/ops/conv1d_operation.h"

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

TEST(Conv1dOperationTest, SimpleCreate) {
  const Shape input({1, 4, 1});
  const Shape filter({2, 1, 10});
  const int stride = 1;
  const PaddingType padding = PaddingType::SAME;
  const Shape expected_result({1, 4, 10});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      Conv1dOperation::Create("conv1d", input, filter, stride, padding));
  EXPECT_THAT(op, OperationArgsAre("conv1d", {input, filter}, expected_result));
  EXPECT_EQ(op.input_value(), input);
  EXPECT_EQ(op.filter(), filter);
  EXPECT_EQ(op.stride(), 1);
  EXPECT_EQ(op.padding(), padding);
}

TEST(Conv1dOperationTest, SimpleCreateBadShape) {
  const Shape input({1, 4, 1});
  const Shape filter({2, 6 /*bad, should be 1=# of input channels*/, 10});
  const int stride = 1;
  const PaddingType padding = PaddingType::SAME;
  EXPECT_THAT(Conv1dOperation::Create("conv1d", input, filter, stride, padding),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, CreateForGraph) {
  const ConstantOperation input("input", DoubleTensor(Shape({1, 4, 3})));
  const ConstantOperation filter("filter", DoubleTensor(Shape({2, 3, 5})));
  const int stride = 1;
  const PaddingType padding = PaddingType::SAME;
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op_args_pair,
                             Conv1dOperation::CreateForGraph(
                                 "conv1d", &input, &filter, stride, padding));
  EXPECT_THAT(op_args_pair.second, ElementsAre(&input, &filter));
  const Conv1dOperation& op = op_args_pair.first;
  EXPECT_THAT(op,
              OperationArgsAre("conv1d", {Shape({1, 4, 3}), Shape({2, 3, 5})},
                               Shape({1, 4, 5})));
  EXPECT_EQ(op.input_value(), Shape({1, 4, 3}));
  EXPECT_EQ(op.filter(), Shape({2, 3, 5}));
}

namespace {

Operation::Options MakeOptions(const int stride,
                               const PaddingType padding_type) {
  Operation::Options result;
  result.integer_options[Conv1dOperation::kOptionsStrideKey] = stride;
  result.string_options[Conv1dOperation::kOptionsPaddingKey] =
      ToString(padding_type);
  return result;
}

}  // namespace

TEST(Conv1dOperationTest, GenericCreate) {
  const Shape input({1, 4, 1});
  const Shape filter({2, 1, 10});
  const int stride = 1;
  const PaddingType padding = PaddingType::SAME;
  const Shape result({1, 4, 10});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      Conv1dOperation::GenericCreate("conv1d", {input, filter}, result,
                                     MakeOptions(stride, padding)));
  EXPECT_THAT(op, OperationArgsAre("conv1d", {input, filter}, result));
  EXPECT_EQ(op.input_value(), input);
  EXPECT_EQ(op.filter(), filter);
  EXPECT_EQ(op.stride(), 1);
  EXPECT_EQ(op.padding(), padding);
}

TEST(Conv1dOperationTest, GenericCreateWrongNumberInputs) {
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}) /* Missing filter*/},
                  Shape({1, 4, 10}), MakeOptions(1, PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, GenericCreateBadOption) {
  Operation::Options options = MakeOptions(1, PaddingType::SAME);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}), Shape({2, 1, 10})},
                  Shape({1, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, GenericCreateInvalidPaddingString) {
  Operation::Options options = MakeOptions(1, PaddingType::SAME);
  options.string_options[Conv1dOperation::kOptionsPaddingKey] =
      "bad_padding_value";
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}), Shape({2, 1, 10})},
                  Shape({1, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, GenericCreateNoStride) {
  Operation::Options options;
  options.integer_options["not_stride_key"] = 1;
  options.string_options[Conv1dOperation::kOptionsPaddingKey] =
      ToString(PaddingType::SAME);
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}), Shape({2, 1, 10})},
                  Shape({1, 4, 10}), options),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, GenericCreateIncompatibleShapes) {
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}), Shape({2, 2, 2})},
                  Shape({1, 4, 10}), MakeOptions(1, PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

TEST(Conv1dOperationTest, GenericCreateBadOutputShape) {
  EXPECT_THAT(Conv1dOperation::GenericCreate(
                  "conv1d", {Shape({1, 4, 1}), Shape({2, 1, 10})},
                  Shape({1, 4, 5}), MakeOptions(1, PaddingType::SAME)),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
