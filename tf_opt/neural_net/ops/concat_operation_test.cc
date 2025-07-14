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

#include "tf_opt/neural_net/ops/concat_operation.h"

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

TEST(ConcatOperationTest, SimpleInitialize) {
  const Shape first({5});
  const Shape second({3});
  const Shape expected_result({8});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ConcatOperation::Create("concat1", {first, second}, 0));
  EXPECT_THAT(op,
              OperationArgsAre("concat1", {first, second}, expected_result));
  EXPECT_EQ(op.axis(), 0);
}

TEST(ConcatOperationTest, SimpleInitializeIncompatibleShapes) {
  const Shape first({5, 2});
  const Shape second({3});
  EXPECT_THAT(ConcatOperation::Create("concat1", {first, second}, 0),
              StatusIs(kInvalidArgument));
}

namespace {

Operation::Options MakeOptions(int axis) {
  Operation::Options result;
  result.integer_options[ConcatOperation::kOptionsAxisKey] = axis;
  return result;
}

}  // namespace

TEST(ConcatOperationTest, GenericInitialize) {
  const Shape first({5});
  const Shape second({3});
  const Shape result({8});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ConcatOperation::GenericCreate("concat1", {first, second},
                                                    result, MakeOptions(0)));
  EXPECT_THAT(op, OperationArgsAre("concat1", {first, second}, result));
  EXPECT_EQ(op.axis(), 0);
}

TEST(ConcatOperationTest, GenericInitializeNoAxis) {
  const Shape first({5});
  const Shape second({3});
  const Shape result({8});
  EXPECT_THAT(ConcatOperation::GenericCreate("concat1", {first, second}, result,
                                             Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(ConcatOperationTest, GenericInitializeBadOutputShape) {
  const Shape first({5});
  const Shape second({3});
  const Shape result({10});
  EXPECT_THAT(ConcatOperation::GenericCreate("concat1", {first, second}, result,
                                             MakeOptions(0)),
              StatusIs(kInvalidArgument));
}

TEST(ConcatOperationTest, GenericInitializeInvalidInputShapes) {
  const Shape first({5});
  const Shape second({3, 2});
  const Shape result({7});
  EXPECT_THAT(ConcatOperation::GenericCreate("concat1", {first, second}, result,
                                             MakeOptions(0)),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
