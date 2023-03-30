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

#include "tf_opt/neural_net/ops/matmul_operation.h"

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

TEST(MatmulOperationTest, SimpleCreate) {
  const Shape left({2, 4});
  const Shape right({4, 3});
  const Shape expected_result({2, 3});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op,
                             MatmulOperation::Create("matmul1", left, right));
  EXPECT_EQ(op.left(), left);
  EXPECT_EQ(op.right(), right);
  EXPECT_THAT(op, OperationArgsAre("matmul1", {left, right}, expected_result));
}

TEST(MatmulOperationTest, SimpleInitializeIncompatibleShapes) {
  const Shape left({2, 4});
  const Shape right({3, 4});
  EXPECT_THAT(MatmulOperation::Create("matmul1", left, right),
              StatusIs(kInvalidArgument));
}

TEST(MatmulOperationTest, GenericCreate) {
  const Shape left({2, 4});
  const Shape right({4, 3});
  const Shape result({2, 3});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op, MatmulOperation::GenericCreate(
                                                "matmul1", {left, right},
                                                result, Operation::Options()));
  EXPECT_EQ(op.left(), left);
  EXPECT_EQ(op.right(), right);
  EXPECT_THAT(op, OperationArgsAre("matmul1", {left, right}, result));
}

TEST(MatmulOperationTest, GenericCreateWrongNumberInputs) {
  EXPECT_THAT(
      MatmulOperation::GenericCreate("matmul1", {Shape({3, 2})}, Shape({3, 2}),
                                     Operation::Options()),
      StatusIs(kInvalidArgument));
}

TEST(MatmulOperationTest, GenericCreateBadOutputShape) {
  const Shape left({2, 4});
  const Shape right({4, 3});
  const Shape result({3, 3});
  EXPECT_THAT(MatmulOperation::GenericCreate("matmul1", {left, right}, result,
                                             Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(MatmulOperationTest, GenericCreateBadExtraOption) {
  const Shape left({2, 4});
  const Shape right({4, 3});
  const Shape result({2, 3});
  Operation::Options bad_option;
  bad_option.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(MatmulOperation::GenericCreate("matmul1", {left, right}, result,
                                             bad_option),
              StatusIs(kInvalidArgument));
}

TEST(MatmulOperationTest, GenericCreateIncompatibleInputShapes) {
  const Shape left({2, 4});
  const Shape right({5, 3});
  const Shape result({2, 3});
  EXPECT_THAT(MatmulOperation::GenericCreate("matmul", {left, right}, result,
                                             Operation::Options()),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
