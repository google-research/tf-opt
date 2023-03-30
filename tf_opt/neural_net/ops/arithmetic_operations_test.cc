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

#include "tf_opt/neural_net/ops/arithmetic_operations.h"

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

template <typename OpType>
class BinaryArithmeticOperationTest : public ::testing::Test {};

using OpTestTypes = ::testing::Types<AddOperation, DivideOperation,
                                     MultiplyOperation, SubtractOperation>;

TYPED_TEST_SUITE(BinaryArithmeticOperationTest, OpTestTypes);

TYPED_TEST(BinaryArithmeticOperationTest, SimpleCreate) {
  const Shape left({2, 4});
  const Shape right({2, 1});
  const Shape expected_result({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(const TypeParam op,
                             TypeParam::Create("bin_op1", left, right));
  EXPECT_EQ(op.left(), left);
  EXPECT_EQ(op.right(), right);
  EXPECT_THAT(op, OperationArgsAre("bin_op1", {left, right}, expected_result));
}

TYPED_TEST(BinaryArithmeticOperationTest, SimpleCreateIncompatibleShapes) {
  const Shape left({2, 4});
  const Shape right({3, 4});
  EXPECT_THAT(TypeParam::Create("op1", left, right),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(BinaryArithmeticOperationTest, GenericCreate) {
  const Shape left({3, 1, 2});
  const Shape right({1, 5, 2});
  const Shape result({3, 5, 2});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const TypeParam op,
      TypeParam::GenericCreate("bin_op1", {left, right}, result,
                               Operation::Options()));
  EXPECT_EQ(op.left(), left);
  EXPECT_EQ(op.right(), right);
  EXPECT_THAT(op, OperationArgsAre("bin_op1", {left, right}, result));
}

TYPED_TEST(BinaryArithmeticOperationTest, GenericCreateWrongNumberInputs) {
  EXPECT_THAT(TypeParam::GenericCreate("bin_op1", {}, Shape({3, 2}),
                                       Operation::Options()),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(BinaryArithmeticOperationTest, GenericCreateBadOutputShape) {
  EXPECT_THAT(
      TypeParam::GenericCreate("bin_op1", {Shape({3, 2}), Shape({3, 2})},
                               Shape({4, 1}), Operation::Options()),
      StatusIs(kInvalidArgument));
}

TYPED_TEST(BinaryArithmeticOperationTest, GenericCreateBadExtraOption) {
  Operation::Options bad_option;
  bad_option.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(
      TypeParam::GenericCreate("bin_op1", {Shape({3, 2}), Shape({3, 2})},
                               Shape({3, 2}), bad_option),
      StatusIs(kInvalidArgument));
}

TYPED_TEST(BinaryArithmeticOperationTest,
           GenericCreateIncompatibleInputShapes) {
  EXPECT_THAT(
      TypeParam::GenericCreate("bin_op1", {Shape({3, 2}), Shape({4, 2})},
                               Shape({4, 2}), Operation::Options()),
      StatusIs(kInvalidArgument));
}

TYPED_TEST(BinaryArithmeticOperationTest, SerializeTest) {
  const Shape left({2, 4});
  const Shape right({2, 1});
  TFOPT_ASSERT_OK_AND_ASSIGN(const TypeParam op,
                             TypeParam::Create("bin_op1", left, right));
  const std::vector<std::string> inputs({"hello", "goodbye"});
  proto::TensorNode serialized = op.ToProto(inputs);
  EXPECT_EQ(serialized.name(), "bin_op1");
  EXPECT_EQ(serialized.input_names(0), "hello");
  EXPECT_EQ(serialized.input_names(1), "goodbye");
  EXPECT_EQ(serialized.out_dimension().dim_sizes(0), 2);
  EXPECT_EQ(serialized.out_dimension().dim_sizes(1), 4);
}

}  // namespace
}  // namespace tf_opt
