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

#include "tf_opt/tensor/math.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::testing::HasSubstr;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(TensorMathTest, BinaryOpOutputShapeSimple) {
  const Shape left({2, 3, 2});
  const Shape right({2, 3, 2});
  const Shape expected({2, 3, 2});
  EXPECT_THAT(BinaryOpOutputShape(left, right), IsOkAndHolds(expected));
}

TEST(TensorMathTest, BinaryOpOutputShapeExtendAndBroadcast) {
  const Shape left({2, 3, 2});
  const Shape right({3, 1});
  const Shape expected({2, 3, 2});
  EXPECT_THAT(BinaryOpOutputShape(left, right), IsOkAndHolds(expected));
}

TEST(TensorMathTest, BinaryOpOutputShapeScalar) {
  const Shape left({2, 3, 2});
  const Shape right;
  const Shape expected({2, 3, 2});
  EXPECT_THAT(BinaryOpOutputShape(left, right), IsOkAndHolds(expected));
}

TEST(TensorMathTest, BinaryOpOutputIncompatibleShapes) {
  const Shape left({2, 3});
  const Shape right({3, 3});
  EXPECT_THAT(BinaryOpOutputShape(left, right).status(),
              StatusIs(kInvalidArgument, HasSubstr("Incompatible shapes")));
}

TEST(TensorMathTest, MatMulOutputShapeSimple) {
  const Shape left({2, 3});
  const Shape right({3, 4});
  const Shape expected({2, 4});
  EXPECT_THAT(MatMulOutputShape(left, right), IsOkAndHolds(expected));
}

TEST(TensorMathTest, MatMulOutputShapeExtendAndBroadcast) {
  const Shape left({10, 2, 3});
  const Shape right({14, 1, 3, 4});
  const Shape expected({14, 10, 2, 4});
  EXPECT_THAT(MatMulOutputShape(left, right), IsOkAndHolds(expected));
}

TEST(TensorMathTest, MatMulOutputShapeIncompatibleShapesFinalTwo) {
  const Shape left({2, 3});
  const Shape right({2, 3});
  EXPECT_THAT(MatMulOutputShape(left, right).status(),
              StatusIs(kInvalidArgument, HasSubstr("Incompatible shapes")));
}

TEST(TensorMathTest, MatMulOutputShapeIncompatibleShapesUpperLevels) {
  const Shape left({4, 10, 2, 3});
  const Shape right({10, 4, 3, 2});
  EXPECT_THAT(MatMulOutputShape(left, right).status(),
              StatusIs(kInvalidArgument, HasSubstr("Incompatible shapes")));
}

TEST(TensorMathTest, ElementwiseNegate) {
  const DoubleTensor t({{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}});
  const DoubleTensor expected({{-1.0, 2.0, -3.0}, {4.0, -5.0, 6.0}});
  EXPECT_THAT(ElementwiseNegate(t), DoubleTensorNear(expected));
}

TEST(TensorMathTest, ElementwiseRelu) {
  const DoubleTensor t({{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}});
  const DoubleTensor expected({{1.0, 0.0, 3.0}, {0.0, 5.0, 0.0}});
  EXPECT_THAT(ElementwiseRelu(t), DoubleTensorNear(expected));
}

TEST(TensorMathTest, ElementwiseClippedRelu) {
  const DoubleTensor t({{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}});
  const DoubleTensor expected({{1.0, 0.0, 3.0}, {0.0, 4.5, 0.0}});
  EXPECT_THAT(ElementwiseClippedRelu(t, 4.5), DoubleTensorNear(expected));
}

void ExpectSum(const DoubleTensor& t1, const DoubleTensor& t2,
               const DoubleTensor& expected_sum) {
  EXPECT_THAT(Add(t1, t2), DoubleTensorNear(expected_sum)) << "Testing t1 + t2";
  EXPECT_THAT(Add(t2, t1), DoubleTensorNear(expected_sum)) << "Testing t2 + t1";
}

TEST(TensorMathTest, BasicAdd) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  const DoubleTensor expected_sum({{11.0, 22.0, 33.0}, {44.0, 55.0, 66.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathTest, BasicSubtract) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  const DoubleTensor expected_diff(
      {{-9.0, -18.0, -27.0}, {-36.0, -45.0, -54.0}});
  EXPECT_THAT(Subtract(t1, t2), DoubleTensorNear(expected_diff));
}

TEST(TensorMathTest, BasicMultiply) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  const DoubleTensor expected_prod({{10.0, 40.0, 90.0}, {160.0, 250.0, 360.0}});
  EXPECT_THAT(Multiply(t1, t2), DoubleTensorNear(expected_prod));
  EXPECT_THAT(Multiply(t2, t1), DoubleTensorNear(expected_prod));
}

TEST(TensorMathTest, BasicDivide) {
  const DoubleTensor t1({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  const DoubleTensor t2({{1.0, 2.0, 1.5}, {2.0, 10.0, -6.0}});
  const DoubleTensor expected_quotient(
      {{10.0, 10.0, 20.0}, {20.0, 5.0, -10.0}});
  EXPECT_THAT(Divide(t1, t2), DoubleTensorNear(expected_quotient));
}

TEST(TensorMathTest, BasicMaximum) {
  const DoubleTensor t1({{1.0, -2.0, 3.0}, {-4.0, 50.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0, -30.0}, {-40.0, 5.0, 60.0}});
  const DoubleTensor expected_max({{10.0, 20.0, 3.0}, {-4.0, 50.0, 60.0}});
  EXPECT_THAT(ElementwiseMaximum(t1, t2), DoubleTensorNear(expected_max));
  EXPECT_THAT(ElementwiseMaximum(t2, t1), DoubleTensorNear(expected_max));
}

TEST(TensorMathTest, BasicMinimum) {
  const DoubleTensor t1({{1.0, -2.0, 3.0}, {-4.0, 50.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0, -30.0}, {-40.0, 5.0, 60.0}});
  const DoubleTensor expected_minimum({{1.0, -2.0, -30.0}, {-40.0, 5.0, 6.0}});
  EXPECT_THAT(ElementwiseMinimum(t1, t2), DoubleTensorNear(expected_minimum));
  EXPECT_THAT(ElementwiseMinimum(t2, t1), DoubleTensorNear(expected_minimum));
}

TEST(TensorMathTest, BroadcastAdd) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(10.0);
  const DoubleTensor expected_sum({{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathTest, BroadcastSubtract) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(10.0);
  const DoubleTensor expected_diff({{-9.0, -8.0, -7.0}, {-6.0, -5.0, -4.0}});
  EXPECT_THAT(Subtract(t1, t2), DoubleTensorNear(expected_diff));
  const DoubleTensor expected_diff_reverse({{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}});
  EXPECT_THAT(Subtract(t2, t1), DoubleTensorNear(expected_diff_reverse));
}

TEST(TensorMathTest, BroadcastMultiply) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(10.0);
  const DoubleTensor expected_prod({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  EXPECT_THAT(Multiply(t1, t2), DoubleTensorNear(expected_prod));
  EXPECT_THAT(Multiply(t2, t1), DoubleTensorNear(expected_prod));
}

TEST(TensorMathTest, BroadcastDivide) {
  const DoubleTensor t1({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
  const DoubleTensor t2(5.0);
  const DoubleTensor expected_quotient({{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}});
  EXPECT_THAT(Divide(t1, t2), DoubleTensorNear(expected_quotient));
  const DoubleTensor expected_quotient_reverse(
      {{0.5, 0.25, 1.0 / 6.0}, {0.125, 0.1, 1.0 / 12.0}});
  EXPECT_THAT(Divide(t2, t1), DoubleTensorNear(expected_quotient_reverse));
}

TEST(TensorMathTest, BroadcastMaximum) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(3.5);
  const DoubleTensor expected_max({{3.5, 3.5, 3.5}, {4.0, 5.0, 6.0}});
  EXPECT_THAT(ElementwiseMaximum(t1, t2), DoubleTensorNear(expected_max));
  EXPECT_THAT(ElementwiseMaximum(t2, t1), DoubleTensorNear(expected_max));
}

TEST(TensorMathTest, BroadcastMinimum) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(3.5);
  const DoubleTensor expected_min({{1.0, 2.0, 3.0}, {3.5, 3.5, 3.5}});
  EXPECT_THAT(ElementwiseMinimum(t1, t2), DoubleTensorNear(expected_min));
  EXPECT_THAT(ElementwiseMinimum(t2, t1), DoubleTensorNear(expected_min));
}

// ////////// Exhaustive tests on broadcasting logic for binary ops ////////////

TEST(TensorMathTest, BroadcastSameRankDim1Add) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(
      {std::vector<double>({10.0}), std::vector<double>({40.0})});
  const DoubleTensor expected_sum({{11.0, 12.0, 13.0}, {44.0, 45.0, 46.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathTest, BroadcastSameRankDim0Add) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(std::vector<std::vector<double>>({{10.0, 20.0, 30.0}}));
  const DoubleTensor expected_sum({{11.0, 22.0, 33.0}, {14.0, 25.0, 36.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathTest, BroadcastRankSmall) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({10.0, 20.0, 30.0});
  const DoubleTensor expected_sum({{11.0, 22.0, 33.0}, {14.0, 25.0, 36.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathTest, BroadcastBothWithRankLift) {
  const DoubleTensor t1({std::vector<double>{1.0}, std::vector<double>({4.0})});
  const DoubleTensor t2({10.0, 20.0, 30.0});
  const DoubleTensor expected_sum({{11.0, 21.0, 31.0}, {14.0, 24.0, 34.0}});
  ExpectSum(t1, t2, expected_sum);
}

TEST(TensorMathDeathTest, WrongRowsNoBroadcasting) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0}, {40.0, 50.0}});
  ASSERT_DEATH({ Add(t1, t2); }, "");
}

TEST(TensorMathDeathTest, WrongRowsNoBroadcastingFlipped) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0}, {40.0, 50.0}});
  ASSERT_DEATH({ Add(t2, t1); }, "");
}

TEST(TensorMathDeathTest, WrongColumnsNoBroadcasting) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(
      {{10.0, 20.0, 30}, {40.0, 50.0, 60.0}, {70.0, 80.0, 90.0}});
  ASSERT_DEATH({ Add(t1, t2); }, "");
}

TEST(TensorMathDeathTest, WrongColumnsNoBroadcastingFlipped) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2(
      {{10.0, 20.0, 30}, {40.0, 50.0, 60.0}, {70.0, 80.0, 90.0}});
  ASSERT_DEATH({ Add(t2, t1); }, "");
}

// ///////////////////////// matmul tests //////////////////////////////////////

TEST(TensorMathTest, BasicMatMul) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}});
  const DoubleTensor expected_mat_mul({{220.0, 280.0}, {490.0, 640.0}});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

TEST(TensorMathTest, BasicMatMulTransposed) {
  const DoubleTensor t1({{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}});
  const DoubleTensor t2({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor expected_mat_mul(
      {{90.0, 120.0, 150.0}, {190.0, 260.0, 330.0}, {290.0, 400.0, 510.0}});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

TEST(TensorMathTest, MatMulMatrixVector) {
  const DoubleTensor t1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor t2({std::vector<double>({10.0}),
                         std::vector<double>({30.0}),
                         std::vector<double>({50.0})});
  const DoubleTensor expected_mat_mul(
      {std::vector<double>({220.0}), std::vector<double>({490.0})});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

TEST(TensorMathTest, MatMulVectorMatrix) {
  const DoubleTensor t1(std::vector<std::vector<double>>({{1.0, 2.0, 3.0}}));
  const DoubleTensor t2({{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}});
  const DoubleTensor expected_mat_mul(
      std::vector<std::vector<double>>({{220.0, 280.0}}));
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

TEST(TensorMathTest, MatMul3d) {
  const DoubleTensor t1(
      {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}});
  const DoubleTensor t2({{{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}},
                         {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}});
  const DoubleTensor expected_mat_mul(
      {{{220.0, 280.0}, {490.0, 640.0}}, {{6.0, 8.0}, {3.0, 4.0}}});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

TEST(TensorMathTest, MatMul3dBroadcast) {
  const DoubleTensor t1(
      {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}});
  const DoubleTensor t2(std::vector<std::vector<std::vector<double>>>(
      {{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}}));
  const DoubleTensor expected_mat_mul(
      {{{22.0, 28.0}, {49.0, 64.0}}, {{6.0, 8.0}, {3.0, 4.0}}});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

// Like above, but multiply by a matrix that must be padded up to a tensor
// before broadcasting.
TEST(TensorMathTest, MatMul3dPadBroadcast) {
  const DoubleTensor t1(
      {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}});
  const DoubleTensor t2({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  const DoubleTensor expected_mat_mul(
      {{{22.0, 28.0}, {49.0, 64.0}}, {{6.0, 8.0}, {3.0, 4.0}}});
  EXPECT_THAT(MatMul(t1, t2), DoubleTensorNear(expected_mat_mul));
}

// TODO: matmul should have death tests as well.


}  // namespace
}  // namespace tf_opt
