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

#include "tf_opt/tensor/convolve.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
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
constexpr double kTolerance = 1e-5;

TEST(Conv2dTest, SimpleValidPadding) {
  DoubleTensor input({{1.0, 2.0, 3.0},  //
                      {4.0, 5.0, 6.0},
                      {7.0, 8.0, 9.0}});
  input.ReshapeInPlace(Shape({1, 3, 3, 1}));
  DoubleTensor filter({{1.0, -1.0, 1.0},  //
                       {2.0, 0.0, -1.0},
                       {0.0, -1.0, 2.0}});
  filter.ReshapeInPlace(Shape({3, 3, 1, 1}));
  const auto kExpectedOutput = DoubleTensor::FromFlatData(
      Shape({1, 1, 1, 1}), {1.0 - 2.0 + 3.0 + 8.0 - 6.0 - 8.0 + 18.0});
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                              PaddingType::VALID)),
              IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

TEST(Conv2dTest, ValidPaddingBatch) {
  DoubleTensor input({{{1.0, 2.0, 3.0},  //
                       {4.0, 5.0, 6.0},
                       {7.0, 8.0, 9.0}},
                      {{-1.0, -2.0, -3.0},  //
                       {-4.0, -5.0, -6.0},
                       {-7.0, -8.0, -9.0}}});
  input.ReshapeInPlace(Shape({2, 3, 3, 1}));
  DoubleTensor filter({{1.0, -1.0, 1.0},  //
                       {2.0, 0.0, -1.0},
                       {0.0, -1.0, 2.0}});
  filter.ReshapeInPlace(Shape({3, 3, 1, 1}));
  constexpr double kResult = 1.0 - 2.0 + 3.0 + 8.0 - 6.0 - 8.0 + 18.0;
  const auto kExpectedOutput =
      DoubleTensor::FromFlatData(Shape({2, 1, 1, 1}), {kResult, -kResult});
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                              PaddingType::VALID)),
              IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

// TODO: Delete this once Tensor supports dot.
double DotProd(const std::vector<double>& a, const std::vector<double>& b) {
  CHECK_EQ(a.size(), b.size());
  double result = 0;
  for (int i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

TEST(Conv2dTest, SimpleSamePadding) {
  DoubleTensor input({{1.0, 2.0, 3.0},  //
                      {4.0, 5.0, 6.0},
                      {7.0, 8.0, 9.0}});
  input.ReshapeInPlace(Shape({1, 3, 3, 1}));
  DoubleTensor filter({{1.0, -1.0, 1.0},  //
                       {2.0, 0.0, -1.0},
                       {0.0, -1.0, 2.0}});
  filter.ReshapeInPlace(Shape({3, 3, 1, 1}));

  TFOPT_ASSERT_OK_AND_ASSIGN(
      const DoubleTensor actual,
      (Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                      PaddingType::SAME)));
  EXPECT_EQ(actual.dimension(), Shape({1, 3, 3, 1}));
  EXPECT_NEAR(
      actual.value({0, 1, 1, 0}),
      DotProd({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, -1, 1, 2, 0, -1, 0, -1, 2}),
      kTolerance);
  EXPECT_NEAR(actual.value({0, 0, 0, 0}), DotProd({1, 2, 4, 5}, {0, -1, -1, 2}),
              kTolerance);
  EXPECT_NEAR(actual.value({0, 2, 2, 0}), DotProd({5, 6, 8, 9}, {1, -1, 2, 0}),
              kTolerance);
}

TEST(Conv2dTest, Same2By2) {
  DoubleTensor input({{1.0, 2.0},  //
                      {3.0, 4.0}});
  input.ReshapeInPlace(Shape({1, 2, 2, 1}));
  DoubleTensor filter({{1.0, -1.0},  //
                       {2.0, 0.0}});
  filter.ReshapeInPlace(Shape({2, 2, 1, 1}));
  DoubleTensor kExpectedResult({{5.0, 10.0},  //
                                {-1.0, 4.0}});
  kExpectedResult.ReshapeInPlace(Shape({1, 2, 2, 1}));
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                              PaddingType::SAME)),
              IsOkAndHolds(DoubleTensorNear(kExpectedResult)));
}

TEST(Conv2dTest, Valid2By2) {
  DoubleTensor input({{1.0, 2.0},  //
                      {3.0, 4.0}});
  input.ReshapeInPlace(Shape({1, 2, 2, 1}));
  DoubleTensor filter({{1.0, -1.0},  //
                       {2.0, 0.0}});
  filter.ReshapeInPlace(Shape({2, 2, 1, 1}));
  const auto kExpectedResult =
      DoubleTensor::FromFlatData(Shape({1, 1, 1, 1}), {5.0});
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                              PaddingType::VALID)),
              IsOkAndHolds(DoubleTensorNear(kExpectedResult)));
}

TEST(Conv2dTest, Channels2In3Out) {
  const auto kInput =
      DoubleTensor::FromFlatData(Shape({1, 1, 1, 2}), {2.0, 10.0});
  DoubleTensor filter({{1.0, -1.0, 0.0},  //
                       {1.0, 1.0, 2.0}});
  filter.ReshapeInPlace(Shape({1, 1, 2, 3}));
  const auto kExpectedResult =
      DoubleTensor::FromFlatData(Shape({1, 1, 1, 3}), {12.0, 8.0, 20.0});
  EXPECT_THAT((Conv2d<double, double, double>(kInput, filter, Position2D(1, 1),
                                              PaddingType::SAME)),
              IsOkAndHolds(DoubleTensorNear(kExpectedResult)));
}

// Forked from:
// cs/third_party/tensorflow/core/kernels/conv_ops_test.cc?cl=176903365.
// See ConvOpTest::HandwrittenConv().
TEST(Conv2dTest, SameLargeTest_ConvOpTestFork) {
  DoubleTensor input({{1.0, 2.0, 3.0, 4.0},  //
                      {5.0, 6.0, 7.0, 8.0},
                      {9.0, 10.0, 11.0, 12.0}});
  input.ReshapeInPlace(Shape({1, 3, 4, 1}));
  DoubleTensor filter({{1.0, 4.0, 7.0},  //
                       {2.0, 5.0, 8.0},
                       {3.0, 6.0, 9.0}});
  filter.ReshapeInPlace(Shape({3, 3, 1, 1}));
  DoubleTensor expected_result({{105.0, 150.0, 183.0, 95.0},  //
                                {235.0, 312.0, 357.0, 178.0},
                                {187.0, 234.0, 261.0, 121.0}});
  expected_result.ReshapeInPlace(Shape({1, 3, 4, 1}));
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 1),
                                              PaddingType::SAME)),
              IsOkAndHolds(DoubleTensorNear(expected_result)));
}

// Forked from:
// cs/third_party/tensorflow/core/kernels/conv_ops_test.cc?cl=176903365.
// See ConvOpTest::AnisotropicStrides().
TEST(Conv2dTest, ValidStrideXTest_ConvOpTestFork) {
  DoubleTensor input({{3.0, 2.0, 1.0, -1.0, -2.0, -3.0},  //
                      {4.0, 3.0, 2.0, -2.0, -3.0, -4.0},
                      {5.0, 4.0, 3.0, -3.0, -4.0, -5.0}});
  input.ReshapeInPlace(Shape({1, 3, 6, 1}));
  DoubleTensor filter({{1.0, 2.0},  //
                       {3.0, 4.0}});
  filter.ReshapeInPlace(Shape({2, 2, 1, 1}));
  DoubleTensor expected_result({{31.0, -23.0},  //
                                {41.0, -33.0}});
  expected_result.ReshapeInPlace(Shape({1, 2, 2, 1}));
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(1, 3),
                                              PaddingType::VALID)),
              IsOkAndHolds(DoubleTensorNear(expected_result)));
}

// Forked from:
// cs/third_party/tensorflow/core/kernels/conv_ops_test.cc?cl=176903365.
// Like ConvOpTest::AnisotropicStrides(), but transposed.
TEST(Conv2dTest, ValidStrideYTest_ConvOpTestForkTranspose) {
  DoubleTensor input({{3.0, 4.0, 5.0},  //
                      {2.0, 3.0, 4.0},
                      {1.0, 2.0, 3.0},
                      {-1.0, -2.0, -3.0},
                      {-2.0, -3.0, -4.0},
                      {-3.0, -4.0, -5.0}});
  input.ReshapeInPlace(Shape({1, 6, 3, 1}));
  DoubleTensor filter({{1.0, 3.0},  //
                       {2.0, 4.0}});
  filter.ReshapeInPlace(Shape({2, 2, 1, 1}));

  DoubleTensor expected_result({{31.0, 41.0},  //
                                {-23.0, -33.0}});
  expected_result.ReshapeInPlace(Shape({1, 2, 2, 1}));
  EXPECT_THAT((Conv2d<double, double, double>(input, filter, Position2D(3, 1),
                                              PaddingType::VALID)),
              IsOkAndHolds(DoubleTensorNear(expected_result)));
}

// For setting up error tests.
struct SimpleConv2dBuilder {
  DoubleTensor input;
  DoubleTensor filter;
  Position2D strides;
  PaddingType padding;

  SimpleConv2dBuilder()
      : input(DoubleTensor::FromFlatData(Shape({1, 2, 2, 1}),
                                         {1.0, 2.0, 3.0, 4.0})),
        filter(DoubleTensor::FromFlatData(Shape({2, 2, 1, 1}),
                                          {1.0, -1.0, 2.0, 0.0})),
        strides(1, 1),
        padding(PaddingType::SAME) {}

  absl::StatusOr<DoubleTensor> MakeConv2d() const {
    return Conv2d<double, double, double>(input, filter, strides, padding);
  }

  absl::StatusOr<Shape> RunConv2dOutputShape() const {
    return Conv2dOutputShape(input.dimension(), filter.dimension(), strides,
                             padding);
  }
};

TEST(Conv2dTest, IllegalStrideCol) {
  SimpleConv2dBuilder builder;
  builder.strides.col = 0;
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument, HasSubstr("Expected stride col > 0")));
}

TEST(Conv2dTest, IllegalStrideRow) {
  SimpleConv2dBuilder builder;
  builder.strides.row = -3;
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument, HasSubstr("Expected stride row > 0")));
}

TEST(Conv2dTest, BadInputRank) {
  SimpleConv2dBuilder builder;
  builder.input.ReshapeInPlace(Shape({2, 2, 1}));
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected input shape to have rank four")));
}

TEST(Conv2dTest, BadFilterRank) {
  SimpleConv2dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({2, 2}));
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected filter shape to have rank four")));
}

TEST(Conv2dTest, InputFilterChannelMismatch) {
  SimpleConv2dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({1, 1, 2, 2}));
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument,
                       HasSubstr("should be equal to filter input channels")));
}

TEST(Conv2dOutputShapeTest, SimpleValidPadding) {
  const Shape kInput({1, 4, 4, 1});
  const Shape kFilter({3, 3, 1, 1});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(1, 1), PaddingType::VALID),
      IsOkAndHolds(Shape({1, 2, 2, 1})));
}

TEST(Conv2dOutputShapeTest, SimpleSamePadding) {
  const Shape kInput({1, 4, 4, 1});
  const Shape kFilter({3, 3, 1, 1});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(1, 1), PaddingType::SAME),
      IsOkAndHolds(Shape({1, 4, 4, 1})));
}

TEST(Conv2dOutputShapeTest, SameStride) {
  const Shape kInput({1, 4, 4, 1});
  const Shape kFilter({3, 3, 1, 1});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(2, 2), PaddingType::SAME),
      IsOkAndHolds(Shape({1, 2, 2, 1})));
}

TEST(Conv2dOutputShapeTest, ValidBatch) {
  const Shape kInput({10, 4, 4, 1});
  const Shape kFilter({3, 3, 1, 1});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(1, 1), PaddingType::VALID),
      IsOkAndHolds(Shape({10, 2, 2, 1})));
}

TEST(Conv2dOutputShapeTest, ValidInChannels) {
  const Shape kInput({1, 4, 4, 5});
  const Shape kFilter({3, 3, 5, 1});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(1, 1), PaddingType::VALID),
      IsOkAndHolds(Shape({1, 2, 2, 1})));
}

TEST(Conv2dOutputShapeTest, ValidOutputChannels) {
  const Shape kInput({1, 4, 4, 1});
  const Shape kFilter({3, 3, 1, 5});
  EXPECT_THAT(
      Conv2dOutputShape(kInput, kFilter, Position2D(1, 1), PaddingType::VALID),
      IsOkAndHolds(Shape({1, 2, 2, 5})));
}

TEST(Conv2dOutputShapeTest, IllegalStrideCol) {
  SimpleConv2dBuilder builder;
  builder.strides.col = 0;
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument, HasSubstr("Expected stride col > 0")));
}

TEST(Conv2dOutputShapeTest, IllegalStrideRow) {
  SimpleConv2dBuilder builder;
  builder.strides.row = -3;
  EXPECT_THAT(builder.MakeConv2d(),
              StatusIs(kInvalidArgument, HasSubstr("Expected stride row > 0")));
}

TEST(Conv2dOutputShapeTest, BadInputRank) {
  SimpleConv2dBuilder builder;
  builder.input.ReshapeInPlace(Shape({4}));
  EXPECT_THAT(builder.RunConv2dOutputShape(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected input shape to have rank four")));
}

TEST(Conv2dOutputShapeTest, BadFilterRank) {
  SimpleConv2dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({2, 2, 1, 1, 1}));
  EXPECT_THAT(builder.RunConv2dOutputShape(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected filter shape to have rank four")));
}

TEST(Conv2dOutputShapeTest, InputFilterChannelMismatch) {
  SimpleConv2dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({1, 1, 2, 2}));
  EXPECT_THAT(builder.RunConv2dOutputShape(),
              StatusIs(kInvalidArgument,
                       HasSubstr("should be equal to filter input channels")));
}

TEST(Conv1dOutputShapeTest, SimpleValidPadding) {
  const Shape kInput({1, 4, 1});
  const Shape kFilter({3, 1, 1});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 1, PaddingType::VALID),
              IsOkAndHolds(Shape({1, 2, 1})));
}

TEST(Conv1dTest, SimpleValidPadding) {
  const auto kInput =
      DoubleTensor::FromFlatData(Shape({1, 4, 1}), {4.0, 5.0, 6.0, 7.0});
  const auto kFilter =
      DoubleTensor::FromFlatData(Shape({3, 1, 1}), {2.0, 0.0, -1.0});
  const auto kExpectedOutput =
      DoubleTensor::FromFlatData(Shape({1, 2, 1}), {2.0, 3.0});

  EXPECT_THAT(
      (Conv1d<double, double, double>(kInput, kFilter, 1, PaddingType::VALID)),
      IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

TEST(Conv1dOutputShapeTest, SimpleSamePadding) {
  const Shape kInput({1, 4, 1});
  const Shape kFilter({3, 1, 1});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 1, PaddingType::SAME),
              IsOkAndHolds(Shape({1, 4, 1})));
}

TEST(Conv1dTest, SimpleSamePadding) {
  const auto kInput =
      DoubleTensor::FromFlatData(Shape({1, 4, 1}), {4.0, 5.0, 6.0, 7.0});
  const auto kFilter =
      DoubleTensor::FromFlatData(Shape({3, 1, 1}), {2.0, 0.0, -1.0});
  const auto kExpectedOutput =
      DoubleTensor::FromFlatData(Shape({1, 4, 1}), {-5.0, 2.0, 3.0, 12.0});
  EXPECT_THAT(
      (Conv1d<double, double, double>(kInput, kFilter, 1, PaddingType::SAME)),
      IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

TEST(Conv1dOutputShapeTest, SameStride) {
  const Shape kInput({1, 4, 1});
  const Shape kFilter({3, 1, 1});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 2, PaddingType::SAME),
              IsOkAndHolds(Shape({1, 2, 1})));
}

TEST(Conv1dTest, SameStride) {
  const auto kInput =
      DoubleTensor::FromFlatData(Shape({1, 4, 1}), {4.0, 5.0, 6.0, 7.0});
  const auto kFilter =
      DoubleTensor::FromFlatData(Shape({3, 1, 1}), {2.0, 0.0, -1.0});
  const auto kExpectedOutput =
      DoubleTensor::FromFlatData(Shape({1, 2, 1}), {2.0, 12.0});
  EXPECT_THAT(
      (Conv1d<double, double, double>(kInput, kFilter, 2, PaddingType::SAME)),
      IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

TEST(Conv1dOutputShapeTest, ValidBatch) {
  const Shape kInput({10, 4, 1});
  const Shape kFilter({3, 1, 1});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 1, PaddingType::VALID),
              IsOkAndHolds(Shape({10, 2, 1})));
}

TEST(Conv1dTest, ValidBatch) {
  DoubleTensor input({{4.0, 5.0, 6.0, 7.0}, {-4.0, -5.0, -6.0, -7.0}});
  input.ReshapeInPlace(Shape({2, 4, 1}));
  const auto kFilter =
      DoubleTensor::FromFlatData(Shape({3, 1, 1}), {2.0, 0.0, -1.0});
  DoubleTensor expected_output({{2.0, 3.0}, {-2.0, -3.0}});
  expected_output.ReshapeInPlace(Shape({2, 2, 1}));
  EXPECT_THAT(
      (Conv1d<double, double, double>(input, kFilter, 1, PaddingType::VALID)),
      IsOkAndHolds(DoubleTensorNear(expected_output)));
}

TEST(Conv1dOutputShapeTest, ValidInChannels) {
  const Shape kInput({1, 4, 5});
  const Shape kFilter({3, 5, 1});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 1, PaddingType::VALID),
              IsOkAndHolds(Shape({1, 2, 1})));
}

TEST(Conv1dTest, ValidInputChannels) {
  DoubleTensor input({{4.0, -4.0}, {5.0, -5.0}, {6.0, -6.0}});
  input.ReshapeInPlace(Shape({1, 3, 2}));
  DoubleTensor filter({{2.0, -10}, {0.0, 10.0}, {-1.0, 10.0}});
  filter.ReshapeInPlace(Shape({3, 2, 1}));
  const auto kExpectedOutput = DoubleTensor::FromFlatData(
      Shape({1, 1, 1}), {8.0 - 6.0 + 40.0 - 50.0 - 60.0});
  EXPECT_THAT(
      (Conv1d<double, double, double>(input, filter, 1, PaddingType::VALID)),
      IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

TEST(Conv1dOutputShapeTest, ValidOutputChannels) {
  const Shape kInput({1, 4, 1});
  const Shape kFilter({3, 1, 5});
  EXPECT_THAT(Conv1dOutputShape(kInput, kFilter, 1, PaddingType::VALID),
              IsOkAndHolds(Shape({1, 2, 5})));
}

TEST(Conv1dTest, ValidOutputChannels) {
  const auto kInput =
      DoubleTensor::FromFlatData(Shape({1, 3, 1}), {4.0, 5.0, 6.0});
  DoubleTensor filter({{2.0, -10}, {0.0, 10.0}, {-1.0, 10.0}});
  filter.ReshapeInPlace(Shape({3, 1, 2}));
  const auto kExpectedOutput = DoubleTensor::FromFlatData(
      Shape({1, 1, 2}), {8.0 - 6.0, -40.0 + 50.0 + 60.0});
  EXPECT_THAT(
      (Conv1d<double, double, double>(kInput, filter, 1, PaddingType::VALID)),
      IsOkAndHolds(DoubleTensorNear(kExpectedOutput)));
}

// For setting up error dests.
struct SimpleConv1dBuilder {
  DoubleTensor input;
  DoubleTensor filter;
  int stride = 1;
  PaddingType padding = PaddingType::SAME;

  SimpleConv1dBuilder() {
    input = DoubleTensor(std::vector<double>({3.0, 4.0}));
    input.ReshapeInPlace(Shape({1, 2, 1}));
    filter = DoubleTensor(std::vector<double>({2.0, -1.0}));
    filter.ReshapeInPlace(Shape({2, 1, 1}));
  }

  absl::StatusOr<DoubleTensor> MakeConv1d() const {
    return Conv1d<double, double, double>(input, filter, stride, padding);
  }

  absl::StatusOr<Shape> RunConv1dOutputShape() const {
    return Conv1dOutputShape(input.dimension(), filter.dimension(), stride,
                             padding);
  }
};

TEST(Conv1dOutputShapeTest, IllegalStrideCol) {
  SimpleConv1dBuilder builder;
  builder.stride = 0;
  EXPECT_THAT(builder.RunConv1dOutputShape(),
              StatusIs(kInvalidArgument, HasSubstr("on conv1d inside conv2d")));
}

TEST(Conv1dOutputShapeTest, BadInputRank) {
  SimpleConv1dBuilder builder;
  builder.input.ReshapeInPlace(Shape({2}));
  EXPECT_THAT(builder.RunConv1dOutputShape(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected input shape to have rank three")));
}

TEST(Conv1dOutputShapeTest, BadFilterRank) {
  SimpleConv1dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({2, 1, 1, 1, 1}));
  EXPECT_THAT(builder.RunConv1dOutputShape(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected filter shape to have rank three")));
}

TEST(Conv1dOutputShapeTest, InputFilterChannelMismatch) {
  SimpleConv1dBuilder builder;
  builder.filter.ReshapeInPlace(Shape({1, 2, 1}));
  EXPECT_THAT(builder.RunConv1dOutputShape(),
              StatusIs(kInvalidArgument, HasSubstr("on conv1d inside conv2d")));
}

}  // namespace
}  // namespace tf_opt
