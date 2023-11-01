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

#include "tf_opt/tensor/pooling.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/tensor_testing.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {
namespace {

using ::testing::HasSubstr;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

class MaxPoolTest : public ::testing::Test {
 protected:
  absl::StatusOr<Shape> OutputShape() const {
    return Pool2dOutputShape(input_shape_, ksize_, stride_, padding_);
  }

  DoubleTensor DoMaxPool() const {
    return MaxPool(input_, ksize_, stride_, padding_);
  }

  Shape input_shape_ = Shape({1, 3, 3, 1});
  Position2D ksize_ = Position2D(2, 2);
  Position2D stride_ = Position2D(1, 1);
  PaddingType padding_ = PaddingType::VALID;

  Shape output_shape_ = Shape({1, 2, 2, 1});
  DoubleTensor input_ = DoubleTensor::FromFlatData(
      input_shape_, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
};

using MaxPoolDeathTest = MaxPoolTest;

TEST_F(MaxPoolTest, OutputShapeBasic) {
  EXPECT_THAT(OutputShape(), IsOkAndHolds(output_shape_));
}

TEST_F(MaxPoolTest, MaxPoolBasic) {
  const auto expected =
      DoubleTensor::FromFlatData(Shape({1, 2, 2, 1}), {5.0, 6.0, 8.0, 9.0});
  EXPECT_THAT(DoMaxPool(), DoubleTensorEquals(expected));
}

TEST_F(MaxPoolTest, OutputShapeTall) {
  ksize_.row = 1;
  EXPECT_THAT(OutputShape(), IsOkAndHolds(Shape({1, 3, 2, 1})));
}

TEST_F(MaxPoolTest, MaxPoolTall) {
  ksize_.row = 1;
  const auto expected =
      DoubleTensor::FromFlatData(Shape({1, 3, 2, 1}), {2.0, 3.0,  //
                                                       5.0, 6.0,  //
                                                       8.0, 9.0});
  EXPECT_THAT(DoMaxPool(), DoubleTensorEquals(expected));
}

TEST_F(MaxPoolTest, OutputSmallWindowWithStride) {
  ksize_.row = 1;
  stride_.row = 2;
  EXPECT_THAT(OutputShape(), IsOkAndHolds(output_shape_));
}

TEST_F(MaxPoolTest, MaxPoolSmallWindowWithStride) {
  ksize_.row = 1;
  stride_.row = 2;
  const auto expected =
      DoubleTensor::FromFlatData(Shape({1, 2, 2, 1}), {2.0, 3.0,  //
                                                       8.0, 9.0});
  EXPECT_THAT(DoMaxPool(), DoubleTensorEquals(expected));
}

TEST_F(MaxPoolTest, OutputShapeBatched) {
  input_shape_ = Shape({10, 3, 3, 1});
  EXPECT_THAT(OutputShape(), IsOkAndHolds(Shape({10, 2, 2, 1})));
}

TEST_F(MaxPoolTest, MaxpoolBatched) {
  input_shape_ = Shape({2, 3, 3, 1});
  input_ = DoubleTensor::FromFlatData(
      input_shape_, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,  //
                     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0});
  const auto expected =
      DoubleTensor::FromFlatData(Shape({2, 2, 2, 1}), {5.0, 6.0, 8.0, 9.0,  //
                                                       15.0, 16.0, 18.0, 19.0});
  EXPECT_THAT(DoMaxPool(), DoubleTensorEquals(expected));
}

TEST_F(MaxPoolTest, OutputShapePaddingSame) {
  padding_ = PaddingType::SAME;
  EXPECT_THAT(OutputShape(), IsOkAndHolds(Shape({1, 3, 3, 1})));
}

TEST_F(MaxPoolTest, MaxpoolPaddingSame) {
  padding_ = PaddingType::SAME;
  // Make the output a little more interesting, otherwise its mostly 9.0.
  input_.set_value({0, 2, 2, 0}, 0.5);
  const auto expected =
      DoubleTensor::FromFlatData(Shape({1, 3, 3, 1}), {5.0, 6.0, 6.0,  //
                                                       8.0, 8.0, 6.0,  //
                                                       8.0, 8.0, 0.5});
  EXPECT_THAT(DoMaxPool(), DoubleTensorEquals(expected));
}

TEST_F(MaxPoolTest, OutputShapeBadRankOnInput) {
  input_shape_ = Shape({3, 3, 1});
  EXPECT_THAT(
      OutputShape(),
      StatusIs(kInvalidArgument, HasSubstr("Expected input to be rank four")));
}

TEST_F(MaxPoolDeathTest, MaxPoolBadRankOnInput) {
  input_.ReshapeInPlace(Shape({3, 3, 1}));
  EXPECT_DEATH(DoMaxPool(), "");
}

TEST_F(MaxPoolTest, OutputShapeBadArguments) {
  ksize_.row = -1;
  EXPECT_THAT(OutputShape(), StatusIs(kInvalidArgument));
}

TEST_F(MaxPoolDeathTest, MaxPoolBadArguments) {
  ksize_.row = -1;
  EXPECT_DEATH(DoMaxPool(), "");
}

}  // namespace
}  // namespace tf_opt
