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

#include "tf_opt/tensor/reduce.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/bounds/bounds.h"
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

TEST(ReduceMaxTest, ReduceOutputShapeOnRankOne) {
  const Shape input_shape({4});
  const Shape expected;
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceOutputShape(input_shape, axes), IsOkAndHolds(expected));
}

TEST(ReduceMaxTest, ReduceOutputShapeOnHigherDim) {
  const Shape input_shape({4, 7, 2});
  const Shape expected({4, 2});
  const std::vector<int64_t> axes = {1};
  EXPECT_THAT(ReduceOutputShape(input_shape, axes), IsOkAndHolds(expected));
}

TEST(ReduceMaxTest, ReduceOutputShapeOnMultiDim) {
  const Shape input_shape({4, 7, 2});
  const Shape expected({4});
  const std::vector<int64_t> axes = {1, 2};
  EXPECT_THAT(ReduceOutputShape(input_shape, axes), IsOkAndHolds(expected));
}

TEST(ReduceMaxTest, ReduceOutputShapeBadAxis) {
  const Shape input_shape({4, 7, 2});
  const std::vector<int64_t> axes = {5};
  EXPECT_THAT(
      ReduceOutputShape(input_shape, axes),
      StatusIs(kInvalidArgument, HasSubstr("axis=5 should have been in")));
}

TEST(ReduceMaxTest, ReduceOutputShapeBadAxisMulti) {
  const Shape input_shape({4, 7, 2});
  const std::vector<int64_t> axes = {0, 5};
  EXPECT_THAT(
      ReduceOutputShape(input_shape, axes),
      StatusIs(kInvalidArgument, HasSubstr("axis=5 should have been in")));
}

TEST(ReduceMaxTest, ReduceOutputShapeBadAxisMultiDuplicates) {
  const Shape input_shape({4, 7, 2});
  const std::vector<int64_t> axes = {0, 0};
  EXPECT_THAT(ReduceOutputShape(input_shape, axes),
              StatusIs(kInvalidArgument, HasSubstr("contains duplicates")));
}

TEST(ReduceMaxTest, ReduceOutputShapeBadAxisMultiNotSorted) {
  const Shape input_shape({4, 7, 2});
  const std::vector<int64_t> axes = {1, 0};
  EXPECT_THAT(ReduceOutputShape(input_shape, axes),
              StatusIs(kInvalidArgument, HasSubstr("not sorted")));
}

TEST(ReduceMaxTest, ReduceMaxRankOneTensor) {
  const DoubleTensor input({10.0, 14.0, 12.0});
  const DoubleTensor expected(14.0);
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceMax(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMinTest, ReduceMinRankOneTensor) {
  const DoubleTensor input({10.0, 14.0, 12.0});
  const DoubleTensor expected(10.0);
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceMin(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMeanTest, ReduceMeanRankOneTensor) {
  const DoubleTensor input({10.0, 14.0, 12.0});
  const DoubleTensor expected(12.0);
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceMean(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceSumTest, ReduceSumRankOneTensor) {
  const DoubleTensor input({10.0, 14.0, 12.0});
  const DoubleTensor expected(36.0);
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceSum(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMaxTest, ReduceMaxAxisZero) {
  const DoubleTensor input({{10.0, 14.0, 12.0}, {13.0, 11.0, 15.0}});
  const DoubleTensor expected({13.0, 14.0, 15.0});
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceMax(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMaxTest, ReduceMaxAxisOne) {
  const DoubleTensor input({{10.0, 14.0, 12.0}, {13.0, 11.0, 15.0}});
  const DoubleTensor expected({14.0, 15.0});
  const std::vector<int64_t> axes = {1};
  EXPECT_THAT(ReduceMax(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMaxTest, ReduceMaxAxisMulti) {
  const DoubleTensor input({{10.0, 14.0, 12.0}, {13.0, 11.0, 15.0}});
  const DoubleTensor expected({15.0});
  const std::vector<int64_t> axes = {0, 1};
  EXPECT_THAT(ReduceMax(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMaxTest, ReduceMaxAxisMulti3D) {
  const DoubleTensor input({{{10.0, 14.0, 12.0}, {13.0, 11.0, 15.0}},
                            {{10.0, 14.0, 12.0}, {13.0, 11.0, 16.0}}});
  const DoubleTensor expected({15.0, 16.0});
  const std::vector<int64_t> axes = {1, 2};
  EXPECT_THAT(ReduceMax(input, axes), DoubleTensorNear(expected));
}

TEST(ReduceMaxTest, ReduceMaxAxisBounds) {
  const BoundsTensor input(
      {{Bounds(10.0, 15.0), Bounds(14.0, 15.0), Bounds(12.0, 13.0)},
       {Bounds(13.0, 14.0), Bounds(11.0, 12.0), Bounds(10.0, 16.0)}});
  const BoundsTensor expected(
      {Bounds(13.0, 15.0), Bounds(14.0, 15.0), Bounds(12.0, 16.0)});
  const std::vector<int64_t> axes = {0};
  EXPECT_THAT(ReduceMax(input, axes), BoundsTensorNear(expected, 1e-5));
}

TEST(ReduceMaxDeathTest, ReduceMaxBadAxis) {
  const DoubleTensor input(Shape({4, 7, 2}));
  const std::vector<int64_t> axes = {5};
  EXPECT_DEATH(ReduceMax(input, axes), "axis=5 should have been in");
}

TEST(ReduceMaxDeathTest, ReduceMaxBadAxisMulti) {
  const DoubleTensor input(Shape({4, 7, 2}));
  const std::vector<int64_t> axes = {0, 5};
  EXPECT_DEATH(ReduceMax(input, {0, 5}), "axis=5 should have been in");
}

TEST(ReduceMaxTest, ReduceAll) {
  const DoubleTensor input({{10.0, 14.0, 9.0}, {17.0, 11.0, 15.0}});
  EXPECT_DOUBLE_EQ(ReduceMax(input), 17.0);
}

TEST(ReduceMinTest, ReduceAll) {
  const DoubleTensor input({{10.0, 14.0, 9.0}, {17.0, 11.0, 15.0}});
  EXPECT_DOUBLE_EQ(ReduceMin(input), 9.0);
}

TEST(ReduceSumTest, ReduceAll) {
  const DoubleTensor input({{10.0, 14.0, 9.0}, {17.0, 11.0, 15.0}});
  EXPECT_NEAR(ReduceSum(input), 76.0, 1e-9);
}

TEST(ReduceMeanTest, ReduceAll) {
  const DoubleTensor input({{10.0, 14.0, 9.0}, {17.0, 11.0, 17.0}});
  EXPECT_NEAR(ReduceMean(input), 13.0, 1e-9);
}

}  // namespace
}  // namespace tf_opt
