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

#include "tf_opt/tensor/concat.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

std::pair<int, int64_t> MakeEntry(int first, int64_t second) {
  return std::make_pair(first, second);
}

// See example on class documentation.
TEST(ConcatLookupTableTest, BasicLookups) {
  const internal::ConcatLookupTable table({3, 5, 4});

  // The first list.
  EXPECT_EQ(table.Lookup(0), MakeEntry(0, 0));
  EXPECT_EQ(table.Lookup(1), MakeEntry(0, 1));
  EXPECT_EQ(table.Lookup(2), MakeEntry(0, 2));

  // The second list.
  EXPECT_EQ(table.Lookup(3), MakeEntry(1, 0));
  EXPECT_EQ(table.Lookup(4), MakeEntry(1, 1));
  EXPECT_EQ(table.Lookup(5), MakeEntry(1, 2));
  EXPECT_EQ(table.Lookup(6), MakeEntry(1, 3));
  EXPECT_EQ(table.Lookup(7), MakeEntry(1, 4));

  // The third list.
  EXPECT_EQ(table.Lookup(8), MakeEntry(2, 0));
  EXPECT_EQ(table.Lookup(9), MakeEntry(2, 1));
  EXPECT_EQ(table.Lookup(10), MakeEntry(2, 2));
  EXPECT_EQ(table.Lookup(11), MakeEntry(2, 3));
}

TEST(ConcatTest, SimpleConcat1D) {
  const DoubleTensor input1({1.0, 2.0, 3.0});
  const DoubleTensor input2({4.0, 5.0, 6.0});
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 0),
              IsOkAndHolds(Shape({6})));
  const DoubleTensor expected_output({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  EXPECT_THAT(Concat<double>({&input1, &input2}, 0),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, SimpleConcat2DAxis0) {
  const DoubleTensor input1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor input2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 0),
              IsOkAndHolds(Shape({4, 3})));
  const DoubleTensor expected_output(
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
  EXPECT_THAT(Concat<double>({&input1, &input2}, 0),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, SimpleConcat2DAxis1) {
  const DoubleTensor input1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensor input2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 1),
              IsOkAndHolds(Shape({2, 6})));
  const DoubleTensor expected_output(
      {{1.0, 2.0, 3.0, 7.0, 8.0, 9.0}, {4.0, 5.0, 6.0, 10.0, 11.0, 12.0}});
  EXPECT_THAT(Concat<double>({&input1, &input2}, 1),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, UnevenConcat2DAxis0) {
  // Shape: [1, 3].
  const auto input1 = DoubleTensor::CreateMatrix({{1.0, 2.0, 3.0}});
  // Shape: [2, 3].
  const DoubleTensor input2({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});

  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 0),
              IsOkAndHolds(Shape({3, 3})));

  const DoubleTensor expected_output(
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  EXPECT_THAT(Concat<double>({&input1, &input2}, 0),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, UnevenConcat2DAxis1) {
  // Shape: [2, 4].
  const DoubleTensor input1({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
  // Shape: [2, 3].
  const DoubleTensor input2({{9.0, 10.0, 11.0}, {12.0, 13.0, 14.0}});

  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 1),
              IsOkAndHolds(Shape({2, 7})));

  const DoubleTensor expected_output({{1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0},
                                      {5.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0}});
  EXPECT_THAT(Concat<double>({&input1, &input2}, 1),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, Concat3DAxis0) {
  DoubleTensor input1(Shape({2, 2, 2}));
  DoubleTensor input2(Shape({2, 2, 2}));
  for (int i = 0; i < 8; ++i) {
    (*input1.mutable_flat_values())[i] = i;
    (*input2.mutable_flat_values())[i] = i + 8;
  }
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 0),
              IsOkAndHolds(Shape({4, 2, 2})));

  const DoubleTensor output = Concat<double>({&input1, &input2}, 0);
  EXPECT_EQ(output.value({0, 0, 0}), 0.0);
  EXPECT_EQ(output.value({0, 0, 1}), 1.0);
  EXPECT_EQ(output.value({0, 1, 0}), 2.0);
  EXPECT_EQ(output.value({0, 1, 1}), 3.0);
  EXPECT_EQ(output.value({1, 0, 0}), 4.0);
  EXPECT_EQ(output.value({2, 0, 0}), 8.0);
  EXPECT_EQ(output.value({2, 0, 1}), 9.0);
  EXPECT_EQ(output.value({2, 1, 0}), 10.0);
  EXPECT_EQ(output.value({3, 1, 1}), 15.0);
}

TEST(ConcatTest, Concat3DAxis1) {
  DoubleTensor input1(Shape({2, 2, 2}));
  DoubleTensor input2(Shape({2, 2, 2}));
  for (int i = 0; i < 8; ++i) {
    (*input1.mutable_flat_values())[i] = i;
    (*input2.mutable_flat_values())[i] = i + 8;
  }
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 1),
              IsOkAndHolds(Shape({2, 4, 2})));

  const DoubleTensor output = Concat<double>({&input1, &input2}, 1);
  EXPECT_EQ(output.value({0, 0, 0}), 0.0);
  EXPECT_EQ(output.value({0, 0, 1}), 1.0);
  EXPECT_EQ(output.value({0, 1, 0}), 2.0);
  EXPECT_EQ(output.value({0, 1, 1}), 3.0);
  EXPECT_EQ(output.value({1, 0, 0}), 4.0);
  EXPECT_EQ(output.value({1, 1, 1}), 7.0);

  EXPECT_EQ(output.value({0, 2, 0}), 8.0);
  EXPECT_EQ(output.value({0, 2, 1}), 9.0);
  EXPECT_EQ(output.value({0, 3, 0}), 10.0);
  EXPECT_EQ(output.value({0, 3, 1}), 11.0);
  EXPECT_EQ(output.value({1, 2, 0}), 12.0);
  EXPECT_EQ(output.value({1, 3, 1}), 15.0);
}

TEST(ConcatTest, Concat3DAxis2) {
  DoubleTensor input1(Shape({2, 2, 2}));
  DoubleTensor input2(Shape({2, 2, 2}));
  for (int i = 0; i < 8; ++i) {
    (*input1.mutable_flat_values())[i] = i;
    (*input2.mutable_flat_values())[i] = i + 8;
  }
  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension()}, 2),
              IsOkAndHolds(Shape({2, 2, 4})));

  const DoubleTensor output = Concat<double>({&input1, &input2}, 2);
  EXPECT_EQ(output.value({0, 0, 0}), 0.0);
  EXPECT_EQ(output.value({0, 0, 1}), 1.0);
  EXPECT_EQ(output.value({0, 1, 0}), 2.0);
  EXPECT_EQ(output.value({0, 1, 1}), 3.0);
  EXPECT_EQ(output.value({1, 0, 0}), 4.0);
  EXPECT_EQ(output.value({1, 1, 1}), 7.0);

  EXPECT_EQ(output.value({0, 0, 2}), 8.0);
  EXPECT_EQ(output.value({0, 0, 3}), 9.0);
  EXPECT_EQ(output.value({0, 1, 2}), 10.0);
  EXPECT_EQ(output.value({0, 1, 3}), 11.0);
  EXPECT_EQ(output.value({1, 0, 2}), 12.0);
  EXPECT_EQ(output.value({1, 1, 3}), 15.0);
}

TEST(ConcatTest, ConcatManyAxis0) {
  const auto input1 = DoubleTensor::CreateMatrix({{1.0, 2.0}, {3.0, 4.0}});
  const auto input2 = DoubleTensor::CreateMatrix({{5.0, 6.0}, {7.0, 8.0}});
  const auto input3 = DoubleTensor::CreateMatrix({{9.0, 10.0}, {11.0, 12.0}});
  const auto input4 = DoubleTensor::CreateMatrix({{13.0, 14.0}, {15.0, 16.0}});

  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension(),
                                 input3.dimension(), input4.dimension()},
                                0),
              IsOkAndHolds(Shape({8, 2})));

  const DoubleTensor expected_output =
      DoubleTensor::CreateMatrix({{1.0, 2.0},
                                  {3.0, 4.0},
                                  {5.0, 6.0},
                                  {7.0, 8.0},
                                  {9.0, 10.0},
                                  {11.0, 12.0},
                                  {13.0, 14.0},
                                  {15.0, 16.0}});
  EXPECT_THAT(Concat<double>({&input1, &input2, &input3, &input4}, 0),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, ConcatManyAxis1) {
  const auto input1 = DoubleTensor::CreateMatrix({{1.0, 2.0}, {3.0, 4.0}});
  const auto input2 = DoubleTensor::CreateMatrix({{5.0, 6.0}, {7.0, 8.0}});
  const auto input3 = DoubleTensor::CreateMatrix({{9.0, 10.0}, {11.0, 12.0}});
  const auto input4 = DoubleTensor::CreateMatrix({{13.0, 14.0}, {15.0, 16.0}});

  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension(),
                                 input3.dimension(), input4.dimension()},
                                1),
              IsOkAndHolds(Shape({2, 8})));
  const DoubleTensor expected_output(
      {{1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0},
       {3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0}});

  EXPECT_THAT(Concat<double>({&input1, &input2, &input3, &input4}, 1),
              DoubleTensorNear(expected_output));
}

TEST(ConcatTest, ConcatDirectManyAxis1) {
  const auto input1 = DoubleTensor::CreateMatrix({{1.0, 2.0}, {3.0, 4.0}});
  const auto input2 = DoubleTensor::CreateMatrix({{5.0, 6.0}, {7.0, 8.0}});
  const auto input3 = DoubleTensor::CreateMatrix({{9.0, 10.0}, {11.0, 12.0}});
  const auto input4 = DoubleTensor::CreateMatrix({{13.0, 14.0}, {15.0, 16.0}});

  ASSERT_THAT(ConcatOutputShape({input1.dimension(), input2.dimension(),
                                 input3.dimension(), input4.dimension()},
                                1),
              IsOkAndHolds(Shape({2, 8})));
  const DoubleTensor expected_output(
      {{1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0},
       {3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0}});

  EXPECT_THAT(ConcatDirect<double>({input1, input2, input3, input4}, 1),
              DoubleTensorNear(expected_output));
}

TEST(ConcatOutputShapeTest, NoInputs) {
  EXPECT_THAT(ConcatOutputShape({}, 0),
              StatusIs(kInvalidArgument,
                       "Concat must have at least one input, found none."));
}

TEST(ConcatOutputShapeTest, UnequalRank) {
  EXPECT_THAT(
      ConcatOutputShape({Shape({3}), Shape({2, 3})}, 0),
      StatusIs(kInvalidArgument,
               ::testing::HasSubstr(
                   "All inputs to concat must have shapes with equal rank")));
}

TEST(ConcatOutputShapeTest, AxisLow) {
  EXPECT_THAT(
      ConcatOutputShape({Shape({3}), Shape({3})}, -1),
      StatusIs(kInvalidArgument, ::testing::HasSubstr("axis must be in [0..")));
}

TEST(ConcatOutputShapeTest, AxisHigh) {
  EXPECT_THAT(
      ConcatOutputShape({Shape({3}), Shape({3})}, 2),
      StatusIs(kInvalidArgument, ::testing::HasSubstr("axis must be in [0..")));
}

TEST(ConcatOutputShapeTest, DisagreeOffAxis) {
  EXPECT_THAT(
      ConcatOutputShape({Shape({3, 4}), Shape({2, 3})}, 0),
      StatusIs(
          kInvalidArgument,
          ::testing::HasSubstr(
              "Inputs to concat must agree in every dimension except axis")));
}

TEST(ConcatDeathTestTest, ConcatOnBadInput) {
  const DoubleTensor input1({1.0, 2.0, 3.0});
  const DoubleTensor input2({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  ASSERT_DEATH(Concat<double>({&input1, &input2}, 0),
               "All inputs to concat must have shapes with equal rank");
}

}  // namespace
}  // namespace tf_opt
