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

#include "tf_opt/tensor/tensor.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::std::vector;
using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(DoubleTensorTest, EmptyOneDTensor) {
  const DoubleTensor t(Shape({7}));
  ASSERT_EQ(7, t.size());
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(0, t.value({i}));
  }
}

TEST(DoubleTensorTest, DefaultRankZeroTensor) {
  DoubleTensor t;
  EXPECT_EQ(t.size(), 1);
  EXPECT_EQ(t.dimension(), Shape());
  EXPECT_EQ(t.value({}), 0.0);
  EXPECT_THAT(t.flat_values(), ElementsAre(0.0));
  t.set_value({}, 5.0);
  EXPECT_EQ(t.value({}), 5.0);
  EXPECT_THAT(t.flat_values(), ElementsAre(5.0));
}

TEST(DoubleTensorTest, RankZeroTensorConstructed) {
  DoubleTensor t1(5.0);
  EXPECT_EQ(t1.dimension(), Shape());
  EXPECT_EQ(t1.size(), 1);
  EXPECT_THAT(t1.flat_values(), ElementsAre(5.0));
  DoubleTensor t2;
  t2.set_value({}, 5.0);
  EXPECT_THAT(t1, DoubleTensorEquals(t2));
}

TEST(DoubleTensorTest, SetValueOneDTensor) {
  DoubleTensor t(Shape({5}));
  t.set_value({1}, 4.5);
  t.set_value({4}, -1.1);

  EXPECT_EQ(0, t.value({0}));
  EXPECT_EQ(4.5, t.value({1}));
  EXPECT_EQ(0, t.value({2}));
  EXPECT_EQ(0, t.value({3}));
  EXPECT_EQ(-1.1, t.value({4}));
}

TEST(DoubleTensorTest, FlatValuesOneD) {
  DoubleTensor t(Shape({5}));
  t.set_value({1}, 4.5);
  t.set_value({4}, -1.1);
  EXPECT_THAT(t.flat_values(), ElementsAre(0, 4.5, 0, 0, -1.1));
}

TEST(DoubleTensorTest, ThirdDimensionTensor) {
  DoubleTensor t(Shape({2, 2, 2}));
  t.set_value({0, 0, 1}, 4.5);
  t.set_value({0, 1, 0}, 5.5);
  t.set_value({1, 0, 0}, 6.5);
  t.set_value({1, 0, 1}, 7.5);
  EXPECT_THAT(t.flat_values(), ElementsAre(0, 4.5, 5.5, 0, 6.5, 7.5, 0, 0));

  EXPECT_EQ(0, t.value({0, 0, 0}));
  EXPECT_EQ(4.5, t.value({0, 0, 1}));
  EXPECT_EQ(5.5, t.value({0, 1, 0}));
  EXPECT_EQ(0, t.value({0, 1, 1}));
  EXPECT_EQ(6.5, t.value({1, 0, 0}));
  EXPECT_EQ(7.5, t.value({1, 0, 1}));
  EXPECT_EQ(0, t.value({1, 1, 0}));
  EXPECT_EQ(0, t.value({1, 1, 1}));
}

TEST(DoubleTensorTest, FillConstructor) {
  DoubleTensor t(Shape({3, 1, 5}), 4.0);
  EXPECT_EQ(t.dimension(), Shape({3, 1, 5}));
  std::vector<double> expected_flat_values(15, 4.0);
  EXPECT_THAT(t.flat_values(), ContainerEq(expected_flat_values));
}

TEST(DoubleTensorTest, Vector1DConstructor) {
  DoubleTensor t({3.0, 1.0, 5.0});
  EXPECT_EQ(t.dimension(), Shape({3}));
  EXPECT_THAT(t.flat_values(), ElementsAre(3.0, 1.0, 5.0));
}

TEST(DoubleTensorTest, StaticVectorCreation) {
  DoubleTensor t = DoubleTensor::CreateVector({3.0, 1.0, 5.0});
  EXPECT_EQ(t.dimension(), Shape({3}));
  EXPECT_THAT(t.flat_values(), ElementsAre(3.0, 1.0, 5.0));
}

TEST(DoubleTensorTest, Vector2DConstructor) {
  DoubleTensor t({{3.0, 1.0, 5.0}, {10.0, 11.0, 12.0}});
  EXPECT_EQ(t.dimension(), Shape({2, 3}));
  EXPECT_THAT(t.flat_values(), ElementsAre(3.0, 1.0, 5.0, 10.0, 11.0, 12.0));
}

TEST(DoubleTensorTest, StaticMatrixCreation) {
  DoubleTensor t =
      DoubleTensor::CreateMatrix({{3.0, 1.0, 5.0}, {10.0, 11.0, 12.0}});
  EXPECT_EQ(t.dimension(), Shape({2, 3}));
  EXPECT_THAT(t.flat_values(), ElementsAre(3.0, 1.0, 5.0, 10.0, 11.0, 12.0));
}

TEST(DoubleTensorTest, Vector3DConstructor) {
  DoubleTensor t({{{3.0, 1.0, 5.0}, {10.0, 11.0, 12.0}},
                  {{-3.0, -1.0, -5.0}, {-10.0, -11.0, -12.0}}});
  EXPECT_EQ(t.dimension(), Shape({2, 2, 3}));
  EXPECT_THAT(
      t.flat_values(),
      ContainerEq(std::vector<double>({3.0, 1.0, 5.0, 10.0, 11.0, 12.0, -3.0,
                                       -1.0, -5.0, -10.0, -11.0, -12.0})));
}

TEST(DoubleTensorTest, FromFlatData) {
  auto t = DoubleTensor::FromFlatData(Shape({2, 2}), {2.0, 3.0, 4.0, 5.0});
  EXPECT_EQ(t.dimension(), Shape({2, 2}));
  EXPECT_THAT(t.flat_values(), ElementsAre(2.0, 3.0, 4.0, 5.0));
}

TEST(DoubleTensorDeathTest, FromFlatDataBadShape) {
  ASSERT_DEATH(DoubleTensor::FromFlatData(Shape({3, 2}), {2.0, 3.0, 4.0, 5.0}),
               "");
}

TEST(DoubleTensorTest, ReshapeInPlace) {
  DoubleTensor tensor({{2.0, 3.0}, {4.0, 5.0}});
  tensor.ReshapeInPlace(Shape({4}));
  const DoubleTensor expected({2.0, 3.0, 4.0, 5.0});
  EXPECT_THAT(tensor, DoubleTensorEquals(expected));
}

TEST(DoubleTensorDeathTest, ReshapeInPlaceBadSize) {
  DoubleTensor tensor({{2.0, 3.0}, {4.0, 5.0}});
  ASSERT_DEATH(tensor.ReshapeInPlace(Shape({5})), "");
}

TEST(DoubleTensorTest, Reshape) {
  DoubleTensor init({{2.0, 3.0}, {4.0, 5.0}});
  DoubleTensor init_copy({{2.0, 3.0}, {4.0, 5.0}});
  DoubleTensor reshaped = init.Reshape(Shape({4}));
  const DoubleTensor expected({2.0, 3.0, 4.0, 5.0});
  EXPECT_THAT(reshaped, DoubleTensorEquals(expected));
  EXPECT_THAT(init, DoubleTensorEquals(init_copy));
}

TEST(DoubleTensorDeathTest, ReshapeBadSize) {
  DoubleTensor tensor({{2.0, 3.0}, {4.0, 5.0}});
  ASSERT_DEATH(tensor.Reshape(Shape({5})), "");
}

TEST(TensorTest, VectorSlice) {
  const Tensor<std::string> tensor(
      std::vector<std::vector<std::string>>({{"a", "b"}, {"c", "d"}}));
  EXPECT_THAT(tensor.VectorSlice({0, -1}), ElementsAre("a", "b"));
  EXPECT_THAT(tensor.VectorSlice({1, -1}), ElementsAre("c", "d"));
  EXPECT_THAT(tensor.VectorSlice({-1, 0}), ElementsAre("a", "c"));
  EXPECT_THAT(tensor.VectorSlice({-1, 1}), ElementsAre("b", "d"));
}

TEST(TensorDeathTest, VectorSliceWrongSize) {
  const Tensor<std::string> tensor(
      std::vector<std::vector<std::string>>({{"a", "b"}, {"c", "d"}}));
  ASSERT_DEATH({ tensor.VectorSlice({-1}); }, "");
}

TEST(TensorDeathTest, VectorSliceNoFreeIndex) {
  const Tensor<std::string> tensor(
      std::vector<std::vector<std::string>>({{"a", "b"}, {"c", "d"}}));
  ASSERT_DEATH({ tensor.VectorSlice({1, 0}); }, "");
}

TEST(TensorDeathTest, VectorSliceTwoFreeIndices) {
  const Tensor<std::string> tensor(
      std::vector<std::vector<std::string>>({{"a", "b"}, {"c", "d"}}));
  ASSERT_DEATH({ tensor.VectorSlice({-1, -1}); }, "");
}

TEST(TensorDeathTest, VectorSliceIndexOutOfBounds) {
  const Tensor<std::string> tensor(
      std::vector<std::vector<std::string>>({{"a", "b"}, {"c", "d"}}));
  ASSERT_DEATH({ tensor.VectorSlice({2, -1}); }, "");
}

TEST(Tensor, SqueezeBasic) {
  const DoubleTensor t =
      DoubleTensor::FromFlatData(Shape({1, 3, 1, 1}), {2.0, 3.0, 4.0});
  const DoubleTensor expected = DoubleTensor({2.0, 3.0, 4.0});
  EXPECT_THAT(t.Squeeze(), DoubleTensorEquals(expected));
  DoubleTensor t2 = t;
  t2.SqueezeInPlace();
  EXPECT_THAT(t2, DoubleTensorEquals(expected));
}

TEST(Tensor, SqueezeToScalar) {
  const DoubleTensor t = DoubleTensor::FromFlatData(Shape({1, 1, 1}), {4.0});
  const DoubleTensor expected = DoubleTensor(4.0);
  EXPECT_THAT(t.Squeeze(), DoubleTensorEquals(expected));
  DoubleTensor t2 = t;
  t2.SqueezeInPlace();
  EXPECT_THAT(t2, DoubleTensorEquals(expected));
}

TEST(Tensor, SqueezeOnDims) {
  const DoubleTensor t =
      DoubleTensor::FromFlatData(Shape({1, 3, 1, 1}), {2.0, 3.0, 4.0});
  EXPECT_THAT(t.Squeeze({2, 3}), DoubleTensorEquals(DoubleTensor::FromFlatData(
                                     Shape({1, 3}), {2.0, 3.0, 4.0})));
  EXPECT_THAT(t.Squeeze({0, 3}), DoubleTensorEquals(DoubleTensor::FromFlatData(
                                     Shape({3, 1}), {2.0, 3.0, 4.0})));
  EXPECT_THAT(t.Squeeze({0, 2, 3}),
              DoubleTensorEquals(
                  DoubleTensor::FromFlatData(Shape({3}), {2.0, 3.0, 4.0})));
  EXPECT_THAT(t.Squeeze({0}), DoubleTensorEquals(DoubleTensor::FromFlatData(
                                  Shape({3, 1, 1}), {2.0, 3.0, 4.0})));
  DoubleTensor t2 = t;
  t2.SqueezeInPlace({0});
  EXPECT_THAT(t2, DoubleTensorEquals(DoubleTensor::FromFlatData(
                      Shape({3, 1, 1}), {2.0, 3.0, 4.0})));
}

TEST(Tensor, CanSqueeze) {
  const DoubleTensor t =
      DoubleTensor::FromFlatData(Shape({1, 3, 1, 1}), {2.0, 3.0, 4.0});
  TFOPT_EXPECT_OK(t.ValidateSqueeze({0}));
  TFOPT_EXPECT_OK(t.ValidateSqueeze({0, 2}));
  EXPECT_THAT(t.ValidateSqueeze({}),
              StatusIs(kInvalidArgument,
                       "Cannot call Squeeze(axes) with an empty axes list."));
  EXPECT_THAT(
      t.ValidateSqueeze({5}),
      StatusIs(kInvalidArgument, HasSubstr("all squeezed axes must fall in")));
  EXPECT_THAT(
      t.ValidateSqueeze({0, 5}),
      StatusIs(kInvalidArgument, HasSubstr("all squeezed axes must fall in")));
  EXPECT_THAT(
      t.ValidateSqueeze({-1}),
      StatusIs(kInvalidArgument, HasSubstr("all squeezed axes must fall in")));
  EXPECT_THAT(
      t.ValidateSqueeze({1}),
      StatusIs(kInvalidArgument,
               HasSubstr("all squeezed axes must have dimension size of 1")));
  EXPECT_THAT(
      t.ValidateSqueeze({0, 1}),
      StatusIs(kInvalidArgument,
               HasSubstr("all squeezed axes must have dimension size of 1")));
}

TEST(TensorDeathTest, Squeeze) {
  const DoubleTensor t =
      DoubleTensor::FromFlatData(Shape({1, 3, 1, 1}), {2.0, 3.0, 4.0});
  ASSERT_DEATH(t.Squeeze({0, 1}), "");
}

TEST(Tensor, CanExpandDims) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  TFOPT_EXPECT_OK(t.ValidateExpandDims(0));
  TFOPT_EXPECT_OK(t.ValidateExpandDims(1));
  TFOPT_EXPECT_OK(t.ValidateExpandDims(2));
  EXPECT_THAT(
      t.ValidateExpandDims(-1),
      StatusIs(kInvalidArgument, HasSubstr("To call ExpandDims on a tensor")));
  EXPECT_THAT(
      t.ValidateExpandDims(3),
      StatusIs(kInvalidArgument, HasSubstr("To call ExpandDims on a tensor")));
}

TEST(Tensor, ExpandDims) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  EXPECT_THAT(t.ExpandDims(0),
              DoubleTensorEquals(DoubleTensor::FromFlatData(
                  Shape({1, 2, 3}), {2.0, 3.0, 4.0, 5.0, 6.0, 7.0})));
  EXPECT_THAT(t.ExpandDims(1),
              DoubleTensorEquals(DoubleTensor::FromFlatData(
                  Shape({2, 1, 3}), {2.0, 3.0, 4.0, 5.0, 6.0, 7.0})));
  EXPECT_THAT(t.ExpandDims(2),
              DoubleTensorEquals(DoubleTensor::FromFlatData(
                  Shape({2, 3, 1}), {2.0, 3.0, 4.0, 5.0, 6.0, 7.0})));
  DoubleTensor t2 = t;
  t2.ExpandDimsInPlace(1);
  EXPECT_THAT(t2, DoubleTensorEquals(DoubleTensor::FromFlatData(
                      Shape({2, 1, 3}), {2.0, 3.0, 4.0, 5.0, 6.0, 7.0})));
}

TEST(TensorDeathTest, BadExpandDims) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  ASSERT_DEATH(t.ExpandDims(4), "");
}

TEST(Tensor, CanSlice) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  TFOPT_EXPECT_OK(t.ValidateSlice({0, 0}, {2, 3}));
  TFOPT_EXPECT_OK(t.ValidateSlice({0, 0}, {1, 1}));
  TFOPT_EXPECT_OK(t.ValidateSlice({1, 2}, {1, 1}));
  TFOPT_EXPECT_OK(t.ValidateSlice({1, 0}, {0, 0}));

  EXPECT_THAT(t.ValidateSlice({0, 0, 0}, {1, 1}),
              StatusIs(kInvalidArgument,
                       HasSubstr("begin_indices has 3 dimensions != 2")));

  EXPECT_THAT(
      t.ValidateSlice({0, 0}, {1}),
      StatusIs(kInvalidArgument, HasSubstr("sizes has 1 dimensions != 2")));

  EXPECT_THAT(t.ValidateSlice({0, -1}, {1, 1}),
              StatusIs(kInvalidArgument, HasSubstr("must be nonnegative")));

  EXPECT_THAT(t.ValidateSlice({0, 0}, {1, -1}),
              StatusIs(kInvalidArgument, HasSubstr("must be nonnegative")));

  EXPECT_THAT(
      t.ValidateSlice({0, 0}, {4, 3}),
      StatusIs(kInvalidArgument,
               HasSubstr("requesting out of bounds indices in tensor slice")));
  EXPECT_THAT(
      t.ValidateSlice({0, 2}, {1, 2}),
      StatusIs(kInvalidArgument,
               HasSubstr("requesting out of bounds indices in tensor slice")));
}

TEST(Tensor, Slice) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  EXPECT_THAT(t.Slice({0, 0}, {2, 3}), DoubleTensorEquals(t));
  EXPECT_THAT(t.Slice({0, 0}, {1, 1}),
              DoubleTensorEquals(DoubleTensor::CreateMatrix({{2.0}})));
  EXPECT_THAT(
      t.Slice({0, 1}, {2, 2}),
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{3.0, 4.0}, {6.0, 7.0}})));
  EXPECT_THAT(t.Slice({1, 2}, {1, 1}),
              DoubleTensorEquals(DoubleTensor::CreateMatrix({{7.0}})));
  EXPECT_THAT(t.Slice({1, 0}, {0, 0}),
              DoubleTensorEquals(DoubleTensor(Shape({0, 0}))));
}

TEST(TensorDeathTest, BadSlice) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  ASSERT_DEATH(t.Slice({0, 0, 0}, {1, 1}), "");
}

TEST(Tensor, SubTensorIndex) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  EXPECT_THAT(
      t.SubTensor(0, /*keep_dims=*/true),
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{2.0, 3.0, 4.0}})));
  EXPECT_THAT(t.SubTensor(0, /*keep_dims=*/false),
              DoubleTensorEquals(DoubleTensor({2.0, 3.0, 4.0})));
  EXPECT_THAT(
      t.SubTensor(1, /*keep_dims=*/true),
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{5.0, 6.0, 7.0}})));
  EXPECT_THAT(t.SubTensor(1, /*keep_dims=*/false),
              DoubleTensorEquals(DoubleTensor({5.0, 6.0, 7.0})));
}

TEST(Tensor, SubTensorOutputArg) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  DoubleTensor target(Shape({3}));
  t.SubTensor(0, &target, /*keep_dims=*/false);
  EXPECT_THAT(target, DoubleTensorEquals(DoubleTensor({2.0, 3.0, 4.0})));
}

TEST(Tensor, SubTensorBig) {
  const DoubleTensor t(
      {{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}}, {{2.1, 3.1, 4.1}, {5.1, 6.1, 7.1}}});
  EXPECT_THAT(t.SubTensor(0, /*keep_dims=*/false),
              DoubleTensorEquals(DoubleTensor::CreateMatrix(
                  {{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}})));
}

TEST(Tensor, SubTensorRange) {
  const DoubleTensor t({{2.0, 3.0}, {4.0, 5.0}, {6.0, 7.0}});
  EXPECT_THAT(t.SubTensor(0, 1),
              DoubleTensorEquals(DoubleTensor::CreateMatrix({{2.0, 3.0}})));
  EXPECT_THAT(
      t.SubTensor(0, 2),
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{2.0, 3.0}, {4.0, 5.0}})));
  EXPECT_THAT(
      t.SubTensor(1, 2),
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{4.0, 5.0}, {6.0, 7.0}})));
  EXPECT_THAT(t.SubTensor(0, 3), DoubleTensorEquals(t));
}

TEST(Tensor, SubTensorRangeOutputArgCorrectShape) {
  const DoubleTensor t({{2.0, 3.0}, {4.0, 5.0}, {6.0, 7.0}});
  DoubleTensor result(Shape({2, 2}));
  t.SubTensor(1, 2, &result);
  const auto expected = DoubleTensor::CreateMatrix({{4.0, 5.0}, {6.0, 7.0}});
  EXPECT_THAT(result, DoubleTensorEquals(expected));
  DoubleTensor result_needs_reshape(Shape({4}));
  t.SubTensor(1, 2, &result_needs_reshape);
  EXPECT_THAT(result, DoubleTensorEquals(expected));
}

TEST(Tensor, SubTensorRangeOutputArgWrongShape) {
  const DoubleTensor t({{2.0, 3.0}, {4.0, 5.0}, {6.0, 7.0}});
  DoubleTensor result;
  t.SubTensor(1, 2, &result);
  EXPECT_THAT(
      result,
      DoubleTensorEquals(DoubleTensor::CreateMatrix({{4.0, 5.0}, {6.0, 7.0}})));
}

TEST(TensorDeathTest, SubTensorRangeScalar) {
  const DoubleTensor t(5.0);

  EXPECT_DEATH(t.SubTensor(0, 1),
               HasSubstr("SubTensor() cannot be called on scalars."));
}

TEST(TensorDeathTest, SubTensorRangeOutOfBounds) {
  const DoubleTensor t({5.0, 6.0, 7.0});
  EXPECT_DEATH(t.SubTensor(1, 3),
               HasSubstr("exceeds first dimension of tensor"));
}

TEST(DoubleTensorTest, DeprecatedDoubleTensorToProto) {
  const DoubleTensor t(
      vector<vector<double>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
  proto::ParameterValue p;
  DoubleTensorToProto(t, &p);
  EXPECT_THAT(p.dimension().dim_sizes(), ElementsAre(2, 3));
  EXPECT_THAT(p.value(), ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
}

TEST(DoubleTensorTest, DeprecatedDoubleTensorFromProto) {
  proto::ParameterValue p;
  p.mutable_dimension()->add_dim_sizes(2);
  p.mutable_dimension()->add_dim_sizes(3);
  p.add_value(1.0);
  p.add_value(2.0);
  p.add_value(3.0);
  p.add_value(4.0);
  p.add_value(5.0);
  p.add_value(6.0);
  DoubleTensor t = ProtoToDoubleTensor(p);
  const DoubleTensor expected(
      vector<vector<double>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
  EXPECT_EQ(t, expected);
}

TEST(DoubleTensorTest, DoubleTensorToProto) {
  const DoubleTensor t({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  const DoubleTensorProto p = DoubleTensorToProto(t);
  EXPECT_THAT(p.shape().dimensions(), ElementsAre(2, 3));
  EXPECT_THAT(p.values(), ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
}

TEST(DoubleTensorTest, DoubleTensorFromProto) {
  DoubleTensorProto p;
  p.mutable_shape()->add_dimensions(2);
  p.mutable_shape()->add_dimensions(3);
  p.add_values(1.0);
  p.add_values(2.0);
  p.add_values(3.0);
  p.add_values(4.0);
  p.add_values(5.0);
  p.add_values(6.0);
  DoubleTensor t = ProtoToDoubleTensor(p);
  const DoubleTensor expected({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  EXPECT_THAT(t, DoubleTensorEquals(expected));
}

TEST(DoubleTensorTest, HasInfiniteOrNan) {
  DoubleTensor t({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  EXPECT_FALSE(HasInfiniteOrNan(t));
  t.set_value({0, 1}, -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(HasInfiniteOrNan(t));
  t.set_value({0, 1}, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(HasInfiniteOrNan(t));
  t.set_value({0, 1}, std::nan(""));
  EXPECT_TRUE(HasInfiniteOrNan(t));
  t.set_value({0, 1}, 8.0);
  EXPECT_FALSE(HasInfiniteOrNan(t));
}

TEST(TensorTest, TensorDimension) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  EXPECT_EQ(TensorDimension(t), Shape({2, 3}));
}

TEST(TensorTest, TensorSize) {
  const DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  EXPECT_EQ(TensorSize(t), 6);
}

TEST(TensorTest, TensorReshapeInPlace) {
  DoubleTensor t({{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}});
  TensorReshapeInPlace(&t, Shape({3, 2}));
  const DoubleTensor expected =
      DoubleTensor::CreateMatrix({{2.0, 3.0}, {4.0, 5.0}, {6.0, 7.0}});
  EXPECT_THAT(t, DoubleTensorEquals(expected));
}

TEST(BoundsTensorTest, TensorToString) {
  Tensor<Bounds> tensor = Tensor<Bounds>::FromFlatData(
      Shape({2, 2}),
      {Bounds(-1, 1), Bounds(-2, 2), Bounds(-3, 3), Bounds(-4, 4)});
  EXPECT_EQ(tensor.ToString(),
            "shape: 2,2, values: [[-1,1], [-2,2], [-3,3], [-4,4]]");
}

}  // namespace
}  // namespace tf_opt
