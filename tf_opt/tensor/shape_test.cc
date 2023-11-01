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

#include "tf_opt/tensor/shape.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/open_source/protocol_buffer_matchers.h"

namespace tf_opt {
namespace {

using ::std::vector;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

TEST(ShapeTest, ScalarDimensionDefault) {
  const Shape shape;
  EXPECT_EQ(1, shape.size());
  EXPECT_EQ(0, shape.num_dimensions());
  EXPECT_THAT(shape.dimension_sizes(), IsEmpty());
  EXPECT_EQ(shape.FlattenIndex({}), 0);
  EXPECT_THAT(shape.ExpandIndex(0), IsEmpty());
  EXPECT_EQ(shape, Shape());
  EXPECT_FALSE(shape != Shape());
  EXPECT_NE(shape, Shape({3}));
  EXPECT_FALSE(shape == Shape({3}));
}

TEST(ShapeTest, ScalarDimensionEmpty) {
  const Shape shape1(std::vector<int64_t>{});
  EXPECT_EQ(1, shape1.size());
  EXPECT_EQ(0, shape1.num_dimensions());
  const Shape shape2;
  EXPECT_EQ(shape1, shape2);
}

TEST(ShapeTest, MultiIndexIsValidSimple) {
  const Shape shape({4, 6, 2});
  EXPECT_TRUE(shape.MultiIndexIsValid({0, 0, 0}));
  EXPECT_TRUE(shape.MultiIndexIsValid({1, 1, 1}));
  EXPECT_TRUE(shape.MultiIndexIsValid({3, 5, 1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({3, 7, 1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({3, 5, -1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({3, 5, 1, 1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({0, 0, 0, 0}));
  EXPECT_FALSE(shape.MultiIndexIsValid({0, 0}));
  EXPECT_FALSE(shape.MultiIndexIsValid({}));
}

TEST(ShapeTest, MultiIndexIsValidScalar) {
  const Shape shape;
  EXPECT_TRUE(shape.MultiIndexIsValid({}));
  EXPECT_FALSE(shape.MultiIndexIsValid({0}));
  EXPECT_FALSE(shape.MultiIndexIsValid({-1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({1}));
  EXPECT_FALSE(shape.MultiIndexIsValid({0, 0}));
}

TEST(ShapeTest, ScalarDimensionEqualtiy) {
  const Shape shape;
  EXPECT_EQ(shape, Shape());
  EXPECT_FALSE(shape != Shape());
  EXPECT_NE(shape, Shape({3}));
  EXPECT_FALSE(shape == Shape({3}));
}

TEST(ShapeTest, ScalarDimensionDeprecatedProtoRoundTrip) {
  const Shape shape1;
  const Shape shape2 = Shape(shape1.AsProto());
  EXPECT_EQ(1, shape2.size());
  EXPECT_THAT(shape2.dimension_sizes(), IsEmpty());
  EXPECT_EQ(shape1, shape2);
}

TEST(ShapeTest, ScalarDimensionProtoRoundTrip) {
  const Shape shape1;
  const Shape shape2 = Shape(shape1.AsShapeProto());
  EXPECT_EQ(1, shape2.size());
  EXPECT_THAT(shape2.dimension_sizes(), IsEmpty());
  EXPECT_EQ(shape1, shape2);
}

TEST(ShapeTest, SingleDimension) {
  const Shape shape({7});
  EXPECT_EQ(7, shape.size());
  EXPECT_EQ(1, shape.num_dimensions());
  EXPECT_THAT(shape.dimension_sizes(), ElementsAre(7));
  for (int i = 0; i < 7; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(i, shape.FlattenIndex({i}));
    EXPECT_THAT(shape.ExpandIndex(i), ElementsAre(i));
  }
}

TEST(ShapeDeathTest, SingleDimensionSizeNegative) {
  ASSERT_DEATH(Shape({-3}), "");
}

TEST(ShapeDeathTest, singleDimensionIndexNegative) {
  const Shape shape({7});
  ASSERT_DEATH(shape.FlattenIndex({-2}), "");
}

TEST(ShapeDeathTest, SingleDimensionIndexBig) {
  const Shape shape({7});
  ASSERT_DEATH(shape.FlattenIndex({7}), "");
}

TEST(ShapeDeathTest, SingleDimensionIndexLowDim) {
  const Shape shape({7});
  ASSERT_DEATH(shape.FlattenIndex({}), "");
}

TEST(ShapeDeathTest, SingleDimensionIndexHighDim) {
  const Shape shape({7});
  ASSERT_DEATH(shape.FlattenIndex({1, 2}), "");
}

TEST(ShapeTest, SecondDimension) {
  const Shape shape({7, 5});
  EXPECT_EQ(35, shape.size());
  EXPECT_EQ(2, shape.num_dimensions());
  EXPECT_THAT(shape.dimension_sizes(), ElementsAre(7, 5));
  EXPECT_EQ(0, shape.FlattenIndex({0, 0}));
  EXPECT_THAT(shape.ExpandIndex(0), ElementsAre(0, 0));
  EXPECT_EQ(1, shape.FlattenIndex({0, 1}));
  EXPECT_THAT(shape.ExpandIndex(1), ElementsAre(0, 1));
  EXPECT_EQ(4, shape.FlattenIndex({0, 4}));
  EXPECT_THAT(shape.ExpandIndex(4), ElementsAre(0, 4));
  EXPECT_EQ(5, shape.FlattenIndex({1, 0}));
  EXPECT_THAT(shape.ExpandIndex(5), ElementsAre(1, 0));
  EXPECT_EQ(9, shape.FlattenIndex({1, 4}));
  EXPECT_THAT(shape.ExpandIndex(9), ElementsAre(1, 4));
  EXPECT_EQ(34, shape.FlattenIndex({6, 4}));
  EXPECT_THAT(shape.ExpandIndex(34), ElementsAre(6, 4));
}

TEST(ShapeTest, RoundTrip) {
  const Shape shape({3, 6, 4});
  for (int64_t i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(i, shape.FlattenIndex(shape.ExpandIndex(i)));
  }
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 6; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        const vector<int64_t> multi_index({i, j, k});
        EXPECT_THAT(shape.ExpandIndex(shape.FlattenIndex(multi_index)),
                    ElementsAre(i, j, k));
      }
    }
  }
}

TEST(ShapeDeathTest, SecondDimensionFirstIndexBig) {
  const Shape shape({7, 5});
  ASSERT_DEATH(shape.FlattenIndex({7, 3}), "");
}

TEST(ShapeDeathTest, SecondDimensionSecondIndexBig) {
  const Shape shape({7, 5});
  ASSERT_DEATH(shape.FlattenIndex({3, 6}), "");
}

TEST(ShapeTest, DeprecatedProtoToDim) {
  proto::Dimension proto_dim;
  proto_dim.add_dim_sizes(5);
  proto_dim.add_dim_sizes(3);
  proto_dim.add_dim_sizes(4);
  const Shape shape(proto_dim);
  EXPECT_EQ(60, shape.size());
  EXPECT_THAT(shape.dimension_sizes(), ElementsAre(5, 3, 4));
}

TEST(ShapeTest, DeprecatedProtoRoundTrip) {
  proto::Dimension proto_dim;
  proto_dim.add_dim_sizes(5);
  proto_dim.add_dim_sizes(3);
  proto_dim.add_dim_sizes(4);
  const Shape shape(proto_dim);
  const proto::Dimension round_trip = shape.AsProto();
  EXPECT_THAT(proto_dim, ::tf_opt::testing::EqualsProto(round_trip));
}

TEST(ShapeTest, ProtoToDim) {
  ShapeProto shape_proto;
  shape_proto.add_dimensions(5);
  shape_proto.add_dimensions(3);
  shape_proto.add_dimensions(4);
  const Shape shape(shape_proto);
  EXPECT_EQ(60, shape.size());
  EXPECT_THAT(shape.dimension_sizes(), ElementsAre(5, 3, 4));
}

TEST(ShapeTest, ProtoRoundTrip) {
  ShapeProto shape_proto;
  shape_proto.add_dimensions(5);
  shape_proto.add_dimensions(3);
  shape_proto.add_dimensions(4);
  const Shape shape(shape_proto);
  const ShapeProto round_trip = shape.AsShapeProto();
  EXPECT_THAT(shape_proto, ::tf_opt::testing::EqualsProto(round_trip));
}

TEST(ShapeTest, OperatorsWhenEqual) {
  const Shape a({3, 6, 2});
  const Shape b({3, 6, 2});
  EXPECT_EQ(a, b);
  // NOTE: we explicitly want to make sure the != operator runs,
  // we cannot use any version of EXPECT_EQ/EXPECT_NE.
  EXPECT_FALSE(a != b);
}

TEST(ShapeTest, OperatorsWhenNotEqual) {
  const Shape a({3, 6, 4});
  const Shape b({3, 6, 2});
  EXPECT_FALSE(a == b);
  // See above note.
  EXPECT_NE(a, b);
}

TEST(ShapeTest, ToString) {
  const Shape shape({3, 6, 4});
  const std::string dim_string = shape.ToString();
  EXPECT_THAT(dim_string, HasSubstr("3"));
  EXPECT_THAT(dim_string, HasSubstr("6"));
  EXPECT_THAT(dim_string, HasSubstr("4"));
  EXPECT_THAT(dim_string, Not(HasSubstr("17")));
}

TEST(ShapeTest, StreamOp) {
  const Shape shape({3, 6, 4});
  std::ostringstream stream;
  stream << shape;
  std::string s = stream.str();
  EXPECT_THAT(s, HasSubstr("3"));
  EXPECT_THAT(s, HasSubstr("6"));
  EXPECT_THAT(s, HasSubstr("4"));
  EXPECT_THAT(s, Not(HasSubstr("17")));
}

TEST(ShapeDeathTest, DeprecatedProtoToDimBadData) {
  proto::Dimension proto_dim;
  proto_dim.add_dim_sizes(5);
  proto_dim.add_dim_sizes(-2);
  proto_dim.add_dim_sizes(4);
  ASSERT_DEATH(Shape{proto_dim}, "");
}

TEST(ShapeDeathTest, ProtoToDimBadData) {
  ShapeProto proto_dim;
  proto_dim.add_dimensions(5);
  proto_dim.add_dimensions(-2);
  proto_dim.add_dimensions(4);
  ASSERT_DEATH(Shape{proto_dim}, "");
}

TEST(ShapeTest, FromVector) {
  Shape shape = Shape::FromVector(std::vector<int>({100, 3, 1}));
  EXPECT_EQ(shape, Shape({3}));
}

TEST(ShapeTest, FromVector2D) {
  Shape shape = Shape::FromVector2D(
      std::vector<std::vector<int>>({{100, 3, 1}, {0, 0, 0}}));
  EXPECT_EQ(shape, Shape({2, 3}));
}

TEST(ShapeTest, FromVector3D) {
  Shape shape = Shape::FromVector3D(std::vector<std::vector<std::vector<int>>>(
      {{{100, 3, 1}, {0, 0, 0}}, {{10, 10, 10}, {10, 10, 10}}}));
  EXPECT_EQ(shape, Shape({2, 2, 3}));
}

TEST(ShapeDeathTest, FromVector2DRagged) {
  ASSERT_DEATH(
      {
        Shape::FromVector2D(
            std::vector<std::vector<int>>({{100, 3, 1}, {0, 0}}));
      },
      "");
}

TEST(ShapeTest, FromVector3DRaggedColumns) {
  ASSERT_DEATH(
      {
        Shape::FromVector3D(std::vector<std::vector<std::vector<int>>>(
            {{{100, 3, 1}, {0, 0, 0}}, {{10, 10, 10}}}));
      },
      "");
}

TEST(ShapeTest, FromVector3DRaggedRows) {
  ASSERT_DEATH(
      {
        Shape::FromVector3D(std::vector<std::vector<std::vector<int>>>(
            {{{100, 3, 1}, {0, 0, 0, 0}}, {{10, 10, 10}, {10, 10, 10}}}));
      },
      "");
}

}  // namespace
}  // namespace tf_opt
