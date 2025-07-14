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

#include "tf_opt/tensor/embedding_lookup.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/bounds/bounds.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

class EmbeddingLookupOutputShapeTest : public ::testing::Test {
 public:
  EmbeddingLookupOutputShapeTest()
      : kParamsShape({kNumClasses, kEmbeddingDimension}),
        kIdsShape({kBatchSize, kNumLookups, kNumClasses}),
        kResultShape({kBatchSize, kNumLookups, kEmbeddingDimension}) {}

 protected:
  static constexpr int kNumLookups = 3;
  static constexpr int kNumClasses = 100;
  static constexpr int kEmbeddingDimension = 10;
  static constexpr int kBatchSize = 1;
  const Shape kParamsShape;
  const Shape kIdsShape;
  const Shape kResultShape;
};

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeSimple) {
  EXPECT_THAT(EmbeddingLookupOutputShape(kParamsShape, kIdsShape),
              IsOkAndHolds(kResultShape));
}

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeMatrixOut) {
  const int embedding_rows_out = 10;
  const int embedding_cols_out = 10;
  const Shape params_shape(
      {kNumClasses, embedding_rows_out, embedding_cols_out});
  const Shape ids_shape({kBatchSize, kNumLookups, kNumClasses});
  const Shape expected_result_shape(
      {kBatchSize, kNumLookups, embedding_rows_out, embedding_cols_out});
  EXPECT_THAT(EmbeddingLookupOutputShape(params_shape, ids_shape),
              IsOkAndHolds(expected_result_shape));
}

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeMultidimensionalInput) {
  const int num_lookups_rows = 3;
  const int num_lookups_cols = 5;
  const Shape params_shape({kNumClasses, kEmbeddingDimension});
  const Shape ids_shape(
      {kBatchSize, num_lookups_rows, num_lookups_cols, kNumClasses});
  const Shape expected_result_shape(
      {kBatchSize, num_lookups_rows, num_lookups_cols, kEmbeddingDimension});
  EXPECT_THAT(EmbeddingLookupOutputShape(params_shape, ids_shape),
              IsOkAndHolds(expected_result_shape));
}

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeBadParamsRank) {
  // params_shape below should be num_classes by embedding_dimension.
  const Shape bad_params_shape({kEmbeddingDimension});
  EXPECT_THAT(EmbeddingLookupOutputShape(bad_params_shape, kIdsShape),
              StatusIs(kInvalidArgument,
                       "Rank of params must be at least two, found: 1"));
}

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeBadIdsRank) {
  // ids_shape should be num_lookups by num_classes.
  const Shape bad_ids_shape({kNumClasses});
  EXPECT_THAT(
      EmbeddingLookupOutputShape(kParamsShape, bad_ids_shape),
      StatusIs(kInvalidArgument, "Rank of ids must be at least two, found: 1"));
}

TEST_F(EmbeddingLookupOutputShapeTest, OutputShapeMismatched) {
  const Shape kIncompatibleIdsShape({kBatchSize, kNumLookups, kNumClasses + 2});
  EXPECT_THAT(EmbeddingLookupOutputShape(kParamsShape, kIncompatibleIdsShape),
              StatusIs(kInvalidArgument, "Incompatible ids and params shapes"));
}

TEST(EmbeddingLookupTest, SimpleEmbedding1Lookup) {
  const auto ids = DoubleTensor::CreateMatrix({{1.0, 0.0, 0.0}});
  const DoubleTensor weights({{-0.2, -0.1}, {-0.3, 0.6}, {-1.0, 0.0}});
  const auto expected_output = DoubleTensor::CreateMatrix({{-0.2, -0.1}});
  EXPECT_THAT(EmbeddingLookup<double>(weights, ids),
              DoubleTensorNear(expected_output));
}

TEST(EmbeddingLookupTest, SimpleEmbedding1Lookup_BoundsTensor) {
  const auto ids =
      BoundsTensor::CreateMatrix({{Bounds(1.0), Bounds(0.0), Bounds(0.0)}});
  const DoubleTensor weights({{-0.2, -0.1}, {-0.3, 0.6}, {-1.0, 0.0}});
  const auto expected_output =
      BoundsTensor::CreateMatrix({{Bounds(-0.2), Bounds(-0.1)}});

  EXPECT_THAT(EmbeddingLookup<Bounds>(weights, ids),
              BoundsTensorNear(expected_output));
}

TEST(EmbeddingLookupTest, SimpleEmbedding2Lookups) {
  const DoubleTensor ids({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const DoubleTensor weights({{-0.2, -0.1}, {-0.3, 0.6}, {-1.0, 0.0}});
  const DoubleTensor expected_output =
      DoubleTensor::CreateMatrix({{-0.3, 0.6}, {-1.0, 0.0}});
  EXPECT_THAT(EmbeddingLookup<double>(weights, ids),
              DoubleTensorNear(expected_output));
}

}  // namespace
}  // namespace tf_opt
