// Copyright 2021 The tf.opt Authors.
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

#include "tf_opt/neural_net/ops/embedding_lookup_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

constexpr int kNumLookups = 3;
constexpr int kNumClasses = 100;
constexpr int kEmbeddingDimension = 10;
constexpr int kBatchSize = 1;

Shape ParamsShape() { return Shape({kNumClasses, kEmbeddingDimension}); }

Shape IdsShape() { return Shape({kBatchSize, kNumLookups, kNumClasses}); }

Shape IncompatibleIdsShape() {
  return Shape({kBatchSize, kNumLookups, kNumClasses + 2});
}

Shape ResultShape() {
  return Shape({kBatchSize, kNumLookups, kEmbeddingDimension});
}

TEST(EmbeddingLookupOperationTest, OutputShapeSimple) {
  EXPECT_THAT(EmbeddingLookupOperation::OutputShape(ParamsShape(), IdsShape()),
              IsOkAndHolds(ResultShape()));
}



TEST(EmbeddingLookupOperationTest, SimpleCreate) {
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, EmbeddingLookupOperation::Create(
                         "embedding_lookup1", ParamsShape(), IdsShape()));
  EXPECT_EQ(op.params(), ParamsShape());
  EXPECT_EQ(op.ids(), IdsShape());
  EXPECT_THAT(op, OperationArgsAre("embedding_lookup1",
                                   {ParamsShape(), IdsShape()}, ResultShape()));
}

TEST(EmbeddingLookupOperationTest, SimpleCreateIncompatibleShapes) {
  EXPECT_THAT(EmbeddingLookupOperation::Create(
                  "embedding_lookup1", ParamsShape(), IncompatibleIdsShape()),
              StatusIs(kInvalidArgument));
}

TEST(EmbeddingLookupOperationTest, GenericCreate) {
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, EmbeddingLookupOperation::GenericCreate(
                         "embedding_lookup1", {ParamsShape(), IdsShape()},
                         ResultShape(), Operation::Options()));
  EXPECT_EQ(op.params(), ParamsShape());
  EXPECT_EQ(op.ids(), IdsShape());
  EXPECT_THAT(op, OperationArgsAre("embedding_lookup1",
                                   {ParamsShape(), IdsShape()}, ResultShape()));
}

TEST(EmbeddingLookupOperationTest, GenericCreateWrongNumberInputs) {
  EXPECT_THAT(EmbeddingLookupOperation::GenericCreate(
                  "embedding_lookup1", {ParamsShape()}, ResultShape(),
                  Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(EmbeddingLookupOperationTest, GenericCreateBadOption) {
  Operation::Options options;
  options.integer_options["bad_key"] = 2;
  EXPECT_THAT(EmbeddingLookupOperation::GenericCreate(
                  "embedding_lookup1", {ParamsShape(), IdsShape()},
                  ResultShape(), options),
              StatusIs(kInvalidArgument));
}

TEST(EmbeddingLookupOperationTest, GenericCreateIncompatibleInputShapes) {
  EXPECT_THAT(EmbeddingLookupOperation::GenericCreate(
                  "embedding_lookup1", {ParamsShape(), IncompatibleIdsShape()},
                  ResultShape(), Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(EmbeddingLookupOperationTest, GenericCreateBadResultShape) {
  EXPECT_THAT(EmbeddingLookupOperation::GenericCreate(
                  "embedding_lookup1", {ParamsShape(), IdsShape()},
                  Shape({2, 3}), Operation::Options()),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
