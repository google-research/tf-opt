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

#include "tf_opt/tensor/batch_iterator.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::StrEq;
using ::testing::UnorderedElementsAre;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(BatchIteratorTest, SimpleIteration) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0, 7.0})}});
  BatchIterator<double> it(&features, 2);
  EXPECT_EQ(it.dataset_size(), 5);
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 2);
  EXPECT_THAT(it.current_batch(),
              UnorderedElementsAre(Pair(
                  StrEq("x"), DoubleTensorEquals(DoubleTensor({3.0, 4.0})))));

  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 2);
  EXPECT_THAT(it.current_batch(),
              UnorderedElementsAre(Pair(
                  StrEq("x"), DoubleTensorEquals(DoubleTensor({5.0, 6.0})))));

  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 1);
  EXPECT_THAT(
      it.current_batch(),
      UnorderedElementsAre(Pair(
          StrEq("x"), DoubleTensorEquals(DoubleTensor::CreateVector({7.0})))));

  ASSERT_FALSE(it.Advance());
  ASSERT_FALSE(it.Advance());
}

TEST(BatchIteratorTest, Reset) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0, 7.0})}});
  BatchIterator<double> it(&features, 2);
  ASSERT_TRUE(it.Advance());
  ASSERT_TRUE(it.Advance());
  ASSERT_TRUE(it.Advance());
  ASSERT_FALSE(it.Advance());
  it.Reset();
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 2);
  EXPECT_THAT(it.current_batch(),
              UnorderedElementsAre(Pair(
                  StrEq("x"), DoubleTensorEquals(DoubleTensor({3.0, 4.0})))));
  ASSERT_TRUE(it.Advance());
  ASSERT_TRUE(it.Advance());
  ASSERT_FALSE(it.Advance());
}

TEST(BatchIteratorTest, BigBatch) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0, 7.0})}});
  BatchIterator<double> it(&features, 10);
  EXPECT_EQ(it.dataset_size(), 5);
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 5);
  EXPECT_THAT(
      it.current_batch(),
      UnorderedElementsAre(Pair(StrEq("x"), DoubleTensorEquals(DoubleTensor(
                                                {3.0, 4.0, 5.0, 6.0, 7.0})))));
  ASSERT_FALSE(it.Advance());
}

TEST(BatchIteratorTest, MultipleInputs) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0})},
       {"y", DoubleTensor({3.1, 4.1, 5.1, 6.1})}});
  BatchIterator<double> it(&features, 2);
  EXPECT_EQ(it.dataset_size(), 4);
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 2);
  EXPECT_THAT(
      it.current_batch(),
      UnorderedElementsAre(
          Pair(StrEq("x"), DoubleTensorEquals(DoubleTensor({3.0, 4.0}))),
          Pair(StrEq("y"), DoubleTensorEquals(DoubleTensor({3.1, 4.1})))));
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 2);
  EXPECT_THAT(
      it.current_batch(),
      UnorderedElementsAre(
          Pair(StrEq("x"), DoubleTensorEquals(DoubleTensor({5.0, 6.0}))),
          Pair(StrEq("y"), DoubleTensorEquals(DoubleTensor({5.1, 6.1})))));
  ASSERT_FALSE(it.Advance());
}

TEST(BatchIteratorTest, HighDimension) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x",
        DoubleTensor({{{3.0, 4.0}, {5.0, 6.0}}, {{7.0, 8.0}, {9.0, 10.0}}})}});
  BatchIterator<double> it(&features, 1);
  EXPECT_EQ(it.dataset_size(), 2);
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 1);
  EXPECT_THAT(it.current_batch(),
              UnorderedElementsAre(Pair(
                  StrEq("x"), DoubleTensorEquals(DoubleTensor::FromFlatData(
                                  Shape({1, 2, 2}), {3.0, 4.0, 5.0, 6.0})))));
  ASSERT_TRUE(it.Advance());
  EXPECT_EQ(it.current_batch_size(), 1);
  EXPECT_THAT(it.current_batch(),
              UnorderedElementsAre(Pair(
                  StrEq("x"), DoubleTensorEquals(DoubleTensor::FromFlatData(
                                  Shape({1, 2, 2}), {7.0, 8.0, 9.0, 10.0})))));
  ASSERT_FALSE(it.Advance());
}

TEST(BatchIteratorTest, DatasetSizeOneInput) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0})}});
  EXPECT_THAT(BatchIterator<double>::CanBatchAndDatasetSize(features),
              IsOkAndHolds(4));
}

TEST(BatchIteratorTest, DatasetSizeMultiInput) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0, 5.0, 6.0})},
       {"y", DoubleTensor({3.1, 4.1, 5.1, 6.1})}});
  EXPECT_THAT(BatchIterator<double>::CanBatchAndDatasetSize(features),
              IsOkAndHolds(4));
}

TEST(BatchIteratorTest, DatasetSizeScalarInput) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor(3.0)}});
  EXPECT_THAT(
      BatchIterator<double>::CanBatchAndDatasetSize(features),
      StatusIs(kInvalidArgument,
               HasSubstr("all features should have at least one dimension")));
}

TEST(BatchIteratorTest, DatasetSizeDisagreement) {
  const absl::flat_hash_map<std::string, DoubleTensor> features(
      {{"x", DoubleTensor({3.0, 4.0})}, {"y", DoubleTensor({3.1, 4.1, 5.1})}});
  EXPECT_THAT(BatchIterator<double>::CanBatchAndDatasetSize(features),
              StatusIs(kInvalidArgument, HasSubstr("to match")));
}

TEST(BatchIteratorTest, EmptyFeatures) {
  const absl::flat_hash_map<std::string, DoubleTensor> features;
  EXPECT_THAT(BatchIterator<double>::CanBatchAndDatasetSize(features),
              IsOkAndHolds(0));
  BatchIterator<double> iterator(&features, 1);
  EXPECT_EQ(iterator.dataset_size(), 0);
  EXPECT_FALSE(iterator.Advance());
}

}  // namespace
}  // namespace tf_opt
