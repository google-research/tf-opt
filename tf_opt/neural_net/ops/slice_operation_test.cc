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

#include "tf_opt/neural_net/ops/slice_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(SliceOperationTest, SimpleCreate) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, SliceOperation::Create("s1", input_shape, begin, sizes));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.begin(), begin);
  EXPECT_EQ(op.sizes(), sizes);
}

TEST(SliceOperationTest, SimpleCreateInvalidSize) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({4, 2});
  EXPECT_THAT(SliceOperation::Create("s1", input_shape, begin, sizes),
              StatusIs(kInvalidArgument));
}

Operation::Options MakeOptions(const std::vector<int64_t>& begin,
                               const std::vector<int64_t>& sizes) {
  Operation::Options options;
  options.integer_list_options[SliceOperation::kOptionsBeginKey] = begin;
  options.integer_list_options[SliceOperation::kOptionsSizeKey] = sizes;
  return options;
}

TEST(SliceOperationTest, GenericCreate) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      SliceOperation::GenericCreate("s1", {input_shape}, output_shape,
                                    MakeOptions(begin, sizes)));
  EXPECT_THAT(op, OperationArgsAre("s1", {input_shape}, output_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.begin(), begin);
  EXPECT_EQ(op.sizes(), sizes);
}

TEST(SliceOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  EXPECT_THAT(
      SliceOperation::GenericCreate("s1", {input_shape, input_shape},
                                    output_shape, MakeOptions(begin, sizes)),
      StatusIs(kInvalidArgument));
}

TEST(SliceOperationTest, GenericCreateBadOption) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  auto options = MakeOptions(begin, sizes);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(
      SliceOperation::GenericCreate("s1", {input_shape}, output_shape, options),
      StatusIs(kInvalidArgument));
}

TEST(SliceOperationTest, GenericCreateMissingBegin) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  auto options = MakeOptions(begin, sizes);
  options.integer_list_options.erase(SliceOperation::kOptionsBeginKey);
  EXPECT_THAT(
      SliceOperation::GenericCreate("s1", {input_shape}, output_shape, options),
      StatusIs(kInvalidArgument));
}

TEST(SliceOperationTest, GenericCreateMissingSizes) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 2});
  auto options = MakeOptions(begin, sizes);
  options.integer_list_options.erase(SliceOperation::kOptionsSizeKey);
  EXPECT_THAT(
      SliceOperation::GenericCreate("s1", {input_shape}, output_shape, options),
      StatusIs(kInvalidArgument));
}

TEST(SliceOperationTest, GenericCreateInvalidSize) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({4, 2});
  const Shape output_shape({4, 2});
  EXPECT_THAT(SliceOperation::GenericCreate("s1", {input_shape}, output_shape,
                                            MakeOptions(begin, sizes)),
              StatusIs(kInvalidArgument));
}

TEST(SliceOperationTest, GenericCreateBadOutputShape) {
  const Shape input_shape({3, 3});
  const std::vector<int64_t> begin({1, 1});
  const std::vector<int64_t> sizes({2, 2});
  const Shape output_shape({2, 6});
  EXPECT_THAT(SliceOperation::GenericCreate("s1", {input_shape}, output_shape,
                                            MakeOptions(begin, sizes)),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
