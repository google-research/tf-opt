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

#include "tf_opt/neural_net/ops/reduce_operations.h"

#include <memory>
#include <string>

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

template <typename OpType>
class LinearReduceOperationTest : public ::testing::Test {};

template <typename OpType>
class NonlinearReduceOperationTest : public ::testing::Test {};

using LinearOpTestTypes =
    ::testing::Types<ReduceMeanOperation, ReduceSumOperation>;
using NonlinearOpTestTypes =
    ::testing::Types<ReduceMaxOperation, ReduceMinOperation>;

TYPED_TEST_SUITE(LinearReduceOperationTest, LinearOpTestTypes);
TYPED_TEST_SUITE(NonlinearReduceOperationTest, NonlinearOpTestTypes);

TYPED_TEST(LinearReduceOperationTest, SimpleCreate) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {1};
  const Shape expected_result({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op,
                             TypeParam::Create("reduce1", input, axes));
  EXPECT_THAT(op, OperationArgsAre("reduce1", {input}, expected_result));
  EXPECT_EQ(op.input(), input);
  EXPECT_EQ(op.axes(), axes);
}

TYPED_TEST(NonlinearReduceOperationTest, SimpleCreate) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {1};
  const Shape expected_result({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      TypeParam::Create("reduce1", input, axes,
                        MaximumImplementationType::kOptimalBigM));
  EXPECT_THAT(op, OperationArgsAre("reduce1", {input}, expected_result));
  EXPECT_EQ(op.input(), input);
  EXPECT_EQ(op.axes(), axes);
  EXPECT_EQ(op.formulation(), MaximumImplementationType::kOptimalBigM);
}

TYPED_TEST(LinearReduceOperationTest, CreateBadInput) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {10};
  EXPECT_THAT(TypeParam::Create("reduce1", input, axes),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(NonlinearReduceOperationTest, CreateBadInput) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {10};
  EXPECT_THAT(TypeParam::Create("reduce1", input, axes,
                                MaximumImplementationType::kOptimalBigM),
              StatusIs(kInvalidArgument));
}


Operation::Options MakeOptions(const std::vector<int64_t>& axes) {
  Operation::Options options;
  options.integer_list_options[reduce::kOptionsAxesKey] = axes;
  return options;
}

Operation::Options MakeOptions(const std::vector<int64_t>& axes,
                               const MaximumImplementationType formulation) {
  Operation::Options options;
  options.integer_list_options[reduce::kOptionsAxesKey] = axes;
  options.string_options[reduce::kOptionsFormulationKey] =
      ToString(formulation);
  return options;
}

TYPED_TEST(LinearReduceOperationTest, GenericCreate) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const Shape result({2, 6});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      TypeParam::GenericCreate("reduce1", {input}, result, MakeOptions(axes)));
  EXPECT_THAT(op, OperationArgsAre("reduce1", {input}, result));
  EXPECT_EQ(op.input(), input);
  EXPECT_EQ(op.axes(), axes);
}

TYPED_TEST(NonlinearReduceOperationTest, GenericCreate) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const MaximumImplementationType formulation =
      MaximumImplementationType::kOptimalBigM;
  const Shape result({2, 6});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, TypeParam::GenericCreate("reduce1", {input}, result,
                                              MakeOptions(axes, formulation)));
  EXPECT_THAT(op, OperationArgsAre("reduce1", {input}, result));
  EXPECT_EQ(op.input(), input);
  EXPECT_EQ(op.axes(), axes);
  EXPECT_EQ(op.formulation(), MaximumImplementationType::kOptimalBigM);
}

TYPED_TEST(LinearReduceOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const Shape result({2, 6});
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input, input}, result,
                                       MakeOptions(axes)),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(NonlinearReduceOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const MaximumImplementationType formulation =
      MaximumImplementationType::kOptimalBigM;
  const Shape result({2, 6});
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input, input}, result,
                                       MakeOptions(axes, formulation)),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(LinearReduceOperationTest, GenericCreateBadOutputShape) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const Shape result({2, 10});  // Should be {2, 6}
  EXPECT_THAT(
      TypeParam::GenericCreate("reduce1", {input}, result, MakeOptions(axes)),
      StatusIs(kInvalidArgument));
}

TYPED_TEST(NonlinearReduceOperationTest, GenericCreateBadOutputShape) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const MaximumImplementationType formulation =
      MaximumImplementationType::kOptimalBigM;
  const Shape result({2, 10});  // Should be {2, 6}
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input}, result,
                                       MakeOptions(axes, formulation)),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(LinearReduceOperationTest, GenericCreateBadExtraOption) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const Shape result({2, 6});
  Operation::Options options = MakeOptions(axes);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input}, result, options),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(NonlinearReduceOperationTest, GenericCreateBadExtraOption) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const MaximumImplementationType formulation =
      MaximumImplementationType::kOptimalBigM;
  const Shape result({2, 6});
  Operation::Options options = MakeOptions(axes, formulation);
  options.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input}, result, options),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(LinearReduceOperationTest, GenericCreateMissingAxis) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const Shape result({2, 6});
  Operation::Options options;
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input}, result, options),
              StatusIs(kInvalidArgument));
}

TYPED_TEST(NonlinearReduceOperationTest, GenericCreateMissingAxis) {
  const Shape input({2, 6, 4});
  const std::vector<int64_t> axes = {2};
  const MaximumImplementationType formulation =
      MaximumImplementationType::kOptimalBigM;
  const Shape result({2, 6});
  Operation::Options options = MakeOptions(axes, formulation);
  options.integer_list_options.erase(reduce::kOptionsAxesKey);
  EXPECT_THAT(TypeParam::GenericCreate("reduce1", {input}, result, options),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
