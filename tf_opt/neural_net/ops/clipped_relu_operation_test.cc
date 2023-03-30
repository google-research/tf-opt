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

#include "tf_opt/neural_net/ops/clipped_relu_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/neuron/clipped_relu_impl_type.h"
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

TEST(ClippedReluOperationTest, SimpleCreate) {
  const Shape input_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ClippedReluOperation::Create(
                         "cr1", input_shape, 6.0,
                         ClippedReluImplementationType::kIncrementalBigM));
  EXPECT_THAT(op, OperationArgsAre("cr1", {input_shape}, input_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.formulation(), ClippedReluImplementationType::kIncrementalBigM);
  EXPECT_EQ(op.cap(), 6.0);
}

TEST(ClippedReluOperationTest, SimpleCreateBadCap) {
  const Shape input_shape({2, 4});
  EXPECT_THAT(ClippedReluOperation::Create(
                  "cr1", input_shape, -1.0,
                  ClippedReluImplementationType::kIncrementalBigM),
              StatusIs(kInvalidArgument));
}

Operation::Options MakeOptions(const double cap,
                               std::string clipped_relu_impl_name = "") {
  Operation::Options options;
  options.double_options[ClippedReluOperation::kOptionsCapKey] = cap;
  if (!clipped_relu_impl_name.empty()) {
    options.string_options[ClippedReluOperation::kOptionsFormulationKey] =
        std::move(clipped_relu_impl_name);
  }
  return options;
}

TEST(ClippedReluOperationTest, GenericCreate) {
  const Shape input_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      ClippedReluOperation::GenericCreate(
          "cr1", {input_shape}, input_shape,
          MakeOptions(
              6.0, ClippedReluOperation::OptionsFormulationIncrementalBigM())));
  EXPECT_THAT(op, OperationArgsAre("cr1", {input_shape}, input_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.formulation(), ClippedReluImplementationType::kIncrementalBigM);
  EXPECT_EQ(op.cap(), 6.0);
}

TEST(ClippedReluOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({2, 4});
  EXPECT_THAT(
      ClippedReluOperation::GenericCreate("cr1", {input_shape, input_shape},
                                          input_shape, MakeOptions(6.0)),
      StatusIs(kInvalidArgument));
}

TEST(ClippedReluOperationTest, GenericCreateBadOutputShape) {
  EXPECT_THAT(ClippedReluOperation::GenericCreate(
                  "cr1", {Shape({3, 2})}, Shape({4, 1}), MakeOptions(6.0)),
              StatusIs(kInvalidArgument));
}

TEST(ClippedReluOperationTest, GenericCreateMissingCap) {
  Operation::Options bad_options;
  EXPECT_THAT(ClippedReluOperation::GenericCreate("cr1", {Shape({3, 2})},
                                                  Shape({3, 2}), bad_options),
              StatusIs(kInvalidArgument));
}

TEST(ClippedReluOperationTest, GenericCreateBadCap) {
  EXPECT_THAT(ClippedReluOperation::GenericCreate(
                  "cr1", {Shape({3, 2})}, Shape({3, 2}), MakeOptions(-1.0)),
              StatusIs(kInvalidArgument));
}

TEST(ClippedReluOperationTest, GenericCreateBadFormulation) {
  EXPECT_THAT(
      ClippedReluOperation::GenericCreate("cr1", {Shape({3, 2})}, Shape({3, 2}),
                                          MakeOptions(6.0, "bad_formulation")),
      StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
