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

#include "tf_opt/neural_net/ops/relu_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/neuron/relu_impl_type.h"
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

TEST(ReluOperationTest, SimpleCreate) {
  const Shape input_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ReluOperation::Create("relu1", input_shape,
                                           ReluImplementationType::kBigM));
  EXPECT_THAT(op, OperationArgsAre("relu1", {input_shape}, input_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.formulation(), ReluImplementationType::kBigM);
}

Operation::Options MakeOptions(std::string relu_impl_name = "") {
  Operation::Options options;
  if (!relu_impl_name.empty()) {
    options.string_options[ReluOperation::kOptionsFormulationKey] =
        std::move(relu_impl_name);
  }
  return options;
}

TEST(ReluOperationTest, GenericCreate) {
  const Shape input_shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ReluOperation::GenericCreate(
                         "relu1", {input_shape}, input_shape,
                         MakeOptions(ReluOperation::OptionsFormulationBigM())));
  EXPECT_THAT(op, OperationArgsAre("relu1", {input_shape}, input_shape));
  EXPECT_EQ(op.input(), input_shape);
  EXPECT_EQ(op.formulation(), ReluImplementationType::kBigM);
}

TEST(ReluOperationTest, GenericCreateWrongNumberInputs) {
  const Shape input_shape({2, 4});
  EXPECT_THAT(ReluOperation::GenericCreate("relu1", {input_shape, input_shape},
                                           input_shape, MakeOptions()),
              StatusIs(kInvalidArgument));
}

TEST(ReluOperationTest, GenericCreateBadOutputShape) {
  EXPECT_THAT(ReluOperation::GenericCreate("relu1", {Shape({3, 2})},
                                           Shape({4, 1}), MakeOptions()),
              StatusIs(kInvalidArgument));
}

TEST(ReluOperationTest, GenericCreateBadFormulation) {
  EXPECT_THAT(
      ReluOperation::GenericCreate("relu1", {Shape({3, 2})}, Shape({3, 2}),
                                   MakeOptions("bad_formulation")),
      StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
