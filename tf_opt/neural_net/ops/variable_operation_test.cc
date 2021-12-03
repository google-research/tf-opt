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

#include "tf_opt/neural_net/ops/variable_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(VariableOperationTest, SimpleCreate) {
  const Shape shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op,
                             VariableOperation::Create("v1", shape));
  EXPECT_THAT(op, OperationArgsAre("v1", {}, shape));
}

TEST(VariableOperationTest, GenericCreate) {
  const Shape shape({2, 4});
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op,
      VariableOperation::GenericCreate("v1", {}, shape, Operation::Options()));
  EXPECT_THAT(op, OperationArgsAre("v1", {}, shape));
}

TEST(VariableOperationTest, GenericCreateWrongNumberInputs) {
  const Shape shape({2, 4});
  EXPECT_THAT(VariableOperation::GenericCreate("relu1", {shape, shape}, shape,
                                               Operation::Options()),
              StatusIs(kInvalidArgument));
}

TEST(VariableOperationTest, GenericCreateBadOption) {
  const Shape shape({2, 4});
  Operation::Options options;
  options.string_options["bad_option"] = "bad_value";
  EXPECT_THAT(VariableOperation::GenericCreate("relu1", {}, shape, options),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
