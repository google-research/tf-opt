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

#include "tf_opt/neural_net/ops/constant_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor_testing.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;

TEST(ConstantOperationTest, SimpleCreate) {
  TFOPT_ASSERT_OK_AND_ASSIGN(
      const auto op, ConstantOperation::Create("c1", DoubleTensor({4.0, 5.0})));
  EXPECT_THAT(op, OperationArgsAre("c1", {}, Shape({2})));
  EXPECT_THAT(op.value(), DoubleTensorEquals(DoubleTensor({4.0, 5.0})));
}

TEST(ConstantOperationTestDeathTest, GenericCreateDies) {
  EXPECT_DEATH((void)ConstantOperation::GenericCreate("c1", {}, Shape(),
                                                      Operation::Options()),
               "");
}

}  // namespace
}  // namespace tf_opt
