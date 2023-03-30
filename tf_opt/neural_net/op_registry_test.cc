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

#include "tf_opt/neural_net/op_registry.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {
namespace {

using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

TEST(OpRegistryTest, MakeOperationSuccess) {
  const Shape left({2, 3});
  const Shape right({1, 3});
  const Shape output = left;

  TFOPT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Operation> op,
      op_registry::MakeOperation(proto::ADD, "add", {left, right}, output,
                                 Operation::Options()));
  EXPECT_THAT(*op, OperationArgsAre("add", {left, right}, output));
}

TEST(OpRegistryTest, MakeOperationFailure) {
  const Shape left({2, 3});
  // Incompatible shapes for add.
  const Shape right({1, 5});
  const Shape output = left;

  EXPECT_THAT(op_registry::MakeOperation(proto::ADD, "add", {left, right}, left,
                                         Operation::Options()),
              StatusIs(kInvalidArgument));
}

}  // namespace
}  // namespace tf_opt
