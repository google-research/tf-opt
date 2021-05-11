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

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {

VariableOperation::VariableOperation(std::string op_name, Shape shape)
    : Operation(std::move(op_name), {}, std::move(shape)) {}

absl::StatusOr<VariableOperation> VariableOperation::Create(std::string op_name,
                                                            Shape shape) {
  return VariableOperation(std::move(op_name), std::move(shape));
}

MaybeForGraph<VariableOperation> VariableOperation::CreateForGraph(
    std::string op_name, Shape shape) {
  return FromMaybeCreated(Create(std::move(op_name), std::move(shape)), {});
}

absl::StatusOr<VariableOperation> VariableOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("VariableOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 0));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsEmpty(options.size()));
  return Create(std::move(op_name), std::move(output_shape));
}

proto::TensorNode VariableOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::INPUT);

  *result.mutable_out_dimension() = output_shape().AsProto();
  // TODO: we need to persist the integer bit so this round trips
  // cleanly.
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
