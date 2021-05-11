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

#include "tf_opt/neural_net/ops/matmul_operation.h"

#include <utility>

#include "glog/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/math.h"

namespace tf_opt {

MatmulOperation::MatmulOperation(std::string op_name,
                                 std::vector<Shape> input_shapes,
                                 Shape output_shape)
    : Operation(std::move(op_name), std::move(input_shapes),
                std::move(output_shape)) {}

absl::StatusOr<MatmulOperation> MatmulOperation::Create(std::string op_name,
                                                        Shape left_shape,
                                                        Shape right_shape) {
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         MatMulOutputShape(left_shape, right_shape));
  return MatmulOperation(std::move(op_name),
                         {std::move(left_shape), std::move(right_shape)},
                         std::move(output_shape));
}

MaybeForGraph<MatmulOperation> MatmulOperation::CreateForGraph(
    std::string op_name, const Operation* left, const Operation* right) {
  return FromMaybeCreated(
      Create(std::move(op_name), left->output_shape(), right->output_shape()),
      {left, right});
}

absl::StatusOr<MatmulOperation> MatmulOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("MatmulOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 2));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsEmpty(options.size()));
  TFOPT_ASSIGN_OR_RETURN(MatmulOperation result,
                         Create(std::move(op_name), std::move(input_shapes[0]),
                                std::move(input_shapes[1])),
                         _ << validator.base_error_message());
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, result.output_shape()));
  return std::move(result);
}

proto::TensorNode MatmulOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::MAT_MUL);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  result.add_input_names(inputs[1]);
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
