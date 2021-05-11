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

#include "tf_opt/neural_net/ops/concat_operation.h"

#include "glog/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/concat.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

constexpr const char ConcatOperation::kOptionsAxisKey[];

ConcatOperation::ConcatOperation(std::string op_name,
                                 std::vector<Shape> input_shapes,
                                 Shape output_shape, const int axis)
    : Operation(std::move(op_name), std::move(input_shapes),
                std::move(output_shape)),
      axis_(axis) {}

absl::StatusOr<ConcatOperation> ConcatOperation::Create(
    std::string op_name, std::vector<Shape> input_shapes, const int axis) {
  OperationValidator validator("ConcatOperation", op_name);
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         ConcatOutputShape(input_shapes, axis),
                         _ << validator.base_error_message());
  return ConcatOperation(std::move(op_name), std::move(input_shapes),
                         std::move(output_shape), axis);
}

MaybeForGraph<ConcatOperation> ConcatOperation::CreateForGraph(
    std::string op_name, const std::vector<const Operation*>& inputs,
    const int axis) {
  std::vector<Shape> input_shapes;
  input_shapes.reserve(inputs.size());
  for (const Operation* input : inputs) {
    input_shapes.push_back(input->output_shape());
  }
  return FromMaybeCreated(Create(std::move(op_name), input_shapes, axis),
                          inputs);
}

absl::StatusOr<ConcatOperation> ConcatOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes,
    const Shape output_shape, const Options& options) {
  OperationValidator validator("ConcatOperation", op_name);
  TFOPT_ASSIGN_OR_RETURN(const int axis,
                         validator.IntegerOption(options, kOptionsAxisKey));
  TFOPT_ASSIGN_OR_RETURN(
      ConcatOperation result,
      Create(std::move(op_name), std::move(input_shapes), axis));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, result.output_shape()));
  return std::move(result);
}

proto::TensorNode ConcatOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::CONCAT);
  for (const std::string& input : inputs) {
    result.add_input_names(input);
  }
  proto::Options::IntegerOption* option =
      result.mutable_options()->add_integer_options();
  option->set_name(kOptionsAxisKey);
  option->set_value(axis());
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
