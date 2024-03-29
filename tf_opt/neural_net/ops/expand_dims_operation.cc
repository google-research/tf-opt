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

#include "tf_opt/neural_net/ops/expand_dims_operation.h"

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

constexpr const char ExpandDimsOperation::kOptionsAxisKey[];

absl::StatusOr<ExpandDimsOperation> ExpandDimsOperation::Create(
    std::string op_name, Shape input_shape, const int axis) {
  OperationValidator validator("ExpandDimsOperation", op_name);
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         internal::ExpandDimsShape(input_shape, axis),
                         _ << validator.base_error_message());
  return ExpandDimsOperation(std::move(op_name), std::move(input_shape),
                             std::move(output_shape), axis);
}

absl::StatusOr<ExpandDimsOperation> ExpandDimsOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("ExpandDimsOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_ASSIGN_OR_RETURN(const int axis,
                         validator.IntegerOption(options, kOptionsAxisKey));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 1));
  // Options validation performed in ExpandDimsShape.
  TFOPT_ASSIGN_OR_RETURN(
      ExpandDimsOperation op,
      Create(std::move(op_name), std::move(input_shapes[0]), axis));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return std::move(op);
}

proto::TensorNode ExpandDimsOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::EXPAND_DIMS);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);

  proto::Options::IntegerOption& axis_option =
      *result.mutable_options()->add_integer_options();
  axis_option.set_name(kOptionsAxisKey);
  axis_option.set_value(axis_);

  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
