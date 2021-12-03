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

#include "tf_opt/neural_net/ops/conv1d_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/convolve.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

constexpr const char Conv1dOperation::kOptionsStrideKey[];
constexpr const char Conv1dOperation::kOptionsPaddingKey[];

Conv1dOperation::Conv1dOperation(std::string op_name, Shape input_value_shape,
                                 Shape filter_shape, Shape output_shape,
                                 const int stride, const PaddingType padding)
    : Operation(std::move(op_name),
                {std::move(input_value_shape), std::move(filter_shape)},
                std::move(output_shape)),
      stride_(stride),
      padding_(padding) {}

absl::StatusOr<Conv1dOperation> Conv1dOperation::Create(
    std::string op_name, Shape input_value_shape, Shape filter_shape,
    const int stride, const PaddingType padding) {
  TFOPT_ASSIGN_OR_RETURN(
      Shape output_shape,
      Conv1dOutputShape(input_value_shape, filter_shape, stride, padding));
  return Conv1dOperation(std::move(op_name), std::move(input_value_shape),
                         std::move(filter_shape), std::move(output_shape),
                         stride, padding);
}

absl::StatusOr<Conv1dOperation> Conv1dOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("Conv1dOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 2));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 2));
  TFOPT_ASSIGN_OR_RETURN(const int stride,
                         validator.IntegerOption(options, kOptionsStrideKey));
  TFOPT_ASSIGN_OR_RETURN(const std::string& padding_name,
                         validator.StringOption(options, kOptionsPaddingKey));
  PaddingType padding_type;
  if (!PaddingTypeFromString(padding_name, &padding_type)) {
    return validator.OperationValidationError("Invalid padding string");
  }

  TFOPT_ASSIGN_OR_RETURN(
      auto op,
      Create(op_name, input_shapes[0], input_shapes[1], stride, padding_type),
      _ << validator.base_error_message());
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, op.output_shape()));
  return std::move(op);
}

proto::TensorNode Conv1dOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::CONV1D);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  result.add_input_names(inputs[1]);
  proto::Options::StringOption& padding_option =
      *result.mutable_options()->add_string_options();
  padding_option.set_name(kOptionsPaddingKey);
  padding_option.set_value(ToString(padding()));

  proto::Options::IntegerOption& stride_option =
      *result.mutable_options()->add_integer_options();
  stride_option.set_name(kOptionsStrideKey);
  stride_option.set_value(stride_);
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
