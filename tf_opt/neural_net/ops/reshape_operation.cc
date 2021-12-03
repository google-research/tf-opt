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

#include "tf_opt/neural_net/ops/reshape_operation.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {

absl::StatusOr<ReshapeOperation> ReshapeOperation::Create(std::string op_name,
                                                          Shape input_shape,
                                                          Shape output_shape) {
  OperationValidator validator("ReshapeOperation", op_name);
  if (input_shape.size() != output_shape.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        validator.base_error_message(), "input_shape: ", input_shape.ToString(),
        "has ", input_shape.size(),
        " elements, but output_shape: ", output_shape.ToString(), " has ",
        output_shape.size(), " elements, must be equal to reshape."));
  }
  return ReshapeOperation(std::move(op_name), std::move(input_shape),
                          std::move(output_shape));
}

absl::StatusOr<ReshapeOperation> ReshapeOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("ReshapeOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsEmpty(options.size()));
  TFOPT_ASSIGN_OR_RETURN(ReshapeOperation op,
                         Create(std::move(op_name), std::move(input_shapes[0]),
                                std::move(output_shape)));
  return op;
}

proto::TensorNode ReshapeOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::RESHAPE);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
