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

#include "tf_opt/neural_net/ops/embedding_lookup_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/embedding_lookup.h"

namespace tf_opt {

EmbeddingLookupOperation::EmbeddingLookupOperation(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape)
    : Operation(std::move(op_name), std::move(input_shapes),
                std::move(output_shape)) {}

absl::StatusOr<Shape> EmbeddingLookupOperation::OutputShape(
    const Shape& params_shape, const Shape& ids_shape) {
  return EmbeddingLookupOutputShape(params_shape, ids_shape);
}

absl::StatusOr<EmbeddingLookupOperation> EmbeddingLookupOperation::Create(
    std::string op_name, Shape params_shape, Shape ids_shape) {
  OperationValidator validator("EmbeddingLookupOperation", op_name);
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         OutputShape(params_shape, ids_shape),
                         _ << validator.base_error_message());
  return EmbeddingLookupOperation(
      std::move(op_name), {std::move(params_shape), std::move(ids_shape)},
      output_shape);
}

absl::StatusOr<EmbeddingLookupOperation>
EmbeddingLookupOperation::GenericCreate(std::string op_name,
                                        std::vector<Shape> input_shapes,
                                        Shape output_shape,
                                        const Options& options) {
  OperationValidator validator("EmbeddingNode", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 2));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsEmpty(options.size()));
  TFOPT_ASSIGN_OR_RETURN(EmbeddingLookupOperation op,
                         Create(std::move(op_name), std::move(input_shapes[0]),
                                std::move(input_shapes[1])));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return std::move(op);
}

proto::TensorNode EmbeddingLookupOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::EMBEDDING_LOOKUP);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  result.add_input_names(inputs[1]);
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
