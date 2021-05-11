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

#include "tf_opt/neural_net/ops/embedding_lookup_operation.h"

#include <utility>

#include "glog/logging.h"
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

MaybeForGraph<EmbeddingLookupOperation>
EmbeddingLookupOperation::CreateForGraph(std::string op_name,
                                         const Operation* params,
                                         const Operation* ids) {
  return FromMaybeCreated(
      Create(std::move(op_name), params->output_shape(), ids->output_shape()),
      {params, ids});
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

}  // namespace tf_opt
