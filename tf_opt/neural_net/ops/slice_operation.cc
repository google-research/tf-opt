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

#include "tf_opt/neural_net/ops/slice_operation.h"

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

constexpr const char SliceOperation::kOptionsBeginKey[];
constexpr const char SliceOperation::kOptionsSizeKey[];

absl::StatusOr<SliceOperation> SliceOperation::Create(
    std::string op_name, Shape input_shape, std::vector<int64_t> begin,
    std::vector<int64_t> sizes) {
  OperationValidator validator("SliceOperation", op_name);
  TFOPT_ASSIGN_OR_RETURN(Shape result_shape,
                         internal::SliceShape(input_shape, begin, sizes),
                         _ << validator.base_error_message());
  return SliceOperation(std::move(op_name), std::move(input_shape),
                        std::move(result_shape), std::move(begin),
                        std::move(sizes));
}

MaybeForGraph<SliceOperation> SliceOperation::CreateForGraph(
    std::string op_name, const Operation* input, std::vector<int64_t> begin,
    std::vector<int64_t> sizes) {
  return FromMaybeCreated(Create(std::move(op_name), input->output_shape(),
                                 std::move(begin), std::move(sizes)),
                          {input});
}

absl::StatusOr<SliceOperation> SliceOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("SliceOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 2));
  TFOPT_ASSIGN_OR_RETURN(
      std::vector<int64_t> begin,
      validator.IntegerListOption(options, kOptionsBeginKey));
  TFOPT_ASSIGN_OR_RETURN(std::vector<int64_t> size,
                         validator.IntegerListOption(options, kOptionsSizeKey));
  TFOPT_ASSIGN_OR_RETURN(SliceOperation op,
                         Create(std::move(op_name), std::move(input_shapes[0]),
                                std::move(begin), std::move(size)));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return op;
}

}  // namespace tf_opt