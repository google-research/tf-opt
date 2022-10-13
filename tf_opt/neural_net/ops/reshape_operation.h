// Copyright 2022 The tf.opt Authors.
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

#ifndef TF_OPT_SHARED_OPS_RESHAPE_OPERATION_H_
#define TF_OPT_SHARED_OPS_RESHAPE_OPERATION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Given a single tensor with n elements and a target shape with n elements,
// produces an output tensor in the target shape with the input data.  The
// "flattened order" remains unchanged, i.e. if input is
//
// [[2, 4],[6, 8]]
//
// (has shape [2,2]) and a tensor with shape [1, 4] is requested, output will be
//
// [[2, 4, 6, 8]],
//
// as for both tensors, the underlying flat storage (which is row-major), is
//
// [2, 4, 6, 8].
class ReshapeOperation : public Operation {
 public:
  static absl::StatusOr<ReshapeOperation> Create(std::string op_name,
                                                 Shape input_shape,
                                                 Shape output_shape);

  // Expected input format:
  //   input_shapes: The shape a single tensor, to be reshaped.
  //   output_shape: The shape to produce, same number of elements as input.
  //   options: Must be empty.
  static absl::StatusOr<ReshapeOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  ReshapeOperation(std::string op_name, Shape input_shape, Shape output_shape)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)) {}
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_RESHAPE_OPERATION_H_
