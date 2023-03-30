// Copyright 2023 The tf.opt Authors.
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

#ifndef TF_OPT_SHARED_OPS_EXPAND_DIMS_OPERATION_H_
#define TF_OPT_SHARED_OPS_EXPAND_DIMS_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Reshapes the input by inserting an extra dimension of size 1.
//
// E.g.
//    x = [[1, 2], [3, 4]]                    (shape [2, 2])
//    expand_dims(x, axis=0)
//      => [[[1, 2], [3, 4]]]                 (shape [1, 2, 2])
//    expand_dims(x, axis=1)
//      => [[[1, 2]], [[3, 4]]]               (shape [2, 1, 2])
//    expand_dims(x, axis=2)
//      => [[[1], [2]], [[3], [4]]]           (shape [2, 2, 1])
class ExpandDimsOperation : public Operation {
 public:
  static constexpr const char kOptionsAxisKey[] = "axis";

  static absl::StatusOr<ExpandDimsOperation> Create(std::string op_name,
                                                    Shape input_shape,
                                                    int axis);

  // Expected input format:
  //   input_shapes: The shape of the tensor to expand,
  //       input_shapes.size() == 1.
  //   output_shape: The shape to produce, the input shape with a 1 inserted at
  //       options[axis].
  //   options: Must contain the integer "axis", the index to add a 1 in the
  //       in the shape.
  static absl::StatusOr<ExpandDimsOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  int axis() const { return axis_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  ExpandDimsOperation(std::string op_name, Shape input_shape,
                      Shape output_shape, int axis)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        axis_(axis) {}

  int axis_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_EXPAND_DIMS_OPERATION_H_
