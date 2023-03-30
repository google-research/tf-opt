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

#ifndef TF_OPT_SHARED_OPS_SQUEEZE_OPERATION_H_
#define TF_OPT_SHARED_OPS_SQUEEZE_OPERATION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Reshapes the input to remove one or more dimensions of size one.
//
// Output is:
//   * If "axes" option is provided and is nonempty, reshapes the input to
//     delete the dimensions with indices given by axes.  Requires that all
//     deleted dimensions have dimension size 1.
//   * Else, reshapes the input to delete all dimensions with dimension size
//     one.
//
// E.g. if the input shape is [1, 3, 1, 2, 1], then
//   axis = [2, 4] => output shape is [1, 3, 2]
//   axis not set => output shape is [3, 2].
class SqueezeOperation : public Operation {
 public:
  static constexpr const char kOptionsAxesKey[] = "axes";

  static absl::StatusOr<SqueezeOperation> Create(std::string op_name,
                                                 Shape input_shape,
                                                 std::vector<int> axes);

  // Expected input format:
  //   input_shapes: The shape of the tensor to squeeze,
  //       input_shapes.size() == 1.
  //   output_shape: The shape to produce, the input shape with some of the
  //       dimensions of size one removed, as controlled by axes.
  //   options: May either be empty, or contain an IntegerList, "axes".
  static absl::StatusOr<SqueezeOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  const std::vector<int>& axes() const { return axes_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  SqueezeOperation(std::string op_name, Shape input_shape, Shape output_shape,
                   std::vector<int> axes)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        axes_(std::move(axes)) {}

  std::vector<int> axes_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_SQUEEZE_OPERATION_H_
