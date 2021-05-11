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

#ifndef TF_OPT_SHARED_OPS_CONV1D_OPERATION_H_
#define TF_OPT_SHARED_OPS_CONV1D_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {

// Compares to to tf.nn.conv1d.
class Conv1dOperation : public Operation {
 public:
  static constexpr const char kOptionsPaddingKey[] = "padding";
  static constexpr const char kOptionsStrideKey[] = "stride";

  static absl::StatusOr<Conv1dOperation> Create(std::string op_name,
                                                Shape input_value_shape,
                                                Shape filter_shape, int stride,
                                                PaddingType padding);

  // TODO: replace this by a variadic template function.
  static MaybeForGraph<Conv1dOperation> CreateForGraph(
      std::string op_name, const Operation* input_value,
      const Operation* filter, int stride, PaddingType padding);

  // Expected input format:
  //   input_shapes: Shapes of two tensors, first the "value" and then the
  //       "filters", as in tf.nn.conv1d:
  //         1. value: has shape [batch, columns, in_channels],
  //         2. filter: has shape [filter_columns, in_channels, out_channels].
  //   output_shape: The shape [batch, out_columns, out_channels].
  //   options: Must have integer key stride taking a positive value. Must have
  //       a string key padding with value ToString(PaddingType::SAME) or
  //       ToString(PaddingType::VALID) (see window.h).
  static absl::StatusOr<Conv1dOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input_value() const { return input_shape(0); }
  const Shape& filter() const { return input_shape(1); }
  PaddingType padding() const { return padding_; }
  int stride() const { return stride_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

 private:
  Conv1dOperation(std::string op_name, Shape input_value_shape,
                  Shape filter_shape, Shape output_shape, int stride,
                  PaddingType padding);

  // Set from options.
  int stride_;
  PaddingType padding_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_CONV1D_OPERATION_H_
