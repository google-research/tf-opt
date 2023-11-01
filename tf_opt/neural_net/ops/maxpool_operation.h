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

#ifndef TF_OPT_SHARED_OPS_MAXPOOL_OPERATION_H_
#define TF_OPT_SHARED_OPS_MAXPOOL_OPERATION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/neuron/maximum_impl_type.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {

// Analogous to tf.nn.max_pool, the max pooling operator.
//
// Given an input tensor, window size, strides, and padding type, produces
// a new tensor which corresponds to the maximum value of a series of windows in
// the input tensor calculated according to the parameters.
//
// The input is 4D of the form (batch, height, width, channels), but in this
// implementation, batch and channels are considered independently and the
// operation is performed only over the x and y components (see todo below).
//
// For example: ignoring batch and channels, if the strides are one, the window
// size is 2x2, and x and y components of the input tensor are:
//
//  {{1,2,1},  then the output is  {{3,2},
//   {3,1,0},                       {3,5}}.
//   {2,0,5}},
//
// The windows skip row or columns according to the values of strides. A value
// of one means no row or column is skipped, as illustrated above. A value of
// two means every second row or column is considered. Strides are computed
// according to https://www.tensorflow.org/api_guides/python/nn#Convolution.
// Also, see tf_opt/optimize/neuron/window.h for more information on how windows
// are computed.
//
// For details on the formulation for maximum, see
// tf_opt/optimize/neuron/maximum*.h.
//
// TODO: Only the spatial version of MaxPool is implemented. In other
// words, ksize and stride defined in the documentation of tf.nn.max_pool are
// restricted to the form (1, *, *, 1).
class MaxpoolOperation : public Operation {
 public:
  static constexpr char kOptionsFormulationKey[] = "formulation";
  static constexpr char kOptionsFormulationDefault[] = "default";
  static constexpr char kOptionsStrideRowKey[] = "stride_row";
  static constexpr char kOptionsStrideColKey[] = "stride_col";
  static constexpr char kOptionsWindowHeightKey[] = "ksize_height";
  static constexpr char kOptionsWindowWidthKey[] = "ksize_width";
  static constexpr char kOptionsPaddingKey[] = "padding";

  static const char* OptionsFormulation(MaximumImplementationType max_impl);
  static std::vector<std::string> AllMaxPoolImplementations();

  // Returns the output shape of the max pooling operation.
  static absl::StatusOr<Shape> OutputShape(const Shape& input_shape,
                                           const Position2D& window_size,
                                           const Position2D& strides,
                                           const PaddingType& padding);

  static absl::StatusOr<MaxpoolOperation> Create(
      std::string op_name, Shape input_shape, Position2D ksize,
      Position2D strides, PaddingType padding,
      MaximumImplementationType formulation = kDefaultMaximum);

  // Expected input format:
  //   input_shapes: The dimensions of a single tensor of shape [batch, rows,
  //       columns, in_channels].
  //   output_shape: The shape of the resulting tensor.
  //   options: May contain a string option with key kOptionsFormulationKey, to
  //       pick a MIP formulation for Maximum().
  //       Must have integer keys stride_row, stride_col, window_height, and
  //       window_width all taking positive values.
  //       Must have string key padding with value ToString(PaddingType::SAME)
  //       or ToString(PaddingType::VALID) (see window.h).
  static absl::StatusOr<MaxpoolOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  Position2D ksize() const { return ksize_; }
  Position2D stride() const { return stride_; }
  PaddingType padding() const { return padding_; }
  MaximumImplementationType formulation() const { return formulation_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  MaxpoolOperation(std::string op_name, Shape input_shape, Shape output_shape,
                   Position2D ksize, Position2D strides,
                   const PaddingType padding,
                   const MaximumImplementationType formulation)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        ksize_(ksize),
        stride_(strides),
        padding_(padding),
        formulation_(formulation) {}

  Position2D ksize_;
  Position2D stride_;
  PaddingType padding_;
  // TODO: move this to MIP world.
  MaximumImplementationType formulation_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_MAXPOOL_OPERATION_H_
