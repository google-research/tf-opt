// Copyright 2025 The tf.opt Authors.
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

// Compute convolutions on Tensors.
//
// For details, see the documentation of tf.nn.conv1d(...) and tf.nn.conv2d(...)
// in the TensorFlow Python API:
//   * https://www.tensorflow.org/api_docs/python/tf/nn/conv1d,
//   * https://www.tensorflow.org/api_docs/python/tf/nn/conv2d.
//
// Typical Conv1d use:
// DoubleTensor input = ...;   // [batch, width, in_channel]
// DoubleTensor filter = ...;  // [filter_width, in_channel, out_channel]
// int stride = 1;
// PaddingType padding = PaddingType::SAME;
// DoubleTensor result = Conv1d<double, double, double>(input, filter,
//                                                      stride, padding);
//
// Typical Conv2d use:
// DoubleTensor input = ...;   // [batch, height, width, in_channel]
// DoubleTensor filter = ...;  // [filter_height, filter_width,
//                             //  in_channel, out_channel]
// int stride_row = 1;
// int stride_col = 1;
// PaddingType padding = PaddingType::SAME;
// DoubleTensor result = Conv2d<double, double, double>(
//   input, filter, Position2D(stride_row, stride_col), padding);
//
// Note on templates in this file (for both Conv1d and Conv2d):
//
// Template types must support multiplication and addition via operator
// overload.  E.g.,
// MPTensor result = Conv2d<operations_research::LinearExpr,
//                          operations_research::LinearExpr, double>(...);
// will compile, but
// MPTensor result = Conv2d<operations_research::LinearExpr,
//                          operations_research::LinearExpr,
//                          operations_research::LinearExpr>(...);
// will not (LinearExpr cannot be multiplied together).

#ifndef TF_OPT_TENSOR_CONVOLVE_H_
#define TF_OPT_TENSOR_CONVOLVE_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "ortools/base/logging.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {

namespace internal {

absl::Status Conv1dValidateShapes(const Shape& input_shape,
                                  const Shape& filter_shape);

absl::Status Conv2dValidateShapes(const Shape& input_shape,
                                  const Shape& filter_shape);

}  // namespace internal

// These shape structs wrap a pointer to Shape. They were created for
// readability and to centralize assumptions on dimension ordering.

// Wrapper to dimensions of the input tensor for conv2d.
// Assumes the order 'NHWC': batch, height, width, and channels.
struct Conv2dInputShape {
  const Shape* shape;  // Pointer is not owned.

  explicit Conv2dInputShape(const Shape* shape)
      : shape(CHECK_NOTNULL(shape)) {
    CHECK_EQ(shape->num_dimensions(), 4);
  }

  int64_t batch() const { return shape->dimension_size(0); }
  int64_t height() const { return shape->dimension_size(1); }
  int64_t width() const { return shape->dimension_size(2); }
  int64_t channels() const { return shape->dimension_size(3); }
  Position2D RegionSize() const { return Position2D(height(), width()); }
};

// Wrapper to dimensions of the filter parameter for conv2d.
// Assumes the order: height, width, in_channels, and out_channels.
struct Conv2dFilterShape {
  const Shape* shape;  // Pointer is not owned.

  explicit Conv2dFilterShape(const Shape* shape)
      : shape(CHECK_NOTNULL(shape)) {
    CHECK_EQ(shape->num_dimensions(), 4);
  }

  int64_t height() const { return shape->dimension_size(0); }
  int64_t width() const { return shape->dimension_size(1); }
  int64_t in_channels() const { return shape->dimension_size(2); }
  int64_t out_channels() const { return shape->dimension_size(3); }
  Position2D RegionSize() const { return Position2D(height(), width()); }
};

// Wrapper to dimensions of the input tensor for conv1d.
// Assumes the order 'NWC': batch, width, and channels.
struct Conv1dInputShape {
  const Shape* shape;  // Pointer is not owned.

  explicit Conv1dInputShape(const Shape* shape)
      : shape(CHECK_NOTNULL(shape)) {
    CHECK_EQ(shape->num_dimensions(), 3);
  }

  int64_t batch() const { return shape->dimension_size(0); }
  int64_t width() const { return shape->dimension_size(1); }
  int64_t channels() const { return shape->dimension_size(2); }

  // Create shape for conv2d with height = 1 and matching parameters.
  Shape shape2d() const {
    return Shape({shape->dimension_size(0), 1, shape->dimension_size(1),
                  shape->dimension_size(2)});
  }
};

// Wrapper to dimensions of the filter parameter for conv1d.
// Assumes the order: width, in_channels, and out_channels.
struct Conv1dFilterShape {
  const Shape* shape;  // Pointer is not owned.

  explicit Conv1dFilterShape(const Shape* shape)
      : shape(CHECK_NOTNULL(shape)) {
    CHECK_EQ(shape->num_dimensions(), 3);
  }

  int64_t width() const { return shape->dimension_size(0); }
  int64_t in_channels() const { return shape->dimension_size(1); }
  int64_t out_channels() const { return shape->dimension_size(2); }

  // Create shape for conv2d with height = 1 and matching parameters.
  Shape shape2d() const {
    return Shape({1, shape->dimension_size(0), shape->dimension_size(1),
                  shape->dimension_size(2)});
  }
};

// Equivalent to (Python) TF's tf.nn.conv2d when strides list is given by
// [1, strides.row, strides.col, 1] from tfopt Conv2d args.
//
// Tensor shapes:
//   input: Must be rank 4, format: [batch, height, width, in_channel].
//   filter: Must be rank 4, format:
//       [filter_height, filter_width, in_channel, out_channel], where
//       in_channel in input and filter match.
//   result: Will be rank 4, format: [batch, height, width, out_channel].
//
// Returns error if shapes are invalid.
//
// For a visualization of how strides and padding work, see
// https://github.com/vdumoulin/conv_arithmetic.
template <typename ResultType, typename InputType, typename FilterType>
absl::StatusOr<Tensor<ResultType>> Conv2d(const Tensor<InputType>& input,
                                          const Tensor<FilterType>& filter,
                                          const Position2D strides,
                                          const PaddingType padding_type) {
  TFOPT_RETURN_IF_ERROR(
      internal::Conv2dValidateShapes(input.dimension(), filter.dimension()));
  const Conv2dInputShape input_shape(&input.dimension());
  const Conv2dFilterShape filter_shape(&filter.dimension());

  WindowExtractor2D window_extractor;
  TFOPT_RETURN_IF_ERROR(window_extractor.Initialize(
      input_shape.RegionSize(), filter_shape.RegionSize(),
      Position2D(strides.row, strides.col), padding_type));

  const int64_t output_batch = input_shape.batch();
  const int64_t output_height = window_extractor.output_size().row;
  const int64_t output_width = window_extractor.output_size().col;
  const int64_t output_channels = filter_shape.out_channels();
  const Shape output_dimension(
      {output_batch, output_height, output_width, output_channels});

  Tensor<ResultType> result = Tensor<ResultType>(output_dimension);
  // Loop through all the output pixels.
  for (int64_t ob = 0; ob < output_batch; ++ob) {
    for (int64_t oy = 0; oy < output_height; ++oy) {
      for (int64_t ox = 0; ox < output_width; ++ox) {
        for (int64_t oc = 0; oc < output_channels; ++oc) {
          const int64_t ib = ob;
          // Compute the center in the input.
          const Rectangle rectangle =
              window_extractor.GetWindow(Position2D(oy, ox));

          // We need a "zero-like" value to start adding. Depending on
          // ResultType, we cannot rely on: (a) 0 or 0.0 being implicitly
          // convertible to ResultType, or (b) having a default value, e.g.
          // if ResultType is an int, then the default value is garbage memory,
          // not zero. However, a Tensor<ResultType> begins "zeroed" out
          // correctly.
          ResultType conv_val = result.ValueSpan({ob, oy, ox, oc});
          for (int64_t iy = rectangle.start.row;
               iy < rectangle.start.row + rectangle.size.row; ++iy) {
            for (int64_t ix = rectangle.start.col;
                 ix < rectangle.start.col + rectangle.size.col; ++ix) {
              if (!window_extractor.IsPadding(Position2D(iy, ix))) {
                for (int64_t ic = 0; ic < input_shape.channels(); ic++) {
                  const FilterType coef =
                      filter.ValueSpan({iy - rectangle.start.row,
                                        ix - rectangle.start.col, ic, oc});
                  conv_val += coef * input.ValueSpan({ib, iy, ix, ic});
                }
              }
            }
          }
          result.SetValueSpan({ob, oy, ox, oc}, conv_val);
        }
      }
    }
  }
  return result;
}

// Equivalent to tf.nn.conv1d(input, filters, stride, padding).
//
// Tensor shapes:
//   input: Must be rank 3, format: [batch, width, in_channel].
//   filter: Must be rank 3, format: [filter_width, in_channel, out_channel],
//       where in_channel and input and filter match.
//   result: Will be rank 3, format: [batch, width, out_channel].
//
// Returns error if shapes are invalid.
template <typename ResultType, typename InputType, typename FilterType>
absl::StatusOr<Tensor<ResultType>> Conv1d(const Tensor<InputType>& input,
                                          const Tensor<FilterType>& filter,
                                          const int stride,
                                          const PaddingType padding_type) {
  TFOPT_RETURN_IF_ERROR(
      internal::Conv1dValidateShapes(input.dimension(), filter.dimension()));
  const Conv1dInputShape input_shape(&input.dimension());
  const Conv1dFilterShape filter_shape(&filter.dimension());

  const Tensor<InputType> conv2d_input = input.Reshape(input_shape.shape2d());
  const Tensor<FilterType> conv2d_filter =
      filter.Reshape(filter_shape.shape2d());
  const Position2D strides(1, stride);
  TFOPT_ASSIGN_OR_RETURN(
      Tensor<ResultType> conv2d_result,
      (Conv2d<ResultType, InputType, FilterType>(conv2d_input, conv2d_filter,
                                                 strides, padding_type)),
      _ << "on conv1d inside conv2d");

  const Shape& conv2d_shape = conv2d_result.dimension();
  CHECK_EQ(conv2d_shape.num_dimensions(), 4);
  CHECK_EQ(conv2d_shape.dimension_size(1), 1);
  const Shape result_shape({conv2d_shape.dimension_size(0),
                            conv2d_shape.dimension_size(2),
                            conv2d_shape.dimension_size(3)});
  conv2d_result.ReshapeInPlace(result_shape);
  return conv2d_result;
}

// Returns the output shape of the Conv1d operator, given the input parameters.
// Also validates the operation, but does not perform it.
absl::StatusOr<Shape> Conv1dOutputShape(const Shape& input_shape,
                                        const Shape& filter_shape, int stride,
                                        PaddingType padding_type);

// Returns the output shape of the Conv2d operator, given the input parameters.
// Also validates the operation, but does not perform it.
absl::StatusOr<Shape> Conv2dOutputShape(const Shape& input_shape,
                                        const Shape& filter_shape,
                                        Position2D strides,
                                        PaddingType padding_type);
}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_CONVOLVE_H_
