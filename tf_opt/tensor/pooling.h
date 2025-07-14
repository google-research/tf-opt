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

#ifndef TF_OPT_TENSOR_POOLING_H_
#define TF_OPT_TENSOR_POOLING_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/element_operations.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {

absl::StatusOr<Shape> Pool2dOutputShape(const Shape& input_shape,
                                        const Position2D& window_size,
                                        const Position2D& strides,
                                        const PaddingType& padding);

template <typename T>
Tensor<T> MaxPool(const Tensor<T>& input, const Position2D& window_size,
                  const Position2D& strides, const PaddingType& padding);

// ////////////////////////// Implementation details ///////////////////////////

namespace internal {

template <typename ResultType, typename InputType, typename PoolElementOperator>
Tensor<ResultType> Pool(const Tensor<InputType>& input,
                        const Position2D& window_size,
                        const Position2D& strides, const PaddingType& padding,
                        const PoolElementOperator& element_operator) {
  const InputType padding_value(0.0);
  // Input tensor is assumed to be of form (batch, height, width, channels).
  CHECK_EQ(input.dimension().num_dimensions(), 4);
  const int64_t input_height = input.dimension().dimension_size(1);
  const int64_t input_width = input.dimension().dimension_size(2);
  const Shape output_shape =
      Pool2dOutputShape(input.dimension(), window_size, strides, padding)
          .value();
  const int64_t output_batch = output_shape.dimension_size(0);
  const int64_t output_height = output_shape.dimension_size(1);
  const int64_t output_width = output_shape.dimension_size(2);
  const int64_t output_channels = output_shape.dimension_size(3);
  WindowExtractor2D window_extractor;
  TFOPT_CHECK_OK(window_extractor.Initialize(
      Position2D(input_height, input_width), window_size, strides, padding));
  Tensor<ResultType> result(output_shape);

  int64_t output_flat_index = 0;
  // NOTE: for output_flat_index to be in the right order, we must
  // loop in the order (batch, y, x, channel), as the output is defined.
  for (int64_t ob = 0; ob < output_batch; ++ob) {
    for (int64_t oy = 0; oy < output_height; ++oy) {
      for (int64_t ox = 0; ox < output_width; ++ox) {
        for (int64_t oc = 0; oc < output_channels; ++oc) {
          const int64_t ib = ob;
          const int64_t ic = oc;
          const Rectangle rectangle =
              window_extractor.GetWindow(Position2D(oy, ox));
          std::vector<InputType> input_window;
          bool padding_found = false;
          for (int64_t iy = rectangle.start.row;
               iy < rectangle.start.row + rectangle.size.row; ++iy) {
            for (int64_t ix = rectangle.start.col;
                 ix < rectangle.start.col + rectangle.size.col; ++ix) {
              if (!window_extractor.IsPadding(Position2D(iy, ix))) {
                input_window.push_back(input.ValueSpan({ib, iy, ix, ic}));
              } else {
                if (!padding_found) {
                  input_window.push_back(padding_value);
                  padding_found = true;
                }
              }
            }
          }
          result.SetValueSpan(
              {ob, oy, ox, oc},
              element_operator(input_window, output_flat_index));
          ++output_flat_index;
        }
      }
    }
  }
  return result;
}

}  // namespace internal

template <typename T>
Tensor<T> MaxPool(const Tensor<T>& input, const Position2D& window_size,
                  const Position2D& strides, const PaddingType& padding) {
  MaxAllElements<T> element;
  return internal::Pool<T, T, MaxAllElements<T>>(input, window_size, strides,
                                                 padding, element);
}
}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_POOLING_H_
