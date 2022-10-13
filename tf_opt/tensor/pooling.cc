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

#include "tf_opt/tensor/pooling.h"

#include <cstdint>

#include "absl/status/statusor.h"

namespace tf_opt {

absl::StatusOr<Shape> Pool2dOutputShape(const Shape& input_shape,
                                        const Position2D& window_size,
                                        const Position2D& strides,
                                        const PaddingType& padding) {
  if (input_shape.num_dimensions() != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected input to be rank four, with shape (batch, "
                     "height, width, channels), but had shape: ",
                     input_shape.ToString()));
  }
  const int64_t input_height = input_shape.dimension_size(1);
  const int64_t input_width = input_shape.dimension_size(2);

  WindowExtractor2D window_extractor;
  TFOPT_RETURN_IF_ERROR(window_extractor.Initialize(
      Position2D(input_height, input_width), window_size, strides, padding));

  const int64_t output_batch = input_shape.dimension_size(0);
  const int64_t output_height = window_extractor.output_size().row;
  const int64_t output_width = window_extractor.output_size().col;
  const int64_t output_channels = input_shape.dimension_size(3);
  return Shape({output_batch, output_height, output_width, output_channels});
}

}  // namespace tf_opt
