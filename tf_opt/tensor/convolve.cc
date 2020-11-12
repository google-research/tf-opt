// Copyright 2020 The tf.opt Authors.
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

#include "tf_opt/tensor/convolve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tf_opt {

using ::absl::InvalidArgumentError;
using ::absl::OkStatus;
using ::absl::Status;

absl::StatusOr<Shape> Conv1dOutputShape(const Shape& input_shape,
                                        const Shape& filter_shape,
                                        const int stride,
                                        const PaddingType padding_type) {
  TFOPT_RETURN_IF_ERROR(
      internal::Conv1dValidateShapes(input_shape, filter_shape));
  const Conv1dInputShape input(&input_shape);
  const Conv1dFilterShape filter(&filter_shape);

  TFOPT_ASSIGN_OR_RETURN(Shape result2d,
                         Conv2dOutputShape(input.shape2d(), filter.shape2d(),
                                           Position2D(1, stride), padding_type),
                         _ << "on conv1d inside conv2d");
  CHECK_EQ(result2d.num_dimensions(), 4);
  return Shape({result2d.dimension_size(0), result2d.dimension_size(2),
                result2d.dimension_size(3)});  // Removes height.
}

absl::StatusOr<Shape> Conv2dOutputShape(const Shape& input_shape,
                                        const Shape& filter_shape,
                                        const Position2D strides,
                                        const PaddingType padding_type) {
  TFOPT_RETURN_IF_ERROR(
      internal::Conv2dValidateShapes(input_shape, filter_shape));
  const Conv2dInputShape input(&input_shape);
  const Conv2dFilterShape filter(&filter_shape);

  WindowExtractor2D window_extractor;
  TFOPT_RETURN_IF_ERROR(window_extractor.Initialize(
      input.RegionSize(), filter.RegionSize(),
      Position2D(strides.row, strides.col), padding_type));

  const int64_t output_batch = input.batch();
  const int64_t output_height = window_extractor.output_size().row;
  const int64_t output_width = window_extractor.output_size().col;
  const int64_t output_channels = filter.out_channels();
  return Shape({output_batch, output_height, output_width, output_channels});
}

namespace internal {

Status Conv1dValidateShapes(const Shape& input_shape,
                            const Shape& filter_shape) {
  if (input_shape.num_dimensions() != 3) {
    return InvalidArgumentError(
        absl::StrCat("Expected input shape to have rank three, found: ",
                     input_shape.ToString()));
  }
  if (filter_shape.num_dimensions() != 3) {
    return InvalidArgumentError(
        absl::StrCat("Expected filter shape to have rank three, found: ",
                     filter_shape.ToString()));
  }
  return OkStatus();
}

Status Conv2dValidateShapes(const Shape& input_shape,
                            const Shape& filter_shape) {
  if (input_shape.num_dimensions() != 4) {
    return InvalidArgumentError(
        absl::StrCat("Expected input shape to have rank four, found: ",
                     input_shape.ToString()));
  }
  if (filter_shape.num_dimensions() != 4) {
    return InvalidArgumentError(
        absl::StrCat("Expected filter shape to have rank four, found: ",
                     filter_shape.ToString()));
  }
  Conv2dInputShape input(&input_shape);
  Conv2dFilterShape filter(&filter_shape);
  if (input.channels() != filter.in_channels()) {
    return InvalidArgumentError(absl::StrCat(
        "Num input channels: ", input.channels(),
        " (input format [batch, height, width, in_channels], shape=",
        input_shape.ToString(),
        ") should be equal to filter input channels: ", filter.in_channels(),
        "(filter format [filter_height, filter_width, in_channels, ",
        "out_channels], shape=", filter_shape.ToString(), ")"));
  }
  return OkStatus();
}

}  // namespace internal

}  // namespace tf_opt
