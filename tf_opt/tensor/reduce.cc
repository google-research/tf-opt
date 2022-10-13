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

#include "tf_opt/tensor/reduce.h"

#include <cstdint>

#include "absl/status/statusor.h"

namespace tf_opt {

absl::StatusOr<Shape> ReduceOutputShape(const Shape& input_shape,
                                        const std::vector<int64_t>& axes) {
  std::vector<int64_t> output_dims;
  std::vector<bool> marked_for_removal(input_shape.num_dimensions(), false);

  for (int i = 0; i < axes.size(); ++i) {
    const int64_t axis = axes[i];
    if (i > 0 && axes[i] <= axes[i - 1]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "axes vector is not sorted or contains duplicates at index ", i,
          "."));
    }
    if (axis < 0 || axis >= input_shape.num_dimensions()) {
      return absl::InvalidArgumentError(
          absl::StrCat("axis=", axis, " should have been in[0..rank(input)=",
                       input_shape.num_dimensions(), ")."));
    }
    marked_for_removal[axis] = true;
  }
  for (int i = 0; i < input_shape.num_dimensions(); ++i) {
    if (marked_for_removal[i]) continue;
    output_dims.push_back(input_shape.dimension_size(i));
  }
  return Shape(output_dims);
}

}  // namespace tf_opt
