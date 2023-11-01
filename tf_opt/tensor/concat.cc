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

#include "tf_opt/tensor/concat.h"

#include <cstdint>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"

namespace tf_opt {

absl::StatusOr<Shape> ConcatOutputShape(const std::vector<Shape>& input_shapes,
                                        int axis) {
  if (input_shapes.empty()) {
    return absl::InvalidArgumentError(
        "Concat must have at least one input, found none.");
  }
  const Shape& base_shape = input_shapes[0];
  for (int i = 1; i < input_shapes.size(); ++i) {
    if (base_shape.num_dimensions() != input_shapes[i].num_dimensions()) {
      return absl::InvalidArgumentError(
          absl::StrCat("All inputs to concat must have shapes with equal rank "
                       "(num_dimensions()), but rank at position 0 was: ",
                       base_shape.num_dimensions(), " and rank at position ", i,
                       " was: ", input_shapes[i].num_dimensions()));
    }
  }
  if (axis < 0 || axis >= base_shape.num_dimensions()) {
    return absl::InvalidArgumentError(
        absl::StrCat("axis must be in [0..input_shapes[0].num_dimensions()=",
                     base_shape.num_dimensions(), "), but found axis=", axis));
  }
  int axis_size = 0;
  for (int i = 0; i < input_shapes.size(); ++i) {
    const Shape& input_shape = input_shapes[i];
    // Check that all the dimensions except the final dimension have the same
    // value.
    for (int j = 0; j < base_shape.num_dimensions(); ++j) {
      if (j == axis) {
        axis_size += input_shape.dimension_sizes()[j];
      } else {
        if (input_shape.dimension_sizes()[j] !=
            base_shape.dimension_sizes()[j]) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Inputs to concat must agree in every dimension except axis=",
              axis, " but input 0=", base_shape.ToString(), "input ", i, "=",
              input_shape.ToString(), " disagree on dimension: ", j));
        }
      }
    }
  }
  std::vector<int64_t> result_dims(base_shape.dimension_sizes());
  result_dims[axis] = axis_size;
  return Shape(result_dims);
}

namespace internal {

ConcatLookupTable::ConcatLookupTable(const std::vector<int64_t>& list_sizes) {
  int cummulative_sum = 0;
  cumulative_list_sizes_.reserve(list_sizes.size());
  for (int i = 0; i < list_sizes.size(); ++i) {
    cumulative_list_sizes_.push_back(cummulative_sum);
    cummulative_sum += list_sizes[i];
    for (int64_t j = 0; j < list_sizes[i]; j++) {
      concat_index_to_init_list_index_.push_back(i);
    }
  }
}

std::pair<int, int64_t> ConcatLookupTable::Lookup(int64_t concat_index) const {
  CHECK_GE(concat_index, 0);
  CHECK_LT(concat_index, concat_index_to_init_list_index_.size());
  const int64_t list = concat_index_to_init_list_index_[concat_index];
  const int64_t position_in_list_out =
      concat_index - cumulative_list_sizes_[list];
  return {list, position_in_list_out};
}

}  // namespace internal

}  // namespace tf_opt
