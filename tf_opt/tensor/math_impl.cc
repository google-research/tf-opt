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

#include "tf_opt/tensor/math_impl.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/statusor.h"

namespace tf_opt {
namespace internal {

std::vector<int64_t> Broadcaster::PaddedMultiIndex(
    const int64_t broadcast_index) const {
  std::vector<int64_t> multi_index =
      broadcast_shape_.ExpandIndex(broadcast_index);

  for (int i = 0; i < multi_index.size(); i++) {
    if (padded_shape_.dimension_size(i) == 1) {
      multi_index[i] = 0;
    }
  }
  return multi_index;
}

int64_t Broadcaster::BroadcastIndexToTrueIndex(
    const int64_t broadcast_index) const {
  return padded_shape_.FlattenIndex(PaddedMultiIndex(broadcast_index));
}

std::vector<int64_t> Broadcaster::BroadcastIndexToMatmulSliceArg(
    const int64_t broadcast_index, MultiplicationPosition mult_pos) const {
  std::vector<int64_t> multi_index = PaddedMultiIndex(broadcast_index);
  switch (mult_pos) {
    case MultiplicationPosition::kLeft:
      multi_index[multi_index.size() - 1] = -1;
      break;
    case MultiplicationPosition::kRight:
      multi_index[multi_index.size() - 2] = -1;
      break;
    default:
      LOG(FATAL);
  }
  const int64_t amount_padding =
      padded_shape_.num_dimensions() - true_shape_.num_dimensions();
  if (amount_padding > 0) {
    CHECK_LT(amount_padding, multi_index.size());
    multi_index.erase(multi_index.begin(),
                      multi_index.begin() + amount_padding);
  }
  return multi_index;
}

Shape BroadcastPadIfNeeded(const Shape& shape,
                           const int64_t target_num_dimensions) {
  if (target_num_dimensions > shape.num_dimensions()) {
    const int64_t num_ones = target_num_dimensions - shape.num_dimensions();
    std::vector<int64_t> result(num_ones, 1);
    result.insert(result.end(), shape.dimension_sizes().begin(),
                  shape.dimension_sizes().end());
    return Shape(result);
  } else {
    return shape;
  }
}

int64_t MaxNumDimensions(const Shape& shape_left, const Shape& shape_right) {
  return std::max(shape_left.num_dimensions(), shape_right.num_dimensions());
}

absl::StatusOr<Shape> ResultShape(const Shape& padded_left,
                                  const Shape& padded_right) {
  const int num_dimensions = padded_left.num_dimensions();
  CHECK_EQ(padded_right.num_dimensions(), num_dimensions);

  std::vector<int64_t> output_size(num_dimensions);
  for (int i = 0; i < num_dimensions; i++) {
    const int64_t a_size = padded_left.dimension_size(i);
    const int64_t b_size = padded_right.dimension_size(i);
    if (a_size != 1 && b_size != 1 && a_size != b_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Incompatible shapes left: ", padded_left.ToString(),
          " and right: ", padded_right.ToString(), " at index: ", i));
    }
    output_size[i] = std::max(a_size, b_size);
  }
  return Shape(output_size);
}

absl::StatusOr<Shape> MatMulResultShape(const Shape& padded_left,
                                        const Shape& padded_right) {
  const int num_dimensions = padded_left.num_dimensions();
  CHECK_EQ(padded_right.num_dimensions(), num_dimensions);
  std::vector<int64_t> output_size(num_dimensions);
  for (int i = 0; i < num_dimensions - 2; i++) {
    const int64_t left_size = padded_left.dimension_size(i);
    const int64_t right_size = padded_right.dimension_size(i);
    if (left_size != 1 && right_size != 1 && left_size != right_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("Incompatible shapes a: ", padded_left.ToString(),
                       " and b: ", padded_right.ToString(), " at index: ", i));
    }
    output_size[i] = std::max(left_size, right_size);
  }
  const int left_height = padded_left.dimension_size(num_dimensions - 2);
  const int left_width = padded_left.dimension_size(num_dimensions - 1);
  const int right_height = padded_right.dimension_size(num_dimensions - 2);
  const int right_width = padded_right.dimension_size(num_dimensions - 1);
  if (left_width != right_height) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incompatible shapes left: ", padded_left.ToString(), " and right: ",
        padded_right.ToString(), " last dimension of left=", left_width,
        " does not agree with next to last dimension of right=", right_height));
  }
  const int out_height = left_height;
  const int out_width = right_width;
  output_size[num_dimensions - 2] = out_height;
  output_size[num_dimensions - 1] = out_width;
  return Shape(output_size);
}

}  // namespace internal
}  // namespace tf_opt
