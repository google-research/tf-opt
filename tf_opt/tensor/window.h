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

// Provides rectangular (2D) windows to be sweeped over a Tensor, used for
// example for convolutions and pooling for images. Supports strides and
// padding.
//
// See:
// https://www.tensorflow.org/api_guides/python/nn#Convolution
// Visualizations: https://github.com/vdumoulin/conv_arithmetic

#ifndef TF_OPT_TENSOR_WINDOW_H_
#define TF_OPT_TENSOR_WINDOW_H_

#include <cstdint>
#include <iostream>

#include "glog/logging.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tf_opt {

enum class PaddingType { SAME, VALID };

const char* ToString(PaddingType padding);

ABSL_MUST_USE_RESULT bool PaddingTypeFromString(absl::string_view padding_name,
                                                PaddingType* padding_out);

PaddingType PaddingTypeFromStringOrDie(absl::string_view padding_name);

std::ostream& operator<<(std::ostream& stream, const PaddingType& padding);

// A row-column pair. Note that the order is (row, col); that is, (y, x).
struct Position2D {
  int64_t row = 0;
  int64_t col = 0;

  Position2D() {}
  constexpr Position2D(int64_t row, int64_t col) : row(row), col(col) {}
};

inline bool operator==(const Position2D& lhs, const Position2D& rhs) {
  return lhs.row == rhs.row && lhs.col == rhs.col;
}
inline bool operator!=(const Position2D& lhs, const Position2D& rhs) {
  return !(lhs == rhs);
}

// A rectangle with start position and size.
struct Rectangle {
  Position2D start;
  Position2D size;

  Rectangle(Position2D start, Position2D size) : start(start), size(size) {}
  Rectangle(int64_t start_row, int64_t start_col, int64_t height, int64_t width)
      : start(Position2D(start_row, start_col)),
        size(Position2D(width, height)) {}
};

// Extracts rectangular windows (for e.g. a 2D convolutional or pooling
// operation), given input and window dimensions, strides, and padding. Each
// position within the output tensor is mapped to a window in the input tensor
// (through the GetWindow() function).
class WindowExtractor2D {
 public:
  // Computes padding information and output shape in preparation for the
  // GetWindow() function. Input sizes, window sizes, and stride values must be
  // positive, and if padding type is "valid", the window size cannot be larger
  // than the input size in either dimension.
  absl::Status Initialize(Position2D input_size, Position2D window_size,
                          Position2D strides, PaddingType padding_type);

  // Given a position in the output, returns as output parameters the
  // corresponding window for the input. Returns whether an error has occurred.
  //
  // Note: This window may include positions that are negative or beyond the
  // dimensions of the input, representing padding, and the caller is
  // responsible for treating them as zero.
  Rectangle GetWindow(Position2D output_position) const;

  // Returns true if a position is part of padding.
  bool IsPadding(Position2D position) const;

  // Returns the size of the output.
  Position2D output_size() const { return output_size_; }

  // Returns the padding type used in the window extractor.
  PaddingType padding() const { return padding_; }

 private:
  Position2D input_size_;
  Position2D output_size_;
  Position2D window_size_;
  Position2D strides_;

  PaddingType padding_ = PaddingType::SAME;

  int64_t padding_top_ = 0;
  int64_t padding_bottom_ = 0;
  int64_t padding_left_ = 0;
  int64_t padding_right_ = 0;
};

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_WINDOW_H_
