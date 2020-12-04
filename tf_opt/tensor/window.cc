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

#include "tf_opt/tensor/window.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {
namespace {


constexpr const char kSame[] = "SAME";
constexpr const char kValid[] = "VALID";

absl::Status CheckInitializeArgumentsPositive(const Position2D input_size,
                                              const Position2D window_size,
                                              const Position2D strides) {
  if (input_size.row <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected input height > 0, found: ", input_size.row));
  }
  if (input_size.col <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected input width > 0, found: ", input_size.col));
  }
  if (window_size.row <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected window height > 0, found: ", window_size.row));
  }
  if (window_size.col <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected window width > 0, found: ", window_size.col));
  }
  if (strides.row <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected stride row > 0, found: ", strides.row));
  }
  if (strides.col <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected stride col > 0, found: ", strides.col));
  }
  return absl::OkStatus();
}

int64_t DivRoundUp(const int64_t num, const int64_t denom) {
  return static_cast<int64_t>(std::ceil(static_cast<double>(num) / denom));
}

// This calculation is a bit tricky, but is derived from here:
// https://www.tensorflow.org/api_guides/python/nn#Convolution,
// see also
// https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding.
int64_t SamePaddingSize(const int64_t input_size, const int64_t stride_size,
                        const int64_t window_size) {
  if (input_size % stride_size == 0) {
    return std::max<int64_t>(window_size - stride_size, 0);
  } else {
    return std::max<int64_t>(window_size - (input_size % stride_size), 0);
  }
}

}  // namespace

const char* ToString(PaddingType padding) {
  switch (padding) {
    case PaddingType::SAME:
      return kSame;
    case PaddingType::VALID:
      return kValid;
  }
}

bool PaddingTypeFromString(absl::string_view padding_name,
                           PaddingType* padding_out) {
  CHECK_NE(padding_out, nullptr);
  if (padding_name == kSame) {
    *padding_out = PaddingType::SAME;
    return true;
  } else if (padding_name == kValid) {
    *padding_out = PaddingType::VALID;
    return true;
  } else {
    return false;
  }
}

PaddingType PaddingTypeFromStringOrDie(absl::string_view padding_name) {
  PaddingType result;
  CHECK(PaddingTypeFromString(padding_name, &result))
      << "Unknown padding type: " << padding_name;
  return result;
}

std::ostream& operator<<(std::ostream& stream, const PaddingType& padding) {
  stream << ToString(padding);
  return stream;
}

absl::Status WindowExtractor2D::Initialize(const Position2D input_size,
                                           const Position2D window_size,
                                           const Position2D strides,
                                           const PaddingType padding_type) {
  input_size_ = input_size;
  window_size_ = window_size;
  strides_ = strides;
  padding_ = padding_type;
  TFOPT_RETURN_IF_ERROR(
      CheckInitializeArgumentsPositive(input_size_, window_size_, strides_));

  // Compute the padding.  Padding is zero when VALID (default value).
  if (padding_ == PaddingType::SAME) {
    const int64_t pad_height =
        SamePaddingSize(input_size_.row, strides_.row, window_size_.row);
    const int64_t pad_width =
        SamePaddingSize(input_size_.col, strides_.col, window_size_.col);
    padding_top_ = pad_height / 2;
    padding_left_ = pad_width / 2;
    padding_bottom_ = pad_height - padding_top_;
    padding_right_ = pad_width - padding_left_;
  } else {
    padding_top_ = 0;
    padding_left_ = 0;
    padding_bottom_ = 0;
    padding_right_ = 0;
  }

  // Build the output shape.
  if (padding_ == PaddingType::SAME) {
    output_size_.row = DivRoundUp(input_size_.row, strides_.row);
    output_size_.col = DivRoundUp(input_size_.col, strides_.col);
  } else if (padding_ == PaddingType::VALID) {
    output_size_.row =
        DivRoundUp(input_size_.row - window_size_.row + 1, strides_.row);
    output_size_.col =
        DivRoundUp(input_size_.col - window_size_.col + 1, strides_.col);
  }

  if (output_size_.row <= 0 || output_size_.col <= 0) {
    return absl::InvalidArgumentError(
        "Output dimension is nonpositive; window does not fit in input");
  }

  return absl::OkStatus();
}

Rectangle WindowExtractor2D::GetWindow(Position2D output_position) const {
  CHECK_GE(output_position.col, 0);
  CHECK_LT(output_position.col, output_size_.col);
  CHECK_GE(output_position.row, 0);
  CHECK_LT(output_position.row, output_size_.row);

  const int64_t start_row = output_position.row * strides_.row - padding_top_;
  const int64_t start_col = output_position.col * strides_.col - padding_left_;
  return Rectangle(Position2D(start_row, start_col), window_size_);
}

bool WindowExtractor2D::IsPadding(Position2D position) const {
  return (position.row < 0 || position.row >= input_size_.row ||
          position.col < 0 || position.col >= input_size_.col);
}

}  // namespace tf_opt
