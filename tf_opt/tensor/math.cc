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

#include "tf_opt/tensor/math.h"

#include "glog/logging.h"
#include "absl/status/statusor.h"

namespace tf_opt {

absl::StatusOr<Shape> BinaryOpOutputShape(const Shape& left,
                                          const Shape& right) {
  const int max_dim = internal::MaxNumDimensions(left, right);
  const Shape pad_left = internal::BroadcastPadIfNeeded(left, max_dim);
  const Shape pad_right = internal::BroadcastPadIfNeeded(right, max_dim);
  return internal::ResultShape(pad_left, pad_right);
}

absl::StatusOr<Shape> MatMulOutputShape(const Shape& left, const Shape& right) {
  CHECK_GE(left.num_dimensions(), 2);
  CHECK_GE(right.num_dimensions(), 2);
  const int max_dim = internal::MaxNumDimensions(left, right);
  const Shape pad_left = internal::BroadcastPadIfNeeded(left, max_dim);
  const Shape pad_right = internal::BroadcastPadIfNeeded(right, max_dim);
  return internal::MatMulResultShape(pad_left, pad_right);
}

}  // namespace tf_opt
