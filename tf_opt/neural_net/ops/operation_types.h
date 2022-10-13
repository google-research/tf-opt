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

// This existence of this file is a quirk to avoid circular dependencies with
// operation_visitor.h.  Users should basically not need to use this value.
#ifndef TF_OPT_SHARED_OPS_OPERATION_TYPES_H_
#define TF_OPT_SHARED_OPS_OPERATION_TYPES_H_

namespace tf_opt {

enum class BinaryArithmeticOpType {
  kAdd = 1,
  kSubtract = 2,
  kMultiply = 3,
  kDivide = 4
};

enum class LinearReduction { kSum = 1, kMean = 2 };

enum class NonlinearReduction { kMax = 1, kMin = 2 };

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_OPERATION_TYPES_H_
