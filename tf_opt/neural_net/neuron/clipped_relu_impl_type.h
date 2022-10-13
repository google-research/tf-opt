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

// Provides an enum ClippedReluImplementationType listing the clipped ReLU
// implementations, along with utilities to convert between the enum and string.
#ifndef LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_CLIPPED_RELU_CLIPPED_RELU_IMPL_TYPE_H_
#define LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_CLIPPED_RELU_CLIPPED_RELU_IMPL_TYPE_H_

#include <iostream>

#include "absl/strings/string_view.h"

namespace tf_opt {

enum class ClippedReluImplementationType {
  kCompositeDirect,
  kCompositeExtended,
  kExtendedYExclusion,
  kExtendedXExclusion,
  kUnaryBigM,
  kIncrementalBigM,
};

constexpr ClippedReluImplementationType kDefaultClippedRelu =
    ClippedReluImplementationType::kUnaryBigM;

const char* ToString(ClippedReluImplementationType clipped_relu_impl);

bool ClippedReluImplFromString(
    absl::string_view impl_name,
    ClippedReluImplementationType* clipped_relu_impl_out);

ClippedReluImplementationType ClippedReluImplFromStringOrDie(
    absl::string_view impl_name);

std::ostream& operator<<(
    std::ostream& stream,
    const ClippedReluImplementationType& clipped_relu_impl);

}  // namespace tf_opt

#endif  // LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_CLIPPED_RELU_CLIPPED_RELU_IMPL_TYPE_H_
