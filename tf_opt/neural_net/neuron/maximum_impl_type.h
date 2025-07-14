// Copyright 2025 The tf.opt Authors.
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

// Provides an enum MaximumImplementationType listing the maximum
// implementations, along with utilities to convert between the enum and string.
#ifndef LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_MAXIMUM_MAXIMUM_IMPL_TYPE_H_
#define LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_MAXIMUM_MAXIMUM_IMPL_TYPE_H_

#include <iostream>
#include <vector>

#include "absl/strings/string_view.h"

namespace tf_opt {

enum class MaximumImplementationType {
  kBigM,
  kExtended,
  kTightenedBigM,
  kOptimalBigM,
  kLogarithmicBigM,
  kEpigraph,
};

constexpr MaximumImplementationType kDefaultMaximum =
    MaximumImplementationType::kTightenedBigM;

std::vector<MaximumImplementationType> AllMaximumImplementations();

std::vector<MaximumImplementationType> AllExactMaximumImplementations();

const char* ToString(MaximumImplementationType maximum_impl);

bool MaximumImplFromString(absl::string_view impl_name,
                           MaximumImplementationType* maximum_impl_out);

MaximumImplementationType MaximumImplFromStringOrDie(
    absl::string_view impl_name);

std::ostream& operator<<(std::ostream& stream,
                         const MaximumImplementationType& maximum_impl);

}  // namespace tf_opt

#endif  // LEARNING_BRAIN_RESEARCH_TF_OPT_OPTIMIZE_NEURON_MAXIMUM_MAXIMUM_IMPL_TYPE_H_
