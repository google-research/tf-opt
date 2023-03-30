// Copyright 2023 The tf.opt Authors.
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

// Provides an enum ReluImplementationType listing the ReLU implementations,
// along with utilities to convert between the enum and string.
#ifndef TF_OPT_NEURAL_NET_NEURON_RELU_IMPL_TYPE_H_
#define TF_OPT_NEURAL_NET_NEURON_RELU_IMPL_TYPE_H_

#include <iostream>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace tf_opt {

enum class ReluImplementationType {
  kBigM,
  kMultipleChoice,
  kMultipleChoiceSimplified,
  kIdealExponential,
  kBigMRelaxation,
};

constexpr ReluImplementationType kDefaultRelu = ReluImplementationType::kBigM;

const char* ToString(ReluImplementationType relu_impl);

ABSL_MUST_USE_RESULT bool ReluImplFromString(
    absl::string_view impl_name, ReluImplementationType* relu_impl_out);

ReluImplementationType ReluImplFromStringOrDie(absl::string_view impl_name);

std::ostream& operator<<(std::ostream& stream,
                         const ReluImplementationType& relu_impl);

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_NEURON_RELU_IMPL_TYPE_H_
