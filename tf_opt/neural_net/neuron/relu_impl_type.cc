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

#include "tf_opt/neural_net/neuron/relu_impl_type.h"

#include <iostream>

#include "ortools/base/logging.h"
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace tf_opt {
namespace {

constexpr const char kBigMName[] = "big_m";
constexpr const char kMultipleChoiceName[] = "multiple_choice";
constexpr const char kMultipleChoiceSimplifiedName[] =
    "multiple_choice_simplified";
constexpr const char kIdealExponentialName[] = "ideal_exponential";
constexpr const char kBigMRelaxation[] = "big_m_relaxation";

}  // namespace

const char* ToString(ReluImplementationType relu_impl) {
  switch (relu_impl) {
    case ReluImplementationType::kBigM:
      return kBigMName;
    case ReluImplementationType::kMultipleChoice:
      return kMultipleChoiceName;
    case ReluImplementationType::kMultipleChoiceSimplified:
      return kMultipleChoiceSimplifiedName;
    case ReluImplementationType::kIdealExponential:
      return kIdealExponentialName;
    case ReluImplementationType::kBigMRelaxation:
      return kBigMRelaxation;
    default:
      LOG(FATAL) << "Unknown ReluImplementationType " << relu_impl;
  }
}

bool ReluImplFromString(absl::string_view impl_name,
                        ReluImplementationType* relu_impl_out) {
  CHECK_NE(relu_impl_out, nullptr);
  if (impl_name == kBigMName) {
    *relu_impl_out = ReluImplementationType::kBigM;
    return true;
  } else if (impl_name == kMultipleChoiceName) {
    *relu_impl_out = ReluImplementationType::kMultipleChoice;
    return true;
  } else if (impl_name == kMultipleChoiceSimplifiedName) {
    *relu_impl_out = ReluImplementationType::kMultipleChoiceSimplified;
    return true;
  } else if (impl_name == kIdealExponentialName) {
    *relu_impl_out = ReluImplementationType::kIdealExponential;
    return true;
  } else if (impl_name == kBigMRelaxation) {
    *relu_impl_out = ReluImplementationType::kBigMRelaxation;
    return true;
  } else {
    return false;
  }
}

ReluImplementationType ReluImplFromStringOrDie(absl::string_view impl_name) {
  ReluImplementationType result;
  CHECK(ReluImplFromString(impl_name, &result))
      << "Unrecognized formulation name for relu: " << impl_name;
  return result;
}

std::ostream& operator<<(std::ostream& stream,
                         const ReluImplementationType& relu_impl) {
  stream << ToString(relu_impl);
  return stream;
}

}  // namespace tf_opt
