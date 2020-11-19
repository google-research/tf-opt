// Copyright 2020 The tf.opt Authors.
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

#include "tf_opt/neural_net/neuron/maximum_impl_type.h"

#include <iostream>

#include "glog/logging.h"
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace tf_opt {

std::vector<MaximumImplementationType> AllMaximumImplementations() {
  return {MaximumImplementationType::kBigM,
          MaximumImplementationType::kExtended,
          MaximumImplementationType::kTightenedBigM,
          MaximumImplementationType::kOptimalBigM,
          MaximumImplementationType::kLogarithmicBigM,
          MaximumImplementationType::kEpigraph};
}

std::vector<MaximumImplementationType> AllExactMaximumImplementations() {
  std::vector<MaximumImplementationType> impls = AllMaximumImplementations();
  impls.erase(std::remove(impls.begin(), impls.end(),
                          MaximumImplementationType::kEpigraph),
              impls.end());
  return impls;
}

namespace {

constexpr const char kBigMName[] = "big_m";
constexpr const char kExtendedName[] = "extended";
constexpr const char kTightenedBigMName[] = "tightened_big_m";
constexpr const char kOptimalBigMName[] = "optimal_big_m";
constexpr const char kLogarithmicBigMName[] = "logarithmic_big_m";
constexpr const char kEpigraphName[] = "epigraph";

}  // namespace

const char* ToString(MaximumImplementationType maximum_impl) {
  switch (maximum_impl) {
    case MaximumImplementationType::kBigM:
      return kBigMName;
    case MaximumImplementationType::kExtended:
      return kExtendedName;
    case MaximumImplementationType::kTightenedBigM:
      return kTightenedBigMName;
    case MaximumImplementationType::kOptimalBigM:
      return kOptimalBigMName;
    case MaximumImplementationType::kLogarithmicBigM:
      return kLogarithmicBigMName;
    case MaximumImplementationType::kEpigraph:
      return kEpigraphName;
    default:
      LOG(FATAL) << "Unknown MaximumImplementationType " << maximum_impl;
  }
}

bool MaximumImplFromString(absl::string_view impl_name,
                           MaximumImplementationType* maximum_impl_out) {
  CHECK_NE(maximum_impl_out, nullptr);
  if (impl_name == kBigMName) {
    *maximum_impl_out = MaximumImplementationType::kBigM;
    return true;
  } else if (impl_name == kExtendedName) {
    *maximum_impl_out = MaximumImplementationType::kExtended;
    return true;
  } else if (impl_name == kTightenedBigMName) {
    *maximum_impl_out = MaximumImplementationType::kTightenedBigM;
    return true;
  } else if (impl_name == kOptimalBigMName) {
    *maximum_impl_out = MaximumImplementationType::kOptimalBigM;
    return true;
  } else if (impl_name == kLogarithmicBigMName) {
    *maximum_impl_out = MaximumImplementationType::kLogarithmicBigM;
    return true;
  } else if (impl_name == kEpigraphName) {
    *maximum_impl_out = MaximumImplementationType::kEpigraph;
    return true;
  } else {
    return false;
  }
}

MaximumImplementationType MaximumImplFromStringOrDie(
    absl::string_view impl_name) {
  MaximumImplementationType result;
  CHECK(MaximumImplFromString(impl_name, &result))
      << "Unrecognized formulation name for maximum: " << impl_name;
  return result;
}

std::ostream& operator<<(std::ostream& stream,
                         const MaximumImplementationType& maximum_impl) {
  stream << ToString(maximum_impl);
  return stream;
}

}  // namespace tf_opt
