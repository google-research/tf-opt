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

#include "tf_opt/neural_net/neuron/clipped_relu_impl_type.h"

#include "ortools/base/logging.h"

namespace tf_opt {

namespace {

constexpr const char kFormulationCompositeDirect[] = "composite_direct";
constexpr const char kFormulationCompositeExtended[] = "composite_extended";
constexpr const char kFormulationExtendedXExclusion[] = "extended_x_exclusion";
constexpr const char kFormulationExtendedYExclusion[] = "extended_y_exclusion";
constexpr const char kFormulationUnaryBigM[] = "unary_big_m";
constexpr const char kFormulationIncrementalBigM[] = "incremental_big_m";

}  // namespace

const char* ToString(ClippedReluImplementationType clipped_relu_impl) {
  switch (clipped_relu_impl) {
    case ClippedReluImplementationType::kCompositeDirect:
      return kFormulationCompositeDirect;
    case ClippedReluImplementationType::kCompositeExtended:
      return kFormulationCompositeExtended;
    case ClippedReluImplementationType::kExtendedXExclusion:
      return kFormulationExtendedXExclusion;
    case ClippedReluImplementationType::kExtendedYExclusion:
      return kFormulationExtendedYExclusion;
    case ClippedReluImplementationType::kUnaryBigM:
      return kFormulationUnaryBigM;
    case ClippedReluImplementationType::kIncrementalBigM:
      return kFormulationIncrementalBigM;
    default:
      LOG(FATAL) << "Unknown ClippedReluImplementationType "
                 << clipped_relu_impl;
  }
}

bool ClippedReluImplFromString(
    absl::string_view impl_name,
    ClippedReluImplementationType* clipped_relu_impl_out) {
  CHECK_NE(clipped_relu_impl_out, nullptr);
  if (impl_name == kFormulationCompositeDirect) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kCompositeDirect;
    return true;
  } else if (impl_name == kFormulationCompositeExtended) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kCompositeExtended;
    return true;
  } else if (impl_name == kFormulationExtendedXExclusion) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kExtendedXExclusion;
    return true;
  } else if (impl_name == kFormulationExtendedYExclusion) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kExtendedYExclusion;
    return true;
  } else if (impl_name == kFormulationUnaryBigM) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kUnaryBigM;
    return true;
  } else if (impl_name == kFormulationIncrementalBigM) {
    *clipped_relu_impl_out = ClippedReluImplementationType::kIncrementalBigM;
    return true;
  } else {
    return false;
  }
}

ClippedReluImplementationType ClippedReluImplFromStringOrDie(
    absl::string_view impl_name) {
  ClippedReluImplementationType result;
  CHECK(ClippedReluImplFromString(impl_name, &result))
      << "Unrecognized formulation name for clipped relu: " << impl_name;
  return result;
}

std::ostream& operator<<(
    std::ostream& stream,
    const ClippedReluImplementationType& clipped_relu_impl) {
  stream << ToString(clipped_relu_impl);
  return stream;
}

}  // namespace tf_opt
