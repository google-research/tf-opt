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

#include "tf_opt/neural_net/operation_evaluator.h"

#include "absl/status/status.h"

namespace tf_opt {
namespace internal {

absl::Status CheckInputShapesAreCorrect(
    const Operation* operation, const std::vector<Shape>& input_shapes) {
  if (operation->input_shapes().size() != input_shapes.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Node: ", operation->name(),
                     " expected: ", operation->input_shapes().size(),
                     " inputs, but found: ", input_shapes.size()));
  }
  for (int i = 0; i < input_shapes.size(); ++i) {
    if (operation->input_shapes()[i] != input_shapes[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Node: ", operation->name(), " input ", i,
          " expected shape: ", operation->input_shapes()[i].ToString(),
          " inputs, but found: ", input_shapes[i].ToString()));
    }
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tf_opt
