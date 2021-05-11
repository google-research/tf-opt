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

#include "tf_opt/neural_net/ops/reduce_operations.h"

namespace tf_opt {
namespace reduce {

const absl::string_view kOptionsAxesKey = "axes";
const absl::string_view kOptionsFormulationKey = "formulation";
const absl::string_view kOptionsFormulationDefault = "default";

absl::string_view OptionsFormulation(const MaximumImplementationType max_impl) {
  return ToString(max_impl);
}
std::vector<std::string> AllNonlinearReduceImplementations() {
  std::vector<std::string> reduce_max_impls;
  for (const MaximumImplementationType max_impl : AllMaximumImplementations()) {
    reduce_max_impls.push_back(ToString(max_impl));
  }
  return reduce_max_impls;
}
}  // namespace reduce
}  // namespace tf_opt
