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

#ifndef TF_OPT_NEURAL_NET_OPERATION_TESTING_H_
#define TF_OPT_NEURAL_NET_OPERATION_TESTING_H_

#include "gmock/gmock.h"
#include "absl/types/span.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

testing::Matcher<const Operation&> OperationArgsAre(
    absl::string_view name, absl::Span<const Shape> input_shapes,
    const Shape& output_shape);

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OPERATION_TESTING_H_
