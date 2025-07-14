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

#ifndef TF_OPT_NEURAL_NET_OP_REGISTRY_H_
#define TF_OPT_NEURAL_NET_OP_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {
namespace op_registry {

// Given an op_type, and the arguments to Operation::GenericCreate, produces a
// new Operation instance of this type, or returns an error.
absl::StatusOr<std::unique_ptr<Operation>> MakeOperation(
    proto::OpType op_type, std::string op_name, std::vector<Shape> input_shapes,
    Shape output_shape, const Operation::Options& options);

}  // namespace op_registry
}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OP_REGISTRY_H_
