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

// This is a forwarding header.  Several files need to depend on every
// operation, they should just depend on this.
#ifndef TF_OPT_SHARED_OPS_ALL_OPERATIONS_H_
#define TF_OPT_SHARED_OPS_ALL_OPERATIONS_H_

#include "tf_opt/neural_net/ops/arithmetic_operations.h"
#include "tf_opt/neural_net/ops/clipped_relu_operation.h"
#include "tf_opt/neural_net/ops/concat_operation.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/neural_net/ops/conv1d_operation.h"
#include "tf_opt/neural_net/ops/conv2d_operation.h"
#include "tf_opt/neural_net/ops/embedding_lookup_operation.h"
#include "tf_opt/neural_net/ops/expand_dims_operation.h"
#include "tf_opt/neural_net/ops/matmul_operation.h"
#include "tf_opt/neural_net/ops/maxpool_operation.h"
#include "tf_opt/neural_net/ops/reduce_operations.h"
#include "tf_opt/neural_net/ops/relu_operation.h"
#include "tf_opt/neural_net/ops/reshape_operation.h"
#include "tf_opt/neural_net/ops/slice_operation.h"
#include "tf_opt/neural_net/ops/squeeze_operation.h"
#include "tf_opt/neural_net/ops/variable_operation.h"

#endif  // TF_OPT_SHARED_OPS_ALL_OPERATIONS_H_
