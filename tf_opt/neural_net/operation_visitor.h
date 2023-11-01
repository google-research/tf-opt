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

#ifndef TF_OPT_NEURAL_NET_OPERATION_VISITOR_H_
#define TF_OPT_NEURAL_NET_OPERATION_VISITOR_H_

#include "tf_opt/neural_net/ops/operation_types.h"

namespace tf_opt {

template <BinaryArithmeticOpType T>
class BinaryArithmeticOperation;
class ClippedReluOperation;
class ConcatOperation;
class ConstantOperation;
class Conv1dOperation;
class Conv2dOperation;
class EmbeddingLookupOperation;
class ExpandDimsOperation;
template <LinearReduction T>
class LinearReduceOperation;
class MatmulOperation;
class MaxpoolOperation;
template <NonlinearReduction T>
class NonlinearReduceOperation;
class ReluOperation;
class ReshapeOperation;
class SliceOperation;
class SqueezeOperation;
class VariableOperation;

// All subclasses of Operation have a visit method that enables "double
// dispatch" (dispatch based of both the visitor and the operation), see
// "Visitor Pattern", e.g.
// https://sourcemaking.com/design_patterns/visitor/cpp/2.
class OperationVisitor {
 public:
  virtual ~OperationVisitor() {}

  // Keep alphabetical!
  virtual void Visit(
      const BinaryArithmeticOperation<BinaryArithmeticOpType::kAdd>&
          operation) = 0;  // AddOperation
  virtual void Visit(
      const BinaryArithmeticOperation<BinaryArithmeticOpType::kDivide>&
          operation) = 0;  // DivideOperation
  virtual void Visit(
      const BinaryArithmeticOperation<BinaryArithmeticOpType::kMultiply>&
          operation) = 0;  // MultiplyOperation
  virtual void Visit(
      const BinaryArithmeticOperation<BinaryArithmeticOpType::kSubtract>&
          operation) = 0;  // SubtractOperation
  virtual void Visit(const ClippedReluOperation& operation) = 0;
  virtual void Visit(const ConcatOperation& operation) = 0;
  virtual void Visit(const ConstantOperation& operation) = 0;
  virtual void Visit(const Conv1dOperation& operation) = 0;
  virtual void Visit(const Conv2dOperation& operation) = 0;
  virtual void Visit(const EmbeddingLookupOperation& operation) = 0;
  virtual void Visit(const ExpandDimsOperation& operation) = 0;
  virtual void Visit(const LinearReduceOperation<LinearReduction::kMean>&
                         operation) = 0;  // ReduceMeanOperation
  virtual void Visit(const LinearReduceOperation<LinearReduction::kSum>&
                         operation) = 0;  // ReduceSumOperation
  virtual void Visit(const MatmulOperation& operation) = 0;
  virtual void Visit(const MaxpoolOperation& operation) = 0;
  virtual void Visit(const NonlinearReduceOperation<NonlinearReduction::kMax>&
                         operation) = 0;  // ReduceMaxOperation
  virtual void Visit(const NonlinearReduceOperation<NonlinearReduction::kMin>&
                         operation) = 0;  // ReduceMinOperation
  virtual void Visit(const ReluOperation& operation) = 0;
  virtual void Visit(const ReshapeOperation& operation) = 0;
  virtual void Visit(const SliceOperation& operation) = 0;
  virtual void Visit(const SqueezeOperation& operation) = 0;
  virtual void Visit(const VariableOperation& operation) = 0;
};

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OPERATION_VISITOR_H_
