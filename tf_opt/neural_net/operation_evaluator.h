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

#ifndef TF_OPT_NEURAL_NET_OPERATION_EVALUATOR_H_
#define TF_OPT_NEURAL_NET_OPERATION_EVALUATOR_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/neural_net/ops/all_operations.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {
namespace internal {

absl::Status CheckInputShapesAreCorrect(const Operation* operation,
                                        const std::vector<Shape>& input_shapes);

}  // namespace internal

// An OperationVisitor that computes a 'ResultType' as a function of
// 'InputTensorType's when visiting each operation. The inputs and outputs are
// stored as state on each call to Visit(). To use this class, do not call
// Visit() or Operation::Accept() directly, instead, call Evaluate().
//
// Typically, use OperationEvaluator<T> (ResultType == InputTensorType) or
// UnsafeOperationEvaluator<T> (ResultType == StatusOr<InputType>) defined
// below instead of using this directly.
//
// The Visit() methods are implemented by delegating to EvaluateXXX() methods,
// which subclasses are responsible for implementing.
//
// When adding a new Operation XXX, you must add a new protected unimplemented
// method EvaluateXXX() and a new Visit(XXX) method that delegates to
// EvaluateXXX().
//
// ResultType must be moveable, need not be copyable.
template <typename ResultType, typename InputTensorType>
class AbstractOperationEvaluator : public OperationVisitor {
 public:
  ResultType Evaluate(const Operation* operation,
                      const std::vector<const InputTensorType*>& inputs) {
    inputs_ = inputs;
    operation->Accept(this);
    inputs_.clear();
    return std::move(result_);
  }

  // NOTE: Users can ignore the Visit() methods, see class comment.
  void Visit(const AddOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateAdd);
  }

  void Visit(const ClippedReluOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateClippedRelu);
  }

  void Visit(const ConcatOperation& operation) override {
    DoVisitN(operation, &AbstractOperationEvaluator::EvaluateConcat);
  }

  void Visit(const ConstantOperation& operation) override {
    DoVisit0(operation, &AbstractOperationEvaluator::EvaluateConstant);
  }

  void Visit(const Conv1dOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateConv1d);
  }

  void Visit(const Conv2dOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateConv2d);
  }

  void Visit(const DivideOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateDivide);
  }

  void Visit(const EmbeddingLookupOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateEmbeddingLookup);
  }

  void Visit(const ExpandDimsOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateExpandDims);
  }

  void Visit(const MatmulOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateMatmul);
  }

  void Visit(const MaxpoolOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateMaxpool);
  }

  void Visit(const MultiplyOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateMultiply);
  }

  void Visit(const ReduceMaxOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateReduceMax);
  }

  void Visit(const ReduceMeanOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateReduceMean);
  }

  void Visit(const ReduceMinOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateReduceMin);
  }

  void Visit(const ReduceSumOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateReduceSum);
  }

  void Visit(const ReluOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateRelu);
  }

  void Visit(const ReshapeOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateReshape);
  }

  void Visit(const SliceOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateSlice);
  }

  void Visit(const SqueezeOperation& operation) override {
    DoVisit1(operation, &AbstractOperationEvaluator::EvaluateSqueeze);
  }

  void Visit(const SubtractOperation& operation) override {
    DoVisit2(operation, &AbstractOperationEvaluator::EvaluateSubtract);
  }

  void Visit(const VariableOperation& operation) override {
    DoVisit0(operation, &AbstractOperationEvaluator::EvaluateVariable);
  }

 protected:
  // Will only be called on the arguments to EvaluateXXX.
  virtual Shape GetShape(const InputTensorType& tensor) const = 0;

  // NOTE: Subclasses must implement each of the EvaluateXXX() methods,
  // see class description.
  virtual ResultType EvaluateAdd(const AddOperation& operation,
                                 const InputTensorType& left,
                                 const InputTensorType& right) = 0;

  virtual ResultType EvaluateClippedRelu(const ClippedReluOperation& operation,
                                         const InputTensorType& input) = 0;

  virtual ResultType EvaluateConcat(
      const ConcatOperation& operation,
      const std::vector<const InputTensorType*>& inputs) = 0;

  virtual ResultType EvaluateConstant(const ConstantOperation& operation) = 0;

  virtual ResultType EvaluateConv1d(const Conv1dOperation& operation,
                                    const InputTensorType& value,
                                    const InputTensorType& filters) = 0;

  virtual ResultType EvaluateConv2d(const Conv2dOperation& operation,
                                    const InputTensorType& value,
                                    const InputTensorType& filters) = 0;

  virtual ResultType EvaluateDivide(const DivideOperation& operation,
                                    const InputTensorType& left,
                                    const InputTensorType& right) = 0;

  virtual ResultType EvaluateEmbeddingLookup(
      const EmbeddingLookupOperation& operation, const InputTensorType& params,
      const InputTensorType& ids) = 0;

  virtual ResultType EvaluateExpandDims(const ExpandDimsOperation& operation,
                                        const InputTensorType& input) = 0;

  virtual ResultType EvaluateMatmul(const MatmulOperation& operation,
                                    const InputTensorType& left,
                                    const InputTensorType& right) = 0;

  virtual ResultType EvaluateMaxpool(const MaxpoolOperation& operation,
                                     const InputTensorType& input) = 0;

  virtual ResultType EvaluateMultiply(const MultiplyOperation& operation,
                                      const InputTensorType& left,
                                      const InputTensorType& right) = 0;

  virtual ResultType EvaluateReduceMax(const ReduceMaxOperation& operation,
                                       const InputTensorType& input) = 0;
  virtual ResultType EvaluateReduceMean(const ReduceMeanOperation& operation,
                                        const InputTensorType& input) = 0;
  virtual ResultType EvaluateReduceMin(const ReduceMinOperation& operation,
                                       const InputTensorType& input) = 0;
  virtual ResultType EvaluateReduceSum(const ReduceSumOperation& operation,
                                       const InputTensorType& input) = 0;

  virtual ResultType EvaluateRelu(const ReluOperation& operation,
                                  const InputTensorType& input) = 0;

  virtual ResultType EvaluateReshape(const ReshapeOperation& operation,
                                     const InputTensorType& input) = 0;

  virtual ResultType EvaluateSlice(const SliceOperation& operation,
                                   const InputTensorType& input) = 0;

  virtual ResultType EvaluateSqueeze(const SqueezeOperation& operation,
                                     const InputTensorType& input) = 0;

  virtual ResultType EvaluateSubtract(const SubtractOperation& operation,
                                      const InputTensorType& left,
                                      const InputTensorType& right) = 0;

  virtual ResultType EvaluateVariable(const VariableOperation& variable) = 0;

 private:
  // Helper for Visit methods on operations that take in a list of inputs.
  // OpType is the subclass of Operation to invoke.  'Method' is a function
  // pointer to an EvaluateXXX() method with signature
  // ResultType (AbstractOperationEvaluator::*)(
  //     const OpType&, const std::vector<const InputType*>&).
  template <typename OpType, typename Method>
  void DoVisitN(const OpType& operation, Method m) {
    EnsureEvaluateReady(&operation);
    result_ = (this->*m)(operation, inputs_);
  }

  // Helper for Visit methods on operations that take in two inputs.
  // OpType is the subclass of Operation to invoke.  'Method' is a function
  // pointer to an EvaluateXXX() method with signature
  // ResultType (AbstractOperationEvaluator::*)(const OpType&, const InputType&,
  //                                            const InputType&).
  template <typename OpType, typename Method>
  void DoVisit2(const OpType& operation, Method m) {
    EnsureEvaluateReady(&operation);
    result_ = (this->*m)(operation, *inputs_[0], *inputs_[1]);
  }

  // Helper for Visit methods on operations that take in one input.
  // OpType is the subclass of Operation to invoke.  'Method' is a function
  // pointer to an EvaluateXXX() method with signature
  // ResultType (AbstractOperationEvaluator::*)(const OpType&,
  //                                            const InputType&).
  template <typename OpType, typename Method>
  void DoVisit1(const OpType& operation, Method m) {
    EnsureEvaluateReady(&operation);
    result_ = (this->*m)(operation, *inputs_[0]);
  }

  // Helper for Visit methods on operations that take in no inputs.
  // OpType is the subclass of Operation to invoke.  'Method' is a function
  // pointer to an EvaluateXXX() method with signature
  // ResultType (AbstractOperationEvaluator::*)(const OpType&).
  template <typename OpType, typename Method>
  void DoVisit0(const OpType& operation, Method m) {
    EnsureEvaluateReady(&operation);
    result_ = (this->*m)(operation);
  }

  // Prepares inputs_ to evaluate, CHECK fails on a shape error.
  void EnsureEvaluateReady(const Operation* operation) {
    std::vector<Shape> input_shapes;
    for (const InputTensorType* const input : inputs_) {
      input_shapes.push_back(GetShape(*input));
    }
    TFOPT_CHECK_OK(
        internal::CheckInputShapesAreCorrect(operation, input_shapes));
  }

  std::vector<const InputTensorType*> inputs_;
  ResultType result_;
};

// An AbstractOperationEvaluator where every operation cannot fail.
template <typename T>
using OperationEvaluator = AbstractOperationEvaluator<T, T>;

// An AbstractOperationEvaluator where every operation might fail.
// TODO: Delete this once we have a linearity operation evaluator. This
// is only needed because for MIP, we can fail on a Matmul if both inputs are
// MPTensors.
template <typename T>
using UnsafeOperationEvaluator =
    AbstractOperationEvaluator<absl::StatusOr<T>, T>;

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OPERATION_EVALUATOR_H_
