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

// Provides operations analogous to: tf.max.reduce_{max, min, mean, sum}.
//
// Given an input tensor and a list of axes to eliminate, produces a new tensor
// with the dimensions from axes removed, where for each output element, we
// compute a component-wise {maximum, minimum, mean, sum} over the eliminated
// dimensions.  For example:
//
// x = {{10, 14, 12},{13, 11, 15}}
// ReduceMax(x, axis=[0]) => {13, 14, 15} (shape = (3))
// ReduceMax(x, axis=[1]) => {14, 15} (shape = (2))
// ReduceMax(x, axis=[0, 1]) => 15 (shape = ())
//
// The public API of this file is the following types (they are typedefs):
//   ReduceMaxOperation
//   ReduceMinOperation
//   ReduceSumOperation
//   ReduceMeanOperation
//
// TODO: When no axis is specified, we should reduce everything to a
// scalar.
#ifndef TF_OPT_SHARED_OPS_REDUCE_OPERATIONS_H_
#define TF_OPT_SHARED_OPS_REDUCE_OPERATIONS_H_

#include <string>
#include <utility>
#include <vector>

#include "ortools/base/logging.h"
#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tf_opt/neural_net/neuron/maximum_impl_type.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/neural_net/ops/operation_types.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/reduce.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {
namespace reduce {

// For all reduce operations
ABSL_CONST_INIT extern const absl::string_view kOptionsAxesKey;

// For reduce_max and reduce_min
ABSL_CONST_INIT extern const absl::string_view kOptionsFormulationKey;
ABSL_CONST_INIT extern const absl::string_view kOptionsFormulationDefault;
absl::string_view OptionsFormulation(const MaximumImplementationType max_impl);
std::vector<std::string> AllNonlinearReduceImplementations();

}  // namespace reduce

template <LinearReduction R>
class LinearReduceOperation : public Operation {
 public:
  static absl::StatusOr<LinearReduceOperation<R>> Create(
      std::string op_name, Shape input_shape, const std::vector<int64_t>& axes);

  // Expected input format:
  //   input_shapes: The shape of a single tensor, to be reduced,
  //   output_shape: The shape of the result, like the input with the index
  //       from axis deleted.
  //   options: Must contain int option kOptionsAxisKey, the axis to reduce on.
  static absl::StatusOr<LinearReduceOperation<R>> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  const std::vector<int64_t>& axes() const { return axes_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  LinearReduceOperation(std::string op_name, Shape input_shape,
                        Shape output_shape, const std::vector<int64_t>& axes)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        axes_(axes) {}

  std::vector<int64_t> axes_;
};

using ReduceSumOperation = LinearReduceOperation<LinearReduction::kSum>;
using ReduceMeanOperation = LinearReduceOperation<LinearReduction::kMean>;

template <NonlinearReduction R>
class NonlinearReduceOperation : public Operation {
 public:
  static absl::StatusOr<Shape> OutputShape(const Shape& input_shape,
                                           const std::vector<int64_t>& axes) {
    return ReduceOutputShape(input_shape, axes);
  }

  static absl::StatusOr<NonlinearReduceOperation<R>> Create(
      std::string op_name, Shape input_shape, const std::vector<int64_t>& axes,
      MaximumImplementationType formulation = kDefaultMaximum);

  // Expected input format:
  //   input_shapes: The shape of a single tensor, to be reduced,
  //   output_shape: The shape of the result, like the input with the index
  //       from axis deleted.
  //   options: Must contain int option kOptionsAxisKey, the axis to reduce on.
  //            May optionally specify a string option with key
  //            kOptionsFormulationKey to pick a MIP formulation of max.
  static absl::StatusOr<NonlinearReduceOperation<R>> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  const std::vector<int64_t>& axes() const { return axes_; }
  MaximumImplementationType formulation() const { return formulation_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  NonlinearReduceOperation(std::string op_name, Shape input_shape,
                           Shape output_shape, const std::vector<int64_t>& axes,
                           MaximumImplementationType formulation)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        axes_(axes),
        formulation_(formulation) {}

  std::vector<int64_t> axes_;
  MaximumImplementationType formulation_;
};

// Supports multiple MIP formulations for modeling the max relationship over
// LinearExpr.  See the options below, and tf_opt/optimize/neuron/maximum*.h
// for more details.
//
// A special formulation is the "epigraph" formulation, which models the
// epigraph of the max function (y >= x for each input x). This is used for
// cases when the optimum is known to fall in the max function, such as when
// minimizing the output and for implementing a linearization of the softmax
// output in verification problems
using ReduceMaxOperation = NonlinearReduceOperation<NonlinearReduction::kMax>;
using ReduceMinOperation = NonlinearReduceOperation<NonlinearReduction::kMin>;

// //////////////// LinearReduceOperation Implementation ///////////////////////

template <LinearReduction R>
absl::StatusOr<LinearReduceOperation<R>> LinearReduceOperation<R>::Create(
    std::string op_name, Shape input_shape, const std::vector<int64_t>& axes) {
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         ReduceOutputShape(input_shape, axes));
  return LinearReduceOperation<R>(std::move(op_name), std::move(input_shape),
                                  std::move(output_shape), axes);
}

template <LinearReduction R>
absl::StatusOr<LinearReduceOperation<R>>
LinearReduceOperation<R>::GenericCreate(std::string op_name,
                                        std::vector<Shape> input_shapes,
                                        Shape output_shape,
                                        const Options& options) {
  OperationValidator validator("LinearReductionOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 1));
  TFOPT_ASSIGN_OR_RETURN(
      const std::vector<int64_t> axes,
      validator.IntegerListOption(options, reduce::kOptionsAxesKey));
  TFOPT_ASSIGN_OR_RETURN(
      LinearReduceOperation<R> op,
      Create(std::move(op_name), std::move(input_shapes[0]), axes));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return std::move(op);
}
template <LinearReduction R>
proto::TensorNode LinearReduceOperation<R>::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  switch (R) {
    case LinearReduction::kMean:
      result.set_op_type(proto::OpType::REDUCE_MEAN);
      break;
    case LinearReduction::kSum:
      result.set_op_type(proto::OpType::REDUCE_SUM);
      break;
  }
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  proto::Options::IntegerListOption* axes =
      result.mutable_options()->add_integer_list_options();
  axes->set_name(reduce::kOptionsAxesKey.data(),
                 reduce::kOptionsAxesKey.size());
  *axes->mutable_value() = {axes_.begin(), axes_.end()};
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

// ////////////// NonlinearReduceOperation Implementation //////////////////////

template <NonlinearReduction R>
absl::StatusOr<NonlinearReduceOperation<R>> NonlinearReduceOperation<R>::Create(
    std::string op_name, Shape input_shape, const std::vector<int64_t>& axes,
    MaximumImplementationType formulation) {
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape, OutputShape(input_shape, axes));
  return NonlinearReduceOperation<R>(std::move(op_name), std::move(input_shape),
                                     std::move(output_shape), axes,
                                     formulation);
}

template <NonlinearReduction R>
absl::StatusOr<NonlinearReduceOperation<R>>
NonlinearReduceOperation<R>::GenericCreate(std::string op_name,
                                           std::vector<Shape> input_shapes,
                                           Shape output_shape,
                                           const Options& options) {
  OperationValidator validator("NonlinearReduceOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 2));
  TFOPT_ASSIGN_OR_RETURN(
      const std::vector<int64_t> axes,
      validator.IntegerListOption(options, reduce::kOptionsAxesKey));
  MaximumImplementationType formulation = kDefaultMaximum;
  {
    std::string formulation_name =
        std::string(reduce::kOptionsFormulationDefault);
    if (options.string_options.contains(reduce::kOptionsFormulationKey)) {
      formulation_name =
          options.string_options.at(reduce::kOptionsFormulationKey);
    }
    if (formulation_name != reduce::kOptionsFormulationDefault &&
        !formulation_name.empty()) {
      if (!MaximumImplFromString(formulation_name, &formulation)) {
        return validator.OperationValidationError(absl::StrCat(
            "Unrecognized formulation name for maximum: ", formulation_name));
      }
    }
  }
  TFOPT_ASSIGN_OR_RETURN(NonlinearReduceOperation<R> op,
                         Create(std::move(op_name), std::move(input_shapes[0]),
                                axes, formulation));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return std::move(op);
}

template <NonlinearReduction R>
proto::TensorNode NonlinearReduceOperation<R>::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  switch (R) {
    case NonlinearReduction::kMax:
      result.set_op_type(proto::OpType::REDUCE_MAX);
      break;
    case NonlinearReduction::kMin:
      result.set_op_type(proto::OpType::REDUCE_MIN);
      break;
  }
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  proto::Options::IntegerListOption* axes =
      result.mutable_options()->add_integer_list_options();
  axes->set_name(reduce::kOptionsAxesKey.data(),
                 reduce::kOptionsAxesKey.size());
  *axes->mutable_value() = {axes_.begin(), axes_.end()};
  if (formulation_ != kDefaultMaximum) {
    proto::Options::StringOption* formulation =
        result.mutable_options()->add_string_options();
    formulation->set_name(reduce::kOptionsFormulationKey.data(),
                          reduce::kOptionsFormulationKey.size());
    formulation->set_value(ToString(formulation_));
  }
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_REDUCE_OPERATIONS_H_
