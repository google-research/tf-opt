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

#ifndef TF_OPT_SHARED_OPS_CLIPPED_RELU_OPERATION_H_
#define TF_OPT_SHARED_OPS_CLIPPED_RELU_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/neuron/clipped_relu_impl_type.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Given an input MPTensor or DoubleTensor x, computes the output MPTensor
// y = min(max(x, 0), cap_). The min and max are applied componentwise.
//
// Multiple MIP formulations for y = min(max(a,0),cap_) (where a is a
// LinearExpr) are supported, as specified by setting the string option
// "formulation". Pick either "default" to just do what the solver thinks is
// best, or use one of ClippedReluOperation::OptionsFormulation* (see
// clipped_relu_impls.h for details). If "formulation" key is missing, "default"
// is used automatically.
//
// cap_ is a double option from TensorNode::Options with key "cap".
class ClippedReluOperation : public Operation {
 public:
  static constexpr const char kOptionsCapKey[] = "cap";
  static constexpr const char kOptionsFormulationKey[] = "formulation";
  static constexpr const char kOptionsFormulationDefault[] = "default";
  static const char* OptionsFormulationCompositeDirect();
  static const char* OptionsFormulationCompositeExtended();
  static const char* OptionsFormulationExtendedXExclusion();
  static const char* OptionsFormulationExtendedYExclusion();
  static const char* OptionsFormulationUnaryBigM();
  static const char* OptionsFormulationIncrementalBigM();

  static absl::StatusOr<ClippedReluOperation> Create(
      std::string op_name, Shape input_shape, double cap,
      ClippedReluImplementationType formulation = kDefaultClippedRelu);

  // Expected input format:
  //   input_shapes: the dimensions of a single tensor to transform.
  //   output_shape: The shape of the resulting tensor, same as the input.
  //   options: May contain a string option with key kOptionsFormulationKey, to
  //       pick a MIP formulation for ClippedReLU().  Must contain a double
  //       option with key kOptionsCapKey, the upper bound.
  static absl::StatusOr<ClippedReluOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  const double cap() const { return cap_; }
  const ClippedReluImplementationType formulation() const {
    return formulation_;
  }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  ClippedReluOperation(std::string op_name, Shape input_shape, double cap,
                       ClippedReluImplementationType formulation);
  double cap_;
  // TODO: move this into MIP world.
  ClippedReluImplementationType formulation_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_CLIPPED_RELU_OPERATION_H_
