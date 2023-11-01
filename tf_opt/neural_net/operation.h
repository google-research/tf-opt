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

#ifndef TF_OPT_NEURAL_NET_OPERATION_H_
#define TF_OPT_NEURAL_NET_OPERATION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

class OperationVisitor;

// Base class for an operation in a neural network. Subclassed based on
// operation type.
//
// All subclasses must have a static initialization method with the signature:
//   static absl::StatusOr<AddOperation> GenericCreate(
//       string op_name, std::vector<Shape> input_shapes,
//       Shape output_shape, const Options& options);
class Operation {
 public:
  // Contains additional parameters for an operation, when initialized through
  // GenericInitialize.
  struct Options {
    absl::flat_hash_map<std::string, double> double_options;
    absl::flat_hash_map<std::string, int> integer_options;
    absl::flat_hash_map<std::string, std::string> string_options;
    absl::flat_hash_map<std::string, std::vector<int64_t>> integer_list_options;

    // An empty options object.
    Options() {}

    // Initialize an options object from a proto.
    explicit Options(const proto::Options& proto_options);

    // True if no options are set.
    bool Empty() const;

    // Returns the total number of options.
    int size() const;
  };

  virtual ~Operation() {}

  const std::string& name() const { return name_; }

  const std::vector<Shape>& input_shapes() const { return input_shapes_; }
  const Shape& input_shape(int i) const { return input_shapes_.at(i); }

  const Shape& output_shape() const { return output_shape_; }

  // As used by the "Visitor Pattern", e.g.
  // https://sourcemaking.com/design_patterns/visitor/cpp/2
  //
  // The implementation of this is always the same:
  //   class MyOp : public Operation {
  //     ...
  //     void Accept(OperationVisitor* visitor) const { visitor->Visit(*this); }
  //     ...
  //   };
  //
  // It lets the visitor view the operation as the subclass.
  virtual void Accept(OperationVisitor* visitor) const = 0;

  virtual proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const = 0;

 protected:
  Operation(std::string name, std::vector<Shape> input_shapes,
            Shape output_shape);

 private:
  std::string name_;
  std::vector<Shape> input_shapes_;
  Shape output_shape_;
};

template <typename OperationType, typename... Args>
OperationType CreateOrDie(Args&&... args) {
  return std::move(OperationType::Create(std::forward<Args>(args)...)).value();
}

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OPERATION_H_
