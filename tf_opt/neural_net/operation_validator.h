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

#ifndef TF_OPT_NEURAL_NET_OPERATION_VALIDATOR_H_
#define TF_OPT_NEURAL_NET_OPERATION_VALIDATOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Class with auxiliary functions for validating an Operation. Holds error
// messages. Typically the following is validated:
//   1. Options must be well formed according to operation specification.
//   2. The operation must take a correct number of inputs.
//   3. The shape of the inputs of an operation must be valid.
//   4. The shape of the output of an operation must be valid.
class OperationValidator {
 public:
  OperationValidator(absl::string_view operation_type_name,
                     absl::string_view operation_name);

  // Returns an InvalidArgumentError with a prefix indicating operation name.
  absl::Status OperationValidationError(absl::string_view error_message) const;

  // Returns the value of a double option or an error if it does not exist.
  absl::StatusOr<double> DoubleOption(const Operation::Options& options,
                                      absl::string_view option_name) const;

  // Returns the value of an integer option or an error if it does not exist.
  absl::StatusOr<int> IntegerOption(const Operation::Options& options,
                                    absl::string_view option_name) const;

  // Returns the value of a string option or an error if it does not exist.
  absl::StatusOr<std::string> StringOption(const Operation::Options& options,
                                           absl::string_view option_name) const;

  // Returns the value of an integer list option or an error if it does not
  // exist.
  absl::StatusOr<std::vector<int64_t>> IntegerListOption(
      const Operation::Options& options, absl::string_view option_name) const;

  // Returns an error if there are too many options. This may be used to take
  // into account optional options.
  absl::Status ExpectOptionsSizeAtMost(const int64_t options_size,
                                       const int value) const;

  // Returns an error unless options is empty.
  absl::Status ExpectOptionsEmpty(const int64_t options_size) const;

  // Returns an error unless input size is at most the given value.
  absl::Status ExpectInputSizeAtMost(const int64_t input_size,
                                     const int value) const;

  // Returns an error unless input size is at least the given value.
  absl::Status ExpectInputSizeAtLeast(const int64_t input_size,
                                      const int value) const;

  // Returns an error unless input size equals the given value.
  absl::Status ExpectInputSizeEquals(const int64_t input_size,
                                     const int value) const;

  // Returns an error unless output shape equals the expected shape.
  absl::Status ExpectOutputShapeEquals(const Shape& output_shape,
                                       const Shape& expected_shape) const;

  // Returns the base error message.
  const std::string& base_error_message() const { return base_error_message_; }

 private:
  const std::string base_error_message_;
};

}  // namespace tf_opt

#endif  // TF_OPT_NEURAL_NET_OPERATION_VALIDATOR_H_
