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

#include "tf_opt/neural_net/operation_validator.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tf_opt/open_source/status_macros.h"
#include "ortools/base/map_util.h"

namespace tf_opt {

OperationValidator::OperationValidator(absl::string_view operation_type_name,
                                       absl::string_view operation_name)
    : base_error_message_(absl::StrCat("Failed to validate operation ",
                                       operation_name, " of type ",
                                       operation_type_name, ": ")) {}

absl::Status OperationValidator::OperationValidationError(
    absl::string_view error_message) const {
  return absl::InvalidArgumentError(
      absl::StrCat(base_error_message_, error_message));
}

absl::StatusOr<double> OperationValidator::DoubleOption(
    const Operation::Options& options, absl::string_view option_name) const {
  const auto it = options.double_options.find(option_name);
  if (it == options.double_options.end()) {
    return OperationValidationError(
        absl::StrCat("Required double option not found: ", option_name));
  }
  return it->second;
}

absl::StatusOr<int> OperationValidator::IntegerOption(
    const Operation::Options& options, absl::string_view option_name) const {
  const auto it = options.integer_options.find(option_name);
  if (it == options.integer_options.end()) {
    return OperationValidationError(
        absl::StrCat("Required integer option not found: ", option_name));
  }
  return it->second;
}

absl::StatusOr<std::string> OperationValidator::StringOption(
    const Operation::Options& options, absl::string_view option_name) const {
  const auto it = options.string_options.find(option_name);
  if (it == options.string_options.end()) {
    return OperationValidationError(
        absl::StrCat("Required string option not found: ", option_name));
  }
  return it->second;
}

absl::StatusOr<std::vector<int64_t>> OperationValidator::IntegerListOption(
    const Operation::Options& options, absl::string_view option_name) const {
  const auto it = options.integer_list_options.find(option_name);
  if (it == options.integer_list_options.end()) {
    return OperationValidationError(
        absl::StrCat("Required integer list option not found: ", option_name));
  }
  return it->second;
}

absl::Status OperationValidator::ExpectOptionsSizeAtMost(
    const int64_t options_size, const int value) const {
  if (options_size > value) {
    return OperationValidationError(
        absl::StrCat("Expected number of options at most ", value,
                     ", found: ", options_size));
  }
  return absl::OkStatus();
}

absl::Status OperationValidator::ExpectOptionsEmpty(
    const int64_t options_size) const {
  return ExpectOptionsSizeAtMost(options_size, 0);
}

absl::Status OperationValidator::ExpectInputSizeAtMost(const int64_t input_size,
                                                       const int value) const {
  if (input_size > value) {
    return OperationValidationError(absl::StrCat(
        "Expected number of inputs at most ", value, ", found: ", input_size));
  }
  return absl::OkStatus();
}

absl::Status OperationValidator::ExpectInputSizeAtLeast(
    const int64_t input_size, const int value) const {
  if (input_size < value) {
    return OperationValidationError(absl::StrCat(
        "Expected number of inputs at least ", value, ", found: ", input_size));
  }
  return absl::OkStatus();
}

absl::Status OperationValidator::ExpectInputSizeEquals(const int64_t input_size,
                                                       const int value) const {
  if (input_size != value) {
    return OperationValidationError(
        absl::StrCat("Expected number of inputs equals to ", value,
                     ", found: ", input_size));
  }
  return absl::OkStatus();
}

absl::Status OperationValidator::ExpectOutputShapeEquals(
    const Shape& output_shape, const Shape& expected_shape) const {
  if (output_shape != expected_shape) {
    return OperationValidationError(
        absl::StrCat("Expected output shape: ", expected_shape.ToString(),
                     ", found: ", output_shape.ToString()));
  }
  return absl::OkStatus();
}

}  // namespace tf_opt
