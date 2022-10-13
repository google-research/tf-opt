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

#include "tf_opt/neural_net/operation.h"

#include <cstdint>

#include "ortools/base/logging.h"
#include "ortools/base/map_util.h"

namespace tf_opt {

Operation::Options::Options(const proto::Options& proto_options) {
  for (const proto::Options::DoubleOption& d : proto_options.double_options()) {
    gtl::InsertOrDie(&double_options, d.name(), d.value());
  }
  for (const proto::Options::IntegerOption& i :
       proto_options.integer_options()) {
    gtl::InsertOrDie(&integer_options, i.name(), i.value());
  }
  for (const proto::Options::StringOption& s : proto_options.string_options()) {
    gtl::InsertOrDie(&string_options, s.name(), s.value());
  }
  for (const proto::Options::IntegerListOption& il :
       proto_options.integer_list_options()) {
    gtl::InsertOrDie(
        &integer_list_options, il.name(),
        std::vector<int64_t>(il.value().begin(), il.value().end()));
  }
}

bool Operation::Options::Empty() const {
  return double_options.empty() && integer_options.empty() &&
         string_options.empty() && integer_list_options.empty();
}

int Operation::Options::size() const {
  return static_cast<int>(double_options.size() + integer_options.size() +
                          string_options.size() + integer_list_options.size());
}

Operation::Operation(std::string name, std::vector<Shape> input_shapes,
                     Shape output_shape)
    : name_(std::move(name)),
      input_shapes_(std::move(input_shapes)),
      output_shape_(std::move(output_shape)) {}


}  // namespace tf_opt
