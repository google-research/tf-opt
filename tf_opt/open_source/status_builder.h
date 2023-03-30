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

#ifndef TF_OPT_OPEN_SOURCE_STATUS_BUILDER_H_
#define TF_OPT_OPEN_SOURCE_STATUS_BUILDER_H_

#include <sstream>

#include "absl/status/status.h"

namespace tf_opt {

class StatusBuilder {
 public:
  explicit StatusBuilder(const absl::Status& status)
      : code_(status.code()), needs_delimiter_(!status.message().empty()) {
    ss_ << std::string(status.message());
  }

  operator absl::Status() const {  // NOLINT
    return absl::Status(code_, ss_.str());
  }

  template <class T>
  StatusBuilder& operator<<(const T& t) {
    if (needs_delimiter_) {
      ss_ << "; ";
      needs_delimiter_ = false;
    }
    ss_ << t;
    return *this;
  }

  bool ok() const { return code_ == absl::StatusCode::kOk; }

 private:
  const absl::StatusCode code_;
  std::ostringstream ss_;
  bool needs_delimiter_;
};

}  // namespace tf_opt

#endif  // TF_OPT_OPEN_SOURCE_STATUS_BUILDER_H_
