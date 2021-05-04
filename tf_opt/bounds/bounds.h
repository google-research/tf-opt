// Copyright 2021 The tf.opt Authors.
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

#ifndef TF_OPT_BOUNDS_BOUNDS_H_
#define TF_OPT_BOUNDS_BOUNDS_H_

#include <iostream>
#include <limits>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace tf_opt {

// Bounds-related overloaded functions defined in other files:
// In tf_opt/tensor/math.h:
//   string Tensor<Bounds>::ToString()

// Representation of lower and upper bounds that overloads standard operations
// (e.g. +, -, *, /), allowing for interval arithmetic with a natural syntax.
class Bounds {
 public:
  Bounds() : Bounds(0) {}

  static const Bounds Unbounded() {
    return Bounds(-std::numeric_limits<double>::infinity(),
                  +std::numeric_limits<double>::infinity());
  }

  explicit Bounds(const double d) : lb_(d), ub_(d) {}

  Bounds(const double lb, const double ub) : lb_(lb), ub_(ub) {}

  double lb() const { return lb_; }

  double ub() const { return ub_; }

  // [a,b] + [c,d] = [a+c, b+d]
  Bounds& operator+=(const Bounds rhs) {
    lb_ += rhs.lb();
    ub_ += rhs.ub();
    return *this;
  }

  Bounds& operator+=(const double rhs) {
    *this += Bounds(rhs);
    return *this;
  }

  // [a,b] - [c,d] = [a-d, b-c]
  Bounds& operator-=(const Bounds rhs) {
    lb_ -= rhs.ub();
    ub_ -= rhs.lb();
    return *this;
  }

  Bounds& operator-=(const double rhs) {
    *this -= Bounds(rhs);
    return *this;
  }

  // [a,b] * [c,d] = [min(a*c, a*d, b*c, b*d), max(a*c, a*d, b*c, b*d)]
  Bounds& operator*=(const Bounds rhs) {
    const double ll = lb_ * rhs.lb();
    const double lu = lb_ * rhs.ub();
    const double ul = ub_ * rhs.lb();
    const double uu = ub_ * rhs.ub();
    lb_ = std::min({ll, lu, ul, uu});
    ub_ = std::max({ll, lu, ul, uu});
    return *this;
  }

  Bounds& operator*=(const double rhs) {
    *this *= Bounds(rhs);
    return *this;
  }

  // In most cases:
  // [a,b] / [c,d] = [min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d)]
  // Special cases (in this order of precedence):
  //   If [c,d] == [0,0]: empty set (represented by [-infinity, +infinity])
  //   If [a,b] == [0,0]: [0,0]
  //   If [c,d] contains zero: [-infinity, +infinity]
  Bounds& operator/=(const Bounds rhs) {
    const double a = lb_;
    const double b = ub_;
    // Protect ourselves from computations leading to the lower bound of the
    // resulting interval being equal to +inf and the upper bound being equal
    // to -inf.
    const double c = rhs.lb() != 0.0 ? rhs.lb() : +0.0;
    const double d = rhs.ub() != 0.0 ? rhs.ub() : -0.0;
    if (c == 0.0 && d == 0.0) {
      // The actual result is "empty set". Since we do not support it, we return
      // a larger interval containing "empty set".
      lb_ = -std::numeric_limits<double>::infinity();
      ub_ = std::numeric_limits<double>::infinity();
    } else if (a == 0.0 && b == 0.0) {
      lb_ = 0.0;
      ub_ = 0.0;
    } else if (c < 0.0 && d > 0.0) {
      lb_ = -std::numeric_limits<double>::infinity();
      ub_ = std::numeric_limits<double>::infinity();
    } else {
      const double ll = lb_ / rhs.lb();
      const double lu = lb_ / rhs.ub();
      const double ul = ub_ / rhs.lb();
      const double uu = ub_ / rhs.ub();
      lb_ = std::min({ll, lu, ul, uu});
      ub_ = std::max({ll, lu, ul, uu});
    }
    return *this;
  }

  Bounds& operator/=(const double rhs) {
    *this /= Bounds(rhs);
    return *this;
  }

  // Turns [lb, ub] into [-ub, -lb].
  Bounds operator-() const { return Bounds(-ub_, -lb_); }

  friend Bounds operator+(const Bounds lhs, const Bounds rhs);
  friend Bounds operator-(const Bounds lhs, const Bounds rhs);
  friend Bounds operator*(const Bounds lhs, const Bounds rhs);
  friend Bounds operator/(const Bounds lhs, const Bounds rhs);

  friend std::ostream& operator<<(std::ostream& ostr, const Bounds& bounds);

  std::string ToString() const { return absl::StrCat("[", lb_, ",", ub_, "]"); }

  bool operator==(const Bounds other) const {
    return lb_ == other.lb() && ub_ == other.ub();
  }

  bool operator!=(const Bounds other) const { return !(*this == other); }

 private:
  double lb_;
  double ub_;
};

// [a,b] + [c,d] = [a+c, b+d]
inline Bounds operator+(Bounds lhs, const Bounds rhs) {
  lhs += rhs;
  return lhs;
}

inline Bounds operator+(const double lhs, const Bounds rhs) {
  return Bounds(lhs) + rhs;
}

inline Bounds operator+(const Bounds lhs, const double rhs) {
  return lhs + Bounds(rhs);
}

// [a,b] - [c,d] = [a-d, b-c]
inline Bounds operator-(Bounds lhs, const Bounds rhs) {
  lhs -= rhs;
  return lhs;
}

inline Bounds operator-(const double lhs, const Bounds rhs) {
  return Bounds(lhs) - rhs;
}

inline Bounds operator-(const Bounds lhs, const double rhs) {
  return lhs - Bounds(rhs);
}

// [a,b] * [c,d] = [min(a*c, a*d, b*c, b*d), max(a*c, a*d, b*c, b*d)]
inline Bounds operator*(Bounds lhs, const Bounds rhs) {
  lhs *= rhs;
  return lhs;
}

inline Bounds operator*(const double lhs, const Bounds rhs) {
  return Bounds(lhs) * rhs;
}

inline Bounds operator*(const Bounds lhs, const double rhs) {
  return lhs * Bounds(rhs);
}

// In most cases:
// [a,b] / [c,d] = [min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d)]
// Special cases (in this order of precedence):
//   If [c,d] == [0,0]: empty set (represented by [-infinity, +infinity])
//   If [a,b] == [0,0]: [0,0]
//   If [c,d] contains zero: [-infinity, +infinity]
inline Bounds operator/(Bounds lhs, const Bounds rhs) {
  lhs /= rhs;
  return lhs;
}

inline Bounds operator/(const double lhs, const Bounds rhs) {
  return Bounds(lhs) / rhs;
}

inline Bounds operator/(const Bounds lhs, const double rhs) {
  return lhs / Bounds(rhs);
}

inline Bounds Max(const Bounds b1, const Bounds b2) {
  return Bounds(std::max(b1.lb(), b2.lb()), std::max(b1.ub(), b2.ub()));
}

inline Bounds Max(absl::Span<const Bounds> bounds) {
  if (bounds.empty()) return Bounds::Unbounded();
  Bounds result = Bounds(-std::numeric_limits<double>::infinity());
  for (const Bounds& bound : bounds) {
    result = Max(result, bound);
  }
  return result;
}

inline Bounds Min(const Bounds b1, const Bounds b2) {
  return Bounds(std::min(b1.lb(), b2.lb()), std::min(b1.ub(), b2.ub()));
}

inline Bounds Intersect(const Bounds b1, const Bounds b2) {
  // TODO: This can result in infeasible bounds. Add a representation
  // of empty bounds if needed.
  return Bounds(std::max(b1.lb(), b2.lb()), std::min(b1.ub(), b2.ub()));
}

inline std::ostream& operator<<(std::ostream& ostr, const Bounds& bounds) {
  ostr << '[' << bounds.lb() << ", " << bounds.ub() << ']';
  return ostr;
}

inline Bounds TfOptLowest() {
  return Bounds(-std::numeric_limits<double>::infinity());
}

inline Bounds TfOptHighest() {
  return Bounds(std::numeric_limits<double>::infinity());
}

}  // namespace tf_opt

#endif  // TF_OPT_BOUNDS_BOUNDS_H_
