/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_TF2XLA_DYNAMIC_EXPRESSION_UTILS_H_
#define TENSORFLOW_COMPILER_TF2XLA_DYNAMIC_EXPRESSION_UTILS_H_

#include "tensorflow/compiler/jit/flags.h"
#include "xla/shape_dynexpr.h"

namespace tensorflow {

inline bool ShouldTrackDynamicShapeExpressions() {
  const MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  return flags->tf_xla_enable_dynamic_sizes ||
         flags->tf_xla_cluster_single_dynamic_dim;
}

template <typename Shape>
inline void MaybeAddExpression(Shape* shape, xla::DynExpr* expr) {
  if (ShouldTrackDynamicShapeExpressions()) {
    shape->AddExpression(expr);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_DYNAMIC_EXPRESSION_UTILS_H_
