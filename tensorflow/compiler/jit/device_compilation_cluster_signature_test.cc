/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"

#include "absl/status/status.h"
#include <utility>
#include <vector>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "xla/client/client_library.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
using SignatureHash = DeviceCompilationClusterSignature::Hash;

TEST(DeviceCompilationClusterSignatureTest, SignatureEquality) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({4, 0});
  args[0].constant_value = Tensor(DT_INT32, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature s1,
                          DeviceCompilationClusterSignature::Build(fn, args));

  args[0].type = DT_FLOAT;
  args[0].constant_value = Tensor(DT_FLOAT, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature s2,
                          DeviceCompilationClusterSignature::Build(fn, args));

  args[0].shape = TensorShape({0, 4});
  args[0].constant_value = Tensor(DT_FLOAT, {0, 4});
  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature s3,
                          DeviceCompilationClusterSignature::Build(fn, args));

  std::vector<DeviceCompilationClusterSignature> signatures = {s1, s2, s3};
  for (int i = 0; i < signatures.size(); ++i) {
    for (int j = 0; j < signatures.size(); ++j) {
      EXPECT_EQ(i == j, signatures[i] == signatures[j])
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s2: " << signatures[j].HumanString();
      EXPECT_EQ(i == j,
                signatures[i].HumanString() == signatures[j].HumanString())
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s2: " << signatures[j].HumanString();
      EXPECT_EQ(i == j, SignatureHash()(signatures[i]) ==
                            SignatureHash()(signatures[j]))
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s1_hash: " << SignatureHash()(signatures[i]) << "\n"
          << "s2: " << signatures[j].HumanString() << "\n"
          << "s2_hash: " << SignatureHash()(signatures[j]);
    }
  }
}

TEST(DeviceCompilationClusterSignatureTest, SignatureUniqueness) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].constant_value = Tensor(DT_INT32, {4, 0});

  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({4, 0});

  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature s1,
                          DeviceCompilationClusterSignature::Build(fn, args));

  using std::swap;  // go/using-std-swap
  swap(args[0], args[1]);
  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature s2,
                          DeviceCompilationClusterSignature::Build(fn, args));

  EXPECT_NE(s1.HumanString(), s2.HumanString());
  EXPECT_NE(SignatureHash()(s1), SignatureHash()(s2));
  EXPECT_FALSE(s1 == s2);
}

TEST(DeviceCompilationClusterSignatureTest,
     BuildForNoResourceInputsMatchesArgumentBuild) {
  NameAttrList fn;
  fn.set_name("afunction");

  Tensor constant_tensor(DT_INT32, TensorShape({2}));
  constant_tensor.vec<int32>()(0) = 7;
  constant_tensor.vec<int32>()(1) = 11;

  Tensor parameter_tensor(DT_FLOAT, TensorShape({3, 4}));
  Tensor empty_tensor(DT_INT64, TensorShape({0}));

  std::vector<const Tensor*> inputs = {&constant_tensor, &parameter_tensor,
                                       &empty_tensor};
  std::vector<int> must_be_constant_idxs = {0};

  std::vector<XlaCompiler::Argument> args(3);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = constant_tensor.dtype();
  args[0].shape = constant_tensor.shape();
  args[0].constant_value = constant_tensor;

  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = parameter_tensor.dtype();
  args[1].shape = parameter_tensor.shape();

  args[2].kind = XlaCompiler::Argument::kConstant;
  args[2].type = empty_tensor.dtype();
  args[2].shape = empty_tensor.shape();
  args[2].constant_value = empty_tensor;

  TF_ASSERT_OK_AND_ASSIGN(DeviceCompilationClusterSignature from_args,
                          DeviceCompilationClusterSignature::Build(fn, args));
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceCompilationClusterSignature from_inputs,
      DeviceCompilationClusterSignature::BuildForNoResourceInputs(
          fn, inputs, must_be_constant_idxs));

  EXPECT_EQ(from_args.HumanString(), from_inputs.HumanString());
  EXPECT_EQ(SignatureHash()(from_args), SignatureHash()(from_inputs));
  EXPECT_TRUE(from_args == from_inputs);
}

TEST(DeviceCompilationClusterSignatureTest,
     BuildForNoResourceInputsRejectsResourceInputs) {
  NameAttrList fn;
  fn.set_name("afunction");

  Tensor resource_tensor(DT_RESOURCE, TensorShape({}));
  std::vector<const Tensor*> inputs = {&resource_tensor};

  absl::StatusOr<DeviceCompilationClusterSignature> signature_or =
      DeviceCompilationClusterSignature::BuildForNoResourceInputs(
          fn, inputs, /*must_be_constant_idxs=*/{});

  EXPECT_FALSE(signature_or.ok());
  EXPECT_EQ(signature_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(DeviceCompilationClusterSignatureTest,
     BuildForNoResourceInputsRejectsUnsortedConstantIndices) {
  NameAttrList fn;
  fn.set_name("afunction");

  Tensor tensor(DT_FLOAT, TensorShape({2}));
  std::vector<const Tensor*> inputs = {&tensor, &tensor};

  absl::StatusOr<DeviceCompilationClusterSignature> signature_or =
      DeviceCompilationClusterSignature::BuildForNoResourceInputs(
          fn, inputs, /*must_be_constant_idxs=*/{1, 0});

  EXPECT_FALSE(signature_or.ok());
  EXPECT_EQ(signature_or.status().code(), absl::StatusCode::kInvalidArgument);
}

void BM_BuildSignature(::testing::benchmark::State& state) {
  const int n_args = state.range(0);

  NameAttrList fn;
  fn.set_name("afunction");
  for (int i = 0; i < n_args; i++) {
    (*fn.mutable_attr())[absl::StrCat("T", i)].set_type(DT_FLOAT);
  }
  std::vector<XlaCompiler::Argument> args(n_args);
  for (int i = 0; i < n_args; i++) {
    args[i].kind = (((i % 3) == 0) ? XlaCompiler::Argument::kConstant
                                   : XlaCompiler::Argument::kParameter);
    args[i].type = DT_INT32;
    args[i].shape = TensorShape({4, 0});
    args[i].constant_value = Tensor(DT_INT32, {4, 0});
  }

  for (auto i : state) {
    auto s = DeviceCompilationClusterSignature::Build(fn, args);
    CHECK(s.ok());
    DeviceCompilationClusterSignature sig = std::move(s.value());
  }
}
BENCHMARK(BM_BuildSignature)->Arg(0)->Arg(1)->Arg(2)->Arg(5)->Arg(10);

}  // namespace
}  // namespace tensorflow
