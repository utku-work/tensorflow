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

#include "tensorflow/compiler/jit/device_compilation_profiler.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/tests/device_compiler_test_helper.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace {

TEST(DeviceCompilationProfilerTest, RegisterExecution) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  for (int i = 0; i < 5; ++i) {
    profiler->RegisterExecution(function);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.execution_count, 5);
}

TEST(DeviceCompilationProfilerTest, RegisterCompilation) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  auto listener = std::make_unique<JitCompilationListener>();
  auto listener_ptr = listener.get();
  RegisterXlaActivityListener(std::move(listener));

  NameAttrList function;
  function.set_name("TestFunc");

  std::vector<XlaJitCompilationActivity> expected_activities;
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(profiler->RegisterCompilation(function, 4, false).ok());

    TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
    XlaJitCompilationActivity expected_activity;
    expected_activity.set_cluster_name(function.name());
    expected_activity.set_compile_count(stats.compile_count);
    expected_activity.set_compile_time_us(4);
    expected_activity.set_cumulative_compile_time_us(
        stats.cumulative_compile_time_us);
    expected_activity.set_used_persistent_cache(false);
    expected_activities.push_back(expected_activity);
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.compile_count, 5);
  EXPECT_EQ(stats.cumulative_compile_time_us, 5 * 4);

  // TODO(b/255826209): Use ::testing::EqualsProto once b/135192747 is fixed.
  const auto& actual_activities = listener_ptr->GetListenerHistory();
  EXPECT_EQ(actual_activities.size(), expected_activities.size());
  for (size_t i = 0; i < actual_activities.size(); ++i) {
    EXPECT_EQ(actual_activities[i].SerializeAsString(),
              expected_activities[i].SerializeAsString());
  }
}

TEST(DeviceCompilationProfilerTest, OngoingAsyncCompilations) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  for (int i = 0; i < 5; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 5);

  for (int i = 0; i < 5; ++i) {
    profiler->DecrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 0);

  for (int i = 0; i < 5; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
    profiler->DecrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 0);
}

TEST(DeviceCompilationProfilerTest, RegisterPhaseTiming) {
  // Phase timing samples should accumulate independently from compile counts.
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 7);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 3);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kCacheLookup, 5);

  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.signature_build.sample_count, 2);
  EXPECT_EQ(stats.signature_build.cumulative_time_us, 10);
  EXPECT_EQ(stats.cache_lookup.sample_count, 1);
  EXPECT_EQ(stats.cache_lookup.cumulative_time_us, 5);
}

TEST(DeviceCompilationProfilerTest, RegisterPhaseTimingOnlyComputeMode) {
  SetOnlyRecordXlaCompileOpComputeTimingForTesting(std::optional<bool>(true));

  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 7);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kXlaCompileOpCompute,
      11);

  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.signature_build.sample_count, 0);
  EXPECT_EQ(stats.signature_build.cumulative_time_us, 0);
  EXPECT_EQ(stats.xla_compile_op_compute.sample_count, 1);
  EXPECT_EQ(stats.xla_compile_op_compute.cumulative_time_us, 11);

  SetOnlyRecordXlaCompileOpComputeTimingForTesting(std::nullopt);
}

TEST(DeviceCompilationProfilerTest, CacheHitFastPathEnabledByDefault) {
  SetEnableDeviceCompilationCacheHitFastPathForTesting(std::nullopt);
  EXPECT_TRUE(ShouldEnableDeviceCompilationCacheHitFastPath());
}

TEST(DeviceCompilationProfilerTest, CacheHitFastPathTestingOverride) {
  SetEnableDeviceCompilationCacheHitFastPathForTesting(
      std::optional<bool>(false));
  EXPECT_FALSE(ShouldEnableDeviceCompilationCacheHitFastPath());

  SetEnableDeviceCompilationCacheHitFastPathForTesting(
      std::optional<bool>(true));
  EXPECT_TRUE(ShouldEnableDeviceCompilationCacheHitFastPath());

  SetEnableDeviceCompilationCacheHitFastPathForTesting(std::nullopt);
}

TEST(DeviceCompilationProfilerTest,
     ZeroResourceArgumentShortCircuitEnabledByDefault) {
  SetEnableZeroResourceArgumentShortCircuitForTesting(std::nullopt);
  EXPECT_TRUE(ShouldEnableZeroResourceArgumentShortCircuit());
}

TEST(DeviceCompilationProfilerTest,
     ZeroResourceArgumentShortCircuitTestingOverride) {
  SetEnableZeroResourceArgumentShortCircuitForTesting(
      std::optional<bool>(false));
  EXPECT_FALSE(ShouldEnableZeroResourceArgumentShortCircuit());

  SetEnableZeroResourceArgumentShortCircuitForTesting(
      std::optional<bool>(true));
  EXPECT_TRUE(ShouldEnableZeroResourceArgumentShortCircuit());

  SetEnableZeroResourceArgumentShortCircuitForTesting(std::nullopt);
}

TEST(DeviceCompilationProfilerTest,
     BuildXlaCompilerArgumentsFastPathEnabledByDefault) {
  SetEnableBuildXlaCompilerArgumentsFastPathForTesting(std::nullopt);
  EXPECT_TRUE(ShouldEnableBuildXlaCompilerArgumentsFastPath());
}

TEST(DeviceCompilationProfilerTest,
     BuildXlaCompilerArgumentsFastPathTestingOverride) {
  SetEnableBuildXlaCompilerArgumentsFastPathForTesting(
      std::optional<bool>(false));
  EXPECT_FALSE(ShouldEnableBuildXlaCompilerArgumentsFastPath());

  SetEnableBuildXlaCompilerArgumentsFastPathForTesting(
      std::optional<bool>(true));
  EXPECT_TRUE(ShouldEnableBuildXlaCompilerArgumentsFastPath());

  SetEnableBuildXlaCompilerArgumentsFastPathForTesting(std::nullopt);
}

TEST(DeviceCompilationProfilerTest, DumpCsv) {
  // DumpCsv should write one row per recorded phase sample.
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 7);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kCacheLookup, 5);

  Env* env = Env::Default();
  std::string filename;
  ASSERT_TRUE(env->LocalTempFilename(&filename));

  TF_ASSERT_OK(profiler->DumpCsv(filename));

  std::string contents;
  TF_ASSERT_OK(ReadFileToString(env, filename, &contents));
  EXPECT_NE(contents.find("cluster_name,phase,event_index"),
            std::string::npos);
  EXPECT_NE(contents.find(
                "\"TestFunc\",signature_build,1,1,7\n\"TestFunc\",cache_lookup,2,1,5"),
            std::string::npos);
}

TEST(DeviceCompilationProfilerTest, DumpsCsvOnDestructionWhenEnvVarIsSet) {
  // The destructor should dump the same CSV snapshot when the env var is set.
  Env* env = Env::Default();
  std::string filename;
  ASSERT_TRUE(env->LocalTempFilename(&filename));
  ASSERT_EQ(::setenv("TF_XLA_DEVICE_COMPILATION_PROFILER_CSV_PATH",
                     filename.c_str(), 1),
            0);

  {
    DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
    core::ScopedUnref profiler_ref(profiler);

    NameAttrList function;
    function.set_name("TestFunc");
    profiler->RegisterPhaseTiming(
        function, DeviceCompilationProfiler::CompilePhase::kXlaCompileOpCompute,
        11);
  }

  std::string contents;
  TF_ASSERT_OK(ReadFileToString(env, filename, &contents));
  EXPECT_NE(contents.find("\"TestFunc\",xla_compile_op_compute,1,1,11"),
            std::string::npos);
  ASSERT_EQ(::unsetenv("TF_XLA_DEVICE_COMPILATION_PROFILER_CSV_PATH"), 0);
}

TEST(DeviceCompilationProfilerTest, UpdatesCsvWhenCompileOpTimingIsRecorded) {
  // Writing once per completed compile-op sample keeps earlier phase samples visible.
  Env* env = Env::Default();
  std::string filename;
  ASSERT_TRUE(env->LocalTempFilename(&filename));
  ASSERT_EQ(::setenv("TF_XLA_DEVICE_COMPILATION_PROFILER_CSV_PATH",
                     filename.c_str(), 1),
            0);

  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 9);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kSignatureBuild, 4);
  profiler->RegisterPhaseTiming(
      function, DeviceCompilationProfiler::CompilePhase::kXlaCompileOpCompute,
      13);

  std::string contents;
  TF_ASSERT_OK(ReadFileToString(env, filename, &contents));
  EXPECT_NE(contents.find("\"TestFunc\",signature_build,1,1,9"),
            std::string::npos);
  EXPECT_NE(contents.find("\"TestFunc\",signature_build,2,2,4"),
            std::string::npos);
  EXPECT_NE(contents.find("\"TestFunc\",xla_compile_op_compute,3,1,13"),
            std::string::npos);

  ASSERT_EQ(::unsetenv("TF_XLA_DEVICE_COMPILATION_PROFILER_CSV_PATH"), 0);
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterNotFound) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterFirstExecution) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  profiler->RegisterExecution(function);

  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterMegamorphic) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  const int64_t kCompileThreshold = 10;
  const int64_t kMinExecutionsPerCompile = 50;

  // Register compilation enough times (without registering executions enough
  // times) so that the function is marked megamorphic.
  for (int i = 0; i < kCompileThreshold + 1; ++i) {
    EXPECT_TRUE(profiler->RegisterCompilation(function, 1, false).ok());
  }
  profiler->RegisterExecution(function);

  // Shouldn't compile cluster since it has gone megamorphic.
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_TRUE(stats.is_megamorphic);

  // Always compile for strict compile mode.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));

  // Once a cluster has gone megamorphic, it remains megamorphic (even though
  // it's being executed more frequently now) and shouldn't be compiled again.
  for (int i = 0; i < kCompileThreshold * kMinExecutionsPerCompile + 1; ++i) {
    profiler->RegisterExecution(function);
  }

  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  TF_ASSERT_OK_AND_ASSIGN(stats, profiler->GetCompileStats(function));
  EXPECT_TRUE(stats.is_megamorphic);

  // Always compile for strict compile mode.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterAsync) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  const int64_t kMaxNumOngoingCompilations = 10;
  for (int i = 0; i < kMaxNumOngoingCompilations; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
  }

  // Should allow compilation since this is the first execution.
  profiler->RegisterExecution(function);
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));

  // Should not allow compilation since this is not the first execution and
  // we've already reached the maximum number of ongoing compilations allowed.
  profiler->RegisterExecution(function);
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));

  profiler->DecrementOngoingAsyncCompilations();
  // Should allow compilation since we've decremented the number of ongoing
  // compilations.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterLazy) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  constexpr int64_t kDefaultCompilationThreshold = 2;

  // Should allow compilation since this is the first execution.
  profiler->RegisterExecution(function);
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));

  // Shouldn't allow compilation until compilation has been requested at least
  // kDefaultCompilationThreshold times.
  profiler->RegisterExecution(function);
  for (int current_request_count = 0;
       current_request_count < kDefaultCompilationThreshold;
       ++current_request_count) {
    EXPECT_FALSE(profiler->ShouldCompileCluster(
        function, DeviceCompileMode::kLazy, current_request_count));
  }
  EXPECT_TRUE(profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy,
                                             kDefaultCompilationThreshold));
}

}  // namespace
}  // namespace tensorflow
