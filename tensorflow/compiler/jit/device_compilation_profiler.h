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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {

// Tracks statistics for device compilation and uses these to determine whether
// the given cluster should be compiled or not.
class DeviceCompilationProfiler : public ResourceBase {
 public:
  DeviceCompilationProfiler() = default;
  ~DeviceCompilationProfiler() override;

  enum class CompilePhase {
    kSignatureBuild,
    kCacheLookup,
    kGetVariableInfosFromInputs,
    kLockVariables,
    kSnapshotResourceVariables,
    kBuildXlaCompilerArguments,
    kGetXlaCompilerArgsAndSnapshotVariables,
    kCompileToLocalExecutable,
    kXlaCompileOpCompute,
  };

  struct PhaseTimingStats {
    int64_t sample_count = 0;
    int64_t cumulative_time_us = 0;

    std::string DebugString(const char* phase_name) const {
      return absl::StrCat(phase_name, "={sample_count=", sample_count,
                          ", cumulative_time_us=", cumulative_time_us,
                          "}");
    }
  };

  struct PhaseTimingRecord {
    CompilePhase phase;
    int64_t event_index = 0;
    int64_t phase_sample_index = 0;
    int64_t elapsed_time_us = 0;
  };

  struct ClusterCompileStats {
    // Number of times the cluster has been (re-)compiled.
    int64_t compile_count = 0;

    // The number of times this cluster has been executed.
    int64_t execution_count = 0;

    // Cumulative time spent compiling the cluster.
    int64_t cumulative_compile_time_us = 0;

    // True if we have decided that this cluster is too dynamic (i.e. its shapes
    // change too frequently) to profitably JIT compile.  Once a cluster is
    // tagged megamorphic, it stays megamorphic forever.
    bool is_megamorphic = false;

    // Aggregated phase timings recorded while compiling or preparing the
    // cluster.
    PhaseTimingStats signature_build;
    PhaseTimingStats cache_lookup;
    PhaseTimingStats get_variable_infos_from_inputs;
    PhaseTimingStats lock_variables;
    PhaseTimingStats snapshot_resource_variables;
    PhaseTimingStats build_xla_compiler_arguments;
    PhaseTimingStats get_xla_compiler_args_and_snapshot_variables;
    PhaseTimingStats compile_to_local_executable;
    PhaseTimingStats xla_compile_op_compute;
    std::vector<PhaseTimingRecord> phase_timing_records;

    std::string DebugString() const {
      return absl::StrCat(
          "DeviceCompilationProfiler::ClusterCompileStats {compile_count=",
          compile_count, ", execution_count=", execution_count,
          ", cumulative_compile_time_us=", cumulative_compile_time_us,
        ", is_megamorphic=", is_megamorphic, ", ",
        signature_build.DebugString("signature_build"), ", ",
        cache_lookup.DebugString("cache_lookup"), ", ",
        get_variable_infos_from_inputs.DebugString(
            "get_variable_infos_from_inputs"),
        ", ",
        lock_variables.DebugString("lock_variables"),
        ", ",
        snapshot_resource_variables.DebugString(
            "snapshot_resource_variables"),
        ", ",
        build_xla_compiler_arguments.DebugString(
            "build_xla_compiler_arguments"),
        ", ",
        get_xla_compiler_args_and_snapshot_variables.DebugString(
            "get_xla_compiler_args_and_snapshot_variables"),
        ", ",
        compile_to_local_executable.DebugString(
            "compile_to_local_executable"),
        ", ",
        xla_compile_op_compute.DebugString("xla_compile_op_compute"),
        "}");
    }
  };

  // Returns the compilation statistics for the given cluster.
  absl::StatusOr<ClusterCompileStats> GetCompileStats(
      const NameAttrList& function) const;

  // Determines whether the cluster should be compiled. Creates and inserts an
  // entry into stats (also calls `RegisterExecution`) for `function` if it
  // doesn't already exist.
  virtual bool ShouldCompileCluster(const NameAttrList& function,
                                    DeviceCompileMode compile_mode,
                                    int64_t current_request_count);

  // Registers a cluster execution. Increments the execution count for the given
  // cluster and also determines whether the cluster has gone megamorphic (and
  // sets the megamorphic bit accordingly).
  void RegisterExecution(const NameAttrList& function);

  // Registers a sampled timing for one phase of cluster compilation.
  void RegisterPhaseTiming(const NameAttrList& function, CompilePhase phase,
                           int64_t elapsed_time_us);

  // Registers a cluster compilation. Increments the compilation count and
  // accumulates the compile time for the given cluster. Also broadcasts an
  // XlaJitCompilationActivity.
  virtual absl::Status RegisterCompilation(const NameAttrList& function,
                                           int64_t compile_time_us,
                                           bool used_persistent_cache);

  // Dumps the currently aggregated per-cluster stats as CSV.
  absl::Status DumpCsv(const std::string& path) const;

  void IncrementOngoingAsyncCompilations();
  void DecrementOngoingAsyncCompilations();
  int64_t GetNumOngoingAsyncCompilations() const;
  std::string DebugString() const override;

 private:
  mutable mutex mu_;

  // Maps cluster names to compilation statistics for said cluster.
  absl::flat_hash_map<std::string, ClusterCompileStats> cluster_compile_stats_
      TF_GUARDED_BY(mu_);

  int64_t num_ongoing_compilations_ TF_GUARDED_BY(mu_) = 0;

  DeviceCompilationProfiler(const DeviceCompilationProfiler&) = delete;
  void operator=(const DeviceCompilationProfiler&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_
