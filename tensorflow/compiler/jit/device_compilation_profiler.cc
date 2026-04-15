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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {
namespace {
constexpr char kDeviceCompilationProfilerCsvPathEnvVar[] =
    "TF_XLA_DEVICE_COMPILATION_PROFILER_CSV_PATH";

bool ShouldBeMegamorphic(int64_t compile_count, int64_t execution_count) {
  int64_t kCompileThreshold = 10;
  const int64_t kMinExecutionsPerCompile = 50;

  int64_t tf_xla_threshold_for_megamorphic =
      GetMarkForCompilationPassFlags()->tf_xla_threshold_for_megamorphic;

  // Negative values other that -1 cannot be used
  if (tf_xla_threshold_for_megamorphic < -1) {
    LOG(FATAL) << "The value for the tf_xla_threshold_for_megamorphic flag "
               << "is out of range.\n"
               << "Allowed ranges are (-1) to "
               << std::numeric_limits<int64_t>::max()
               << " got " << tf_xla_threshold_for_megamorphic << ".";
  }

  // -1: setting clusters as Megamorphic is disabled
  //  0 Default behaviour in Tensorflow
  //  Any other number sets the compilation threshold
  if (tf_xla_threshold_for_megamorphic == -1) {
    return false;
  } else if (tf_xla_threshold_for_megamorphic > 0) {
    kCompileThreshold = tf_xla_threshold_for_megamorphic;
  }

  // This heuristic is trying to capture the following property: have we sunk a
  // certain minimum amount of compile time into the cluster that didn't quite
  // "pay off"?
  return compile_count > kCompileThreshold &&
         execution_count < kMinExecutionsPerCompile * compile_count;
}

void RegisterExecutionForCluster(
    const NameAttrList& function,
    DeviceCompilationProfiler::ClusterCompileStats* stats) {
  ++stats->execution_count;

  // The is_megamorphic bit is "sticky".  We assume clusters that have been
  // observed to be megamorphic once stay megamorphic forever.
  if (!stats->is_megamorphic &&
      ShouldBeMegamorphic(stats->compile_count, stats->execution_count)) {
    VLOG(1) << "Marking " << function.name()
            << " as megamorphic, compile_count=" << stats->compile_count
            << " execution_count=" << stats->execution_count;
    stats->is_megamorphic = true;
  }
}

// The number of times a lazy compilation must be requested for a specific
// signature before  we attempt to compile it.
constexpr int64_t kDefaultCompilationThreshold = 2;

// Maximum number of ongoing compilations.
constexpr int64_t kMaxNumOngoingCompilations = kNumAsyncDeviceCompilerThreads;

DeviceCompilationProfiler::ClusterCompileStats& GetOrCreateStatsLocked(
    absl::flat_hash_map<std::string,
                        DeviceCompilationProfiler::ClusterCompileStats>* stats,
    const NameAttrList& function) {
  return stats->emplace(function.name(),
                        DeviceCompilationProfiler::ClusterCompileStats{})
      .first->second;
}

DeviceCompilationProfiler::PhaseTimingStats* GetPhaseTimingStats(
    DeviceCompilationProfiler::ClusterCompileStats* stats,
    DeviceCompilationProfiler::CompilePhase phase) {
  switch (phase) {
    case DeviceCompilationProfiler::CompilePhase::kSignatureBuild:
      return &stats->signature_build;
    case DeviceCompilationProfiler::CompilePhase::kCacheLookup:
      return &stats->cache_lookup;
    case DeviceCompilationProfiler::CompilePhase::kGetXlaCompilerArgsAndSnapshotVariables:
      return &stats->get_xla_compiler_args_and_snapshot_variables;
    case DeviceCompilationProfiler::CompilePhase::kCompileToLocalExecutable:
      return &stats->compile_to_local_executable;
    case DeviceCompilationProfiler::CompilePhase::kXlaCompileOpCompute:
      return &stats->xla_compile_op_compute;
  }

  LOG(FATAL) << "Unknown compile phase.";
}

std::string CsvEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char ch : value) {
    if (ch == '"') {
      escaped.append("\"\"");
    } else {
      escaped.push_back(ch);
    }
  }
  escaped.push_back('"');
  return escaped;
}

}  // namespace

DeviceCompilationProfiler::~DeviceCompilationProfiler() {
  const char* csv_path = std::getenv(kDeviceCompilationProfilerCsvPathEnvVar);
  if (csv_path != nullptr && csv_path[0] != '\0') {
    absl::Status dump_status = DumpCsv(csv_path);
    if (!dump_status.ok()) {
      LOG(ERROR) << "Failed to dump device compilation profiler CSV to "
                 << csv_path << ": " << dump_status;
    }
  }

  mutex_lock lock(mu_);
  cluster_compile_stats_.clear();
}

absl::StatusOr<DeviceCompilationProfiler::ClusterCompileStats>
DeviceCompilationProfiler::GetCompileStats(const NameAttrList& function) const {
  mutex_lock lock(mu_);

  if (auto it = cluster_compile_stats_.find(function.name());
      it != cluster_compile_stats_.end()) {
    return it->second;
  }

  return errors::NotFound("Couldn't find compilation stats for cluster: ",
                          function.name());
}

void DeviceCompilationProfiler::RegisterExecution(
    const NameAttrList& function) {
  mutex_lock lock(mu_);
  RegisterExecutionForCluster(function,
                              &GetOrCreateStatsLocked(&cluster_compile_stats_,
                                                      function));
}

void DeviceCompilationProfiler::RegisterPhaseTiming(
    const NameAttrList& function, CompilePhase phase, int64_t elapsed_time_us) {
  mutex_lock lock(mu_);
  ClusterCompileStats& stats =
      GetOrCreateStatsLocked(&cluster_compile_stats_, function);
  PhaseTimingStats* phase_stats = GetPhaseTimingStats(&stats, phase);
  ++phase_stats->sample_count;
  phase_stats->cumulative_time_us += elapsed_time_us;
}

absl::Status DeviceCompilationProfiler::RegisterCompilation(
    const NameAttrList& function, int64_t compile_time_us,
    bool used_persistent_cache) {
  metrics::UpdateXlaCompilationTime(compile_time_us);

  const std::string& function_name = function.name();

  mutex_lock lock(mu_);
  // Create a stats entry if it doesn't already exist.
  ClusterCompileStats& stats =
      GetOrCreateStatsLocked(&cluster_compile_stats_, function);

  const uint64 compile_time_s = compile_time_us / 1.0e6;
  stats.compile_count++;
  stats.cumulative_compile_time_us += compile_time_us;
  VLOG(1) << "Compiled " << function_name << " " << stats.compile_count
          << " times, compile time: " << compile_time_us
          << " us, cumulative: " << stats.cumulative_compile_time_us
          << " us ("
          << tensorflow::strings::HumanReadableElapsedTime(compile_time_s)
          << " / "
          << tensorflow::strings::HumanReadableElapsedTime(
                 stats.cumulative_compile_time_us / 1.0e6)
          << ")";

  XlaJitCompilationActivity jit_compilation_activity;
  jit_compilation_activity.set_cluster_name(function_name);
  jit_compilation_activity.set_compile_count(stats.compile_count);
  jit_compilation_activity.set_compile_time_us(compile_time_us);
  jit_compilation_activity.set_cumulative_compile_time_us(
      stats.cumulative_compile_time_us);
  jit_compilation_activity.set_used_persistent_cache(used_persistent_cache);
  return BroadcastXlaActivity(std::move(jit_compilation_activity));
}

absl::Status DeviceCompilationProfiler::DumpCsv(const std::string& path) const {
  std::vector<std::pair<std::string, ClusterCompileStats>> stats_snapshot;
  {
    mutex_lock lock(mu_);
    stats_snapshot.reserve(cluster_compile_stats_.size());
    for (const auto& [name, stats] : cluster_compile_stats_) {
      stats_snapshot.emplace_back(name, stats);
    }
  }

  Env* env = Env::Default();
  const std::string dirname = io::Dirname(path);
  if (!dirname.empty() && dirname != path) {
    TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dirname));
  }

  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(path, &file));
  TF_RETURN_IF_ERROR(file->Append(
      "cluster_name,compile_count,execution_count,cumulative_compile_time_us,"
      "is_megamorphic,signature_build_count,signature_build_time_us,"
      "cache_lookup_count,cache_lookup_time_us,"
      "get_xla_compiler_args_and_snapshot_variables_count,"
      "get_xla_compiler_args_and_snapshot_variables_time_us,"
      "compile_to_local_executable_count,"
      "compile_to_local_executable_time_us,"
      "xla_compile_op_compute_count,xla_compile_op_compute_time_us\n"));

  for (const auto& [name, stats] : stats_snapshot) {
    TF_RETURN_IF_ERROR(file->Append(absl::StrCat(
        CsvEscape(name), ",", stats.compile_count, ",",
        stats.execution_count, ",", stats.cumulative_compile_time_us, ",",
        stats.is_megamorphic ? "true" : "false", ",",
        stats.signature_build.sample_count, ",",
        stats.signature_build.cumulative_time_us, ",",
        stats.cache_lookup.sample_count, ",",
        stats.cache_lookup.cumulative_time_us, ",",
        stats.get_xla_compiler_args_and_snapshot_variables.sample_count, ",",
        stats.get_xla_compiler_args_and_snapshot_variables
            .cumulative_time_us,
        ",", stats.compile_to_local_executable.sample_count, ",",
        stats.compile_to_local_executable.cumulative_time_us, ",",
        stats.xla_compile_op_compute.sample_count, ",",
        stats.xla_compile_op_compute.cumulative_time_us, "\n")));
  }

  return file->Close();
}

bool DeviceCompilationProfiler::ShouldCompileCluster(
    const NameAttrList& function, DeviceCompileMode compile_mode,
    int64_t current_request_count) {
  std::optional<int64_t> compile_threshold;
  if (compile_mode == DeviceCompileMode::kLazy) {
    compile_threshold = kDefaultCompilationThreshold;
  } else if (compile_mode == DeviceCompileMode::kAsync) {
    compile_threshold = 0;  // for now, always compile right away.
  }

  if (compile_mode == DeviceCompileMode::kStrict) {
    // Lazy compilation is disabled.
    return true;
  }

  mutex_lock lock(mu_);
  // Create a stats entry if one isn't found and register an execution.
  // Determine eligibility assuming this is the first execution of the cluster
  // and this cluster has never been compiled before.
  auto [it, cluster_not_found] =
      cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{});
  if (cluster_not_found) {
    RegisterExecutionForCluster(function, &it->second);
  }

  // We avoid compiling clusters that have "gone megamorphic" i.e. have an
  // excessive amount of shape dynamism.
  if (it->second.is_megamorphic) {
    BroadcastOptimizationRemark(XlaOptimizationRemark::MEGAMORPHIC_FUNCTION,
                                function.name())
        .IgnoreError();
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it is megamorphic.";
    return false;
  }

  // TODO(b/255826209): Figure out if Lazy compilation is still needed given
  // that we always compile a cluster the first time it is executed (explained
  // below) regardless of compilation mode. If it is not, clean up the related
  // logic.
  // We always compile a cluster the very first time it is executed.  This is an
  // optimistic guess that pays off for statically shaped TensorFlow graphs
  // (since they get the benefit of XLA right away without waiting for warmup)
  // and doesn't hurt much for dynamically shaped TensorFlow graphs (we "pay" at
  // most one cluster-compilation's worth of compile time).
  if (it->second.execution_count == 1) {
    return true;
  }

  if (compile_mode == DeviceCompileMode::kAsync) {
    // Asynchronous compilation is enabled.
    if (num_ongoing_compilations_ >= kMaxNumOngoingCompilations) {
      VLOG(2) << "Not asynchronously compiling cluster " << function.name()
              << " because of too many ongoing compilations.";
      return false;
    }
  }

  bool reached_compile_threshold = current_request_count >= *compile_threshold;
  if (!reached_compile_threshold) {
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it has not reached compile threshold; threshold is "
            << *compile_threshold << " execution count "
            << current_request_count << ".";
  }
  return reached_compile_threshold;
}

void DeviceCompilationProfiler::IncrementOngoingAsyncCompilations() {
  mutex_lock lock(mu_);
  num_ongoing_compilations_++;
}

void DeviceCompilationProfiler::DecrementOngoingAsyncCompilations() {
  mutex_lock lock(mu_);
  num_ongoing_compilations_--;
}

int64_t DeviceCompilationProfiler::GetNumOngoingAsyncCompilations() const {
  mutex_lock lock(mu_);
  return num_ongoing_compilations_;
}

std::string DeviceCompilationProfiler::DebugString() const {
  std::string debug_string =
      "DeviceCompilationProfiler {\ncluster_compile_stats: {\n";
  {
    mutex_lock lock(mu_);

    for (const auto& [key, stats] : cluster_compile_stats_) {
      absl::StrAppend(&debug_string, key, ": ", stats.DebugString(), "\n");
    }
  }

  absl::StrAppend(&debug_string, "}\nnum_ongoing_compilations=",
                  GetNumOngoingAsyncCompilations(), "\n}\n");

  return debug_string;
}

}  // namespace tensorflow
