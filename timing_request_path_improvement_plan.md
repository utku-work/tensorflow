# Request-Path Improvement Plan

## Scope

This note is a planning document only.

It does not propose applying code changes yet. The purpose is to capture the likely improvements for the steady-state request path, based on the timing data where the first record includes compilation and the second record reflects the request-only path.

The main target is the overhead currently attributed to `get_xla_compiler_args_and_snapshot_variables` and the work immediately around it.

## Baseline Observation

For the request-only records, the main issue is no longer backend compilation.

Instead, the hot path is the front-end setup that still happens before the system confirms that a compiled executable is already cached.

For the measured `cluster_0` request-only path:

- `get_xla_compiler_args_and_snapshot_variables = 52 us`
- `build_xla_compiler_arguments = 50 us`
- `signature_build = 19 us`
- `cache_lookup = 5 us`
- `compile_to_local_executable = 46 us`
- `xla_compile_op_compute = 108 us`

The key point is that most of the steady-state request cost is spent preparing arguments and reconstructing state before the cache hit is known.

## Current Logic

Today the request path effectively does the following:

1. Enter `_XlaCompile`.
2. Collect runtime input tensors.
3. Enter `GetXlaCompilerArgsAndSnapshotVariables(...)`.
4. Gather `VariableInfo` objects from resource inputs.
5. Lock variables.
6. Snapshot resource variable values.
7. Build `std::vector<XlaCompiler::Argument>` for all inputs.
8. Enter the device compiler.
9. Build a cluster signature from the materialized XLA arguments.
10. Look up the compilation cache.
11. If cache hit, use the cached executable.
12. If cache miss, continue with the normal compilation path.

The problem with this order is that expensive front-end work is paid before the request learns whether the executable is already available.

## Before Flow

### Current steady-state request flow

```text
XlaCompileOp::Compute
  -> InputsFromContext
  -> GetXlaCompilerArgsAndSnapshotVariables
       -> GetVariableInfosFromInputs
       -> LockVariables
       -> SnapshotResourceVariables
       -> BuildXlaCompilerArguments
  -> DeviceCompiler::CompileImpl
       -> Build signature from XlaCompiler::Argument vector
       -> Cache lookup
       -> Cache hit: return cached executable
       -> Cache miss: compile
```

### Current inefficiency

The cache lookup happens too late.

For request-only traffic, the system is still paying for argument preparation and related setup before discovering that the executable already exists.

## Proposed Improvements

## 1. Add a zero-resource cache-hit fast path

### Idea

For clusters with no resource inputs, build the signature directly from runtime input tensors and try the compilation cache before materializing full `XlaCompiler::Argument` objects.

### Logic

- If `variable_indices` is empty, there is no need to gather variable metadata, lock variables, or snapshot resources.
- If the signature can be built directly from runtime tensors, the cache can be queried much earlier.
- On a cache hit, the request can skip the full helper path.

### Why this matters

This is the highest-value improvement for the measured steady-state path because your hot clusters are zero-resource clusters.

## 2. Build the signature directly from runtime inputs

### Idea

Introduce a direct signature-construction path for the no-resource case.

### Logic

- Constant inputs should still be represented as constants in the signature.
- Non-constant non-resource inputs should be represented using dtype and shape only.
- Empty tensors should preserve the same signature behavior as today.
- Resource inputs should stay on the existing slower path.

### Why this matters

The current flow first constructs `XlaCompiler::Argument` objects and then derives the signature from those arguments. For request-only cache hits, that intermediate argument materialization is unnecessary work.

## 3. Add a cache peek path before full request preparation

### Idea

Allow the cache to be checked without incrementally forcing the full compile-setup path.

### Logic

- Compute the direct signature.
- Ask the cache whether an executable already exists for that signature.
- If the entry is already compiled, use it immediately.
- If the entry is absent or uncompiled, fall back to the existing path.

### Why this matters

This changes the order of work so that the expensive setup is only paid on cache miss or when the cluster shape really requires the full path.

## 4. Short-circuit zero-resource work inside `GetXlaCompilerArgsAndSnapshotVariables`

### Idea

Even without a full fast path, the helper itself can avoid dead work when there are no resource inputs.

### Logic

- If `variable_indices.empty()`, skip:
  - `GetVariableInfosFromInputs`
  - `LockVariables`
  - `SnapshotResourceVariables`
- Move directly to argument construction.

### Why this matters

This is smaller than the direct-signature fast path, but it is low-risk and removes obviously unnecessary work for zero-resource clusters.

## 5. Reduce per-input overhead in `BuildXlaCompilerArguments`

### Idea

Make argument construction more linear and less branch-heavy for constant-heavy clusters.

### Logic

- Replace repeated binary searches over `must_be_constant_idxs` with one forward scan.
- Avoid creating `variable_info_lookup` when `variable_args` is empty.
- Keep the resource-handling logic on the existing path only when resource inputs are present.

### Why this matters

`cluster_0` is constant-heavy, so repeated constant-index checks and setup work show up more clearly there than in `cluster_1`.

## 6. Avoid unnecessary constant materialization on request-only cache hits

### Idea

Do not copy constant tensor payloads into `XlaCompiler::Argument::constant_value` unless the request really needs the full compile path.

### Logic

- For direct signature construction, constants only need to contribute to the signature.
- Full constant-value population should be delayed until cache miss.

### Why this matters

This removes data copying and object population that do not help the steady-state request path.

## After Flow

### Proposed steady-state request flow for zero-resource clusters

```text
XlaCompileOp::Compute
  -> InputsFromContext
  -> If no resource inputs:
       -> Build signature directly from runtime inputs
       -> Cache peek / cache lookup
       -> Cache hit: return cached executable immediately
       -> Cache miss: fall back to full path
  -> Full path fallback:
       -> GetXlaCompilerArgsAndSnapshotVariables
            -> GetVariableInfosFromInputs if needed
            -> LockVariables if needed
            -> SnapshotResourceVariables if needed
            -> BuildXlaCompilerArguments
       -> DeviceCompiler::CompileImpl
            -> Full cache path / compile path
```

### Expected effect of the new flow

For request-only traffic on zero-resource clusters:

- cache decision happens first
- full argument materialization is delayed until needed
- variable-related helper work disappears from the common case
- constant-heavy request overhead should drop noticeably

## Before And After Comparison

### Before

- Full request-preparation work is paid first.
- Signature build happens after argument materialization.
- Cache hit is discovered late.
- Zero-resource clusters still pass through resource-oriented scaffolding.

### After

- Signature is built first for eligible clusters.
- Cache hit is discovered early.
- Full helper work is paid only on miss or unsupported cases.
- Zero-resource request traffic uses a smaller steady-state path.

## Recommended Order Of Work

If these ideas are implemented later, the safest order is:

1. Completed: add a direct signature builder for no-resource inputs.
2. Completed: add a cache peek or early cache-hit path using that signature.
3. Completed implicitly: keep the current path as fallback when the fast path is disabled or misses.
4. Next: add zero-resource short-circuits inside `GetXlaCompilerArgsAndSnapshotVariables`.
5. Then optimize `BuildXlaCompilerArguments` for constant-heavy input sets.

This order isolates correctness risk and makes it easier to compare timing improvements after each change.

## Current Status After Step 2

- Step 1 is implemented.
- Step 2 is implemented.
- Step 3 is already satisfied by the current fallback behavior.

The practical remaining implementation order is therefore:

1. Step 4 first.
2. Step 5 second.

The reason for this order is simple:

- Step 4 is smaller, more local, and lower-risk.
- Step 4 removes obviously dead zero-resource work in one helper.
- Step 5 changes the shared argument-building loop and has broader surface area.
- Doing step 4 first gives a cleaner baseline for measuring step 5.

## Step 3 Status

Step 3 is not really a separate optimization anymore.

It is the fallback behavior that already exists today:

- try the zero-resource cache-hit fast path first
- if it is disabled, unsupported, or misses, fall back to the original request-preparation path

For measurement purposes, the step-2 fast-path flag already acts as the step-3 switch:

- `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH=1` uses the improvement when eligible
- `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH=0` forces the fallback path

So there is no additional step-3 flag to add. The existing step-2 flag already gives the A/B comparison against the fallback.

## Why `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH` Helped Most

This flag produced the large timing win because it is the only change that
reorders the request path.

Before this fast path, even a warm cache hit still paid for the full front-end
preparation path first:

- `GetXlaCompilerArgsAndSnapshotVariables(...)`
- variable-info gathering
- variable locking
- variable snapshotting
- `BuildXlaCompilerArguments(...)`
- signature construction from materialized `XlaCompiler::Argument` objects
- only then the compilation cache lookup

That meant the cache-hit decision was discovered too late.

With `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH=1`, the request does
something much smaller first for the safe case of zero-resource, non-PJRT
clusters:

1. build the cache signature directly from runtime input tensors
2. `Peek(...)` the compilation cache without forcing the normal compile-setup path
3. if the entry is already compiled, return the cached executable immediately
4. otherwise fall back to the existing path unchanged

That is why this flag helped much more than step 4 and step 5.

Step 4 and step 5 only make the old helper path cheaper.
This flag avoids that helper path entirely on eligible warm hits.

### Core logic

The key idea is:

- do an early cache decision
- do it with a direct signature built from raw runtime inputs
- only use it when resource handling is not required
- keep the old path as fallback on miss or unsupported cases

### ASCII flow

Before:

```text
XlaCompileOp::Compute
  -> InputsFromContext
  -> GetXlaCompilerArgsAndSnapshotVariables
       -> GetVariableInfosFromInputs
       -> LockVariables
       -> SnapshotResourceVariables
       -> BuildXlaCompilerArguments
  -> DeviceCompiler::CompileImpl
       -> Build signature from XlaCompiler::Argument vector
       -> Cache lookup
       -> Cache hit: return cached executable
```

After with CACHE_HIT_FAST_PATH:

```text
XlaCompileOp::Compute
  -> InputsFromContext
  -> if !use_pjrt && resources_.empty() && CACHE_HIT_FAST_PATH enabled:
       -> BuildForNoResourceInputs
            -> constants stay value-based
            -> normal inputs use dtype + shape
            -> resource inputs rejected
       -> cache->Peek(signature)
       -> compiled hit: return cached executable immediately
       -> miss: fall back to original path
  -> fallback path:
       -> GetXlaCompilerArgsAndSnapshotVariables
       -> DeviceCompiler::CompileImpl
```

The important difference is that the cache decision moves ahead of the helper
path instead of happening after the helper path.

### Why zero-resource only

```text
Early cache peek needs a correct cache key before the helper path runs.

Case A: zero-resource inputs
---------------------------------------------
raw runtime inputs
     -> direct signature can be built immediately
                -> constant inputs: use full tensor value
                -> normal inputs: use dtype + shape
                -> empty tensors: preserve existing signature behavior
     -> cache->Peek(signature)
     -> hit: return cached executable
     -> miss: fall back

Why this is safe:
     -> all compile-relevant state is already visible in the input tensors


Case B: resource inputs present
---------------------------------------------
raw runtime inputs
     -> resource tensor is only a handle to mutable state
     -> compile-relevant state is NOT just the handle
     -> must first:
                -> GetVariableInfosFromInputs
                -> LockVariables
                -> SnapshotResourceVariables
     -> only after that do we have a consistent resource-backed compile view
     -> then build args / signature / cache path

Why early peek is not safe here:
     -> resource values and shapes can change concurrently
     -> a handle alone is not a sufficient cache key
     -> constant-resource handling may depend on snapped variable contents
```

That is why the current early-peek optimization is really a zero-resource fast
path, not a constant-only fast path.

### Normal tensor vs resource variable

```text
Normal tensor input:
     input tensor itself == compile-relevant value

Resource variable input:
     input tensor == DT_RESOURCE handle
     handle -> TensorFlow variable object -> current mutable tensor value
```

Example:

```python
import tensorflow as tf

x = tf.constant([[1.0, 2.0]])
v = tf.Variable([[3.0, 4.0]])

@tf.function(jit_compile=True)
def uses_tensor(a):
          return a + 1.0

@tf.function(jit_compile=True)
def uses_variable():
          return v * 2.0
```

In `uses_tensor`, `a` is just a normal value tensor.

In `uses_variable`, `v` is a resource variable. The compile path does not just
see a plain `f32[...]` input value. It sees a handle to mutable state, so it
must resolve, lock, and snapshot that state before treating it as safe compile
input state.

### Code snippet

The fast-path flow is effectively this:

```cpp
if (!use_pjrt && resources_.empty() &&
          ShouldEnableDeviceCompilationCacheHitFastPath()) {
     auto compiled_cache_hit_or = TryUseCompiledLocalExecutableCacheHit(
               ctx, function_, platform_info_, constants_, inputs, profiler, &client,
               &kernel, &executable);
     OP_REQUIRES_OK(ctx, compiled_cache_hit_or.status());
     used_compiled_cache_hit_fast_path = *compiled_cache_hit_or;
}

absl::StatusOr<bool> TryUseCompiledLocalExecutableCacheHit(
          OpKernelContext* ctx, const NameAttrList& function,
          const XlaPlatformInfo& platform_info,
          absl::Span<const int> must_be_constant_idxs,
          absl::Span<const Tensor* const> inputs,
          DeviceCompilationProfiler* profiler, xla::LocalClient** client,
          const XlaCompiler::CompilationResult** compilation_result,
          xla::LocalExecutable** executable) {
     TF_ASSIGN_OR_RETURN(
               DeviceCompilationClusterSignature signature,
               DeviceCompilationClusterSignature::BuildForNoResourceInputs(
                         function, inputs, must_be_constant_idxs));

     TF_ASSIGN_OR_RETURN(DeviceType compilation_device_type,
                                                       GetCompilationDeviceType(platform_info.device_type()));

     XlaDeviceCompiler* xla_device_compiler;
     TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaDeviceCompiler>(
               rm->default_container(), "xla_device_compiler", &xla_device_compiler,
               [&](XlaDeviceCompiler** xla_device_compiler) {
                    return BuildXlaDeviceCompiler(ctx->device(), ctx->function_library(),
                                                                                               platform_info, compilation_device_type,
                                                                                               xla_device_compiler);
               }));

     auto cache_value = xla_device_compiler->cache()->Peek(signature);
     if (!cache_value.has_value() ||
               cache_value->compile_state != DeviceCompileState::kCompiled) {
          return false;
     }

     *client = static_cast<xla::LocalClient*>(xla_device_compiler->client());
     *compilation_result = cache_value->compilation_result;
     *executable = cache_value->executable;
     profiler->RegisterExecution(function);
     return true;
}
```

And the direct-signature part works like this:

```cpp
absl::StatusOr<Signature> Signature::BuildForNoResourceInputs(
          const NameAttrList& function, absl::Span<const Tensor* const> inputs,
          absl::Span<const int> must_be_constant_idxs) {
     Signature signature;
     signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));

     for (int64_t input_num = 0; input_num < inputs.size(); ++input_num) {
          const Tensor* input = inputs[input_num];
          if (input->dtype() == DT_RESOURCE) {
               return errors::InvalidArgument("resource input not supported");
          }

          if (is_constant_input(input_num) || input->NumElements() == 0) {
               signature.args.push_back(*input);
          } else {
               signature.args.push_back(TensorSignatureTypeAndShape(*input));
          }
     }

     return signature;
}
```

### Why the later steps moved less

Step 4 and step 5 improve the fallback/helper path.

If the workload is dominated by eligible warm cache hits, the fast path skips
that helper path almost entirely. In that case:

- step 2 removes most of the cost
- step 4 only trims work in a path that is now rarely used
- step 5 only trims work in a path that is now rarely used

So it is expected that `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH`
shows the largest effect.

## Step 4 Plan

### What to do

Add a zero-resource short-circuit inside `GetXlaCompilerArgsAndSnapshotVariables(...)`.

### How

When `variable_indices.empty()` and the step-4 flag is enabled:

- skip `GetVariableInfosFromInputs`
- skip `LockVariables`
- skip `SnapshotResourceVariables`
- call `BuildXlaCompilerArguments(...)` with an empty `variable_infos` span
- return an empty `ResourceVarsSnapshot`

The fallback behavior stays unchanged when:

- the flag is off
- resource inputs are present
- any unsupported condition is reached

### Why do this before step 5

This is the narrowest remaining change and directly removes dead work that is still visible in the fallback path.

### Optional flag for step 4

Use a dedicated env var so it can be measured independently:

- `TF_XLA_DEVICE_COMPILATION_ENABLE_ZERO_RESOURCE_ARG_SHORT_CIRCUIT=1` enables it
- `TF_XLA_DEVICE_COMPILATION_ENABLE_ZERO_RESOURCE_ARG_SHORT_CIRCUIT=0` disables it

Recommended default once implemented:

- enabled by default

### Measurement note for step 4

Step 4 mainly affects the full helper path.

That means its effect is easiest to measure in one of these modes:

- with step 2 disabled, so zero-resource requests still go through the helper
- on cache-miss or compile-path traffic
- on any path where the fast path is intentionally bypassed

If step 2 remains enabled on a pure warm cache-hit workload, step 4 may have little or no visible effect because the helper is skipped entirely.

## Step 5 Plan

### What to do

Optimize `BuildXlaCompilerArguments(...)` for constant-heavy and zero-resource cases.

### How

Do this in a narrow order inside the function:

1. If `variable_args.empty()`, skip building `variable_info_lookup` entirely.
2. Replace repeated `absl::c_binary_search(must_be_constant_idxs, input_num)` calls with a single forward scan over sorted constant indices.
3. Keep resource-handling logic only on the path where resource inputs actually exist.
4. Optionally delay `TryGetDeviceContext(...)` until the code actually needs resource-constant host copies.

The key point is to remove setup and repeated per-input checks from the common zero-resource / constant-heavy path without changing resource semantics.

### Why this comes after step 4

Step 5 touches a shared hot loop and is more likely to create subtle regressions.

After step 4, the remaining cost inside the fallback path will be easier to attribute to the argument builder itself.

### Optional flag for step 5

Use a separate env var so it can be measured independently from step 4:

- `TF_XLA_DEVICE_COMPILATION_ENABLE_BUILD_XLA_COMPILER_ARGUMENTS_FAST_PATH=1` enables it
- `TF_XLA_DEVICE_COMPILATION_ENABLE_BUILD_XLA_COMPILER_ARGUMENTS_FAST_PATH=0` disables it

Recommended default once implemented:

- enabled by default

### Measurement note for step 5

To isolate step 5 cleanly:

- keep the step-5 flag as the only variable under test
- compare with step 4 either fixed on or fixed off
- prefer workloads that still execute `BuildXlaCompilerArguments(...)`

As with step 4, a pure warm-hit zero-resource workload with step 2 enabled may hide most of step-5's effect because the fast path bypasses argument building.

## Suggested Measurement Matrix

Once step 4 and step 5 exist, the clean comparison setup is:

1. Baseline fallback only:
     `TF_XLA_DEVICE_COMPILATION_ENABLE_CACHE_HIT_FAST_PATH=0`
     `TF_XLA_DEVICE_COMPILATION_ENABLE_ZERO_RESOURCE_ARG_SHORT_CIRCUIT=0`
     `TF_XLA_DEVICE_COMPILATION_ENABLE_BUILD_XLA_COMPILER_ARGUMENTS_FAST_PATH=0`
2. Step 4 only:
     step-2 flag off, step-4 flag on, step-5 flag off
3. Step 5 only on top of baseline:
     step-2 flag off, step-4 flag off, step-5 flag on
4. Step 4 + Step 5 together:
     step-2 flag off, step-4 flag on, step-5 flag on
5. Full optimized path:
     step-2 flag on, step-4 flag on, step-5 flag on

This keeps the effect of each step measurable instead of mixing all changes together immediately.

## Risks And Guardrails

The main correctness risks are:

- signature mismatch between the new direct-signature path and the existing argument-based path
- incorrect handling of empty tensors
- incorrect handling of constant inputs
- accidentally applying the fast path to resource-input clusters

Guardrails for later implementation:

- only enable the fast path for zero-resource clusters first
- verify the direct signature exactly matches the existing signature builder
- keep the existing path as fallback on any unsupported case
- add tests that compare the direct-signature result against the current argument-based result

## Summary

The current request-only bottleneck is not real compilation. It is the front-end work paid before the cache hit is known.

The most important improvement is to move cache eligibility earlier by building the signature directly from runtime inputs for zero-resource clusters. After that, the next gains come from trimming dead zero-resource work and reducing per-input overhead in argument construction.

Current status:

- step 1 is implemented and covered by signature tests
- step 2 is implemented and has a runtime on/off flag
- step 3 is already satisfied by the fallback path
- step 4 is implemented and has a runtime on/off flag
- step 5 is implemented and has a runtime on/off flag