#!/usr/bin/env python3
"""Build TensorFlow Serving SavedModels for K/V projection fusion POC.

The baseline model computes K and V with two independent MatMuls. The optimized
model concatenates the K/V weights, computes one wider MatMul, then splits the
result back into K and V. Both SavedModels use the same numeric weights and are
exported under versioned directories that tensorflow_model_server can load.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and locally benchmark baseline vs fused-KV SavedModels."
    )
    parser.add_argument(
        "--export-dir",
        default="/tmp/kv_projection_fusion_poc",
        help="Directory where baseline/1 and fused_kv/1 SavedModels are written.",
    )
    parser.add_argument("--version", default="1", help="SavedModel version directory.")
    parser.add_argument("--input-dim", type=int, default=24)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-xla",
        action="store_true",
        help="Do not mark the exported serving function with jit_compile=True.",
    )
    parser.add_argument(
        "--hlo-dump-dir",
        help="Optional XLA dump directory. Set this only when you want HLO text dumps.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing export directories before saving.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip local benchmark and only write SavedModels.",
    )
    return parser.parse_args()


def configure_hlo_dump(hlo_dump_dir: str | None) -> None:
    if not hlo_dump_dir:
        return

    dump_path = Path(hlo_dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    existing_flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = f"--xla_dump_to={dump_path} --xla_dump_hlo_as_text"
    os.environ["XLA_FLAGS"] = f"{existing_flags} {dump_flags}".strip()


def make_weights(input_dim: int, proj_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w_k": rng.normal(0.0, 0.05, size=(input_dim, proj_dim)).astype(np.float32),
        "b_k": rng.normal(0.0, 0.05, size=(proj_dim,)).astype(np.float32),
        "w_v": rng.normal(0.0, 0.05, size=(input_dim, proj_dim)).astype(np.float32),
        "b_v": rng.normal(0.0, 0.05, size=(proj_dim,)).astype(np.float32),
    }


def build_modules(tf, weights: dict[str, np.ndarray], input_dim: int, use_xla: bool):
    class SeparateKV(tf.Module):
        def __init__(self):
            super().__init__(name="separate_kv")
            self.w_k = tf.Variable(weights["w_k"], name="k_kernel")
            self.b_k = tf.Variable(weights["b_k"], name="k_bias")
            self.w_v = tf.Variable(weights["w_v"], name="v_kernel")
            self.b_v = tf.Variable(weights["b_v"], name="v_bias")
            self.serve = tf.function(
                self._serve,
                input_signature=[
                    tf.TensorSpec([None, input_dim], tf.float32, name="x")
                ],
                jit_compile=use_xla,
            )

        def _serve(self, x):
            k = tf.linalg.matmul(x, self.w_k) + self.b_k
            v = tf.linalg.matmul(x, self.w_v) + self.b_v
            return {"k": k, "v": v}

    class FusedKV(tf.Module):
        def __init__(self):
            super().__init__(name="fused_kv")
            w_kv = np.concatenate([weights["w_k"], weights["w_v"]], axis=1)
            b_kv = np.concatenate([weights["b_k"], weights["b_v"]], axis=0)
            self.w_kv = tf.Variable(w_kv, name="kv_kernel")
            self.b_kv = tf.Variable(b_kv, name="kv_bias")
            self.serve = tf.function(
                self._serve,
                input_signature=[
                    tf.TensorSpec([None, input_dim], tf.float32, name="x")
                ],
                jit_compile=use_xla,
            )

        def _serve(self, x):
            kv = tf.linalg.matmul(x, self.w_kv) + self.b_kv
            k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
            return {"k": k, "v": v}

    return SeparateKV(), FusedKV()


def save_model(tf, module, export_path: Path, keep_existing: bool) -> None:
    if export_path.exists() and not keep_existing:
        shutil.rmtree(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        module,
        str(export_path),
        signatures={"serving_default": module.serve},
    )


def op_counts(serve_fn) -> dict[str, int]:
    interesting_ops = {"MatMul", "BatchMatMulV2", "Split", "SplitV", "AddV2"}
    concrete_fn = serve_fn.get_concrete_function()
    counts = Counter(node.op for node in concrete_fn.graph.as_graph_def().node)
    return {op: counts[op] for op in sorted(interesting_ops) if counts[op]}


def consume_outputs(result) -> None:
    _ = result["k"].numpy()
    _ = result["v"].numpy()


def benchmark(serve_fn, x, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        consume_outputs(serve_fn(x))

    start = time.perf_counter()
    for _ in range(iters):
        consume_outputs(serve_fn(x))
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def max_abs_diff(baseline_out, fused_out) -> float:
    k_diff = np.max(np.abs(baseline_out["k"].numpy() - fused_out["k"].numpy()))
    v_diff = np.max(np.abs(baseline_out["v"].numpy() - fused_out["v"].numpy()))
    return float(max(k_diff, v_diff))


def print_serving_commands(export_dir: Path) -> None:
    baseline_base = export_dir / "baseline"
    fused_base = export_dir / "fused_kv"
    print("\nTensorFlow Serving commands:")
    print(
        "tensorflow_model_server "
        f"--model_name=kv_baseline --model_base_path={baseline_base} --port=8500"
    )
    print(
        "tensorflow_model_server "
        f"--model_name=kv_fused --model_base_path={fused_base} --port=8501"
    )


def main() -> None:
    args = parse_args()
    configure_hlo_dump(args.hlo_dump_dir)

    import tensorflow as tf

    use_xla = not args.no_xla
    export_dir = Path(args.export_dir)
    version = str(args.version)
    weights = make_weights(args.input_dim, args.proj_dim, args.seed)
    baseline, fused = build_modules(tf, weights, args.input_dim, use_xla)

    baseline_path = export_dir / "baseline" / version
    fused_path = export_dir / "fused_kv" / version
    save_model(tf, baseline, baseline_path, args.keep_existing)
    save_model(tf, fused, fused_path, args.keep_existing)

    rng = np.random.default_rng(args.seed + 1)
    x_np = rng.normal(0.0, 1.0, size=(args.batch_size, args.input_dim)).astype(
        np.float32
    )
    x = tf.constant(x_np)

    baseline_out = baseline.serve(x)
    fused_out = fused.serve(x)
    diff = max_abs_diff(baseline_out, fused_out)

    print(f"Exported baseline SavedModel: {baseline_path}")
    print(f"Exported fused-KV SavedModel: {fused_path}")
    print(f"Max absolute output diff: {diff:.8g}")
    print(f"Baseline graph ops: {op_counts(baseline.serve)}")
    print(f"Fused-KV graph ops: {op_counts(fused.serve)}")

    if not args.export_only:
        baseline_ms = benchmark(baseline.serve, x, args.warmup, args.iters)
        fused_ms = benchmark(fused.serve, x, args.warmup, args.iters)
        speedup = baseline_ms / fused_ms if fused_ms else float("inf")
        print(f"Baseline local latency: {baseline_ms:.6f} ms/call")
        print(f"Fused-KV local latency: {fused_ms:.6f} ms/call")
        print(f"Local speedup: {speedup:.3f}x")

    if args.hlo_dump_dir:
        print(f"HLO dumps requested under: {args.hlo_dump_dir}")
    print_serving_commands(export_dir)


if __name__ == "__main__":
    main()