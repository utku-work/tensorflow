#!/usr/bin/env python3
"""Export and validate a SavedModel that mimics the CVR cluster_2 XLA issue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _argv_value(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _bootstrap_xla_dump_flags() -> None:
    if "--dump-hlo" not in sys.argv:
        return
    hlo_dir = _argv_value("--hlo-dir", "xla_issue_lab/saved_model_hlo_dumps")
    os.makedirs(hlo_dir, exist_ok=True)
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (
        f" --xla_dump_to={hlo_dir} --xla_dump_hlo_as_text"
    )


_bootstrap_xla_dump_flags()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf  # noqa: E402
from tensorflow.core.protobuf import saved_model_pb2  # noqa: E402


HISTORY_SPECS = (
    ("click", 45, 2),
    ("noware", 57, 7),
    ("search", 14, 5),
)
INIT_OP_SIGNATURE_KEY = "__saved_model_init_op"


def swish(value: tf.Tensor) -> tf.Tensor:
    return value * tf.nn.sigmoid(value)


class Cluster2IssueModel(tf.Module):
    """SavedModel-shaped version of the synthetic cluster_2 issue."""

    def __init__(
        self,
        *,
        table_rows: int,
        batch_size: int,
        sparse_width: int,
        feature_dim: int,
        chunk_count: int,
        hidden_dim: int,
        expert_dim: int,
        tower_dim: int,
        packed_moe: bool,
        dense_sessions: bool,
    ) -> None:
        super().__init__()
        self.table_rows = table_rows
        self.batch_size = batch_size
        self.sparse_width = sparse_width
        self.feature_dim = feature_dim
        self.chunk_count = chunk_count
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.tower_dim = tower_dim
        self.packed_moe = packed_moe
        self.dense_sessions = dense_sessions

        self.side_dim = 4 * 4 + len(HISTORY_SPECS) * 4
        self.base_dim = feature_dim - self.side_dim
        if self.base_dim <= 0:
            raise ValueError("feature_dim must be larger than synthetic side feature dim")

        self.chunk_sizes = self._make_chunk_sizes(feature_dim, chunk_count)
        self.tables = [self._make_table(table_index) for table_index in range(4)]

        self.expert_w1 = [
            [
                self._weight([chunk_size, hidden_dim], 1000 + expert_id * 100 + chunk_id)
                for chunk_id, chunk_size in enumerate(self.chunk_sizes)
            ]
            for expert_id in range(3)
        ]
        self.expert_b1 = [self._weight([hidden_dim], 2000 + expert_id) for expert_id in range(3)]
        self.expert_w2 = [
            self._weight([hidden_dim, expert_dim], 2100 + expert_id)
            for expert_id in range(3)
        ]
        self.expert_b2 = [self._weight([expert_dim], 2200 + expert_id) for expert_id in range(3)]

        self.gate_w = [self._weight([feature_dim, 2], 2300 + task_id) for task_id in range(3)]
        self.gate_b = [self._weight([2], 2400 + task_id) for task_id in range(3)]
        self.tower_w1 = [self._weight([expert_dim, tower_dim], 2500 + task_id) for task_id in range(3)]
        self.tower_b1 = [self._weight([tower_dim], 2600 + task_id) for task_id in range(3)]
        self.tower_w2 = [self._weight([tower_dim, 1], 2700 + task_id) for task_id in range(3)]
        self.tower_b2 = [self._weight([1], 2800 + task_id) for task_id in range(3)]

        self.packed_w1 = self._weight([feature_dim, 2 * hidden_dim], 3000)
        self.packed_b1 = self._weight([2 * hidden_dim], 3001)
        self.packed_w2 = [self._weight([hidden_dim, expert_dim], 3100 + expert_id) for expert_id in range(2)]
        self.packed_b2 = [self._weight([expert_dim], 3200 + expert_id) for expert_id in range(2)]
        self.packed_gate_w = self._weight([feature_dim, 6], 3300)
        self.packed_gate_b = self._weight([6], 3301)

    @staticmethod
    def _make_chunk_sizes(total: int, chunk_count: int) -> list[int]:
        base = total // chunk_count
        remainder = total % chunk_count
        return [base + (1 if index < remainder else 0) for index in range(chunk_count)]

    def _weight(self, shape: list[int], seed: int, scale: float = 0.02) -> tf.Variable:
        values = tf.random.stateless_normal(shape, seed=[seed, seed + 17], dtype=tf.float32) * scale
        return tf.Variable(values, trainable=False)

    def _make_table(self, table_index: int) -> tf.Variable:
        values = tf.random.stateless_uniform(
            [self.table_rows, 4],
            seed=[10 + table_index, 99 + table_index],
            minval=-1.0,
            maxval=1.0,
            dtype=tf.float32,
        )
        return tf.Variable(values, trainable=False)

    def _table_features(
        self,
        table_ids_0: tf.Tensor,
        table_ids_1: tf.Tensor,
        table_ids_2: tf.Tensor,
        table_ids_3: tf.Tensor,
    ) -> list[tf.Tensor]:
        feature_parts = []
        for table, ids in zip(self.tables, (table_ids_0, table_ids_1, table_ids_2, table_ids_3)):
            feature_parts.append(tf.reduce_mean(tf.gather(table, ids), axis=1))
        return feature_parts

    def _dense_build_session(self, ids: tf.Tensor, lengths: tf.Tensor, max_len: int) -> tuple[tf.Tensor, tf.Tensor]:
        positions = tf.range(max_len, dtype=tf.int32)[tf.newaxis, :]
        mask = positions < lengths[:, tf.newaxis]
        return tf.where(mask, ids, tf.zeros_like(ids)), mask

    def _scatter_build_session(
        self,
        ids: tf.Tensor,
        lengths: tf.Tensor,
        max_len: int,
        multiplier: int,
        history_name: str,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope(f"serve_prep/history_{history_name}_session_generator"):
            positions = tf.range(max_len, dtype=tf.int32)
            target_positions = tf.math.floormod(positions * tf.constant(multiplier, dtype=tf.int32), max_len)
            batch_positions = tf.range(self.batch_size, dtype=tf.int32)
            batch_grid = tf.tile(batch_positions[:, tf.newaxis], [1, max_len])
            target_grid = tf.tile(target_positions[tf.newaxis, :], [self.batch_size, 1])
            scatter_indices = tf.stack([tf.reshape(batch_grid, [-1]), tf.reshape(target_grid, [-1])], axis=1)
            valid = positions[tf.newaxis, :] < lengths[:, tf.newaxis]
            shifted_updates = tf.where(valid, ids + 1, tf.zeros_like(ids))
            shifted_session_ids = tf.scatter_nd(
                scatter_indices,
                tf.reshape(shifted_updates, [-1]),
                [self.batch_size, max_len],
                name="act_idx",
            )
            session_mask = shifted_session_ids > 0
            session_ids = tf.maximum(shifted_session_ids - 1, 0)
            return session_ids, session_mask

    def _history_feature(
        self,
        table_index: int,
        ids: tf.Tensor,
        lengths: tf.Tensor,
        max_len: int,
        multiplier: int,
        history_name: str,
    ) -> tf.Tensor:
        if self.dense_sessions:
            session_ids, session_mask = self._dense_build_session(ids, lengths, max_len)
        else:
            session_ids, session_mask = self._scatter_build_session(ids, lengths, max_len, multiplier, history_name)
        session_emb = tf.gather(self.tables[table_index], session_ids, name=f"model/history_{history_name}_din/act_padding_session")
        mask = tf.cast(session_mask[..., tf.newaxis], tf.float32)
        denominator = tf.maximum(tf.cast(lengths[:, tf.newaxis], tf.float32), 1.0)
        return tf.reduce_sum(session_emb * mask, axis=1) / denominator

    def _features(
        self,
        dense_base: tf.Tensor,
        table_ids_0: tf.Tensor,
        table_ids_1: tf.Tensor,
        table_ids_2: tf.Tensor,
        table_ids_3: tf.Tensor,
        history_click_ids: tf.Tensor,
        history_noware_ids: tf.Tensor,
        history_search_ids: tf.Tensor,
        history_click_lengths: tf.Tensor,
        history_noware_lengths: tf.Tensor,
        history_search_lengths: tf.Tensor,
    ) -> tf.Tensor:
        parts = [dense_base]
        parts.extend(self._table_features(table_ids_0, table_ids_1, table_ids_2, table_ids_3))
        parts.append(self._history_feature(0, history_click_ids, history_click_lengths, HISTORY_SPECS[0][1], HISTORY_SPECS[0][2], HISTORY_SPECS[0][0]))
        parts.append(self._history_feature(1, history_noware_ids, history_noware_lengths, HISTORY_SPECS[1][1], HISTORY_SPECS[1][2], HISTORY_SPECS[1][0]))
        parts.append(self._history_feature(2, history_search_ids, history_search_lengths, HISTORY_SPECS[2][1], HISTORY_SPECS[2][2], HISTORY_SPECS[2][0]))
        return tf.concat(parts, axis=1)

    def _split_expert(self, features: tf.Tensor, expert_id: int) -> tf.Tensor:
        chunks = tf.split(features, self.chunk_sizes, axis=1)
        hidden = tf.zeros([self.batch_size, self.hidden_dim], dtype=tf.float32)
        for chunk, weight in zip(chunks, self.expert_w1[expert_id]):
            hidden = hidden + tf.matmul(chunk, weight)
        hidden = swish(hidden + self.expert_b1[expert_id])
        return swish(tf.matmul(hidden, self.expert_w2[expert_id]) + self.expert_b2[expert_id])

    def _bad_moe(self, features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope("model/common_MoE"):
            experts = [self._split_expert(features, expert_id) for expert_id in range(3)]
            stacked_experts = tf.stack(experts, axis=1, name="stack")
            selected_experts = tf.gather(stacked_experts, tf.constant([0, 1], dtype=tf.int32), axis=1, name="GatherV2_3")
            return selected_experts, self._task_outputs(features, selected_experts)

    def _packed_moe(self, features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        packed_hidden = swish(tf.matmul(features, self.packed_w1) + self.packed_b1)
        expert_hidden = tf.split(packed_hidden, 2, axis=1)
        selected_experts = tf.stack(
            [
                swish(tf.matmul(expert_hidden[expert_id], self.packed_w2[expert_id]) + self.packed_b2[expert_id])
                for expert_id in range(2)
            ],
            axis=1,
        )
        return selected_experts, self._packed_task_outputs(features, selected_experts)

    def _task_outputs(self, features: tf.Tensor, selected_experts: tf.Tensor) -> tf.Tensor:
        task_outputs = []
        for task_id in range(3):
            gate = tf.nn.softmax(tf.matmul(features, self.gate_w[task_id]) + self.gate_b[task_id], axis=-1)
            mixed = tf.einsum("be,ber->br", gate, selected_experts)
            tower = swish(tf.matmul(mixed, self.tower_w1[task_id]) + self.tower_b1[task_id])
            task_outputs.append(tf.matmul(tower, self.tower_w2[task_id]) + self.tower_b2[task_id])
        return tf.concat(task_outputs, axis=1)

    def _packed_task_outputs(self, features: tf.Tensor, selected_experts: tf.Tensor) -> tf.Tensor:
        packed_gate_logits = tf.matmul(features, self.packed_gate_w) + self.packed_gate_b
        gate_logits = tf.split(packed_gate_logits, 3, axis=1)
        task_outputs = []
        for task_id in range(3):
            gate = tf.nn.softmax(gate_logits[task_id], axis=-1)
            mixed = tf.einsum("be,ber->br", gate, selected_experts)
            tower = swish(tf.matmul(mixed, self.tower_w1[task_id]) + self.tower_b1[task_id])
            task_outputs.append(tf.matmul(tower, self.tower_w2[task_id]) + self.tower_b2[task_id])
        return tf.concat(task_outputs, axis=1)

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec([63, 1288], tf.float32, name="dense_base"),
            tf.TensorSpec([63, 64], tf.int32, name="table_ids_0"),
            tf.TensorSpec([63, 64], tf.int32, name="table_ids_1"),
            tf.TensorSpec([63, 64], tf.int32, name="table_ids_2"),
            tf.TensorSpec([63, 64], tf.int32, name="table_ids_3"),
            tf.TensorSpec([63, 45], tf.int32, name="history_click_ids"),
            tf.TensorSpec([63, 57], tf.int32, name="history_noware_ids"),
            tf.TensorSpec([63, 14], tf.int32, name="history_search_ids"),
            tf.TensorSpec([63], tf.int32, name="history_click_lengths"),
            tf.TensorSpec([63], tf.int32, name="history_noware_lengths"),
            tf.TensorSpec([63], tf.int32, name="history_search_lengths"),
        ],
    )
    def serve(
        self,
        dense_base: tf.Tensor,
        table_ids_0: tf.Tensor,
        table_ids_1: tf.Tensor,
        table_ids_2: tf.Tensor,
        table_ids_3: tf.Tensor,
        history_click_ids: tf.Tensor,
        history_noware_ids: tf.Tensor,
        history_search_ids: tf.Tensor,
        history_click_lengths: tf.Tensor,
        history_noware_lengths: tf.Tensor,
        history_search_lengths: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        features = self._features(
            dense_base,
            table_ids_0,
            table_ids_1,
            table_ids_2,
            table_ids_3,
            history_click_ids,
            history_noware_ids,
            history_search_ids,
            history_click_lengths,
            history_noware_lengths,
            history_search_lengths,
        )
        selected_experts, task_outputs = self._packed_moe(features) if self.packed_moe else self._bad_moe(features)
        return {
            "mmoe_input": features,
            "selected_experts": selected_experts,
            "task_0": task_outputs[:, 0:1],
            "task_1": task_outputs[:, 1:2],
            "task_2": task_outputs[:, 2:3],
            "class_ids": tf.zeros([self.batch_size, 1], dtype=tf.int32),
            "rank_ids": tf.ones([self.batch_size, 1], dtype=tf.int32),
        }


def make_example_inputs(*, table_rows: int, batch_size: int, sparse_width: int, base_dim: int) -> dict[str, tf.Tensor]:
    dense_values = tf.linspace(-1.0, 1.0, batch_size * base_dim)
    dense_base = tf.reshape(dense_values, [batch_size, base_dim])
    sparse_base = tf.reshape(tf.range(batch_size * sparse_width, dtype=tf.int32), [batch_size, sparse_width])
    inputs = {"dense_base": dense_base}
    for table_index in range(4):
        inputs[f"table_ids_{table_index}"] = (sparse_base * (11 + table_index * 7) + 101 * table_index) % table_rows

    for history_index, (name, max_len, _multiplier) in enumerate(HISTORY_SPECS):
        raw = tf.reshape(tf.range(batch_size * max_len, dtype=tf.int32), [batch_size, max_len])
        inputs[f"history_{name}_ids"] = (raw * (17 + history_index * 5) + 19 * history_index) % table_rows
        inputs[f"history_{name}_lengths"] = ((tf.range(batch_size, dtype=tf.int32) * (history_index + 3)) % max_len) + 1
    return inputs


def export_saved_model(args: argparse.Namespace) -> Path:
    export_dir = Path(args.export_dir).resolve()
    if export_dir.exists() and args.clean:
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = Cluster2IssueModel(
        table_rows=args.table_rows,
        batch_size=args.batch_size,
        sparse_width=args.sparse_width,
        feature_dim=args.feature_dim,
        chunk_count=args.chunks,
        hidden_dim=args.hidden_dim,
        expert_dim=args.expert_dim,
        tower_dim=args.tower_dim,
        packed_moe=args.packed_moe,
        dense_sessions=args.dense_sessions,
    )
    tf.saved_model.save(model, str(export_dir), signatures={"serving_default": model.serve})
    if args.strip_init_signature:
        strip_init_signature(export_dir)
    return export_dir


def strip_init_signature(export_dir: Path) -> bool:
    saved_model_path = export_dir / "saved_model.pb"
    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString(saved_model_path.read_bytes())
    removed = False
    for meta_graph in saved_model.meta_graphs:
        if INIT_OP_SIGNATURE_KEY in meta_graph.signature_def:
            del meta_graph.signature_def[INIT_OP_SIGNATURE_KEY]
            removed = True
    if removed:
        saved_model_path.write_bytes(saved_model.SerializeToString())
    return removed


def invoke_saved_model(args: argparse.Namespace) -> dict[str, object]:
    loaded = tf.saved_model.load(str(Path(args.export_dir).resolve()))
    signature = loaded.signatures["serving_default"]
    inputs = make_example_inputs(
        table_rows=args.table_rows,
        batch_size=args.batch_size,
        sparse_width=args.sparse_width,
        base_dim=args.feature_dim - (4 * 4 + len(HISTORY_SPECS) * 4),
    )
    outputs = signature(**inputs)
    return {
        name: {
            "shape": tensor.shape.as_list(),
            "dtype": tensor.dtype.name,
            "checksum": float(tf.reduce_sum(tf.cast(tensor, tf.float32)).numpy()),
        }
        for name, tensor in outputs.items()
    }


def count_hlo_ops(hlo_dir: Path) -> dict[str, int]:
    counts = {"dot": 0, "gather": 0, "scatter": 0, "while": 0, "fusion": 0}
    for path in hlo_dir.glob("*before_optimizations.txt"):
        text = path.read_text()
        for op in counts:
            counts[op] += text.count(f" {op}(")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and invoke an XLA-compiled CVR issue SavedModel")
    parser.add_argument("--export-dir", default="xla_issue_lab/saved_models/cluster2_issue/1")
    parser.add_argument("--table-rows", type=int, default=200_000)
    parser.add_argument("--real-table-size", action="store_true", help="Use f32[8388609,4] tables like the real dump")
    parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--sparse-width", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=1316)
    parser.add_argument("--chunks", type=int, default=18)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--expert-dim", type=int, default=128)
    parser.add_argument("--tower-dim", type=int, default=64)
    parser.add_argument("--packed-moe", action="store_true")
    parser.add_argument("--dense-sessions", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strip-init-signature", action="store_true", help="Remove __saved_model_init_op from SignatureDefs for runners that execute every signature")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--invoke", action="store_true")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--hlo-dir", default="xla_issue_lab/saved_model_hlo_dumps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.real_table_size:
        args.table_rows = 8_388_609
    if args.batch_size != 63:
        raise ValueError("The SavedModel signature currently mimics fixed batch size 63, like the real cluster dump")
    if args.feature_dim != 1316:
        raise ValueError("The SavedModel signature currently mimics fixed feature dim 1316, like the real cluster dump")
    if args.clean and args.dump_hlo:
        shutil.rmtree(args.hlo_dir, ignore_errors=True)
        Path(args.hlo_dir).mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "export_dir": str(Path(args.export_dir).resolve()),
        "table_rows": args.table_rows,
        "packed_moe": args.packed_moe,
        "dense_sessions": args.dense_sessions,
        "strip_init_signature": args.strip_init_signature,
    }
    if args.export:
        result["exported"] = str(export_saved_model(args))
    if args.invoke:
        result["outputs"] = invoke_saved_model(args)
    if args.dump_hlo:
        result["before_optimization_hlo_counts"] = count_hlo_ops(Path(args.hlo_dir))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()