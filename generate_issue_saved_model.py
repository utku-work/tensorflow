#!/usr/bin/env python3
"""Export and validate a SavedModel that mimics the CVR cluster_2 XLA issue."""

from __future__ import annotations

import argparse
import json
import os
import re
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
REAL_CLUSTER2_HLO = "module_9287.cluster_2__XlaCompiledKernel_true__XlaHasReferenceVars_true__XlaNumConstantArgs_48__XlaNumResourceArgs_55_.2025.before_optimizations.txt"
HLO_DTYPE_MAP = {
    "f32": tf.float32,
    "s32": tf.int32,
    "s64": tf.int64,
    "pred": tf.bool,
}


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
            with tf.name_scope(f"common_MoE_gate_{task_id}"):
                gate = tf.nn.softmax(tf.matmul(features, self.gate_w[task_id]) + self.gate_b[task_id], axis=-1, name="Softmax")
                mixed = tf.einsum("be,ber->br", gate, selected_experts)
            tower = swish(tf.matmul(mixed, self.tower_w1[task_id]) + self.tower_b1[task_id])
            task_outputs.append(tf.matmul(tower, self.tower_w2[task_id]) + self.tower_b2[task_id])
        return tf.concat(task_outputs, axis=1)

    def _packed_task_outputs(self, features: tf.Tensor, selected_experts: tf.Tensor) -> tf.Tensor:
        packed_gate_logits = tf.matmul(features, self.packed_gate_w) + self.packed_gate_b
        gate_logits = tf.split(packed_gate_logits, 3, axis=1)
        task_outputs = []
        for task_id in range(3):
            with tf.name_scope(f"common_MoE_gate_{task_id}"):
                gate = tf.nn.softmax(gate_logits[task_id], axis=-1, name="Softmax")
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
            "out_0_mmoe_input": features,
            "out_1_selected_experts": selected_experts,
            "out_2_class_ids": tf.zeros([self.batch_size, 1], dtype=tf.int32),
            "out_3_rank_ids": tf.ones([self.batch_size, 1], dtype=tf.int32),
            "out_4_task_0": task_outputs[:, 0:1],
            "out_5_task_1": task_outputs[:, 1:2],
            "out_6_task_2": task_outputs[:, 2:3],
        }


def default_real_hlo_path() -> Path:
    return Path(__file__).resolve().parent.parent / REAL_CLUSTER2_HLO


def parse_hlo_entry_specs(hlo_path: Path) -> list[tf.TensorSpec]:
    text = hlo_path.read_text()
    entry_index = text.find("ENTRY %")
    if entry_index < 0:
        raise ValueError(f"No ENTRY computation found in {hlo_path}")
    open_paren = text.find("(", entry_index)
    close_paren = text.find(") ->", open_paren)
    if open_paren < 0 or close_paren < 0:
        raise ValueError(f"Could not parse ENTRY argument list in {hlo_path}")
    entry_args = text[open_paren + 1 : close_paren]
    specs: list[tf.TensorSpec] = []
    for match in re.finditer(r"arg(\d+)\.\d+:\s+([a-z0-9]+)\[([^\]]*)\]", entry_args):
        arg_index = int(match.group(1))
        hlo_dtype = match.group(2)
        shape_text = match.group(3)
        if arg_index != len(specs):
            raise ValueError(f"Expected arg{len(specs)} but found arg{arg_index} in {hlo_path}")
        if hlo_dtype not in HLO_DTYPE_MAP:
            raise ValueError(f"Unsupported HLO dtype {hlo_dtype!r} in {hlo_path}")
        shape = [] if not shape_text else [int(part) for part in shape_text.split(",")]
        specs.append(tf.TensorSpec(shape, HLO_DTYPE_MAP[hlo_dtype], name=f"arg{arg_index}"))
    if len(specs) != 233:
        raise ValueError(f"Expected 233 ENTRY args from real cluster_2, parsed {len(specs)} from {hlo_path}")
    return specs


class RealCluster2EntryTemplate(tf.Module):
    """HLO-entry-shaped template using the real cluster_2 argument contract."""

    def __init__(self, input_specs: list[tf.TensorSpec]) -> None:
        super().__init__()
        self.serve = tf.function(
            self._serve,
            jit_compile=True,
            input_signature=input_specs,
        ).get_concrete_function()

    def _safe_gather(self, table: tf.Tensor, indices: tf.Tensor, *, name: str) -> tf.Tensor:
        row_count = tf.constant(table.shape.as_list()[0], dtype=indices.dtype)
        safe_indices = tf.math.floormod(indices, row_count)
        return tf.gather(table, safe_indices, name=name)

    def _batch_tile(self, value: tf.Tensor) -> tf.Tensor:
        flat = tf.reshape(tf.cast(value, tf.float32), [1, -1])
        return tf.tile(flat, [63, 1])

    def _reduce_to_batch_feature(self, value: tf.Tensor) -> tf.Tensor:
        rank = value.shape.rank
        if rank is None:
            value = tf.reshape(value, [-1, value.shape.as_list()[-1]])
            reduced = tf.reduce_mean(value, axis=0)
        elif rank <= 1:
            reduced = value
        else:
            reduced = tf.reduce_mean(tf.cast(value, tf.float32), axis=list(range(rank - 1)))
        return self._batch_tile(reduced)

    def _scatter_session_ids(
        self,
        indices: tf.Tensor,
        updates: tf.Tensor,
        max_len: int,
        history_name: str,
    ) -> tf.Tensor:
        with tf.name_scope(f"serve_prep/history_{history_name}_session_generator"):
            shifted_updates = tf.cast(updates, tf.int64) + tf.constant(1, dtype=tf.int64)
            return tf.scatter_nd(
                indices,
                shifted_updates,
                [1, max_len],
                name="act_idx",
            )

    def _session_feature(
        self,
        table: tf.Tensor,
        session_ids: tf.Tensor,
        history_name: str,
    ) -> tf.Tensor:
        gather_ids = tf.maximum(session_ids - tf.constant(1, dtype=session_ids.dtype), tf.constant(0, dtype=session_ids.dtype))
        session_emb = self._safe_gather(table, gather_ids, name=f"model/history_{history_name}_din/act_padding_session")
        mask = tf.cast(session_ids > 0, tf.float32)[..., tf.newaxis]
        return self._reduce_to_batch_feature(session_emb * mask)

    def _indexed_feature(self, table: tf.Tensor, indices: tf.Tensor, name: str) -> tf.Tensor:
        gathered = self._safe_gather(table, indices, name=name)
        if gathered.shape.rank == 2 and gathered.shape.as_list()[0] == 63:
            return tf.cast(gathered, tf.float32)
        return self._reduce_to_batch_feature(gathered)

    def _fit_features(self, parts: list[tf.Tensor], touch: tf.Tensor) -> tf.Tensor:
        features = tf.concat(parts, axis=1)
        width = features.shape.as_list()[1]
        if width is None:
            features = features[:, :1316]
        elif width < 1316:
            features = tf.concat([features, tf.zeros([63, 1316 - width], dtype=tf.float32)], axis=1)
        elif width > 1316:
            features = features[:, :1316]
        return features + touch

    def _touch_all_args(self, entry_args: tuple[tf.Tensor, ...]) -> tf.Tensor:
        touch = tf.constant(0.0, dtype=tf.float32)
        for tensor in entry_args:
            first_value = tf.reshape(tf.cast(tensor, tf.float32), [-1])[0]
            touch = touch + first_value * tf.constant(1.0e-12, dtype=tf.float32)
        return touch

    def _extra_gather_touch(self, args: tuple[tf.Tensor, ...]) -> tf.Tensor:
        pairs = [
            (102, 103), (102, 104), (108, 109), (110, 111), (112, 113), (120, 145),
            (121, 115), (122, 117), (123, 119), (124, 100), (125, 99), (126, 158),
            (127, 160), (128, 98), (129, 158), (130, 159), (131, 160), (136, 157),
            (137, 138), (139, 140), (141, 142), (143, 144), (149, 150), (151, 152),
            (153, 154), (155, 156), (178, 113), (179, 113), (180, 113), (137, 132),
            (139, 133), (141, 134), (143, 135), (120, 146),
        ]
        touch = tf.constant(0.0, dtype=tf.float32)
        for index, (table_arg, index_arg) in enumerate(pairs):
            with tf.name_scope(f"model/extra_gather_fanout_{index}"):
                offset = tf.cast(index, args[index_arg].dtype)
                gathered = self._safe_gather(args[table_arg], args[index_arg] + offset, name="GatherV2")
                touch = touch + tf.reduce_sum(tf.cast(gathered, tf.float32)) * tf.constant(1.0e-12, dtype=tf.float32)
        return touch

    def _extra_dot_touch(self, args: tuple[tf.Tensor, ...]) -> tf.Tensor:
        touch = tf.constant(0.0, dtype=tf.float32)
        for index in range(30):
            lhs = tf.cast(args[index], tf.float32)
            rhs = tf.cast(args[index + 1], tf.float32)
            shared_dim = min(lhs.shape.as_list()[0], rhs.shape.as_list()[0])
            with tf.name_scope(f"model/extra_dot_fanout_{index}"):
                dot = tf.matmul(tf.transpose(lhs[:shared_dim, :]), rhs[:shared_dim, :], name="MatMul")
                touch = touch + tf.reduce_sum(dot) * tf.constant(1.0e-6, dtype=tf.float32)
        return touch

    def _features(self, entry_args: tuple[tf.Tensor, ...], touch: tf.Tensor) -> tf.Tensor:
        args = entry_args
        click_ids = self._scatter_session_ids(args[114], args[115], 45, "click")
        search_ids = self._scatter_session_ids(args[116], args[117], 14, "search")
        noware_ids = self._scatter_session_ids(args[118], args[119], 57, "noware")

        parts = [
            tf.reshape(args[96], [63, 1]),
            tf.cast(args[97], tf.float32),
            self._indexed_feature(args[102], args[104], "model/BuildCommonEmbInput/domain_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[137], args[138], "model/iF_iic/huge_act_emb_lookup/GatherV2"),
            self._indexed_feature(args[139], args[140], "model/iC_iic/huge_act_emb_lookup/GatherV2"),
            self._indexed_feature(args[141], args[142], "model/iR_iic/huge_act_emb_lookup/GatherV2"),
            self._indexed_feature(args[143], args[144], "model/history_huge_act_emb_lookup/GatherV2"),
            self._session_feature(args[137], click_ids, "click"),
            self._session_feature(args[139], noware_ids, "noware"),
            self._session_feature(args[141], search_ids, "search"),
            self._indexed_feature(args[108], args[109], "model/click/click_freq_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[110], args[111], "model/search/search_freq_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[112], args[113], "model/noware/noware_freq_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[120], args[145], "model/BuildCommonEmbInput/GatherV2"),
            self._indexed_feature(args[149], args[150], "model/de/de_time_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[151], args[152], "model/iF_iic/iF_iic_time_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[153], args[154], "model/iC_iic/iC_iic_time_emb_lookup/EmbeddingLookupUnique/GatherV2"),
            self._indexed_feature(args[155], args[156], "model/iR_iic/iR_iic_time_emb_lookup/EmbeddingLookupUnique/GatherV2"),
        ]
        for arg_index in range(96):
            parts.append(self._reduce_to_batch_feature(args[arg_index]))
        return self._fit_features(parts, touch)

    def _moe(self, features: tf.Tensor, args: tuple[tf.Tensor, ...]) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope("model/common_MoE"):
            padded_features = tf.concat([features, tf.zeros([63, 224], dtype=tf.float32)], axis=1)
            hidden_base = swish(tf.matmul(padded_features, args[161]) + args[162])
            expert_outputs = []
            for expert_id, weight_arg in enumerate((166, 167, 168)):
                hidden = hidden_base + tf.reshape(args[181 + expert_id], [1, 64])
                expert = swish(tf.matmul(hidden, tf.transpose(args[weight_arg])))
                expert_outputs.append(expert)
            stacked_experts = tf.stack(expert_outputs, axis=1, name="stack")
            selected_experts = tf.gather(stacked_experts, tf.constant([0, 1], dtype=tf.int32), axis=1, name="GatherV2_3")

            task_outputs = []
            for task_id, gate_arg in enumerate((163, 164, 165)):
                with tf.name_scope(f"common_MoE_gate_{task_id}"):
                    gate = tf.nn.softmax(tf.matmul(features, args[gate_arg]), axis=-1, name="Softmax")
                    mixed = tf.einsum("be,ber->br", gate, selected_experts)
                tower = swish(tf.matmul(mixed, args[166 + task_id]) + args[169 + task_id])
                if task_id == 0:
                    task_outputs.append(tf.matmul(tower, args[173]) + args[176])
                elif task_id == 1:
                    task_outputs.append(tf.matmul(tower, args[174]) + args[177])
                else:
                    task_outputs.append((tf.matmul(tower, args[172]) + args[175])[:, 0:1])
            return selected_experts, tf.concat(task_outputs, axis=1)

    def _serve(self, *entry_args: tf.Tensor) -> dict[str, tf.Tensor]:
        touch = self._touch_all_args(entry_args) + self._extra_gather_touch(entry_args) + self._extra_dot_touch(entry_args)
        features = self._features(entry_args, touch)
        selected_experts, task_outputs = self._moe(features, entry_args)
        return {
            "out_0_mmoe_input": features,
            "out_1_selected_experts": selected_experts,
            "out_2_class_ids": tf.cast(entry_args[97], tf.int32),
            "out_3_rank_ids": tf.reshape(tf.cast(entry_args[104], tf.int32), [63, 1]),
            "out_4_task_0": task_outputs[:, 0:1],
            "out_5_task_1": task_outputs[:, 1:2],
            "out_6_task_2": task_outputs[:, 2:3],
        }


class RealCluster2HeavyDotTemplate(RealCluster2EntryTemplate):
    """Real-entry template that makes the 46 dot ops do larger runtime work."""

    def _extra_dot_touch(self, args: tuple[tf.Tensor, ...]) -> tf.Tensor:
        features = self._fit_features([self._reduce_to_batch_feature(args[index]) for index in range(96)], tf.constant(0.0))
        padded_features = tf.concat([features, tf.zeros([63, 224], dtype=tf.float32)], axis=1)
        touch = tf.constant(0.0, dtype=tf.float32)
        heavy_pairs = [
            (padded_features, args[161]),
            (features, args[163]),
            (features, args[164]),
            (features, args[165]),
        ]
        for index, (lhs, rhs) in enumerate(heavy_pairs):
            with tf.name_scope(f"model/heavy_dot_fanout_{index}"):
                dot = tf.matmul(lhs, rhs, name="MatMul")
                touch = touch + tf.reduce_sum(dot) * tf.constant(1.0e-6, dtype=tf.float32)
        for index in range(26):
            lhs = tf.cast(args[index], tf.float32)
            rhs = tf.cast(args[index + 1], tf.float32)
            shared_dim = min(lhs.shape.as_list()[0], rhs.shape.as_list()[0])
            with tf.name_scope(f"model/extra_dot_fanout_{index}"):
                dot = tf.matmul(tf.transpose(lhs[:shared_dim, :]), rhs[:shared_dim, :], name="MatMul")
                touch = touch + tf.reduce_sum(dot) * tf.constant(1.0e-6, dtype=tf.float32)
        return touch


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


def resolve_real_hlo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    fallback = default_real_hlo_path()
    if path_text == REAL_CLUSTER2_HLO and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find real HLO file {path_text!r}; also tried {fallback}")


def _sequential_indices(shape: tf.TensorShape, dtype: tf.dtypes.DType) -> tf.Tensor:
    element_count = 1
    for dim in shape.as_list():
        element_count *= dim
    return tf.reshape(tf.range(element_count, dtype=dtype), shape.as_list())


def _scatter_indices(update_count: int, max_len: int) -> tf.Tensor:
    rows = tf.zeros([update_count], dtype=tf.int64)
    cols = tf.range(update_count, dtype=tf.int64) % tf.constant(max_len, dtype=tf.int64)
    return tf.stack([rows, cols], axis=1)


def _float_input(shape: tf.TensorShape, index: int) -> tf.Tensor:
    scale = tf.constant(((index % 13) + 1) * 0.01, dtype=tf.float32)
    return tf.ones(shape, dtype=tf.float32) * scale


def make_real_entry_inputs(input_specs: list[tf.TensorSpec]) -> dict[str, tf.Tensor]:
    inputs: dict[str, tf.Tensor] = {}
    for index, spec in enumerate(input_specs):
        shape = tf.TensorShape(spec.shape)
        if spec.dtype == tf.float32:
            inputs[f"arg{index}"] = _float_input(shape, index)
        elif spec.dtype in (tf.int32, tf.int64):
            inputs[f"arg{index}"] = _sequential_indices(shape, spec.dtype)
        elif spec.dtype == tf.bool:
            inputs[f"arg{index}"] = tf.zeros(shape, dtype=tf.bool)
        else:
            raise ValueError(f"Unsupported input dtype {spec.dtype.name!r} for arg{index}")

    inputs["arg114"] = _scatter_indices(44, 45)
    inputs["arg115"] = tf.range(44, dtype=tf.int32)
    inputs["arg116"] = _scatter_indices(13, 14)
    inputs["arg117"] = tf.range(13, dtype=tf.int32)
    inputs["arg118"] = _scatter_indices(56, 57)
    inputs["arg119"] = tf.range(56, dtype=tf.int32)
    return inputs


def export_saved_model(args: argparse.Namespace) -> Path:
    export_dir = Path(args.export_dir).resolve()
    if export_dir.exists() and args.clean:
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.real_entry_template:
        input_specs = parse_hlo_entry_specs(resolve_real_hlo_path(args.real_hlo))
        model = RealCluster2HeavyDotTemplate(input_specs) if args.heavy_dots else RealCluster2EntryTemplate(input_specs)
        signature = model.serve
    else:
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
        signature = model.serve
    tf.saved_model.save(model, str(export_dir), signatures={"serving_default": signature})
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
    if args.real_entry_template:
        inputs = make_real_entry_inputs(parse_hlo_entry_specs(resolve_real_hlo_path(args.real_hlo)))
    else:
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
    parser.add_argument("--real-entry-template", action="store_true", help="Export a 233-input template parsed from the real cluster_2 HLO ENTRY signature")
    parser.add_argument("--heavy-dots", action="store_true", help="With --real-entry-template, spend the matched dot count on larger runtime matmuls for profiling")
    parser.add_argument("--real-hlo", default=REAL_CLUSTER2_HLO, help="Real cluster_2 before_optimizations HLO used for --real-entry-template")
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
    if args.real_entry_template:
        real_hlo_path = resolve_real_hlo_path(args.real_hlo)
    else:
        real_hlo_path = None
    if args.heavy_dots and not args.real_entry_template:
        raise ValueError("--heavy-dots requires --real-entry-template")
    if not args.real_entry_template and args.batch_size != 63:
        raise ValueError("The SavedModel signature currently mimics fixed batch size 63, like the real cluster dump")
    if not args.real_entry_template and args.feature_dim != 1316:
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
        "real_entry_template": args.real_entry_template,
        "heavy_dots": args.heavy_dots,
    }
    if real_hlo_path is not None:
        result["real_hlo"] = str(real_hlo_path)
        result["entry_arg_count"] = len(parse_hlo_entry_specs(real_hlo_path))
    if args.export:
        result["exported"] = str(export_saved_model(args))
    if args.invoke:
        result["outputs"] = invoke_saved_model(args)
    if args.dump_hlo:
        result["before_optimization_hlo_counts"] = count_hlo_ops(Path(args.hlo_dir))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()