#include "tensorflow/cc/saved_model/variable_freezing.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace internal {
namespace {

constexpr int64_t kDefaultMaxTensorBytes = 1LL * 1024 * 1024;

struct FrozenValue {
	Tensor tensor;
};

using FrozenValueMap = absl::flat_hash_map<std::string, FrozenValue>;
using FrozenInputMap = absl::flat_hash_map<std::string, const FrozenValue*>;

bool IsEnabled() {
	return true;
}

int64_t GetMaxFrozenTensorBytes() {
	return kDefaultMaxTensorBytes;
}

bool HasAnyToken(absl::string_view value,
								 std::initializer_list<absl::string_view> tokens) {
	for (absl::string_view token : tokens) {
		if (!token.empty() && value.find(token) != absl::string_view::npos) {
			return true;
		}
	}
	return false;
}

bool HasAnySuffix(absl::string_view value,
									std::initializer_list<absl::string_view> suffixes) {
	for (absl::string_view suffix : suffixes) {
		if (!suffix.empty() && absl::EndsWith(value, suffix)) {
			return true;
		}
	}
	return false;
}

bool IsAllowlistedVariableName(absl::string_view name) {
	const std::string lowered_name = absl::AsciiStrToLower(std::string(name));
	const absl::string_view lowered_view(lowered_name);
	if (HasAnyToken(lowered_view,
									{"embedding", "lookup_table", "/part_", "hash_table"})) {
		return false;
	}
	return HasAnySuffix(lowered_view,
											{"weight", "bias", "kernel", "_w", "/w", "_b",
											 "/b", "beta", "gamma", "moving_mean",
											 "moving_variance", "mean", "variance"});
}

bool ShouldFreezeTensor(const Tensor& tensor, int64_t max_tensor_bytes) {
	return max_tensor_bytes <= 0 || tensor.TotalBytes() <= max_tensor_bytes;
}

std::string BaseNodeName(absl::string_view input) {
	if (!input.empty() && input.front() == '^') {
		input.remove_prefix(1);
	}
	const size_t colon = input.find(':');
	if (colon == absl::string_view::npos) return std::string(input);
	return std::string(input.substr(0, colon));
}

int OutputIndex(absl::string_view input) {
	if (!input.empty() && input.front() == '^') {
		input.remove_prefix(1);
	}
	const size_t colon = input.rfind(':');
	if (colon == absl::string_view::npos) return 0;
	int output_index = 0;
	for (size_t i = colon + 1; i < input.size(); ++i) {
		char ch = input[i];
		if (ch < '0' || ch > '9') return 0;
		output_index = output_index * 10 + (ch - '0');
	}
	return output_index;
}

bool GetFirstDataInput(const NodeDef& node, std::string* input_name) {
	for (const std::string& input : node.input()) {
		if (!input.empty() && input.front() == '^') continue;
		*input_name = BaseNodeName(input);
		return true;
	}
	return false;
}

std::vector<std::string> GetControlInputs(const NodeDef& node) {
	std::vector<std::string> controls;
	for (const std::string& input : node.input()) {
		if (!input.empty() && input.front() == '^') {
			controls.push_back(input);
		}
	}
	return controls;
}

void PreserveInternalAttrs(const NodeDef& original, NodeDef* replacement) {
	absl::flat_hash_map<std::string, AttrValue> internal_attrs;
	for (const auto& attr : original.attr()) {
		if (!attr.first.empty() && attr.first[0] == '_') {
			internal_attrs.insert(attr);
		}
	}

	replacement->clear_attr();
	for (const auto& attr : internal_attrs) {
		(*replacement->mutable_attr())[attr.first] = attr.second;
	}
}

void ReplaceNodeWithConst(const FrozenValue& frozen_value, NodeDef* node) {
	const std::vector<std::string> controls = GetControlInputs(*node);
	const NodeDef original = *node;

	node->set_op("Const");
	node->clear_input();
	for (const std::string& control : controls) {
		node->add_input(control);
	}

	PreserveInternalAttrs(original, node);
	(*node->mutable_attr())["dtype"].set_type(frozen_value.tensor.dtype());
	frozen_value.tensor.AsProtoTensorContent(
			(*node->mutable_attr())["value"].mutable_tensor());
}

bool IsCallNode(const NodeDef& node) {
	return node.op() == "PartitionedCall" ||
				 node.op() == "StatefulPartitionedCall";
}

std::string GetCalledFunctionName(const NodeDef& node) {
	const auto it = node.attr().find("f");
	if (it == node.attr().end() || !it->second.has_func()) return std::string();
	return it->second.func().name();
}

absl::StatusOr<std::string> GetConstStringValue(const NodeDef& node) {
	if (node.op() != "Const") {
		return absl::InvalidArgumentError(
				absl::StrCat("Expected Const node but found ", node.op()));
	}
	const auto dtype_it = node.attr().find("dtype");
	const auto value_it = node.attr().find("value");
	if (dtype_it == node.attr().end() || value_it == node.attr().end() ||
			dtype_it->second.type() != DT_STRING ||
			!value_it->second.has_tensor()) {
		return absl::InvalidArgumentError(
				absl::StrCat("Const node ", node.name(),
								 " is not a string tensor const"));
	}

	const TensorProto& tensor_proto = value_it->second.tensor();
	if (!tensor_proto.string_val().empty()) {
		return tensor_proto.string_val(0);
	}

	Tensor tensor;
	if (!tensor.FromProto(tensor_proto)) {
		return absl::InvalidArgumentError(
				absl::StrCat("Unable to deserialize const tensor ", node.name()));
	}
	if (tensor.dtype() != DT_STRING || tensor.NumElements() != 1) {
		return absl::InvalidArgumentError(
				absl::StrCat("Expected single-element string const ", node.name()));
	}
	return std::string(tensor.flat<tstring>()(0));
}

absl::StatusOr<std::vector<std::string>> GetConstStringValues(
		const NodeDef& node) {
	if (node.op() != "Const") {
		return absl::InvalidArgumentError(
				absl::StrCat("Expected Const node but found ", node.op()));
	}
	const auto dtype_it = node.attr().find("dtype");
	const auto value_it = node.attr().find("value");
	if (dtype_it == node.attr().end() || value_it == node.attr().end() ||
			dtype_it->second.type() != DT_STRING ||
			!value_it->second.has_tensor()) {
		return absl::InvalidArgumentError(
				absl::StrCat("Const node ", node.name(),
								 " is not a string tensor const"));
	}

	const TensorProto& tensor_proto = value_it->second.tensor();
	if (!tensor_proto.string_val().empty()) {
		return std::vector<std::string>(tensor_proto.string_val().begin(),
												 tensor_proto.string_val().end());
	}

	Tensor tensor;
	if (!tensor.FromProto(tensor_proto)) {
		return absl::InvalidArgumentError(
				absl::StrCat("Unable to deserialize const tensor ", node.name()));
	}
	if (tensor.dtype() != DT_STRING) {
		return absl::InvalidArgumentError(
				absl::StrCat("Expected string const tensor ", node.name()));
	}

	std::vector<std::string> values;
	values.reserve(tensor.NumElements());
	auto flat = tensor.flat<tstring>();
	for (int i = 0; i < flat.size(); ++i) {
		values.push_back(std::string(flat(i)));
	}
	return values;
}

absl::StatusOr<absl::flat_hash_map<std::string, const NodeDef*>> BuildNodeMap(
		const GraphDef& graph_def) {
	absl::flat_hash_map<std::string, const NodeDef*> node_map;
	node_map.reserve(graph_def.node_size());
	for (const NodeDef& node : graph_def.node()) {
		node_map[node.name()] = &node;
	}
	return node_map;
}

absl::flat_hash_map<std::string, const NodeDef*> BuildFunctionNodeMap(
		const FunctionDef& function) {
	absl::flat_hash_map<std::string, const NodeDef*> node_map;
	node_map.reserve(function.node_def_size());
	for (const NodeDef& node : function.node_def()) {
		node_map[node.name()] = &node;
	}
	return node_map;
}

std::string ResolveForwardedInputName(
		absl::string_view input_name,
		const absl::flat_hash_map<std::string, const NodeDef*>& node_map) {
	std::string current = BaseNodeName(input_name);
	while (true) {
		const auto it = node_map.find(current);
		if (it == node_map.end() || it->second->op() != "Identity") {
			return current;
		}

		std::string forwarded_name;
		if (!GetFirstDataInput(*it->second, &forwarded_name)) {
			return current;
		}
		current = forwarded_name;
	}
}

std::vector<std::string> CandidateCheckpointKeys(const NodeDef& node) {
	std::vector<std::string> keys;
	const auto shared_name_it = node.attr().find("shared_name");
	const std::string shared_name =
			shared_name_it != node.attr().end() ? shared_name_it->second.s() : "";
	if (!shared_name.empty()) {
		keys.push_back(shared_name);
		keys.push_back(absl::StrCat(shared_name, "/.ATTRIBUTES/VARIABLE_VALUE"));
	}
	keys.push_back(node.name());
	keys.push_back(absl::StrCat(node.name(), "/.ATTRIBUTES/VARIABLE_VALUE"));
	return keys;
}

absl::StatusOr<FrozenValueMap> LoadFrozenVariableV1Values(
		const std::string& export_dir, const GraphDef& graph_def,
		int64_t max_tensor_bytes) {
	bool has_variable_v2 = false;
	for (const NodeDef& node : graph_def.node()) {
		if (node.op() == "VariableV2") {
			has_variable_v2 = true;
			break;
		}
	}
	if (!has_variable_v2) return FrozenValueMap();

	const std::string variables_prefix = io::JoinPath(
			export_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename);
	BundleReader reader(Env::Default(), variables_prefix);
	TF_RETURN_WITH_CONTEXT_IF_ERROR(
			reader.status(), "Unable to load SavedModel variables checkpoint from ",
			variables_prefix);

	TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(graph_def));
	FrozenValueMap frozen_values;
	for (const NodeDef& node : graph_def.node()) {
		if (node.op() != "Assign" || node.input_size() < 2) continue;

		const std::string variable_name = BaseNodeName(node.input(0));
		const auto variable_it = node_map.find(variable_name);
		if (variable_it == node_map.end() ||
				variable_it->second->op() != "VariableV2") {
			continue;
		}
		if (!IsAllowlistedVariableName(variable_name)) continue;
		if (frozen_values.contains(variable_name)) continue;

		std::string tensor_name = variable_name;
		const std::string restore_input = node.input(1);
		const int restore_output_index = OutputIndex(restore_input);
		const std::string restore_name = BaseNodeName(restore_input);
		const auto restore_it = node_map.find(restore_name);
		if (restore_it != node_map.end() &&
				restore_it->second->op() == "RestoreV2" &&
				restore_it->second->input_size() >= 2) {
			const std::string tensor_names_const_name =
					BaseNodeName(restore_it->second->input(1));
			const auto tensor_names_const_it =
					node_map.find(tensor_names_const_name);
			if (tensor_names_const_it != node_map.end()) {
				TF_ASSIGN_OR_RETURN(
						std::vector<std::string> tensor_names,
						GetConstStringValues(*tensor_names_const_it->second));
				if (restore_output_index >= 0 &&
						restore_output_index < tensor_names.size()) {
					tensor_name = tensor_names[restore_output_index];
				} else if (!tensor_names.empty()) {
					tensor_name = tensor_names[0];
				}
			}
		}

		Tensor tensor;
		TF_RETURN_IF_ERROR(reader.Lookup(tensor_name, &tensor));
		if (!ShouldFreezeTensor(tensor, max_tensor_bytes)) {
			continue;
		}
		LOG(INFO) << "[variable_freezing] matched VariableV2 checkpoint key="
						  << tensor_name << " graph_node=" << variable_name;
		frozen_values[variable_name] = FrozenValue{std::move(tensor)};
	}
	return frozen_values;
}

absl::StatusOr<bool> LookupTensorForCheckpointKeys(
		BundleReader* reader, const std::vector<std::string>& candidate_keys,
		Tensor* tensor) {
	for (const std::string& candidate_key : candidate_keys) {
		absl::Status status = reader->Lookup(candidate_key, tensor);
		if (status.ok()) {
			return true;
		}
		if (status.code() != absl::StatusCode::kNotFound) {
			return status;
		}
	}
	return false;
}

absl::StatusOr<FrozenValueMap> LoadFrozenVarHandleValues(
		const std::string& export_dir, const GraphDef& graph_def,
		int64_t max_tensor_bytes) {
	bool has_var_handle = false;
	for (const NodeDef& node : graph_def.node()) {
		if (node.op() == "VarHandleOp") {
			has_var_handle = true;
			break;
		}
	}
	if (!has_var_handle) return FrozenValueMap();

	const std::string variables_prefix = io::JoinPath(
			export_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename);
	BundleReader reader(Env::Default(), variables_prefix);
	TF_RETURN_WITH_CONTEXT_IF_ERROR(
			reader.status(), "Unable to load SavedModel variables checkpoint from ",
			variables_prefix);

	FrozenValueMap frozen_values;
	for (const NodeDef& node : graph_def.node()) {
		if (node.op() != "VarHandleOp") {
			continue;
		}
		const auto shared_name_it = node.attr().find("shared_name");
		const std::string shared_name =
				shared_name_it != node.attr().end() ? shared_name_it->second.s() : "";
		if ((!node.name().empty() && !IsAllowlistedVariableName(node.name())) &&
				(shared_name.empty() || !IsAllowlistedVariableName(shared_name))) {
			continue;
		}

		Tensor tensor;
		TF_ASSIGN_OR_RETURN(bool found,
												LookupTensorForCheckpointKeys(
														&reader, CandidateCheckpointKeys(node), &tensor));
		if (!found || !ShouldFreezeTensor(tensor, max_tensor_bytes)) {
			continue;
		}
		LOG(INFO) << "[variable_freezing] matched VarHandleOp checkpoint graph_node="
						  << node.name() << " shared_name=" << shared_name;
		frozen_values[node.name()] = FrozenValue{std::move(tensor)};
	}
	return frozen_values;
}

absl::Status RewriteTopLevelReadNodes(GraphDef* graph_def,
																			const FrozenValueMap& frozen_values) {
	TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(*graph_def));
	for (NodeDef& node : *graph_def->mutable_node()) {
		if (node.op() != "ReadVariableOp" && node.op() != "Identity") {
			continue;
		}

		std::string source_name;
		if (!GetFirstDataInput(node, &source_name)) continue;
		source_name = ResolveForwardedInputName(source_name, node_map);
		const auto frozen_it = frozen_values.find(source_name);
		if (frozen_it == frozen_values.end()) continue;
		LOG(INFO) << "[variable_freezing] rewrite top-level node=" << node.name()
						  << " op=" << node.op() << " source=" << source_name;
		ReplaceNodeWithConst(frozen_it->second, &node);
	}
	return absl::OkStatus();
}

std::string BuildVisitedKey(absl::string_view function_name,
														const FrozenInputMap& frozen_inputs) {
	std::vector<std::string> keys;
	keys.reserve(frozen_inputs.size());
	for (const auto& entry : frozen_inputs) {
		keys.push_back(entry.first);
	}
	std::sort(keys.begin(), keys.end());
	return absl::StrCat(function_name, "|", absl::StrCat(keys.size()));
}

void BuildFunctionMap(GraphDef* graph_def,
											absl::flat_hash_map<std::string, FunctionDef*>* map) {
	map->clear();
	map->reserve(graph_def->library().function_size());
	for (FunctionDef& function : *graph_def->mutable_library()->mutable_function()) {
		(*map)[function.signature().name()] = &function;
	}
}

absl::Status RewriteFunctionAndDescendants(
		FunctionDef* function, const FrozenInputMap& frozen_inputs,
		absl::flat_hash_map<std::string, FunctionDef*>* function_map,
		absl::flat_hash_map<std::string, bool>* visited) {
	const std::string visited_key =
			BuildVisitedKey(function->signature().name(), frozen_inputs);
	if ((*visited)[visited_key]) {
		return absl::OkStatus();
	}
	(*visited)[visited_key] = true;

	const absl::flat_hash_map<std::string, const NodeDef*> function_node_map =
			BuildFunctionNodeMap(*function);

	for (NodeDef& node : *function->mutable_node_def()) {
		if (node.op() == "ReadVariableOp") {
			std::string input_name;
			if (GetFirstDataInput(node, &input_name)) {
				input_name = ResolveForwardedInputName(input_name, function_node_map);
				const auto frozen_it = frozen_inputs.find(input_name);
				if (frozen_it != frozen_inputs.end()) {
					LOG(INFO) << "[variable_freezing] rewrite function node="
								  << node.name() << " function="
								  << function->signature().name() << " source="
								  << input_name;
					ReplaceNodeWithConst(*frozen_it->second, &node);
				}
			}
			continue;
		}

		if (!IsCallNode(node)) continue;

		const std::string callee_name = GetCalledFunctionName(node);
		if (callee_name.empty()) continue;
		const auto callee_it = function_map->find(callee_name);
		if (callee_it == function_map->end()) continue;

		FrozenInputMap callee_inputs;
		const int arg_count = std::min(node.input_size(),
																	 callee_it->second->signature().input_arg_size());
		for (int i = 0; i < arg_count; ++i) {
			if (!node.input(i).empty() && node.input(i).front() == '^') continue;
			const std::string input_name = ResolveForwardedInputName(
					node.input(i), function_node_map);
			const auto frozen_it = frozen_inputs.find(input_name);
			if (frozen_it == frozen_inputs.end()) continue;
			callee_inputs[callee_it->second->signature().input_arg(i).name()] =
					frozen_it->second;
		}

		if (!callee_inputs.empty()) {
			TF_RETURN_IF_ERROR(RewriteFunctionAndDescendants(
					callee_it->second, callee_inputs, function_map, visited));
		}
	}

	return absl::OkStatus();
}

absl::Status RewriteCapturedFunctionReads(GraphDef* graph_def,
																					const FrozenValueMap& frozen_values) {
	TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(*graph_def));
	absl::flat_hash_map<std::string, FunctionDef*> function_map;
	BuildFunctionMap(graph_def, &function_map);

	absl::flat_hash_map<std::string, bool> visited;
	for (const NodeDef& node : graph_def->node()) {
		if (!IsCallNode(node)) continue;

		const std::string callee_name = GetCalledFunctionName(node);
		if (callee_name.empty()) continue;
		const auto callee_it = function_map.find(callee_name);
		if (callee_it == function_map.end()) continue;

		FrozenInputMap callee_inputs;
		const int arg_count = std::min(node.input_size(),
																	 callee_it->second->signature().input_arg_size());
		for (int i = 0; i < arg_count; ++i) {
			if (!node.input(i).empty() && node.input(i).front() == '^') continue;
			const std::string input_name = ResolveForwardedInputName(
					node.input(i), node_map);
			const auto frozen_it = frozen_values.find(input_name);
			if (frozen_it == frozen_values.end()) continue;
			callee_inputs[callee_it->second->signature().input_arg(i).name()] =
					&frozen_it->second;
		}

		if (!callee_inputs.empty()) {
			TF_RETURN_IF_ERROR(RewriteFunctionAndDescendants(
					callee_it->second, callee_inputs, &function_map, &visited));
		}
	}

	return absl::OkStatus();
}

}  // namespace

absl::Status MaybeFreezeAllowlistedVariableReads(
		const std::string& export_dir, MetaGraphDef* meta_graph_def) {
	if (!IsEnabled() || meta_graph_def == nullptr) {
		return absl::OkStatus();
	}
  LOG(INFO) << "[variable_freezing] graph before freeze:\n"
          << meta_graph_def->graph_def().DebugString();

	GraphDef* graph_def = meta_graph_def->mutable_graph_def();
	TF_ASSIGN_OR_RETURN(FrozenValueMap v1_frozen_values,
										LoadFrozenVariableV1Values(export_dir, *graph_def,
																	GetMaxFrozenTensorBytes()));
	TF_ASSIGN_OR_RETURN(FrozenValueMap frozen_values,
											LoadFrozenVarHandleValues(export_dir, *graph_def,
																								GetMaxFrozenTensorBytes()));
	for (auto& entry : v1_frozen_values) {
		frozen_values[entry.first] = std::move(entry.second);
	}
	LOG(INFO) << "[variable_freezing] export_dir=" << export_dir
					  << " matched_v1=" << v1_frozen_values.size()
					  << " matched_total=" << frozen_values.size();
	if (frozen_values.empty()) {
		return absl::OkStatus();
	}

	TF_RETURN_IF_ERROR(RewriteTopLevelReadNodes(graph_def, frozen_values));
	TF_RETURN_IF_ERROR(RewriteCapturedFunctionReads(graph_def, frozen_values));
  LOG(INFO) << "[variable_freezing] graph after freeze:\n"
          << meta_graph_def->graph_def().DebugString();
	LOG(INFO) << "[variable_freezing] finished rewrites for export_dir="
					  << export_dir;
	return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorflow
