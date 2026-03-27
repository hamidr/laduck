#include "llm_functions.hpp"
#include "model_registry.hpp"
#include "model_source.hpp"
#include "model_storage.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

struct LlmLoadModelData : public FunctionData {
	ClientContext *context = nullptr;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<LlmLoadModelData>();
		copy->context = context;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		return true;
	}
};

static unique_ptr<FunctionData> LlmLoadModelBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<LlmLoadModelData>();
	data->context = &context;
	return std::move(data);
}

static void LlmLoadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<LlmLoadModelData>();
	auto *context = func_data.context;

	auto &path_vec = args.data[0];
	auto &name_vec = args.data[1];
	auto count = args.size();

	auto paths = FlatVector::GetData<string_t>(path_vec);
	auto names = FlatVector::GetData<string_t>(name_vec);

	for (idx_t i = 0; i < count; i++) {
		auto raw_path = paths[i].GetString();
		auto name = names[i].GetString();

		int32_t n_gpu_layers = 99;
		int32_t n_ctx = 2048;

		if (args.ColumnCount() > 2) {
			n_gpu_layers = FlatVector::GetData<int32_t>(args.data[2])[i];
		}
		if (args.ColumnCount() > 3) {
			n_ctx = FlatVector::GetData<int32_t>(args.data[3])[i];
		}

		try {
			auto source = ResolveModelSource(raw_path);
			std::string gguf_path;

			switch (source.type) {
			case ModelSourceType::LOCAL_FILE:
			case ModelSourceType::HUGGINGFACE:
				gguf_path = source.resolved_path;
				break;
			case ModelSourceType::DB_STORAGE:
				gguf_path = LoadModelFromDb(*context, source.db_model_name);
				break;
			}

			ModelRegistry::Instance().Load(name, gguf_path, n_gpu_layers, n_ctx);

			auto models = ModelRegistry::Instance().List();
			std::string desc;
			for (auto &m : models) {
				if (m.name == name) {
					desc = name + " loaded (" + std::to_string(m.n_params) + " params, " + m.quantization + ")";
					break;
				}
			}
			if (desc.empty()) {
				desc = name + " loaded";
			}
			FlatVector::GetData<string_t>(result)[i] = StringVector::AddString(result, desc);
		} catch (std::exception &e) {
			throw InvalidInputException(e.what());
		}
	}
}

void RegisterLlmLoadModelFunction(ExtensionLoader &loader) {
	ScalarFunctionSet load_set("llm_load_model");

	auto fn2 = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, LlmLoadModelFunction,
	                           LlmLoadModelBind);
	load_set.AddFunction(fn2);

	auto fn4 = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
	                           LogicalType::VARCHAR, LlmLoadModelFunction, LlmLoadModelBind);
	load_set.AddFunction(fn4);

	loader.RegisterFunction(load_set);
}

} // namespace laduck
} // namespace duckdb
