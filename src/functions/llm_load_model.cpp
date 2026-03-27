#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

static void LlmLoadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &path_vec = args.data[0];
	auto &name_vec = args.data[1];
	auto count = args.size();

	auto paths = FlatVector::GetData<string_t>(path_vec);
	auto names = FlatVector::GetData<string_t>(name_vec);

	for (idx_t i = 0; i < count; i++) {
		auto path = paths[i].GetString();
		auto name = names[i].GetString();

		int32_t n_gpu_layers = 99;
		int32_t n_ctx = 2048;

		// Optional parameters via additional args
		if (args.ColumnCount() > 2) {
			n_gpu_layers = FlatVector::GetData<int32_t>(args.data[2])[i];
		}
		if (args.ColumnCount() > 3) {
			n_ctx = FlatVector::GetData<int32_t>(args.data[3])[i];
		}

		try {
			ModelRegistry::Instance().Load(name, path, n_gpu_layers, n_ctx);

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
	// llm_load_model(path VARCHAR, model_name VARCHAR) → VARCHAR
	ScalarFunctionSet load_set("llm_load_model");

	load_set.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::VARCHAR, LlmLoadModelFunction));

	// llm_load_model(path, model_name, n_gpu_layers, n_ctx) → VARCHAR
	load_set.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
	    LogicalType::VARCHAR, LlmLoadModelFunction));

	loader.RegisterFunction(load_set);
}

} // namespace laduck
} // namespace duckdb
