#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace laduck {

struct LlmModelsData : public TableFunctionData {
	std::vector<ModelInfo> models;
	mutable idx_t offset = 0;
};

static unique_ptr<FunctionData> LlmModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("name");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("path");

	return_types.push_back(LogicalType::BIGINT);
	names.push_back("parameters");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("quantization");

	return_types.push_back(LogicalType::INTEGER);
	names.push_back("context_size");

	return_types.push_back(LogicalType::INTEGER);
	names.push_back("gpu_layers");

	auto result = make_uniq<LlmModelsData>();
	result->models = ModelRegistry::Instance().List();
	return std::move(result);
}

static void LlmModelsExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<LlmModelsData>();

	idx_t count = 0;
	idx_t max_count = STANDARD_VECTOR_SIZE;

	while (bind_data.offset < bind_data.models.size() && count < max_count) {
		auto &m = bind_data.models[bind_data.offset];

		FlatVector::GetData<string_t>(output.data[0])[count] = StringVector::AddString(output.data[0], m.name);
		FlatVector::GetData<string_t>(output.data[1])[count] = StringVector::AddString(output.data[1], m.path);
		FlatVector::GetData<int64_t>(output.data[2])[count] = m.n_params;
		FlatVector::GetData<string_t>(output.data[3])[count] = StringVector::AddString(output.data[3], m.quantization);
		FlatVector::GetData<int32_t>(output.data[4])[count] = m.context_size;
		FlatVector::GetData<int32_t>(output.data[5])[count] = m.gpu_layers;

		count++;
		bind_data.offset++;
	}

	output.SetCardinality(count);
}

void RegisterLlmModelsFunction(ExtensionLoader &loader) {
	TableFunction llm_models("llm_models", {}, LlmModelsExecute, LlmModelsBind);
	loader.RegisterFunction(llm_models);
}

} // namespace laduck
} // namespace duckdb
