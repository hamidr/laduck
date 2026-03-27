#include "llm_functions.hpp"
#include "model_registry.hpp"
#include "model_storage.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

struct LlmStoreModelData : public TableFunctionData {
	std::string name;
	std::string gguf_path;
	mutable bool done = false;
};

static unique_ptr<FunctionData> LlmStoreModelBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("result");

	auto result = make_uniq<LlmStoreModelData>();
	result->name = input.inputs[0].GetValue<string>();

	auto *entry = ModelRegistry::Instance().Get(result->name);
	if (!entry) {
		throw InvalidInputException("Model '" + result->name + "' is not loaded. Load it first before storing.");
	}
	result->gguf_path = entry->path;

	return std::move(result);
}

static void LlmStoreModelExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<LlmStoreModelData>();
	if (bind_data.done) {
		output.SetCardinality(0);
		return;
	}

	StoreModelToDb(context, bind_data.name, bind_data.gguf_path);

	auto msg = bind_data.name + " stored in database";
	FlatVector::GetData<string_t>(output.data[0])[0] = StringVector::AddString(output.data[0], msg);
	output.SetCardinality(1);
	bind_data.done = true;
}

void RegisterLlmStoreModelFunction(ExtensionLoader &loader) {
	TableFunction func("llm_store_model", {LogicalType::VARCHAR}, LlmStoreModelExecute, LlmStoreModelBind);
	loader.RegisterFunction(func);
}

} // namespace laduck
} // namespace duckdb
