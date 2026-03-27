#include "llm_functions.hpp"
#include "model_storage.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

struct LlmDeleteModelData : public TableFunctionData {
	std::string name;
	mutable bool done = false;
};

static unique_ptr<FunctionData> LlmDeleteModelBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("result");

	auto result = make_uniq<LlmDeleteModelData>();
	result->name = input.inputs[0].GetValue<string>();
	return std::move(result);
}

static void LlmDeleteModelExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<LlmDeleteModelData>();
	if (bind_data.done) {
		output.SetCardinality(0);
		return;
	}

	DeleteModelFromDb(context, bind_data.name);

	auto msg = bind_data.name + " deleted from storage";
	FlatVector::GetData<string_t>(output.data[0])[0] = StringVector::AddString(output.data[0], msg);
	output.SetCardinality(1);
	bind_data.done = true;
}

void RegisterLlmDeleteModelFunction(ExtensionLoader &loader) {
	TableFunction func("llm_delete_model", {LogicalType::VARCHAR}, LlmDeleteModelExecute, LlmDeleteModelBind);
	loader.RegisterFunction(func);
}

} // namespace laduck
} // namespace duckdb
