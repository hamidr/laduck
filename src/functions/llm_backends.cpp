#include "llm_functions.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "ggml-backend.h"

namespace duckdb {
namespace laduck {

struct LlmBackendsData : public TableFunctionData {
	std::vector<std::pair<std::string, std::string>> backends;
	mutable idx_t offset = 0;
};

static unique_ptr<FunctionData> LlmBackendsBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("name");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("description");

	auto result = make_uniq<LlmBackendsData>();

	// Enumerate available ggml backends
	auto n = ggml_backend_dev_count();
	for (size_t i = 0; i < n; i++) {
		auto *dev = ggml_backend_dev_get(i);
		auto *name = ggml_backend_dev_name(dev);
		auto *desc = ggml_backend_dev_description(dev);
		result->backends.emplace_back(name ? name : "unknown", desc ? desc : "");
	}

	return std::move(result);
}

static void LlmBackendsExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<LlmBackendsData>();

	idx_t count = 0;
	while (bind_data.offset < bind_data.backends.size() && count < STANDARD_VECTOR_SIZE) {
		auto &b = bind_data.backends[bind_data.offset];
		FlatVector::GetData<string_t>(output.data[0])[count] = StringVector::AddString(output.data[0], b.first);
		FlatVector::GetData<string_t>(output.data[1])[count] = StringVector::AddString(output.data[1], b.second);
		count++;
		bind_data.offset++;
	}
	output.SetCardinality(count);
}

void RegisterLlmBackendsFunction(ExtensionLoader &loader) {
	TableFunction func("llm_backends", {}, LlmBackendsExecute, LlmBackendsBind);
	loader.RegisterFunction(func);
}

} // namespace laduck
} // namespace duckdb
