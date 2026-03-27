#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

static void LlmUnloadModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(
	    args.data[0], result, args.size(), [&](string_t name_str) {
		    auto name = name_str.GetString();
		    try {
			    ModelRegistry::Instance().Unload(name);
		    } catch (std::exception &e) {
			    throw InvalidInputException(e.what());
		    }
		    return StringVector::AddString(result, name + " unloaded");
	    });
}

void RegisterLlmUnloadModelFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(
	    ScalarFunction("llm_unload_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR, LlmUnloadModelFunction));
}

} // namespace laduck
} // namespace duckdb
