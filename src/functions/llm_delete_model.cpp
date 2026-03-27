#include "llm_functions.hpp"
#include "model_storage.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace laduck {

struct LlmDeleteModelData : public FunctionData {
	ClientContext *context = nullptr;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<LlmDeleteModelData>();
		copy->context = context;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		return true;
	}
};

static unique_ptr<FunctionData> LlmDeleteModelBind(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<LlmDeleteModelData>();
	data->context = &context;
	return std::move(data);
}

static void LlmDeleteModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<LlmDeleteModelData>();
	auto *context = func_data.context;

	UnaryExecutor::Execute<string_t, string_t>(
	    args.data[0], result, args.size(), [&](string_t name_str) {
		    auto name = name_str.GetString();

		    try {
			    DeleteModelFromDb(*context, name);
		    } catch (std::exception &e) {
			    throw InvalidInputException(e.what());
		    }

		    return StringVector::AddString(result, name + " deleted from storage");
	    });
}

void RegisterLlmDeleteModelFunction(ExtensionLoader &loader) {
	auto fn = ScalarFunction("llm_delete_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR, LlmDeleteModelFunction,
	                          LlmDeleteModelBind);
	loader.RegisterFunction(fn);
}

} // namespace laduck
} // namespace duckdb
