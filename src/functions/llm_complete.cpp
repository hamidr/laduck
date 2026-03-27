#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/exception.hpp"

#include "llama.h"
#include "common.h"

#include <string>
#include <vector>

namespace duckdb {
namespace laduck {

static std::string RunInference(LoadedModel &entry, const std::string &prompt, int32_t max_tokens, float temperature,
                                float top_p) {
	std::lock_guard<std::mutex> lock(entry.ctx_mutex);

	auto *model = entry.model;
	auto *ctx = entry.ctx;
	auto *vocab = llama_model_get_vocab(model);

	// Tokenize prompt
	int n_prompt_max = prompt.size() + 32;
	std::vector<llama_token> tokens(n_prompt_max);
	int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_prompt_max, true, true);
	if (n_tokens < 0) {
		return "";
	}
	tokens.resize(n_tokens);

	// Check context fits
	int n_ctx = llama_n_ctx(ctx);
	if (n_tokens + max_tokens > n_ctx) {
		max_tokens = n_ctx - n_tokens;
		if (max_tokens <= 0) {
			return "";
		}
	}

	// Clear KV cache for fresh inference
	llama_kv_cache_clear(ctx);

	// Create batch for prompt
	llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

	if (llama_decode(ctx, batch) != 0) {
		return "";
	}

	// Set up sampler
	auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
	if (temperature <= 0.0f) {
		llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
	} else {
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
		llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
		llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
	}

	// Generate tokens
	std::string output;
	for (int i = 0; i < max_tokens; i++) {
		llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

		if (llama_vocab_is_eog(vocab, new_token)) {
			break;
		}

		char buf[256];
		int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
		if (n > 0) {
			output.append(buf, n);
		}

		// Prepare next batch with the new token
		batch = llama_batch_get_one(&new_token, 1);
		if (llama_decode(ctx, batch) != 0) {
			break;
		}
	}

	llama_sampler_free(smpl);

	return output;
}

// llm_complete(prompt, model_name) with default params
static void LlmCompleteFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(),
	    [&](string_t prompt, string_t model_name, ValidityMask &mask, idx_t idx) {
		    auto *entry = ModelRegistry::Instance().Get(model_name.GetString());
		    if (!entry) {
			    throw InvalidInputException("Model '" + model_name.GetString() + "' is not loaded.");
		    }

		    int32_t max_tokens = 256;
		    float temperature = 0.7f;
		    float top_p = 0.9f;

		    // Read optional params if present
		    if (args.ColumnCount() > 2) {
			    max_tokens = FlatVector::GetData<int32_t>(args.data[2])[idx];
		    }
		    if (args.ColumnCount() > 3) {
			    auto temp_data = FlatVector::GetData<float>(args.data[3]);
			    temperature = temp_data[idx];
		    }
		    if (args.ColumnCount() > 4) {
			    auto top_p_data = FlatVector::GetData<float>(args.data[4]);
			    top_p = top_p_data[idx];
		    }

		    auto text = RunInference(*entry, prompt.GetString(), max_tokens, temperature, top_p);
		    if (text.empty()) {
			    mask.SetInvalid(idx);
			    return string_t();
		    }
		    return StringVector::AddString(result, text);
	    });
}

void RegisterLlmCompleteFunction(ExtensionLoader &loader) {
	ScalarFunctionSet complete_set("llm_complete");

	// llm_complete(prompt VARCHAR, model VARCHAR) → VARCHAR
	complete_set.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::VARCHAR, LlmCompleteFunction));

	// llm_complete(prompt, model, max_tokens, temperature, top_p) → VARCHAR
	complete_set.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::FLOAT, LogicalType::FLOAT},
	    LogicalType::VARCHAR, LlmCompleteFunction));

	loader.RegisterFunction(complete_set);
}

} // namespace laduck
} // namespace duckdb
