#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/exception.hpp"

#include "llama.h"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {
namespace laduck {

static void NormalizeL2(std::vector<float> &vec) {
	float sum = 0.0f;
	for (auto v : vec) {
		sum += v * v;
	}
	float norm = std::sqrt(sum);
	if (norm > 0.0f) {
		for (auto &v : vec) {
			v /= norm;
		}
	}
}

static std::vector<float> RunEmbedding(LoadedModel &entry, const std::string &text) {
	std::lock_guard<std::mutex> lock(entry.inference_mutex);

	auto *model = entry.model;
	auto *vocab = llama_model_get_vocab(model);
	int n_embd = llama_model_n_embd(model);

	// Tokenize
	int n_max = text.size() + 32;
	std::vector<llama_token> tokens(n_max);
	int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_max, true, true);
	if (n_tokens < 0) {
		return {};
	}
	tokens.resize(n_tokens);

	// Create context with embeddings enabled
	auto ctx_params = llama_context_default_params();
	ctx_params.n_ctx = static_cast<uint32_t>(n_tokens);
	ctx_params.n_batch = static_cast<uint32_t>(n_tokens);
	ctx_params.embeddings = true;

	auto *ctx = llama_init_from_model(model, ctx_params);
	if (!ctx) {
		return {};
	}

	// Clear memory
	auto *mem = llama_get_memory(ctx);
	if (mem) {
		llama_memory_clear(mem, true);
	}

	// Build batch with all tokens marked for output
	llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

	if (llama_decode(ctx, batch) != 0) {
		llama_free(ctx);
		return {};
	}

	// Get pooled embeddings (sequence 0)
	const float *embd = nullptr;
	auto pooling = llama_pooling_type(ctx);

	if (pooling == LLAMA_POOLING_TYPE_NONE) {
		// No pooling — get last token's embedding
		embd = llama_get_embeddings_ith(ctx, n_tokens - 1);
	} else {
		// Pooled (mean, cls, etc.) — get sequence embedding
		embd = llama_get_embeddings_seq(ctx, 0);
	}

	std::vector<float> result;
	if (embd) {
		result.assign(embd, embd + n_embd);
		NormalizeL2(result);
	}

	llama_free(ctx);
	return result;
}

static void LlmEmbedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();

	auto &text_vec = args.data[0];
	auto &model_vec = args.data[1];

	auto &result_validity = FlatVector::Validity(result);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &child_vec = ListVector::GetEntry(result);

	idx_t total_floats = 0;

	// First pass: compute embeddings and count total floats
	std::vector<std::vector<float>> embeddings(count);

	UnifiedVectorFormat text_data, model_data;
	text_vec.ToUnifiedFormat(count, text_data);
	model_vec.ToUnifiedFormat(count, model_data);
	auto texts = UnifiedVectorFormat::GetData<string_t>(text_data);
	auto models = UnifiedVectorFormat::GetData<string_t>(model_data);

	for (idx_t i = 0; i < count; i++) {
		auto text_idx = text_data.sel->get_index(i);
		auto model_idx = model_data.sel->get_index(i);

		if (!text_data.validity.RowIsValid(text_idx) || !model_data.validity.RowIsValid(model_idx)) {
			result_validity.SetInvalid(i);
			list_entries[i] = {0, 0};
			continue;
		}

		auto model_name = models[model_idx].GetString();
		auto *entry = ModelRegistry::Instance().Get(model_name);
		if (!entry) {
			throw InvalidInputException("Model '" + model_name + "' is not loaded.");
		}

		auto embd = RunEmbedding(*entry, texts[text_idx].GetString());
		if (embd.empty()) {
			result_validity.SetInvalid(i);
			list_entries[i] = {0, 0};
		} else {
			list_entries[i] = {total_floats, static_cast<idx_t>(embd.size())};
			total_floats += embd.size();
			embeddings[i] = std::move(embd);
		}
	}

	ListVector::SetListSize(result, total_floats);
	ListVector::Reserve(result, total_floats);

	// Second pass: populate child vector
	auto child_data = FlatVector::GetData<float>(child_vec);
	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		if (!embeddings[i].empty()) {
			memcpy(child_data + offset, embeddings[i].data(), embeddings[i].size() * sizeof(float));
			offset += embeddings[i].size();
		}
	}
}

void RegisterLlmEmbedFunction(ExtensionLoader &loader) {
	// llm_embed(text VARCHAR, model VARCHAR) → FLOAT[]
	loader.RegisterFunction(ScalarFunction("llm_embed",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR},
	    LogicalType::LIST(LogicalType::FLOAT),
	    LlmEmbedFunction));
}

} // namespace laduck
} // namespace duckdb
