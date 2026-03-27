#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/exception.hpp"

#include "llama.h"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {
namespace laduck {

struct ClassifyResult {
	std::string label;
	float score;
};

static ClassifyResult RunClassify(LoadedModel &entry, const std::string &text,
                                   const std::vector<std::string> &labels) {
	std::lock_guard<std::mutex> lock(entry.inference_mutex);

	auto *model = entry.model;
	auto *vocab = llama_model_get_vocab(model);

	// Tokenize prompt
	int n_max = text.size() + 32;
	std::vector<llama_token> tokens(n_max);
	int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_max, true, true);
	if (n_tokens < 0) {
		return {"", 0.0f};
	}
	tokens.resize(n_tokens);

	// Create context — only need one forward pass
	auto ctx_params = llama_context_default_params();
	ctx_params.n_ctx = static_cast<uint32_t>(n_tokens + 1);
	ctx_params.n_batch = static_cast<uint32_t>(n_tokens);

	auto *ctx = llama_init_from_model(model, ctx_params);
	if (!ctx) {
		return {"", 0.0f};
	}

	// Decode prompt
	llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
	if (llama_decode(ctx, batch) != 0) {
		llama_free(ctx);
		return {"", 0.0f};
	}

	// Get logits for the last token position (next-token prediction)
	const float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
	if (!logits) {
		llama_free(ctx);
		return {"", 0.0f};
	}

	// For each label, tokenize and get the logit of its first token
	std::vector<float> label_logits;
	label_logits.reserve(labels.size());

	for (auto &label : labels) {
		std::vector<llama_token> label_tokens(label.size() + 8);
		int n_label = llama_tokenize(vocab, label.c_str(), label.size(), label_tokens.data(),
		                              label_tokens.size(), false, false);
		if (n_label <= 0) {
			label_logits.push_back(-1e9f);
			continue;
		}
		label_logits.push_back(logits[label_tokens[0]]);
	}

	// Softmax over candidate logits to get probabilities
	float max_logit = *std::max_element(label_logits.begin(), label_logits.end());
	float sum_exp = 0.0f;
	for (auto l : label_logits) {
		sum_exp += std::exp(l - max_logit);
	}

	int best_idx = 0;
	float best_prob = 0.0f;
	for (size_t i = 0; i < label_logits.size(); i++) {
		float prob = std::exp(label_logits[i] - max_logit) / sum_exp;
		if (prob > best_prob) {
			best_prob = prob;
			best_idx = static_cast<int>(i);
		}
	}

	llama_free(ctx);

	return {labels[best_idx], best_prob};
}

// llm_classify(text, model, labels_list) → struct(label, score)
static void LlmClassifyFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();

	auto &text_vec = args.data[0];
	auto &model_vec = args.data[1];
	auto &labels_vec = args.data[2];

	// Result is a STRUCT {label VARCHAR, score FLOAT}
	auto &children = StructVector::GetEntries(result);
	auto &label_out = *children[0];
	auto &score_out = *children[1];
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto text_val = text_vec.GetValue(i);
		auto model_val = model_vec.GetValue(i);
		auto labels_val = labels_vec.GetValue(i);

		if (text_val.IsNull() || model_val.IsNull() || labels_val.IsNull()) {
			result_validity.SetInvalid(i);
			continue;
		}

		auto model_name = model_val.GetValue<string>();
		auto *entry = ModelRegistry::Instance().Get(model_name);
		if (!entry) {
			throw InvalidInputException("Model '" + model_name + "' is not loaded.");
		}

		// Extract label strings from the list
		auto &label_list = ListValue::GetChildren(labels_val);
		if (label_list.empty()) {
			throw InvalidInputException("Labels list cannot be empty.");
		}

		std::vector<std::string> labels;
		labels.reserve(label_list.size());
		for (auto &v : label_list) {
			labels.push_back(v.GetValue<string>());
		}

		auto text = text_val.GetValue<string>();
		auto res = RunClassify(*entry, text, labels);

		if (res.label.empty()) {
			result_validity.SetInvalid(i);
		} else {
			FlatVector::GetData<string_t>(label_out)[i] = StringVector::AddString(label_out, res.label);
			FlatVector::GetData<float>(score_out)[i] = res.score;
		}
	}
}

void RegisterLlmClassifyFunction(ExtensionLoader &loader) {
	// llm_classify(text VARCHAR, model VARCHAR, labels VARCHAR[]) → STRUCT(label VARCHAR, score FLOAT)
	auto return_type = LogicalType::STRUCT({
	    {"label", LogicalType::VARCHAR},
	    {"score", LogicalType::FLOAT}
	});

	loader.RegisterFunction(ScalarFunction("llm_classify",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)},
	    return_type, LlmClassifyFunction));
}

} // namespace laduck
} // namespace duckdb
