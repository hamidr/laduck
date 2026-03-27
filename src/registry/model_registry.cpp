#include "model_registry.hpp"

#include <stdexcept>

namespace duckdb {
namespace laduck {

LoadedModel::~LoadedModel() {
	if (ctx) {
		llama_free(ctx);
		ctx = nullptr;
	}
	if (model) {
		llama_model_free(model);
		model = nullptr;
	}
}

ModelRegistry &ModelRegistry::Instance() {
	static ModelRegistry instance;
	return instance;
}

void ModelRegistry::InitBackend() {
	std::lock_guard<std::mutex> lock(registry_mutex_);
	if (!backend_initialized_) {
		llama_backend_init();
		backend_initialized_ = true;
	}
}

void ModelRegistry::FreeBackend() {
	std::lock_guard<std::mutex> lock(registry_mutex_);
	models_.clear();
	if (backend_initialized_) {
		llama_backend_free();
		backend_initialized_ = false;
	}
}

void ModelRegistry::Load(const std::string &name, const std::string &path, int32_t n_gpu_layers, int32_t n_ctx) {
	std::lock_guard<std::mutex> lock(registry_mutex_);

	if (models_.count(name)) {
		throw std::runtime_error("Model '" + name + "' is already loaded. Unload it first.");
	}

	auto entry = std::make_unique<LoadedModel>();
	entry->name = name;
	entry->path = path;
	entry->gpu_layers = n_gpu_layers;
	entry->context_size = n_ctx;

	// Load model
	auto model_params = llama_model_default_params();
	model_params.n_gpu_layers = n_gpu_layers;

	entry->model = llama_model_load_from_file(path.c_str(), model_params);
	if (!entry->model) {
		throw std::runtime_error("Failed to load model from '" + path + "'. Is it a valid GGUF file?");
	}

	// Create context
	auto ctx_params = llama_context_default_params();
	ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
	ctx_params.n_batch = static_cast<uint32_t>(n_ctx);

	entry->ctx = llama_init_from_model(entry->model, ctx_params);
	if (!entry->ctx) {
		llama_model_free(entry->model);
		entry->model = nullptr;
		throw std::runtime_error("Failed to create inference context for model '" + name + "'.");
	}

	models_[name] = std::move(entry);
}

void ModelRegistry::Unload(const std::string &name) {
	std::lock_guard<std::mutex> lock(registry_mutex_);

	auto it = models_.find(name);
	if (it == models_.end()) {
		throw std::runtime_error("Model '" + name + "' is not loaded.");
	}

	// Lock the context mutex to wait for any in-flight inference
	{
		std::lock_guard<std::mutex> ctx_lock(it->second->ctx_mutex);
	}

	models_.erase(it);
}

LoadedModel *ModelRegistry::Get(const std::string &name) {
	std::lock_guard<std::mutex> lock(registry_mutex_);

	auto it = models_.find(name);
	if (it == models_.end()) {
		return nullptr;
	}
	return it->second.get();
}

std::vector<ModelInfo> ModelRegistry::List() {
	std::lock_guard<std::mutex> lock(registry_mutex_);

	std::vector<ModelInfo> result;
	result.reserve(models_.size());

	for (auto &pair : models_) {
		auto &entry = pair.second;
		ModelInfo info;
		info.name = entry->name;
		info.path = entry->path;
		info.context_size = entry->context_size;
		info.gpu_layers = entry->gpu_layers;

		if (entry->model) {
			info.n_params = static_cast<int64_t>(llama_model_n_params(entry->model));

			// Detect quantization from model description
			char buf[128] = {0};
			llama_model_desc(entry->model, buf, sizeof(buf));
			info.quantization = buf;
		}

		result.push_back(std::move(info));
	}

	return result;
}

} // namespace laduck
} // namespace duckdb
