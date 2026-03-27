#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace duckdb {
namespace laduck {

struct ModelInfo {
	std::string name;
	std::string path;
	int64_t n_params;
	std::string quantization;
	int32_t context_size;
	int32_t gpu_layers;
};

struct LoadedModel {
	std::string name;
	std::string path;
	llama_model *model = nullptr;
	llama_context *ctx = nullptr;
	std::mutex ctx_mutex;
	int32_t context_size = 2048;
	int32_t gpu_layers = 99;

	~LoadedModel();
	LoadedModel() = default;
	LoadedModel(const LoadedModel &) = delete;
	LoadedModel &operator=(const LoadedModel &) = delete;
};

class ModelRegistry {
public:
	static ModelRegistry &Instance();

	void Load(const std::string &name, const std::string &path, int32_t n_gpu_layers, int32_t n_ctx);
	void Unload(const std::string &name);
	LoadedModel *Get(const std::string &name);
	std::vector<ModelInfo> List();

	void InitBackend();
	void FreeBackend();

private:
	ModelRegistry() = default;
	std::unordered_map<std::string, std::unique_ptr<LoadedModel>> models_;
	std::mutex registry_mutex_;
	bool backend_initialized_ = false;
};

} // namespace laduck
} // namespace duckdb
