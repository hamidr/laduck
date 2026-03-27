#pragma once

#include <string>

namespace duckdb {
namespace laduck {

enum class ModelSourceType {
	LOCAL_FILE,
	HUGGINGFACE,
	DB_STORAGE
};

struct ModelSource {
	ModelSourceType type;
	std::string resolved_path;
	std::string db_model_name;
};

ModelSource ResolveModelSource(const std::string &path_or_uri);

std::string DownloadFromHuggingFace(const std::string &hf_path);

std::string GetCacheDir();

} // namespace laduck
} // namespace duckdb
