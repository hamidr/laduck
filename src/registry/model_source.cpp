#include "model_source.hpp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <array>

namespace duckdb {
namespace laduck {

std::string GetCacheDir() {
	std::string cache_dir;

	const char *xdg = std::getenv("XDG_CACHE_HOME");
	if (xdg && xdg[0] != '\0') {
		cache_dir = std::string(xdg) + "/laduck";
	} else {
		const char *home = std::getenv("HOME");
		if (!home) {
			home = "/tmp";
		}
		cache_dir = std::string(home) + "/.cache/laduck";
	}

	return cache_dir;
}

static void EnsureDirectory(const std::string &path) {
	struct stat st;
	if (stat(path.c_str(), &st) == 0) {
		return;
	}

	// Create parent directories recursively
	size_t pos = 0;
	while ((pos = path.find('/', pos + 1)) != std::string::npos) {
		auto sub = path.substr(0, pos);
		mkdir(sub.c_str(), 0755);
	}
	mkdir(path.c_str(), 0755);
}

ModelSource ResolveModelSource(const std::string &path_or_uri) {
	ModelSource source;

	if (path_or_uri.rfind("hf://", 0) == 0) {
		source.type = ModelSourceType::HUGGINGFACE;
		auto hf_path = path_or_uri.substr(5); // strip "hf://"
		source.resolved_path = DownloadFromHuggingFace(hf_path);
	} else if (path_or_uri.rfind("db://", 0) == 0) {
		source.type = ModelSourceType::DB_STORAGE;
		source.db_model_name = path_or_uri.substr(5); // strip "db://"
	} else {
		source.type = ModelSourceType::LOCAL_FILE;
		source.resolved_path = path_or_uri;
	}

	return source;
}

std::string DownloadFromHuggingFace(const std::string &hf_path) {
	// hf_path format: "owner/repo/filename.gguf"
	// URL: https://huggingface.co/owner/repo/resolve/main/filename.gguf

	// Split into owner/repo and filename
	// Find the second slash to split repo from filename
	auto first_slash = hf_path.find('/');
	if (first_slash == std::string::npos) {
		throw std::runtime_error("Invalid HuggingFace path: '" + hf_path + "'. Expected format: owner/repo/filename.gguf");
	}
	auto second_slash = hf_path.find('/', first_slash + 1);
	if (second_slash == std::string::npos) {
		throw std::runtime_error("Invalid HuggingFace path: '" + hf_path + "'. Expected format: owner/repo/filename.gguf");
	}

	auto owner_repo = hf_path.substr(0, second_slash);
	auto filename = hf_path.substr(second_slash + 1);

	auto url = "https://huggingface.co/" + owner_repo + "/resolve/main/" + filename;

	// Cache path: ~/.cache/laduck/hf/owner/repo/filename.gguf
	auto cache_dir = GetCacheDir() + "/hf/" + owner_repo;
	auto cache_path = cache_dir + "/" + filename;

	// Check if already cached
	struct stat st;
	if (stat(cache_path.c_str(), &st) == 0 && st.st_size > 0) {
		return cache_path;
	}

	// Download with curl
	EnsureDirectory(cache_dir);

	auto tmp_path = cache_path + ".part";
	auto cmd = "curl -L --fail --progress-bar -o '" + tmp_path + "' '" + url + "' 2>&1";

	std::string output;
	std::array<char, 4096> buf;
	FILE *pipe = popen(cmd.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("Failed to execute curl for model download.");
	}
	while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
		output += buf.data();
	}
	int exit_code = pclose(pipe);

	if (exit_code != 0) {
		remove(tmp_path.c_str());
		throw std::runtime_error("Failed to download model from '" + url + "'. curl output: " + output);
	}

	// Rename .part to final path (atomic on same filesystem)
	if (rename(tmp_path.c_str(), cache_path.c_str()) != 0) {
		remove(tmp_path.c_str());
		throw std::runtime_error("Failed to finalize downloaded model file.");
	}

	return cache_path;
}

} // namespace laduck
} // namespace duckdb
