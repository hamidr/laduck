#include "model_storage.hpp"
#include "model_source.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/vector.hpp"

#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

namespace duckdb {
namespace laduck {

static const char *STORAGE_TABLE = "__laduck_models";

void EnsureStorageTable(ClientContext &context) {
	auto query = "CREATE TABLE IF NOT EXISTS " + std::string(STORAGE_TABLE) +
	             " (name VARCHAR PRIMARY KEY, gguf_data BLOB, size_bytes BIGINT, stored_at TIMESTAMP DEFAULT current_timestamp)";
	context.Query(query, false);
}

void StoreModelToDb(ClientContext &context, const std::string &name, const std::string &gguf_path) {
	EnsureStorageTable(context);

	// Check file exists
	struct stat st;
	if (stat(gguf_path.c_str(), &st) != 0) {
		throw std::runtime_error("GGUF file not found: '" + gguf_path + "'");
	}

	// Check if model already stored
	auto check = context.Query("SELECT name FROM " + std::string(STORAGE_TABLE) + " WHERE name = '" + name + "'", false);
	if (check && check->HasError()) {
		throw std::runtime_error("Failed to check storage: " + check->GetError());
	}
	if (check) {
		auto chunk = check->Fetch();
		if (chunk && chunk->size() > 0) {
			throw std::runtime_error("Model '" + name + "' is already stored. Delete it first with llm_delete_model('" + name + "').");
		}
	}

	// Use read_blob to efficiently load the file
	auto insert = "INSERT INTO " + std::string(STORAGE_TABLE) +
	              " (name, gguf_data, size_bytes) VALUES ('" + name + "', read_blob('" + gguf_path + "'), " +
	              std::to_string(st.st_size) + ")";
	auto result = context.Query(insert, false);
	if (result && result->HasError()) {
		throw std::runtime_error("Failed to store model: " + result->GetError());
	}
}

std::string LoadModelFromDb(ClientContext &context, const std::string &name) {
	EnsureStorageTable(context);

	// Extract BLOB to a temp file in cache dir
	auto cache_dir = GetCacheDir() + "/db";

	// Ensure directory exists
	struct stat st;
	if (stat(cache_dir.c_str(), &st) != 0) {
		size_t pos = 0;
		while ((pos = cache_dir.find('/', pos + 1)) != std::string::npos) {
			mkdir(cache_dir.substr(0, pos).c_str(), 0755);
		}
		mkdir(cache_dir.c_str(), 0755);
	}

	auto tmp_path = cache_dir + "/" + name + ".gguf";

	auto query = "SELECT gguf_data FROM " + std::string(STORAGE_TABLE) + " WHERE name = '" + name + "'";
	auto result = context.Query(query, false);
	if (result && result->HasError()) {
		throw std::runtime_error("Failed to load model from storage: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw std::runtime_error("Model '" + name + "' not found in storage.");
	}

	// Get the blob data
	auto blob = FlatVector::GetData<string_t>(chunk->data[0])[0];

	// Write to file
	std::ofstream out(tmp_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("Failed to create temp file: " + tmp_path);
	}
	out.write(blob.GetData(), blob.GetSize());
	out.close();

	return tmp_path;
}

void DeleteModelFromDb(ClientContext &context, const std::string &name) {
	EnsureStorageTable(context);

	// Check if model exists first
	auto check = context.Query("SELECT name FROM " + std::string(STORAGE_TABLE) + " WHERE name = '" + name + "'", false);
	if (check) {
		auto chunk = check->Fetch();
		if (!chunk || chunk->size() == 0) {
			throw std::runtime_error("Model '" + name + "' not found in storage.");
		}
	}

	auto query = "DELETE FROM " + std::string(STORAGE_TABLE) + " WHERE name = '" + name + "'";
	auto result = context.Query(query, false);
	if (result && result->HasError()) {
		throw std::runtime_error("Failed to delete model: " + result->GetError());
	}
}

} // namespace laduck
} // namespace duckdb
