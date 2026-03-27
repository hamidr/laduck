#include "model_storage.hpp"
#include "model_source.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/types/vector.hpp"

#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace duckdb {
namespace laduck {

static const char *STORAGE_TABLE = "__laduck_models";
static const size_t CHUNK_SIZE = 50 * 1024 * 1024; // 50MB per chunk

void StoreModelToDb(ClientContext &context, const std::string &name, const std::string &gguf_path) {
	struct stat st;
	if (stat(gguf_path.c_str(), &st) != 0) {
		throw std::runtime_error("GGUF file not found: '" + gguf_path + "'");
	}
	auto total_size = static_cast<int64_t>(st.st_size);

	// Use a separate connection for all DB writes to avoid deadlocking the caller's transaction
	Connection con(*context.db);

	con.Query("CREATE TABLE IF NOT EXISTS " + std::string(STORAGE_TABLE) +
	          " (name VARCHAR, chunk_idx INTEGER, chunk_data BLOB, total_chunks INTEGER, total_size BIGINT,"
	          " PRIMARY KEY (name, chunk_idx))");

	// Check if already stored
	auto check = con.Query("SELECT chunk_idx FROM " + std::string(STORAGE_TABLE) +
	                        " WHERE name = '" + name + "' LIMIT 1");
	if (check && !check->HasError()) {
		auto chunk = check->Fetch();
		if (chunk && chunk->size() > 0) {
			throw std::runtime_error("Model '" + name + "' is already stored. Delete it first with llm_delete_model('" + name + "').");
		}
	}

	// Read file and insert chunks
	std::ifstream in(gguf_path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("Failed to open GGUF file: '" + gguf_path + "'");
	}

	int32_t total_chunks = static_cast<int32_t>((total_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
	std::vector<char> buf(CHUNK_SIZE);

	con.Query("BEGIN TRANSACTION");

	for (int32_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
		in.read(buf.data(), CHUNK_SIZE);
		auto bytes_read = static_cast<idx_t>(in.gcount());

		// Use prepared statement to avoid SQL encoding overhead
		auto prepared = con.Prepare("INSERT INTO " + std::string(STORAGE_TABLE) +
		                             " (name, chunk_idx, chunk_data, total_chunks, total_size) VALUES ($1, $2, $3, $4, $5)");

		auto result = prepared->Execute(name, chunk_idx,
		                                 Value::BLOB(const_data_ptr_t(buf.data()), bytes_read),
		                                 total_chunks, total_size);
		if (result->HasError()) {
			con.Query("ROLLBACK");
			throw std::runtime_error("Failed to store chunk " + std::to_string(chunk_idx) + ": " + result->GetError());
		}
	}

	con.Query("COMMIT");
}

std::string LoadModelFromDb(ClientContext &context, const std::string &name) {
	auto cache_dir = GetCacheDir() + "/db";

	struct stat st;
	if (stat(cache_dir.c_str(), &st) != 0) {
		size_t pos = 0;
		while ((pos = cache_dir.find('/', pos + 1)) != std::string::npos) {
			mkdir(cache_dir.substr(0, pos).c_str(), 0755);
		}
		mkdir(cache_dir.c_str(), 0755);
	}

	auto tmp_path = cache_dir + "/" + name + ".gguf";

	// Use separate connection for reads too — caller may be in a scalar function
	Connection con(*context.db);

	auto result = con.Query("SELECT chunk_data FROM " + std::string(STORAGE_TABLE) +
	                         " WHERE name = '" + name + "' ORDER BY chunk_idx");
	if (result && result->HasError()) {
		throw std::runtime_error("Failed to load model from storage: " + result->GetError());
	}

	std::ofstream out(tmp_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("Failed to create temp file: " + tmp_path);
	}

	bool found_any = false;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		found_any = true;

		for (idx_t i = 0; i < chunk->size(); i++) {
			auto blob = FlatVector::GetData<string_t>(chunk->data[0])[i];
			out.write(blob.GetData(), blob.GetSize());
		}
	}
	out.close();

	if (!found_any) {
		remove(tmp_path.c_str());
		throw std::runtime_error("Model '" + name + "' not found in storage.");
	}

	return tmp_path;
}

void DeleteModelFromDb(ClientContext &context, const std::string &name) {
	Connection con(*context.db);

	con.Query("CREATE TABLE IF NOT EXISTS " + std::string(STORAGE_TABLE) +
	          " (name VARCHAR, chunk_idx INTEGER, chunk_data BLOB, total_chunks INTEGER, total_size BIGINT,"
	          " PRIMARY KEY (name, chunk_idx))");

	auto check = con.Query("SELECT chunk_idx FROM " + std::string(STORAGE_TABLE) +
	                         " WHERE name = '" + name + "' LIMIT 1");
	if (check && !check->HasError()) {
		auto chunk = check->Fetch();
		if (!chunk || chunk->size() == 0) {
			throw std::runtime_error("Model '" + name + "' not found in storage.");
		}
	}

	auto result = con.Query("DELETE FROM " + std::string(STORAGE_TABLE) + " WHERE name = '" + name + "'");
	if (result && result->HasError()) {
		throw std::runtime_error("Failed to delete model: " + result->GetError());
	}
}

} // namespace laduck
} // namespace duckdb
