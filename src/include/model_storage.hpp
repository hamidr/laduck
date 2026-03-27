#pragma once

#include "duckdb/main/client_context.hpp"

#include <string>

namespace duckdb {
namespace laduck {

void StoreModelToDb(ClientContext &context, const std::string &name, const std::string &gguf_path);

std::string LoadModelFromDb(ClientContext &context, const std::string &name);

void DeleteModelFromDb(ClientContext &context, const std::string &name);

} // namespace laduck
} // namespace duckdb
