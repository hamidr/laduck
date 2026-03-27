#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace laduck {

void RegisterLlmLoadModelFunction(ExtensionLoader &loader);
void RegisterLlmCompleteFunction(ExtensionLoader &loader);
void RegisterLlmModelsFunction(ExtensionLoader &loader);
void RegisterLlmUnloadModelFunction(ExtensionLoader &loader);
void RegisterLlmEmbedFunction(ExtensionLoader &loader);
void RegisterLlmClassifyFunction(ExtensionLoader &loader);
void RegisterLlmBackendsFunction(ExtensionLoader &loader);
void RegisterLlmStoreModelFunction(ExtensionLoader &loader);
void RegisterLlmDeleteModelFunction(ExtensionLoader &loader);

} // namespace laduck
} // namespace duckdb
