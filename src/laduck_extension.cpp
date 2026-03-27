#define DUCKDB_EXTENSION_MAIN

#include "laduck_extension.hpp"
#include "llm_functions.hpp"
#include "model_registry.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	laduck::ModelRegistry::Instance().InitBackend();

	laduck::RegisterLlmLoadModelFunction(loader);
	laduck::RegisterLlmCompleteFunction(loader);
	laduck::RegisterLlmModelsFunction(loader);
	laduck::RegisterLlmUnloadModelFunction(loader);
	laduck::RegisterLlmEmbedFunction(loader);
	laduck::RegisterLlmStoreModelFunction(loader);
	laduck::RegisterLlmDeleteModelFunction(loader);
}

void LaduckExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string LaduckExtension::Name() {
	return "laduck";
}

std::string LaduckExtension::Version() const {
#ifdef EXT_VERSION_LADUCK
	return EXT_VERSION_LADUCK;
#else
	return "0.1.0-dev";
#endif
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(laduck, loader) {
	duckdb::LoadInternal(loader);
}
}
