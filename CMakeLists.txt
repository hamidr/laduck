cmake_minimum_required(VERSION 3.12)

set(TARGET_NAME laduck)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# ---------------------------------------------------------------------------
# llama.cpp — build as a subdirectory, static linking
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Extension sources
# ---------------------------------------------------------------------------
set(EXTENSION_SOURCES
    src/laduck_extension.cpp
    src/registry/model_registry.cpp
    src/registry/model_source.cpp
    src/registry/model_storage.cpp
    src/functions/llm_load_model.cpp
    src/functions/llm_complete.cpp
    src/functions/llm_models.cpp
    src/functions/llm_embed.cpp
    src/functions/llm_unload_model.cpp
    src/functions/llm_store_model.cpp
    src/functions/llm_delete_model.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
target_include_directories(${EXTENSION_NAME} PRIVATE src/include)
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE src/include)

# llama.cpp headers
target_include_directories(${EXTENSION_NAME} PRIVATE third_party/llama.cpp/include third_party/llama.cpp/ggml/include)
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE third_party/llama.cpp/include third_party/llama.cpp/ggml/include)

# ---------------------------------------------------------------------------
# Link llama.cpp
# ---------------------------------------------------------------------------
target_link_libraries(${EXTENSION_NAME} llama common ggml)
target_link_libraries(${LOADABLE_EXTENSION_NAME} llama common ggml)
