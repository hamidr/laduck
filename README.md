# LaDuck

Run LLMs inside DuckDB. Load GGUF models, generate text, compute embeddings, and classify data — all from SQL, fully local, no API keys.

LaDuck embeds [llama.cpp](https://github.com/ggml-org/llama.cpp) directly into DuckDB as an extension. Models run in-process with GPU acceleration on Apple Silicon, NVIDIA, and AMD hardware.

## Quick Start

```sql
LOAD 'laduck.duckdb_extension';

-- Download and load a model from HuggingFace
SELECT llm_load_model('hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q2_k.gguf', 'qwen');

-- Generate text
SELECT llm_complete('Explain SQL joins in one sentence:', 'qwen');

-- Classify rows without training
SELECT product_name,
       llm_classify(description, 'qwen', ['electronics', 'clothing', 'food']) AS category
FROM products;

-- Semantic search with embeddings
SELECT title, list_cosine_similarity(
    llm_embed(content, 'qwen'),
    llm_embed('machine learning tutorial', 'qwen')
) AS relevance
FROM articles
ORDER BY relevance DESC
LIMIT 10;
```

## Functions

### Inference

| Function | Description | Returns |
|----------|-------------|---------|
| `llm_complete(text, model [, max_tokens, temperature, top_p])` | Generate text | `VARCHAR` |
| `llm_embed(text, model)` | Compute embedding vector | `FLOAT[]` |
| `llm_classify(text, model, labels)` | Zero-shot classification (single forward pass) | `STRUCT(label VARCHAR, score FLOAT)` |

### Model Management

| Function | Description | Returns |
|----------|-------------|---------|
| `llm_load_model(source, name [, n_gpu_layers, n_ctx])` | Load a GGUF model | `VARCHAR` |
| `llm_unload_model(name)` | Free model from memory | `VARCHAR` |
| `llm_models()` | List loaded models | Table |
| `llm_backends()` | List available compute backends (CPU, Metal, CUDA, ...) | Table |

### Persistent Storage

| Function | Description | Returns |
|----------|-------------|---------|
| `llm_store_model(name)` | Save loaded model into the DuckDB database | Table |
| `llm_delete_model(name)` | Remove stored model from database | Table |

Store and delete are table functions — use `SELECT * FROM llm_store_model('name')`.

## Model Sources

LaDuck supports three ways to load models:

```sql
-- Local file
SELECT llm_load_model('/path/to/model.gguf', 'mymodel');

-- HuggingFace (downloads and caches to ~/.cache/laduck/)
SELECT llm_load_model('hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q2_k.gguf', 'qwen');

-- From DuckDB storage (after llm_store_model)
SELECT llm_load_model('db://qwen', 'qwen');
```

## Portable Databases

Store models directly in the `.duckdb` file for fully self-contained, portable databases:

```sql
-- Load and store
SELECT llm_load_model('hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q2_k.gguf', 'qwen');
SELECT * FROM llm_store_model('qwen');

-- Later, on another machine — just load from the database
SELECT llm_load_model('db://qwen', 'qwen');
SELECT llm_complete('Hello!', 'qwen');
```

Models are stored as chunked BLOBs (50MB chunks). A 322MB model stores in ~0.5 seconds.

## GPU Backends

LaDuck uses the best available backend automatically. Use `llm_backends()` to see what's available:

```sql
SELECT * FROM llm_backends();
-- ┌─────────┬──────────────┐
-- │  name   │ description  │
-- ├─────────┼──────────────┤
-- │ MTL0    │ Apple M4 Max │
-- │ BLAS    │ Accelerate   │
-- │ CPU     │ Apple M4 Max │
-- └─────────┴──────────────┘
```

Force CPU-only inference by setting `n_gpu_layers` to 0:

```sql
SELECT llm_load_model('model.gguf', 'cpu_model', 0, 2048);
```

## Building

### Prerequisites

- [Nix](https://nixos.org/) with flakes enabled

### Default (Metal on macOS, CPU on Linux)

```bash
nix develop
GEN=ninja make release
```

### NVIDIA (CUDA)

```bash
nix build .#laduck-cuda
```

### AMD (Vulkan — cross-platform)

```bash
nix build .#laduck-vulkan
```

### AMD (ROCm — Linux, better performance)

```bash
nix build .#laduck-rocm
```

### Testing

```bash
nix develop
GEN=ninja make release
build/release/test/unittest --test-dir test
```

### Loading the Extension

```bash
duckdb -cmd "LOAD 'build/release/extension/laduck/laduck.duckdb_extension'"
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  DuckDB                      │
│                                              │
│  SQL Query ──► Scalar/Table Functions        │
│                    │                         │
│         ┌──────────┴──────────┐              │
│         │   LaDuck Extension  │              │
│         │                     │              │
│         │  ┌───────────────┐  │              │
│         │  │ Model Registry│  │              │
│         │  │ (name→model)  │  │              │
│         │  └──────┬────────┘  │              │
│         │         │           │              │
│         │  ┌──────┴────────┐  │              │
│         │  │  llama.cpp    │  │              │
│         │  │  (embedded)   │  │              │
│         │  └───────────────┘  │              │
│         └─────────────────────┘              │
└─────────────────────────────────────────────┘
         │                    │
    ┌────┴────┐    ┌─────────┴────────┐
    │  GGUF   │    │  DuckDB Tables   │
    │  Files  │    │  (BLOB storage)  │
    │ (disk)  │    └──────────────────┘
    └─────────┘
```

## License

MIT — see [LICENSE](LICENSE).
