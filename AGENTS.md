# WebLLM — Agent Instructions

WebLLM runs LLMs entirely in the browser via WebGPU. It exposes an **OpenAI-compatible API** and ships three runtime modes: plain Web Worker, Service Worker, and Chrome Extension Service Worker.

## Commands

```bash
npm install          # install deps
npm run build        # rollup → lib/ + cleanup-index-js.sh
npm run lint         # eslint + prettier check
npm run format       # auto-fix formatting
npm test             # jest --coverage (with coverage thresholds)
npx jest --coverage=false tests/<file>.test.ts  # single test file, faster
```

Pre-commit hooks (Husky + lint-staged) enforce lint/format automatically.

## Architecture

| Layer | Key files | Responsibility |
|-------|-----------|----------------|
| **API** | `src/openai_api_protocols/` | OpenAI-compatible protocol adapters |
| **Engine** | `src/engine.ts` | Model lifecycle, request routing, per-model `CustomLock` |
| **Pipeline** | `src/llm_chat.ts`, `src/embedding.ts` | Inference, KV cache, token sampling via `@mlc-ai/web-runtime` (tvmjs/WebGPU) |
| **Config** | `src/config.ts`, `src/types.ts` | Schemas + validation (`postInitAndCheckGenerationConfigValues()`) |
| **Workers** | `src/web_worker.ts`, `src/service_worker.ts`, `src/extension_service_worker.ts` | Message dispatch, state sync |
| **Cache** | `src/cache_util.ts` | Artifact fetching strategy (IndexedDB / HTTP cache / no-cache for localhost wasm) |
| **Conversation** | `src/conversation.ts` | Message history, role templates |
| **Errors** | `src/error.ts` | Custom error hierarchy (`ConfigValueError` → specific subclasses) |

Public API surface is defined in `src/index.ts`.

## Key Conventions

### Naming
- Private helpers: `_methodName()` prefix
- Errors: `<Concept>Error` (e.g., `DeviceLostError`, `ConfigValueError`)
- Types: suffix with `Config`, `Params`, `Request`, `Response`, `Pipeline`

### Async / Streaming
- Streaming uses `AsyncGenerator<T>`; each request gets a unique ID + generator entry
- `CustomLock` (in `src/support.ts`) serialises requests per model — never bypass it
- Always release the lock on error (existing generators use try/finally)

### Config merging order
```ts
{ ...baseConfig, ...modelRecord.overrides, ...chatOpts }
```
Validate with `postInitAndCheckGenerationConfigValues()` after merging.

### Multi-model state
Engine holds parallel maps keyed by `modelId`:
`loadedModelIdToPipeline`, `loadedModelIdToChatConfig`, `loadedModelIdToLock`, `loadedModelIdToModelType`

### OpenAI API divergence
- **Model is NOT in the request**. Call `engine.reload(modelId)` before API calls.
- Extra options go in `extra_body` (e.g., `latencyBreakdown`, `enable_thinking`).
- Overloads distinguish streaming vs. non-streaming requests.

## Testing Conventions

- Mock entire modules with `jest.mock()` (e.g., `llm_chat`, `embedding`, `@mlc-ai/web-xgrammar`)
- Access private engine internals via `(engine as any)` for test setup
- Mock pipelines expose counters (`decodeLimit`, `prefillCallCount`) to verify call patterns
- See `tests/engine_integration.test.ts` for the canonical integration test pattern

## Pitfalls

- **Localhost wasm is never cached**; same-server wasm uses HTTP cache only; remote wasm uses IndexedDB.
- **`deviceLostIsError` flag** — don't treat intentional unloads as errors; check this flag before throwing on `device.lost`.
- **Streaming cleanup** — incomplete streams hold the per-model lock; always consume the generator or handle abort.
- **`ModelType` is set at load time** — you cannot swap an LLM pipeline for an embedding pipeline without reloading.
- **Emoji/partial character handling in streaming** — decoder may emit U+FFFD; strip trailing replacement chars before yielding.
- **Seed is auto-reset** to `Date.now()` after each streaming request.
- **KV cache reuse** — `compareConversationObject()` enables prefill reuse for unchanged history; avoid mutating history in place.

## Documentation

- User docs: `docs/user/`
- Developer docs: `docs/developer/`
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Examples (each has its own `package.json`): `examples/`
