# WebLLM — High-Level Design

## Purpose

This document describes the major subsystems of WebLLM, the responsibilities each carries, and the contracts between them. Refer to [architecture.md](architecture.md) for the overall layered picture and deployment topologies.

---

## Subsystem Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        WebLLM Library                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 A. Public API Surface                    │   │
│  │  src/index.ts  (barrel exports)                         │   │
│  └───────────────────┬─────────────────────────────────────┘   │
│                      │                                          │
│  ┌───────────────────▼────────────────────────────────────┐    │
│  │         B. OpenAI Protocol Adapters                     │    │
│  │  chat_completion.ts │ completion.ts │ embedding.ts      │    │
│  └───────────────────┬────────────────────────────────────┘    │
│                      │                                          │
│  ┌───────────────────▼────────────────────────────────────┐    │
│  │                 C. Engine                               │    │
│  │  MLCEngine — model lifecycle, request routing, locking  │    │
│  └──────┬──────────────────────────────────────┬──────────┘    │
│         │                                      │               │
│  ┌──────▼──────────┐                  ┌────────▼──────────┐   │
│  │  D. LLM Pipeline│                  │ E. Embedding Pipeline│  │
│  │  LLMChatPipeline│                  │ EmbeddingPipeline  │   │
│  └──────┬──────────┘                  └────────┬──────────┘   │
│         │                                      │               │
│  ┌──────▼──────────────────────────────────────▼──────────┐   │
│  │            F. Runtime & Infrastructure                  │   │
│  │  Config │ Cache │ Conversation │ Integrity │ Support    │   │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              G. Worker Communication Layer               │  │
│  │  WebWorkerMLCEngine{Handler} │ ServiceWorkerMLCEngine{H} │  │
│  │  message.ts (WorkerRequest / WorkerResponse protocol)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## A. Public API Surface

**File:** `src/index.ts`

**Responsibility:** Barrel re-export of all symbols that library consumers should use. Nothing is implemented here — it is purely a surface-definition file that drives tree-shaking in bundlers.

**Key exports:**

| Export | Source | Purpose |
|--------|--------|---------|
| `MLCEngine` | `engine.ts` | Main-thread engine |
| `CreateMLCEngine` | `engine.ts` | Factory helper |
| `WebWorkerMLCEngine` / `Handler` | `web_worker.ts` | Worker proxy + handler |
| `ServiceWorkerMLCEngine` / `Handler` | `service_worker.ts` | SW proxy + handler |
| `prebuiltAppConfig` | `config.ts` | Default model catalogue |
| `hasModelInCache`, `deleteModel*` | `cache_util.ts` | Cache management |
| `verifyIntegrity` | `integrity.ts` | SRI verification utility |
| Error classes | `error.ts` | Typed errors for catch blocks |

**Contracts:** All exported types and classes must remain stable between minor versions.

---

## B. OpenAI Protocol Adapters

**Files:** `src/openai_api_protocols/chat_completion.ts`, `completion.ts`, `embedding.ts`

**Responsibility:** Translate the OpenAI REST API schema into calls on `MLCEngineInterface`. Validation, overload resolution (streaming vs. non-streaming), and request normalisation happen exclusively here.

### Subsystem internals

```
Chat (class)
  └─ Completions (class)          chat_completion.ts
       └─ create(request)
            ├─ postInitAndCheckFields()   ← validate & normalise
            └─ engine.chatCompletion()

Completions (class)               completion.ts
  └─ create(request)
       ├─ postInitAndCheckFields()
       └─ engine.completion()

Embeddings (class)                embedding.ts
  └─ create(request)
       └─ engine.embeddings()
```

**Key behaviours:**
- Streaming vs. non-streaming is resolved at the TypeScript overload level; callers get the correct return type at compile time.
- `postInitAndCheckFields()` applies defaults, checks field compatibility (e.g. `stream_options` only valid with `stream: true`), and rejects unsupported OpenAI fields.
- `model` in the request body is **not used** to select a pipeline — the model must be pre-loaded with `engine.reload()`.

**WebLLM extensions via `extra_body`:**

| Field | Purpose |
|-------|---------|
| `latencyBreakdown` | Include per-phase latency in the response |
| `enable_thinking` | Enable Qwen3-style chain-of-thought thinking blocks |

---

## C. Engine

**File:** `src/engine.ts`

**Responsibility:** Owns the full model lifecycle (load, unload, reload), routes API calls to the correct pipeline, serialises concurrent requests per model via `CustomLock`, and reports initialisation progress.

### State

```
MLCEngine {
  // API namespace objects (thin wrappers delegating back to engine)
  chat:        API.Chat
  completions: API.Completions
  embeddings:  API.Embeddings

  // Per-model state maps
  loadedModelIdToPipeline:   Map<string, LLMChatPipeline | EmbeddingPipeline>
  loadedModelIdToChatConfig: Map<string, ChatConfig>
  loadedModelIdToModelType:  Map<string, ModelType>
  loadedModelIdToLock:       Map<string, CustomLock>

  // Engine-level state
  appConfig:            AppConfig
  initProgressCallback: InitProgressCallback | undefined
  logitProcessorRegistry: Map<string, LogitProcessor> | undefined
  interruptSignal:      boolean
  deviceLostIsError:    boolean
  reloadController:     AbortController | undefined
}
```

### Model Lifecycle

```
reload(modelId, chatOpts?)
  │
  ├─ 1. Unload all existing models (_unloadAllModels)
  │       └─ pipeline.dispose() + clear all four maps
  │
  ├─ 2. For each modelId:
  │       a. findModelRecord() → ModelRecord
  │       b. Load mlc-chat-config.json  (cache_util)
  │       c. Load tokenizer files       (cache_util)
  │       d. tvmjs.detectGPUDevice()    → verify WebGPU availability
  │       e. tvmjs.instantiate()        → load .wasm + weights
  │       f. Instantiate LLMChatPipeline or EmbeddingPipeline
  │       g. Register in all four maps
  │
  └─ 3. Emit InitProgressReport at each stage via initProgressCallback
```

### Request Routing

```
chatCompletion(request)
  │
  ├─ getModelIdToUse(request, loadedModelIds)   ← resolve which model
  ├─ validatePipelineType(modelId) → must be LLM
  ├─ lock = loadedModelIdToLock.get(modelId)
  ├─ await lock.acquire()
  ├─  try {
  │      compareConversationObject() → prefill reuse decision
  │      pipeline.prefillStep()
  │      loop pipeline.decodeStep() → yield chunks
  │   } finally {
  │      lock.release()
  │   }
```

---

## D. LLM Chat Pipeline

**File:** `src/llm_chat.ts`

**Responsibility:** Runs a single autoregressive LLM on the WebGPU device via the TVM virtual machine. Manages the KV cache, token sampling, streaming output, and function call extraction.

### Internal phases

| Phase | Description |
|-------|-------------|
| **Prefill** | Tokenise prompt, run prefill kernel, populate KV cache |
| **Decode** | Autoregressive decode loop: one token per step |
| **Sampling** | Apply repetition/frequency/presence penalties → softmax with temperature → top-p nucleus sample |
| **Grammar constraint** | `web-xgrammar` applies bitmask to logits before sampling when `response_format` is JSON/grammar |
| **Stop detection** | Match stop strings or stop token IDs; set `finishReason` |
| **Tool call extraction** | Parse structured tool call JSON from output message |

### KV Cache Management

- Default: **Paged KV Cache** (transformer attention, supports sliding window)
- Alternative: **RNN State** (for hybrid recurrent models)
- Cache is reused across turns if `compareConversationObject()` confirms history is unchanged (enabling prompt prefix caching).

---

## E. Embedding Pipeline

**File:** `src/embedding.ts`

**Responsibility:** Runs an encoder-only (or encoder-style) model to produce dense vector embeddings. Supports batched inputs up to `maxBatchSize`. Does not support sliding window configurations.

### Differences from LLM pipeline

| Aspect | LLMChatPipeline | EmbeddingPipeline |
|--------|-----------------|-------------------|
| Direction | Autoregressive (decode loop) | Single forward pass |
| Output | Token stream | Float32 embedding vector |
| KV cache | Yes (paged) | No |
| Batching | Single (decode is sequential) | Yes (`maxBatchSize`) |
| Stop handling | Yes | Not applicable |

---

## F. Runtime & Infrastructure

### F.1 Config (`src/config.ts`)

Defines all configuration schemas:

| Schema | Purpose |
|--------|---------|
| `ModelRecord` | Single model entry in the app catalogue (URL, model_id, overrides) |
| `AppConfig` | Full model catalogue + cache backend selection |
| `ChatConfig` | Per-model runtime config loaded from `mlc-chat-config.json` |
| `GenerationConfig` | Sampling parameters (temperature, top_p, max_tokens, etc.) |
| `MLCEngineConfig` | Engine-level config (logLevel, appConfig, callbacks) |

Config merging order at request time:
```
{ ...baseConfig, ...modelRecord.overrides, ...chatOpts }
```
followed by `postInitAndCheckGenerationConfigValues()`.

### F.2 Cache Utilities (`src/cache_util.ts`)

Three cache scopes, each mapped to a `tvmjs.ArtifactCache`:

| Scope | Contents | Backend rules |
|-------|----------|---------------|
| `webllm/model` | Weight shards, tokenizer files | IndexedDB (remote CDN), HTTP cache (same-server), no cache (localhost) |
| `webllm/config` | `mlc-chat-config.json` | Same as model |
| `webllm/wasm` | Compiled `.wasm` model libraries | HTTP cache (same-server / remote), no cache (localhost) |

Public helpers: `hasModelInCache`, `deleteModelInCache`, `deleteModelAllInfoInCache`, `deleteChatConfigInCache`, `deleteModelWasmInCache`.

### F.3 Conversation (`src/conversation.ts`)

`Conversation` tracks the message history as a typed tuple array:
```
messages: Array<[Role, role_name_str, content | undefined]>
```
- Builds the final prompt string (or token ID array) from the `ConvTemplateConfig` template.
- Supports system messages, user/assistant/tool turns, and function call strings.
- `compareConversationObject()` enables KV-cache reuse by comparing two conversation states structurally.

### F.4 Integrity (`src/integrity.ts`)

- Implements **Subresource Integrity (SRI)** verification (sha256, sha384, sha512).
- Hashes downloaded `ArrayBuffer` data using the Web Crypto API and compares against the declared SRI string.
- Behaviour on mismatch is controlled per-model via `ModelIntegrity.onFailure`: `"error"` (default) or `"warn"`.

### F.5 Support Utilities (`src/support.ts`)

| Utility | Purpose |
|---------|---------|
| `CustomLock` | Async mutex; serialises requests per model |
| `findModelRecord()` | Looks up a model by ID in AppConfig |
| `getModelIdToUse()` | Selects the active model when multiple are loaded |
| `getTopProbs()` | Top-k probability selection (CPU-side, for logprobs) |
| `getTokenTableFromTokenizer()` | Builds token string table for grammar masking |
| `getToolCallFromOutputMessage()` | Parses function call JSON from model output |
| `getChunkedPrefillInputData()` | Splits long prompts into prefill chunks |

---

## G. Worker Communication Layer

**Files:** `src/web_worker.ts`, `src/service_worker.ts`, `src/extension_service_worker.ts`, `src/message.ts`

**Responsibility:** Provide a transparent proxy so that application code on the main thread interacts with `MLCEngine` running in a worker, with no change to the API surface.

### Message Protocol (`message.ts`)

All communication uses two typed message envelopes:

```ts
WorkerRequest  { kind: RequestKind;  uuid: string; content: ParamsType }
WorkerResponse { kind: ResponseKind; uuid: string; content: any }
```

`ResponseKind` values:
- `"return"` — successful non-streaming result
- `"throw"` — serialised error
- `"initProgressCallback"` — progress update during model loading

`RequestKind` covers all `MLCEngineInterface` methods plus internal messages (`keepAlive`, `heartbeat`, `setLogLevel`, etc.).

### Handler Pattern

```
WebWorkerMLCEngineHandler (base)
  │  owns: MLCEngine (real)
  │  owns: loadedModelIdToAsyncGenerator (per-model streaming state)
  │
  ├─ onmessage(event) → dispatch by msg.kind
  │     "reload"                   → engine.reload()
  │     "chatCompletionNonStreaming"→ engine.chatCompletion() → postMessage(return)
  │     "chatCompletionStreamInit" → create async generator, stash in map
  │     "completionStreamNextChunk"→ generator.next() → postMessage(chunk or return)
  │     "throw" path              → postMessage(throw)
  │
  └─ postMessage(msg) → self.postMessage / port.postMessage / client.postMessage

ServiceWorkerMLCEngineHandler (extends above)
  │  adds: clientRegistry Map<uuid, Client | MessagePort>
  │  adds: keepAlive / heartbeat handling
  └─ postMessage routes to the specific client that made the request
```

### Proxy Pattern

`WebWorkerMLCEngine` (and its Service Worker variant) implements `MLCEngineInterface` by serialising every method call into a `WorkerRequest` with a unique UUID and returning a `Promise` that resolves when the matching `WorkerResponse` arrives.

Streaming responses are bridged using an async generator that the proxy exposes, internally driven by repeated `completionStreamNextChunk` messages.

---

## Error Subsystem

**File:** `src/error.ts`

Typed error hierarchy (excerpt):

```
Error
├─ ConfigValueError
│   ├─ MinValueError
│   ├─ RangeError
│   ├─ NonNegativeError
│   ├─ DependencyError
│   └─ InvalidNumberStringError
├─ WebGPUNotAvailableError
├─ DeviceLostError
├─ ModelNotFoundError
├─ ModelNotLoadedError
├─ IntegrityError
├─ ContextWindowSizeExceededError
├─ EmbeddingUnsupportedModelError
└─ ... (30+ typed errors)
```

All errors have a distinct `name` property matching the class name, enabling reliable `instanceof` or `error.name` checks across the worker boundary (where prototype chains are lost after serialisation).
