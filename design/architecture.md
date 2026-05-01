# WebLLM — Architecture

## Overview

WebLLM runs large language models (LLMs) entirely inside the browser, using WebGPU as the compute backend. There is no server-side inference; weights, tokenizers, and compiled model libraries are fetched from a CDN (or a custom URL) and cached in the browser. The public API surface is a strict subset of the OpenAI REST API, meaning most client code written against OpenAI's SDK can be used with WebLLM by changing only the import.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser Tab / Page                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      Application Code                        │   │
│  │   engine.chat.completions.create(...)                        │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                           │ OpenAI-compatible API                    │
│  ┌────────────────────────▼─────────────────────────────────────┐   │
│  │                 WebLLM Public API Layer                       │   │
│  │   MLCEngine / WebWorkerMLCEngine / ServiceWorkerMLCEngine     │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                           │                                          │
│         ┌─────────────────▼──────────────────────┐                  │
│         │         MLCEngine  (core)               │                  │
│         │  ┌──────────────┐  ┌────────────────┐  │                  │
│         │  │LLMChatPipeline│  │EmbeddingPipeline│  │                  │
│         │  └──────┬───────┘  └──────┬─────────┘  │                  │
│         └─────────┼────────────────┼─────────────┘                  │
│                   │  TVM/WebGPU    │                                  │
│         ┌─────────▼────────────────▼─────────────┐                  │
│         │       @mlc-ai/web-runtime (tvmjs)        │                  │
│         │         WebGPU Compute Backend           │                  │
│         └────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Topologies

WebLLM supports three deployment modes. All three share the same `MLCEngine` core.

### 1. Main-Thread (Direct)

The engine and pipelines run directly on the page's main thread. Suitable for simple demos and scripts that do not need to keep UI responsive during inference.

```
Page JS ──────────────────────────────► MLCEngine
                                         (main thread)
```

### 2. Web Worker

Inference is offloaded to a dedicated Web Worker. The page holds a `WebWorkerMLCEngine` proxy that serialises calls over `postMessage`. The worker hosts `WebWorkerMLCEngineHandler` which owns the actual `MLCEngine`.

```
Page JS
  │  WebWorkerMLCEngine (proxy)
  │  postMessage / onmessage (WorkerRequest / WorkerResponse)
  ▼
Web Worker
  │  WebWorkerMLCEngineHandler
  └► MLCEngine (real)
```

### 3. Service Worker / Chrome Extension Service Worker

A Service Worker (or an Extension Service Worker) survives page reloads and can serve multiple tabs. Tabs communicate via `postMessage` through the Service Worker's `client` registry. `ServiceWorkerMLCEngineHandler` extends `WebWorkerMLCEngineHandler` with keepAlive / heartbeat and a per-request client routing map.

```
Tab A ──┐
Tab B ──┼──► ServiceWorkerMLCEngineHandler (SW)
Tab C ──┘         └► MLCEngine (real, persists across reloads)
```

---

## Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 5: Public API                                          │
│  src/index.ts — re-exports everything; tree-shakeable         │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: OpenAI Protocol Adapters                            │
│  src/openai_api_protocols/                                    │
│    chat_completion.ts  │  completion.ts  │  embedding.ts      │
│  Validates, normalises, overloads for stream vs non-stream    │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Engine & Worker Façades                             │
│  src/engine.ts            — MLCEngine (orchestration)         │
│  src/web_worker.ts        — WebWorkerMLCEngine{Handler}       │
│  src/service_worker.ts    — ServiceWorkerMLCEngine{Handler}   │
│  src/extension_service_worker.ts                              │
│  src/message.ts           — typed message protocol            │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Inference Pipelines                                 │
│  src/llm_chat.ts     — LLMChatPipeline (autoregressive LLM)  │
│  src/embedding.ts    — EmbeddingPipeline (encoder models)     │
│  src/conversation.ts — Conversation history + prompt building │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Runtime, Config & Utilities                         │
│  src/config.ts       — ChatConfig, AppConfig, GenerationConfig│
│  src/cache_util.ts   — artifact fetch + browser cache mgmt   │
│  src/integrity.ts    — SRI (sha256/384/512) verification      │
│  src/support.ts      — CustomLock, top-p helpers, tool calls  │
│  src/error.ts        — typed error hierarchy                  │
│  src/utils.ts        — array helpers, chat-opt comparisons    │
├──────────────────────────────────────────────────────────────┤
│  Layer 0: External Native Libraries                           │
│  @mlc-ai/web-runtime   (tvmjs — TVM WebGPU runtime)          │
│  @mlc-ai/web-tokenizers (Rust/WASM tokenizer)                │
│  @mlc-ai/web-xgrammar  (grammar-constrained sampling)        │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **WebGPU for inference** | Near-native GPU throughput in the browser without plugins |
| **TVM compiled model libraries (.wasm + weight shards)** | Cross-platform, hardware-adaptive kernel generation via MLC-LLM toolchain |
| **OpenAI-compatible API surface** | Drop-in replacement; no bespoke client SDK needed |
| **Worker-based isolation** | Keeps UI thread responsive; Service Worker enables persistence across page reloads |
| **Per-model `CustomLock`** | Serialises concurrent requests to the same model without blocking other models |
| **Three-scope browser cache** | `webllm/model` (weights), `webllm/config` (JSON configs), `webllm/wasm` (compiled libraries) mapped to either IndexedDB or HTTP cache depending on origin |
| **Streaming via `AsyncGenerator`** | Composable, cancellable token stream with no internal buffering |

---

## Data Flow: Chat Completion (Streaming)

```
App
 │  engine.chat.completions.create({ stream: true, ... })
 ▼
Chat.Completions.create()          [openai_api_protocols/chat_completion.ts]
 │  postInitAndCheckFields()
 ▼
MLCEngine.chatCompletion()         [engine.ts]
 │  acquire CustomLock(modelId)
 │  compareConversationObject() → reuse KV cache if history matches
 ▼
LLMChatPipeline.prefillStep()     [llm_chat.ts]
 │  tokenise → embed → forward (WebGPU prefill kernel)
 │  build KV cache
 ▼
LLMChatPipeline.decodeStep() × N  [llm_chat.ts]
 │  decode kernel → apply penalties → softmax → top-p sample
 │  grammar mask via web-xgrammar (if response_format set)
 │  yield ChatCompletionChunk
 ▼
App receives AsyncIterable<ChatCompletionChunk>
 │
 └  release CustomLock on return / throw / abort
```

---

## Artifact Loading & Caching

```
MLCEngine.reload(modelId)
  │
  ├─ findModelRecord(modelId, appConfig)
  │    └─ prebuiltAppConfig or user-supplied AppConfig
  │
  ├─ Fetch mlc-chat-config.json  → cache "webllm/config"
  │    └─ verifyIntegrity(SRI)   [integrity.ts]
  │
  ├─ Fetch tokenizer files       → cache "webllm/model"
  │    └─ maybeVerifyTokenizerIntegrity()
  │
  ├─ Fetch .wasm model library   → cache "webllm/wasm"
  │    └─ localhost: no cache (always re-fetch)
  │    └─ same-server: HTTP cache only
  │    └─ remote CDN: IndexedDB
  │
  └─ Fetch weight shards         → cache "webllm/model"
       └─ tvmjs.ArtifactCache
```

---

## Multi-Model Support

`MLCEngine` maintains four parallel `Map<modelId, …>` structures:

| Map | Type | Purpose |
|-----|------|---------|
| `loadedModelIdToPipeline` | `LLMChatPipeline \| EmbeddingPipeline` | Active inference pipeline |
| `loadedModelIdToChatConfig` | `ChatConfig` | Per-model runtime config |
| `loadedModelIdToModelType` | `ModelType` | `"LLM"` or `"embedding"` |
| `loadedModelIdToLock` | `CustomLock` | Per-model request serialiser |

A `reload()` call **unloads all** existing models before loading the new set. Models cannot be hot-swapped without a full reload.
