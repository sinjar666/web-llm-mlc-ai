# WebLLM — Low-Level Design

## Purpose

This document details the internal implementation of every major component: data structures, algorithms, concurrency contracts, state machines, and cross-cutting concerns. Refer to [high-level-design.md](high-level-design.md) for component responsibilities and [architecture.md](architecture.md) for the overall system picture.

---

## 1. Configuration System (`src/config.ts`)

### 1.1 Schema Hierarchy

```
AppConfig
  model_list: ModelRecord[]
  useIndexedDBCache?: boolean           // legacy alias
  cacheType?: "indexedDB" | "cache"     // preferred

ModelRecord
  model_id: string                      // unique key used everywhere
  model: string                         // URL (HuggingFace or custom)
  model_lib: string                     // URL to .wasm
  model_type?: ModelType                // "LLM" (default) | "embedding"
  overrides?: ChatOptions               // merged into ChatConfig at load time
  required_features?: string[]          // e.g. ["shader-f16"]
  integrity?: ModelIntegrity            // SRI hashes

ChatConfig                              // loaded from mlc-chat-config.json
  tokenizer_files: string[]
  vocab_size: number
  conv_template: ConvTemplateConfig
  context_window_size: number
  sliding_window_size: number           // -1 if disabled
  attention_sink_size: number
  max_history_size?: number             // RNN/hybrid models
  repetition_penalty: number
  frequency_penalty: number
  presence_penalty: number
  top_p: number
  temperature: number
  bos_token_id?: number

GenerationConfig                        // per-request, merged at request time
  max_tokens?: number | null
  temperature?: number
  top_p?: number
  frequency_penalty?: number
  presence_penalty?: number
  repetition_penalty?: number
  logprobs?: boolean
  top_logprobs?: number
  logit_bias?: Record<string, number>
  seed?: number | null
  stop?: string | string[]
  response_format?: ResponseFormat
  ...
```

### 1.2 Config Merging

At model load time (inside `reloadInternal`):
```ts
const mergedConfig: ChatConfig = {
  ...JSON.parse(configJson),      // mlc-chat-config.json base
  ...modelRecord.overrides,       // ModelRecord-level overrides
  ...chatOpts,                    // Per-reload ChatOptions
};
```

At request time inside the pipeline:
```ts
const genConfig: GenerationConfig = {
  ...chatConfig,                  // base from merged ChatConfig
  ...requestGenerationConfig,     // from the API request body
};
postInitAndCheckGenerationConfigValues(genConfig);
```

`postInitAndCheckGenerationConfigValues` validates:
- `temperature` ∈ (0, ∞) — throws `MinValueError`
- `top_p` ∈ (0, 1] — throws `RangeError`
- `frequency_penalty`, `presence_penalty` ∈ [-2, 2] — throws `RangeError`
- `repetition_penalty` > 0 — throws `MinValueError`
- `max_tokens` > 0 if set — throws `MinValueError`
- `top_logprobs` ∈ [0, 5] if `logprobs` is true — throws `RangeError`
- `logit_bias` values ∈ [-100, 100] — throws `RangeError`
- `seed` must be integer-representable — throws `SeedTypeError`

---

## 2. Engine (`src/engine.ts`)

### 2.1 MLCEngine Class Internals

```ts
class MLCEngine implements MLCEngineInterface {
  // ── API namespace delegates ──────────────────────────────────────
  chat:        API.Chat        // → this.chatCompletion()
  completions: API.Completions // → this.completion()
  embeddings:  API.Embeddings  // → this.embeddings.create()

  // ── Per-model state maps ─────────────────────────────────────────
  private loadedModelIdToPipeline:   Map<string, LLMChatPipeline | EmbeddingPipeline>
  private loadedModelIdToChatConfig: Map<string, ChatConfig>
  private loadedModelIdToModelType:  Map<string, ModelType>
  private loadedModelIdToLock:       Map<string, CustomLock>

  // ── Engine-level state ────────────────────────────────────────────
  private appConfig:              AppConfig
  private logger:                 (msg: string) => void    // loglevel.info
  private logitProcessorRegistry: Map<string, LogitProcessor> | undefined
  private initProgressCallback:   InitProgressCallback | undefined
  private interruptSignal:        boolean        // set by interruptGenerate()
  private deviceLostIsError:      boolean        // false during intentional unload
  private reloadController:       AbortController | undefined
}
```

### 2.2 `reload()` State Machine

```
[IDLE]
  │  reload(modelId[], chatOpts[])
  ▼
[VALIDATE]
  │  check: modelId is unique set
  │  check: chatOpts.length matches modelId.length
  │  throw ReloadArgumentSizeUnmatchedError | ReloadModelIdNotUniqueError
  ▼
[UNLOAD ALL]
  │  deviceLostIsError = false
  │  for each loaded model: pipeline.dispose(), clear all four maps
  │  deviceLostIsError = true
  ▼
[CREATE AbortController]   ← reloadController
  ▼
[FOR EACH modelId]  (sequential, not parallel)
  │  reloadInternal(modelId[i], chatOpts[i])
  │    ├─ findModelRecord(modelId, appConfig)
  │    ├─ fetch mlc-chat-config.json   (configCache)
  │    ├─ verifyIntegrity(config SRI)
  │    ├─ merge ChatConfig
  │    ├─ fetch .wasm                  (wasmCache, with origin-based rules)
  │    ├─ verifyIntegrity(wasm SRI)
  │    ├─ tvmjs.instantiate(wasm)
  │    ├─ tvmjs.detectGPUDevice()
  │    ├─ check required_features (shader-f16, etc.)
  │    ├─ register device.lost handler
  │    ├─ tvm.initWebGPU(device)
  │    ├─ asyncLoadTokenizer()
  │    ├─ tvm.fetchTensorCache()      (weight shards → webllm/model)
  │    ├─ new LLMChatPipeline | EmbeddingPipeline
  │    └─ register in all four maps
  ▼
[IDLE]  (reloadController = undefined)

On AbortError: log warn, return (no throw)
On any other error: propagate up
```

### 2.3 `chatCompletion()` Flow

```
chatCompletion(request: ChatCompletionRequest)
  │
  ├─ postInitAndCheckFieldsChatCompletion(request)
  │    validates fields, sets defaults
  │
  ├─ selectedModelId = getModelIdToUse(request.extra_body?.modelId, loadedIds)
  │    if only one model loaded → use it
  │    if multiple → must be explicitly specified
  │
  ├─ modelType = loadedModelIdToModelType.get(selectedModelId)
  │    if ModelType.embedding → throw IncorrectPipelineLoadedError
  │
  ├─ pipeline = loadedModelIdToPipeline.get(selectedModelId) as LLMChatPipeline
  ├─ chatConfig = loadedModelIdToChatConfig.get(selectedModelId)
  ├─ lock = loadedModelIdToLock.get(selectedModelId)
  │
  ├─ (streaming path)
  │    await lock.acquire()
  │    try {
  │      pipeline.resetKVCache() or reuse (compareConversationObject)
  │      await pipeline.prefillStep(newMessages)
  │      while (!pipeline.stopped()) {
  │        await pipeline.decodeStep()
  │        yield buildChatCompletionChunk(pipeline.getOutputMessage())
  │      }
  │      yield final chunk with finish_reason + usage
  │    } finally { lock.release() }
  │
  └─ (non-streaming path)
       await lock.acquire()
       try { ... same logic, collect all chunks, return ChatCompletion }
       finally { lock.release() }
```

### 2.4 `CustomLock` (`src/support.ts`)

```ts
class CustomLock {
  private _waitQueue: Array<() => void> = [];
  private _locked = false;

  async acquire(): Promise<void> {
    if (!this._locked) {
      this._locked = true;
      return;
    }
    // queue a resolver; wait until previous holder calls release()
    return new Promise<void>((resolve) => {
      this._waitQueue.push(resolve);
    });
  }

  release(): void {
    if (this._waitQueue.length > 0) {
      const next = this._waitQueue.shift()!;
      next(); // transfer lock to next waiter
    } else {
      this._locked = false;
    }
  }
}
```

One `CustomLock` instance per loaded model. This guarantees that:
1. Requests to the same model are strictly serialised (no interleaving of prefill/decode steps).
2. Requests to *different* models can run concurrently (each has its own lock + GPU context).

---

## 3. LLM Chat Pipeline (`src/llm_chat.ts`)

### 3.1 Attributes

```ts
class LLMChatPipeline {
  // ── TVM handles ───────────────────────────────────────────────────
  private tvm:    tvmjs.Instance
  private device: tvmjs.DLDevice          // WebGPU device
  private vm:     tvmjs.VirtualMachine    // compiled model VM
  private prefill:  tvmjs.PackedFunc
  private decoding: tvmjs.PackedFunc
  private embed:    tvmjs.PackedFunc
  private image_embed?: tvmjs.PackedFunc  // only for VLMs

  // ── Sampling functions (TVM PackedFuncs) ──────────────────────────
  private fapplyBitmask, fapplyPenalty, fapplyLogitBias
  private fsoftmaxWithTemperature, fsampleWithTopP
  private fargsortProbs

  // ── KV cache functions ────────────────────────────────────────────
  private fclearKVCaches, fKVCacheAddSequence, fKVCacheRemoveSequence
  private fKVCacheBeginForward, fKVCacheEndForward
  private fKVCacheEnableSlidingWindowForSeq

  // ── State ─────────────────────────────────────────────────────────
  private params:               tvmjs.TVMObject   // model weights
  private kvCache?:             tvmjs.TVMObject   // paged KV cache
  private rnnState?:            tvmjs.TVMObject   // for RNN/hybrid models
  private logitsOnCPU?:         tvmjs.Tensor
  private filledKVCacheLength:  number

  // ── Generation state (reset each prefill) ─────────────────────────
  private outputMessage:        string
  private outputIds:            number[]
  private stopTriggered:        boolean
  private finishReason:         ChatCompletionFinishReason | undefined
  private appearedTokensFreq:   Map<number, number>   // for repetition penalty
  private conversation:         Conversation
}
```

### 3.2 Model ABI Resolution

At construction time, the pipeline inspects the TVM VM's exported function names to determine which ABI to use:

```ts
type ResolvedModelABI = {
  kvStateKind:        "kv_cache" | "rnn_state" | "hybrid"
  prefillABI:         "single" | "batch"   // "prefill" vs "batch_prefill"
  decodeABI:          "single" | "batch"   // "decode"  vs "batch_decode"
  prefillFunctionName: "prefill" | "batch_prefill"
  decodeFunctionName:  "decode"  | "batch_decode"
  needsKVCache:        boolean
  needsRNNState:       boolean
}
```

The resolution checks for the presence of `"create_tir_paged_kv_cache"`, `"create_rnn_state"`, `"batch_prefill"`, `"batch_decode"` in the VM's function registry.

### 3.3 Prefill Step

```
prefillStep(genConfig, inputTokenIds, images?)
  │
  ├─ 1. Build prompt tokens
  │       conversation.getPromptArray() → string[]
  │       tokenizer.encode(strings) → token ID arrays
  │
  ├─ 2. Chunked prefill
  │       split inputIds into chunks of size prefillChunkSize
  │       for each chunk:
  │         if needsKVCache: fKVCacheBeginForward(seqId, len)
  │         if image in this chunk: image_embed() → embed image
  │         embed(inputChunk) → embeddings tensor
  │         prefill(embeddings, kvCache | rnnState, params) → logits
  │         if needsKVCache: fKVCacheEndForward()
  │         filledKVCacheLength += chunkLen
  │
  ├─ 3. Update conversation
  │       conversation.appendReplyHeader(Role.assistant)
  │
  └─ 4. Sample first token (same as decode step)
```

### 3.4 Decode Step

```
decodeStep(genConfig)
  │
  ├─ 1. Forward
  │       fKVCacheBeginForward(seqId, 1) if kv_cache
  │       decoding(embedding_of_last_token, kvCache | rnnState, params) → logits
  │       fKVCacheEndForward() if kv_cache
  │
  ├─ 2. Apply penalties (CPU-side, on logitsOnCPU)
  │       fapplyPenalty(logits, appearedTokensFreq,
  │                     repetition_penalty, frequency_penalty, presence_penalty)
  │
  ├─ 3. Apply grammar bitmask (if response_format ≠ text)
  │       fapplyBitmask(logits, grammarMatcher.nextTokenBitmask())
  │
  ├─ 4. Apply logit_bias (if set)
  │       fapplyLogitBias(logits, tokenIds, biasValues)
  │
  ├─ 5. Softmax with temperature
  │       fsoftmaxWithTemperature(logits, temperature) → probabilities
  │
  ├─ 6. Sample
  │       if seed set: tvm.randomSeed(seed)
  │       nextToken = fsampleWithTopP(probs, top_p, uniform_sample)
  │       seed auto-reset to Date.now() after streaming request
  │
  ├─ 7. Update state
  │       appearedTokensFreq.set(nextToken, count + 1)
  │       outputIds.push(nextToken)
  │       if grammarMatcher: grammarMatcher.acceptToken(nextToken)
  │
  ├─ 8. Decode token to string
  │       outputMessage += tokenizer.decode([...outputIds])
  │       strip trailing U+FFFD replacement characters
  │
  └─ 9. Check stop conditions
         if nextToken ∈ stopTokens → stopTriggered = true, finishReason = "stop"
         if outputMessage contains stopStr → stopTriggered = true, finishReason = "stop"
         if outputIds.length >= max_tokens → stopTriggered = true, finishReason = "length"
         if context window exceeded → throw ContextWindowSizeExceededError
```

### 3.5 KV Cache Reuse

```
compareConversationObject(existing, incoming)
  │
  ├─ compare: config (object equality)
  ├─ compare: messages length
  ├─ compare: each message [role, role_str, content]
  ├─ compare: isTextCompletion, prompt, function_string, use_function_calling
  └─ returns true if identical up to the last message (reply header)

If true:
  → do NOT clear KV cache; resume from filledKVCacheLength
  → only prefill the new messages (efficient prefix caching)
If false:
  → fclearKVCaches()
  → filledKVCacheLength = 0
  → full prefill
```

### 3.6 Grammar Constraint (`@mlc-ai/web-xgrammar`)

When `response_format.type` is `"json_object"` or `"json_schema"`:

```
At prefill start:
  grammarMatcher = xgr.GrammarMatcher(grammar)
  grammar = xgr.Grammar.fromJSONSchema(schema) | xgr.Grammar.builtinJSONGrammar()

Each decode step:
  bitmask = grammarMatcher.nextTokenBitmask()   // allowed token set
  fapplyBitmask(logits, bitmask)                // zero-out disallowed tokens
  grammarMatcher.acceptToken(sampledToken)

`structural_tag` response_format:
  same flow but grammar = xgr.Grammar.fromStructuralTag(...)
```

---

## 4. Embedding Pipeline (`src/embedding.ts`)

### 4.1 Attributes

```ts
class EmbeddingPipeline {
  private tvm:       tvmjs.Instance
  private device:    tvmjs.DLDevice
  private vm:        tvmjs.VirtualMachine
  private prefill:   tvmjs.PackedFunc    // single forward pass
  private params:    tvmjs.TVMObject

  private contextWindowSize: number
  private prefillChunkSize:  number
  private maxBatchSize:      number

  // perf counters
  private curRoundEmbedTotalTokens: number
  private curRoundEmbedTotalTime:   number
}
```

### 4.2 Embedding Flow

```
embed(inputs: string[], mean_pool: boolean, encode_options)
  │
  ├─ validate: inputs.length > 0 (EmbeddingInputEmptyError)
  ├─ validate: sliding_window disabled (EmbeddingSlidingWindowError)
  │
  ├─ tokenize all inputs → token ID arrays
  ├─ validate: no input exceeds contextWindowSize
  │
  ├─ batch inputs up to maxBatchSize
  │   for each batch:
  │     pad to uniform sequence length
  │     tvm.beginScope()
  │       inputTensor  = tvm.empty([batchSize, seqLen], "int32", device)
  │       inputTensor.copyFrom(paddedTokenIds)
  │       outputTensor = prefill(inputTensor, params)
  │       if mean_pool: average non-padding token embeddings
  │       copy output to CPU Float32Array
  │     tvm.endScope()
  │
  └─ return Embedding[] (one per input string)
```

---

## 5. Conversation Manager (`src/conversation.ts`)

### 5.1 Message Storage

```ts
// Each entry is a 3-tuple
type Message = [
  Role,                                                        // user | assistant | tool
  string,                                                      // role display name
  string | ChatCompletionContentPart[] | undefined             // content (undefined = reply header)
]

class Conversation {
  messages: Message[]
  config:   ConvTemplateConfig
  isTextCompletion: boolean
  prompt?: string
  function_string: string
  use_function_calling: boolean
  override_system_message?: string
  private isLastMessageEmptyThinkingReplyHeader: boolean
}
```

### 5.2 Prompt Assembly Algorithm

```
getPromptArray(addSystem, startPos) → Array<string | (string | ImageURL)[]>
  │
  ├─ build system prefix:
  │     system_prompt = config.system_template
  │                       .replace("{system_message}", effective_system_message)
  │     if addSystem && system_prompt ≠ "": ret.push(system_prompt)
  │
  └─ for i in [startPos, messages.length):
       item = messages[i]
       role = item[0], role_str = item[1], content = item[2]
       │
       ├─ if content === undefined  (reply header)
       │     ret.push(config.role_templates[role]
       │                .replace("{assistant_message}", ""))
       │     + config.role_empty_sep
       │
       ├─ if content is string  (text message)
       │     apply role_template placeholders
       │     append role_content_sep and separator
       │
       └─ if content is ContentPart[]  (multimodal)
             for each part: text → string, image_url → ImageURL object
             wrap in array to preserve structure for VLM tokenisation
```

### 5.3 Conversation Population from Request

```
getConversationFromChatCompletionRequest(request, config)
  │
  ├─ validate message ordering rules:
  │     - system message must be first (SystemMessageOrderError)
  │     - messages must alternate user/assistant (MessageOrderError)
  │     - tool messages must follow assistant (MessageOrderError)
  │
  ├─ if request.tools set: inject function_string, set use_function_calling
  │
  ├─ for each message:
  │     append to conversation with appropriate role mapping
  │
  └─ return populated Conversation
```

---

## 6. Cache Utilities (`src/cache_util.ts`)

### 6.1 Cache Backend Selection

```ts
function getCacheBackend(appConfig: AppConfig): "indexedDB" | "cache" {
  // appConfig.useIndexedDBCache is the legacy field
  // appConfig.cacheType is the preferred field
  if (appConfig.cacheType !== undefined) return appConfig.cacheType;
  if (appConfig.useIndexedDBCache === true) return "indexedDB";
  return "cache";  // default: HTTP Cache API
}
```

### 6.2 Cache Scope Mapping

| Scope string | `tvmjs.ArtifactCache` behaviour |
|---|---|
| `"webllm/config"` | Stores JSON config files; uses HTTP Cache or IndexedDB per backend |
| `"webllm/wasm"` | Stores compiled WASM; special fetch logic (see §6.3) |
| `"webllm/model"` | Stores weight shards and tokenizer; uses `tvm.fetchTensorCache()` |

### 6.3 WASM Fetch Strategy

```
wasmUrl origin check:
  contains "localhost"  → fetch() directly, no cache (always fresh for dev)
  not starts with "http"→ resolve relative to baseUrl, fetch() with browser HTTP cache
  remote URL            → wasmCache.fetchWithCache(url, "arraybuffer", signal)
                           → tvmjs ArtifactCache → IndexedDB or Cache API
```

### 6.4 Tokenizer Loading (`asyncLoadTokenizer`)

```
asyncLoadTokenizer(modelUrl, chatConfig, appConfig, logger, integrity?)
  │
  ├─ determine tokenizer files from chatConfig.tokenizer_files
  │     supported: ["tokenizer.json"], ["tokenizer.model"], ["tokenizer.model", "tokenizer.json"]
  │     else: throw UnsupportedTokenizerFilesError
  │
  ├─ for each file:
  │     fetch from modelCache (scope: "webllm/model")
  │     maybeVerifyTokenizerIntegrity(data, filename, url, integrity)
  │
  └─ return Tokenizer.fromBlobRecord({ ... })   // @mlc-ai/web-tokenizers
```

---

## 7. Integrity Verification (`src/integrity.ts`)

### 7.1 SRI String Format

```
"sha256-<base64>"  |  "sha384-<base64>"  |  "sha512-<base64>"
```

Validated by regex: `/^(sha256|sha384|sha512)-([A-Za-z0-9+/]+={0,2})$/`

### 7.2 Verification Algorithm

```ts
async function verifyIntegrity(
  data: ArrayBuffer,
  sri: SRIString,
  url: string,
  onFailure?: "error" | "warn"
): Promise<void> {
  const [algorithm, expectedBase64] = parseSRI(sri);  // validate format
  const hashBuffer = await crypto.subtle.digest(ALGO_MAP[algorithm], data);
  const actualBase64 = btoa(String.fromCharCode(...new Uint8Array(hashBuffer)));

  if (actualBase64 !== expectedBase64) {
    const msg = `Integrity check failed for ${url}`;
    if (onFailure === "warn") { log.warn(msg); return; }
    throw new IntegrityError(msg);
  }
}
```

The Web Crypto API (`crypto.subtle.digest`) is used — always asynchronous, hardware-accelerated where available.

---

## 8. Worker Message Protocol (`src/message.ts`)

### 8.1 Message Envelopes

```ts
interface WorkerRequest {
  kind:    RequestKind
  uuid:    string          // crypto.randomUUID() on the proxy side
  content: ParamsType      // typed per kind (ReloadParams, ChatCompletionNonStreamingParams, …)
}

interface WorkerResponse {
  kind:    "return" | "throw" | "initProgressCallback" | "heartbeat"
  uuid:    string          // echoed back from request
  content: any             // result, error object, or InitProgressReport
}
```

### 8.2 Request Kind → Handler Mapping

| `kind` | Handler action |
|--------|---------------|
| `"reload"` | `engine.reload(params.modelId, params.chatOpts)` |
| `"unload"` | `engine.unload()` |
| `"resetChat"` | `engine.resetChat(params.keepStats, params.modelId)` |
| `"chatCompletionNonStreaming"` | `engine.chatCompletion(request)` → `postMessage(return)` |
| `"chatCompletionStreamInit"` | create `AsyncGenerator`, store in `loadedModelIdToAsyncGenerator[selectedModelId]` |
| `"completionStreamNextChunk"` | `generator.next()` → if done `postMessage(return)` else `postMessage(chunk)` |
| `"completionNonStreaming"` | `engine.completion(request)` → `postMessage(return)` |
| `"completionStreamInit"` | same pattern as chat stream init |
| `"embedding"` | `engine.embeddings.create(request)` → `postMessage(return)` |
| `"forwardTokensAndSample"` | `pipeline.forwardTokensAndSample(...)` (expert API) |
| `"getMessage"` | `pipeline.getMessage()` → `postMessage(return)` |
| `"runtimeStatsText"` | `pipeline.runtimeStatsText()` → `postMessage(return)` |
| `"interruptGenerate"` | `engine.interruptGenerate()` |
| `"getMaxStorageBufferBindingSize"` | query GPU adapter limits |
| `"getGPUVendor"` | query GPU adapter info |
| `"setLogLevel"` | `engine.setLogLevel(level)` |
| `"setAppConfig"` | `engine.setAppConfig(config)` |
| `"keepAlive"` | reply `"heartbeat"` (Service Worker only) |

### 8.3 Streaming Bridge (Proxy Side)

```
WebWorkerMLCEngine.chatCompletion(streamingRequest)
  │
  ├─ uuid = crypto.randomUUID()
  ├─ postMessage({ kind: "chatCompletionStreamInit", uuid, content: { request, selectedModelId } })
  │
  └─ return AsyncGenerator:
       loop:
         postMessage({ kind: "completionStreamNextChunk", uuid, content: { selectedModelId } })
         response = await waitForResponse(uuid)       // resolves on matching uuid
         if response.kind === "return": return        // generator done
         if response.kind === "throw":  throw response.content
         yield response.content                       // ChatCompletionChunk
```

### 8.4 Service Worker Client Registry

```
ServiceWorkerMLCEngineHandler.clientRegistry: Map<uuid, Client | MessagePort>

onmessage(event):
  clientRegistry.set(event.data.uuid, event.source)
  …handle…

postMessage(response):
  client = clientRegistry.get(response.uuid)
  client.postMessage(response)
  if response.kind === "return" | "throw":
    clientRegistry.delete(response.uuid)   // release reference
```

---

## 9. Error Handling

### 9.1 Error Class Taxonomy

```
Error (native)
├─ ConfigValueError
│    ├─ MinValueError(paramName, minValue)
│    ├─ RangeError(paramName, min, max)
│    ├─ NonNegativeError(paramName)
│    ├─ DependencyError(dependent, required, requiredValue)
│    └─ InvalidNumberStringError(paramName, actualValue?)
│
├─ WebGPUNotAvailableError     — navigator.gpu missing or disabled
├─ WebGPUNotFoundError         — tvmjs cannot find GPU device
├─ ShaderF16SupportError       — GPU lacks f16 shader support
├─ FeatureSupportError(feature)— missing required WebGPU feature
├─ DeviceLostError             — GPUDevice.lost fired unexpectedly
│
├─ ModelNotFoundError(modelId)         — not in appConfig
├─ ModelNotLoadedError(requestName)    — API called before reload()
├─ SpecifiedModelNotFoundError         — model ID not in loaded set
├─ MissingModelWasmError               — model_lib URL undefined
├─ ReloadArgumentSizeUnmatchedError    — chatOpts length mismatch
├─ ReloadModelIdNotUniqueError         — duplicate model IDs
├─ IncorrectPipelineLoadedError        — embedding model used as LLM
│
├─ IntegrityError              — SRI hash mismatch
│
├─ ContextWindowSizeExceededError
├─ WindowSizeConfigurationError
├─ WindowSizeSpecificationError
├─ AttentionSinkSizeError
│
├─ EmbeddingUnsupportedModelError
├─ EmbeddingInputEmptyError
├─ EmbeddingSlidingWindowError
├─ EmbeddingExceedContextWindowSizeError
├─ EmbeddingChunkingUnsupportedError
│
├─ UnsupportedTokenizerFilesError
├─ WorkerEngineModelNotLoadedError
├─ UnknownMessageKindError
├─ NoServiceWorkerAPIError
├─ NonWorkerEnvironmentError
└─ ServiceWorkerInitializationError
```

### 9.2 Cross-Worker Error Serialisation

JavaScript prototype chains are lost when objects cross worker boundaries via `postMessage` structured clone. To preserve error identity:

1. Every error class sets `this.name = "ClassName"` in its constructor.
2. The handler catches errors, serialises them as `{ name, message, stack }` JSON, and sends `WorkerResponse { kind: "throw" }`.
3. The proxy reconstructs a plain `Error` with `.name` and `.message` set from the serialised fields.
4. Callers can match on `error.name === "ModelNotLoadedError"` or similar.

---

## 10. Logit Processor Hook

```ts
interface LogitProcessor {
  processLogits(logits: Float32Array): Float32Array
    // Called immediately after prefill/decode forward pass, before sampling.
    // Receives and returns the raw logit array (CPU-side Float32Array).

  processSampledToken(token: number): void
    // Called immediately after sampling, with the selected token ID.

  resetState(): void
    // Called when engine.resetChat() is invoked.
}
```

Registration:
```ts
engine = new MLCEngine({
  logitProcessorRegistry: new Map([
    ["Llama-3.1-8B-Instruct-q4f32_1-MLC", myProcessor]
  ])
});
```

The processor is retrieved at pipeline construction time by model ID and stored inside `LLMChatPipeline`. The `processLogits` call is inserted between the TVM forward pass and `fapplyBitmask` / `fapplyPenalty`.

---

## 11. Interrupt and Abort

### `engine.interruptGenerate()`

Sets `engine.interruptSignal = true`. The decode loop in `LLMChatPipeline.decodeStep()` checks this flag at the top of each iteration and throws `GenerateInterruptedError` if set. The lock is still released via `finally`.

### `AbortController` during reload

`engine.reloadController` is set to a new `AbortController` at the start of `reload()`. Its signal is passed to every `fetchWithCache` call. Calling `engine.unload()` during an in-progress reload triggers the abort. On `DOMException { name: "AbortError" }`, the reload silently returns.

---

## 12. Vision Model Support

When `ModelRecord` includes a vision-capable model and the user sends messages with `image_url` content parts:

```
prefillStep detects image in prompt array
  │
  ├─ getImageDataFromURL(imageURL) → ImageData
  ├─ getRGBArrayFromImageData(imageData) → Float32Array
  ├─ image_embed(rgbTensor, params) → embedding tensor
  │    (replaces the token embedding for the image placeholder token)
  └─ concat with text embeddings → pass to prefill kernel
```

The pipeline checks for `image_embed` function availability in the VM at construction time. Image embedding is always chunked to align with `IMAGE_EMBED_SIZE` constants.
