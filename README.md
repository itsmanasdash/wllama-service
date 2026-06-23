# wllama-service

A framework-agnostic browser LLM service wrapper built on [@wllama/wllama](https://github.com/ngxson/wllama).
Runs GGUF models locally in the browser with WebGPU acceleration and WebAssembly fallback.

## Requirements

Your web server must send these headers for multi-threading and WebGPU to work:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Copy `wllama.wasm` from `node_modules/@wllama/wllama/esm/wasm/wllama.wasm` to your public folder
and pass its URL via `wasmPath`.

## Install

```bash
npm install wllama-service @wllama/wllama
```

## Usage

```typescript
import { WllamaService } from 'wllama-service';

const service = new WllamaService({
  wasmPath: '/wllama/wllama.wasm', // path in your public folder
  nCtx: 2048,
  nGpuLayers: 999, // set 0 to disable WebGPU
});

// Check browser support
const env = service.checkEnvironment();
console.log('WebGPU available:', env.hasWebGPU);

// Load a GGUF file from file input
const result = await service.loadModel(file, (progress) => {
  console.log(`Loading: ${progress}%`);
});

// Generate
const response = await service.generate({
  system: 'You are a helpful assistant.',
  prompt: 'Hello!',
  maxTokens: 256,
  temperature: 0.7,
});

console.log(response.text);
console.log(`Generated in ${response.timeMs}ms`);

// Unload when done
await service.unload();
```