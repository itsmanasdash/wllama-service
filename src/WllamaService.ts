// src/WllamaService.ts

import { Wllama } from "@wllama/wllama/esm/index.js";

import type {
  WllamaServiceConfig,
  GenerateRequest,
  GenerateResult,
  LoadModelResult,
  ConnectionResult,
} from "./types";

export class WllamaService {
  private wllama: Wllama | null = null;
  private modelName = "";
  private readonly config: Required<WllamaServiceConfig>;

  constructor(config: WllamaServiceConfig = {}) {
    this.config = {
      wasmPath: config.wasmPath ?? "/wllama/wllama.wasm",
      nGpuLayers: config.nGpuLayers ?? 999,
      nCtx: config.nCtx ?? 2048,
    };
  }

  /**
   * Check WebGPU without directly accessing navigator.gpu.
   *
   * This avoids requiring @webgpu/types merely for feature detection.
   */
  private hasWebGPU(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  }

  /** Check browser compatibility and capabilities. */
  checkEnvironment(): ConnectionResult {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      return {
        success: false,
        error: "WllamaService can only run in a browser environment.",
      };
    }

    if (typeof WebAssembly === "undefined") {
      return {
        success: false,
        error: "WebAssembly is not supported in this browser.",
      };
    }

    return {
      success: true,
      hasWebGPU: this.hasWebGPU(),
      crossOriginIsolated: window.crossOriginIsolated,
    };
  }

  /** Load a GGUF model file from the user's filesystem. */
  async loadModel(
    file: File,
    onProgress?: (progress: number) => void,
  ): Promise<LoadModelResult> {
    const environment = this.checkEnvironment();

    if (!environment.success) {
      return {
        success: false,
        error: environment.error,
      };
    }

    await this.unload();

    try {
      onProgress?.(10);

      this.wllama = new Wllama({
        default: this.config.wasmPath,
      });

      onProgress?.(30);

      const useWebGPU =
        this.hasWebGPU() && this.config.nGpuLayers > 0;

      await this.wllama.loadModel([file], {
        n_ctx: this.config.nCtx,
        n_gpu_layers: useWebGPU
          ? this.config.nGpuLayers
          : 0,
        jinja: true,
      });

      this.modelName = file.name;

      onProgress?.(100);

      return {
        success: true,
        usedWebGPU: useWebGPU,
      };
    } catch (error: unknown) {
      await this.unload();

      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to load model",
      };
    }
  }

  /**
   * Generate a chat response using the model's embedded chat template.
   *
   * Suitable for instruction-tuned models such as Phi-3 Instruct,
   * Llama Instruct, ChatML models, and similar chat models.
   */
  async generate(
    req: GenerateRequest,
  ): Promise<GenerateResult> {
    if (!this.wllama) {
      return {
        success: false,
        error: "No model loaded. Call loadModel() first.",
      };
    }

    try {
      const startedAt = performance.now();

      const messages: Array<{
        role: "system" | "user";
        content: string;
      }> = [];

      if (req.system) {
        messages.push({
          role: "system",
          content: req.system,
        });
      }

      messages.push({
        role: "user",
        content: req.prompt,
      });

      /*
       * Keep the cast here if the installed @wllama/wllama declaration
       * does not expose the exact createChatCompletion signature.
       */
      const response = await (
        this.wllama as unknown as {
          createChatCompletion(options: {
            messages: Array<{
              role: "system" | "user";
              content: string;
            }>;
            stream: false;
            max_tokens: number;
            temperature: number;
            top_k: number;
            top_p: number;
            stop?: string | string[];
            abortSignal?: AbortSignal;
          }): Promise<{
            choices?: Array<{
              message?: {
                content?: string;
              };
            }>;
          }>;
        }
      ).createChatCompletion({
        messages,
        stream: false,
        max_tokens: req.maxTokens ?? 512,
        temperature: req.temperature ?? 0.7,
        top_k: req.topK ?? 40,
        top_p: req.topP ?? 0.95,
        stop: req.stop,
        abortSignal: req.abortSignal,
      });

      const text =
        response.choices?.[0]?.message?.content ?? "";

      return {
        success: true,
        text,
        timeMs: Math.round(
          performance.now() - startedAt,
        ),
      };
    } catch (error: unknown) {
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Generation failed",
      };
    }
  }

  /**
   * Generate a raw completion without applying a chat template.
   *
   * Suitable for base completion models, manually formatted prompts,
   * few-shot completion, fill-in-the-middle, and custom templates.
   */
  async generateCompletion(
    req: GenerateRequest,
  ): Promise<GenerateResult> {
    if (!this.wllama) {
      return {
        success: false,
        error: "No model loaded. Call loadModel() first.",
      };
    }

    try {
      const startedAt = performance.now();

      const prompt = req.system
        ? `${req.system}\n\n${req.prompt}`
        : req.prompt;

      const response =
        await this.wllama.createCompletion({
          prompt,
          stream: false,
          max_tokens: req.maxTokens ?? 512,
          temperature: req.temperature ?? 0.7,
          top_k: req.topK ?? 40,
          top_p: req.topP ?? 0.95,
          stop: req.stop,
          abortSignal: req.abortSignal,
        });

      const choice = response.choices?.[0];

      console.debug("[wllama] raw completion", {
        text: choice?.text,
        finishReason: choice?.finish_reason,
        usage: response.usage,
        timings: response.timings,
      });

      return {
        success: true,
        text: choice?.text ?? "",
        timeMs: Math.round(
          performance.now() - startedAt,
        ),
      };
    } catch (error: unknown) {
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Generation failed",
      };
    }
  }

  /** Unload the current model and release its resources. */
  async unload(): Promise<void> {
    if (this.wllama) {
      try {
        await this.wllama.exit();
      } catch {
        // Ignore shutdown errors.
      }

      this.wllama = null;
    }

    this.modelName = "";
  }

  /** Whether a model is currently loaded. */
  get isLoaded(): boolean {
    return this.wllama !== null;
  }

  /** Name of the currently loaded model. */
  get currentModel(): string {
    return this.modelName;
  }
}