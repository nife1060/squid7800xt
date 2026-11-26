# squid7800xt

Single-binary, zero-Python **Zig + Vulkan 1.4 Compute (`SPIR-V 1.6`)** inference engine for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini), built and hand-tuned for the **AMD Radeon RX 7800 XT (16 GiB GDDR6, RDNA 3 `gfx1101`)** on Windows 11.

Inspired by [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) and tuned with [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill).

## Highlights

- **4.1× faster than GGUF on RX 7800 XT:** breaks **100 tok/s** (`100.6 tok/s` on 512-token codegen with DSpark MTP, **78–81 tok/s** without speculative decoding) vs. `24–25 tok/s` on `UD-IQ4_XS` GGUF.
- **Fits natively in 16 GiB VRAM:** all 998 target + DSpark MTP weights quantized on-the-fly to **NVFP4** (`13.63 GiB` VRAM), with embeddings (`0.86 GiB`), `markov_w1` (`98 MiB`), and a 3-slot prefix KV cache in DDR5.
- **RDNA 3 Vulkan 1.4 Compute Kernels:**
  - Wave64 + 256-entry LDS `f16vec2` LUT NVFP4 GEMV (`562.1 GB/s` effective on LM head)
  - `SPEC_B` specialization constants & 9-VGPR split-K attention (zero scratch-memory spills)
  - Parallel Wave64 sorted MoE routing, persistent zero-barrier SwiGLU `moe_up`, 3-stream batched controller
  - `VK_KHR_cooperative_matrix` (WMMA) `B=64` batched prefill (`240–301 tok/s`)
  - DSpark 3-layer MTP speculative decoding (`--mtp 7`, adaptive `{2,4,8}` Markov confidence gating + O(1) prompt n-gram lookup)
  - GPU Gumbel-Max sampling (`--temperature`, `--seed`)
- **128K context at full speed:** tiled split attention (one Wave64 per split, all MTP rows × GQA heads share each KV tile, head-major Q4 KV) + automatic **DDR5 expert tier** that keeps the whole KV cache in VRAM: **`44.7 tok/s` at `pos=130,000`** (`22.4 ms`, was `63.2 ms`), `45.4 tok/s` at `65K` (was `23.1`); verified with a 101,782-token needle test.
- **3-Slot DDR5 Prefix Cache:** `11×` faster multi-turn prefill on cache hits (100K-token prompt re-sent: `548 s → 11 s` prefill).
- **Universal OpenAI + `llama.cpp` Router-Mode HTTP/SSE Server:** zero post-init allocations, dual-stack IPv4/IPv6 (`127.0.0.1` + `[::1]`), `Transfer-Encoding: chunked`, OpenAI tool calling (`tools` / `tool_calls` / `role: "tool"`), `/props`, `/models`, `/models/load`, `/tokenize`, `/v1/chat/completions`, `/v1/completions`, and `/completion` — compatible with **`pi`**, **OpenCode**, **Cline**, **Cursor**, and standard OpenAI SDKs.

## Measured Performance (RX 7800 XT + 7800X3D, Windows 11)

| Benchmark | `unsloth` `UD-IQ4_XS` GGUF | `squid7800xt` (NVFP4 + Vulkan 1.4) |
| :--- | :---: | :---: |
| Short-context decode (`mtp=0`) | 24–25 tok/s | **78.6 – 80.6 tok/s** (`11.99 ms/step` at `pos=0`) |
| 512-tok codegen (`mtp=7`, DSpark MTP) | 24–25 tok/s | **100.58 tok/s** (`acc=61.0%`, `dl=2.65`) |
| Cold prefill (`B=64` WMMA) | — | **240 – 301 tok/s** |
| Cached multi-turn prefill (Prefix KV) | full re-prefill | **207.3 ms** (`667/688` cached, `11×` faster) |
| Long context (`pos=65,000`, `ctx=65536`, Q4 KV) | OOM / PCIe spill | **54.6 tok/s** (`18.3 ms` `B=1`; was `23.0`) |
| Long context (`pos=130,000`, `ctx=131072`, Q4 KV) | OOM / PCIe spill | **44.7 tok/s** (`22.4 ms` `B=1`), **~48–52 tok/s** with MTP (was `15.8` / `15.1`) |
| Cold model load (ReBAR upload) | — | **7.2 – 9.6 s** (`13.70 / 15.98 GiB` VRAM) |

Full roofline analysis and A/B experiment logs (including reverted regressions) are in [THEORY.md](THEORY.md), [PLAN.md](PLAN.md), and [ledger.jsonl](ledger.jsonl).

## Requirements

- **OS / GPU:** Windows 11, AMD Radeon RX 7800 XT (or RDNA 3 `gfx11xx` with 16 GiB+ VRAM and Vulkan 1.4 driver)
- **Build tools:** [Zig 0.17.1](https://ziglang.org/) and Khronos [`glslang`](https://github.com/KhronosGroup/glslang/releases) (`vulkan1.4` / `SPIR-V 1.6`)
- **Model files:** `thinkingmachines/Inkling-2-Mini` (`model-00001-of-00001.safetensors`, `tokenizer.json`, and optionally `mtp.safetensors` + `draft_vocab_49k.bin`) placed in `models/inkling-2-mini/`

## Build

```powershell
# Using glslang on PATH:
zig build -Doptimize=fast -Dglslang=glslang

# Or if using portable binaries in tools/:
.\tools\zig\zig.exe build -Doptimize=fast
```

Output binary: `zig-out\bin\squid7800xt.exe` (all compiled SPIR-V shaders are embedded directly in the executable).

## Usage

### Start the OpenAI / `llama.cpp` API Server

```powershell
.\zig-out\bin\squid7800xt.exe serve models/inkling-2-mini --host 127.0.0.1 --port 8000 --ctx 65536 --mtp 7
```

Common flags:
- `--ctx 32768` / `--ctx 65536` / `--ctx 131072` — context window (`Q4` global KV cache)
- `--expert-tier auto|N` — (default `auto`) experts per layer kept in pinned DDR5 instead of VRAM. `auto` probes the free VRAM at startup and moves just enough tail experts out so that the **entire KV cache for `--ctx` stays in VRAM** (each token reads all of the KV but only 8/128 experts per layer, so experts are the cheaper bytes to evict). `0` = all experts in VRAM.
- `--mtp 7` (default) or `--mtp 0` — DSpark speculative draft tokens
- `--port 8000` / `--host 127.0.0.1` (binds both `127.0.0.1` and `[::1]`)
- `--api-key <key>` — optional auth (disabled by default; any key like `none` is accepted)
- `--effort none|low|medium|high` — default thinking effort level

Connect any harness:
- **`pi` (`llama.cpp` provider):** URL `http://127.0.0.1:8000`, API key `none`
- **OpenCode / Cline / OpenAI SDK:** Base URL `http://127.0.0.1:8000/v1`, Model `inkling-2-mini`, API key `none`

### CLI Generation & Benchmarks

```powershell
# Speculative generation (MTP=7)
.\zig-out\bin\squid7800xt.exe generate models/inkling-2-mini --prompt "Write a lock-free ring buffer in Zig." --max-tokens 512 --mtp 7

# Sampled generation (GPU Gumbel-Max)
.\zig-out\bin\squid7800xt.exe generate models/inkling-2-mini --prompt "Explain RDNA 3 Wave64 vs Wave32." --max-tokens 256 --mtp 7 --temperature 0.7 --seed 42

# Hardware roofline microbenchmarks
.\zig-out\bin\squid7800xt.exe ceilings

# Long-context latency sweep (0 -> 130K)
.\zig-out\bin\squid7800xt.exe longctx models/inkling-2-mini --ctx 131072 --mtp 7

# Attention kernel self-test (GPU vs f64 CPU reference: Q4/BF16, global/local/DSpark, B=1..8)
.\zig-out\bin\squid7800xt.exe attncheck models/inkling-2-mini --ctx 8192 --mtp 0
```

> **Tip for maximum context:** on Windows the desktop compositor holds ~1.6–1.8 GiB of the 7800 XT's VRAM when a monitor is attached to it. Driving the display from the Ryzen iGPU (motherboard video output) frees that VRAM; by the planner's own estimate (~15.1 GiB needed at `--ctx 131072` with MTP) that should be enough to keep every expert in VRAM at 128K (not measured).

## Project Layout

```text
build.zig          # Zig + glslang SPIR-V 1.6 build pipeline
src/
  main.zig         # CLI entry point (serve, generate, bench, profile, longctx, attncheck, ceilings)
  engine.zig       # Pre-recorded Vulkan command buffers, DSpark MTP, WMMA prefill, DDR5 prefix cache
  kernels.zig      # Vulkan compute pipeline & SPEC_B specialization manager
  weights.zig      # Streaming safetensors -> NVFP4 GPU packer
  vram_arena.zig   # Multi-slab BDA VRAM allocator with DDR5 fallback
  vulkan.zig       # Zero-dependency dynamic vulkan-1.dll loader
  tokenizer.zig    # Zero-malloc o200k_base BPE tokenizer & Inkling chat template
  server.zig       # Zero-malloc Winsock2 OpenAI + llama.cpp router-mode HTTP/SSE server
  constants.zig    # Comptime-verified Inkling-2-Mini & DSpark geometry
  tiger.zig        # Win32/Winsock2 primitives, timing, always-on assertions
shaders/           # 26 hand-tuned GLSL 4.60 / SPIR-V 1.6 Vulkan compute shaders
THEORY.md          # RX 7800 XT roofline & memory-budget model
PLAN.md            # Hill-climbing experiment log
ledger.jsonl       # Machine-readable experiment results
```
