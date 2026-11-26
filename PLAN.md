# PLAN.md — `squid7800xt` Build & Hill-Climb Log

Target: Single-binary, zero-Python TigerStyle Zig 0.17.1 + Vulkan 1.4 Compute (`SPIR-V 1.6`, RDNA 3 `gfx1101`) inference engine for `thinkingmachines/Inkling-2-Mini` on AMD Radeon RX 7800 XT (16 GiB GDDR6, 624 GB/s spec) + Ryzen 7 7800X3D (32 GiB DDR5).

Baseline: `unsloth/Inkling-2-Mini-GGUF:UD-IQ4_XS` (15.73 GiB) -> **24–25 tok/s** (VRAM spillover over PCIe/DDR5).

## Experiment Log

- **Exp 0 — Hardware Ceilings (`THEORY.md`):**
  - Built zero-dependency Vulkan 1.4 dynamic loader (`vulkan-1.dll`) + GLSL->SPIR-V 1.6 build pipeline (`Zig 0.17.1` + `glslang 16.6.0`).
  - Measured `532.1 GB/s` pure read BW (`DEVICE_LOCAL`), `524.9 GB/s` ReBAR (0% penalty), `25.1 GB/s` PCIe 4.0 DDR5 read, `502.3 GB/s` D2D copy, `16.06 TFLOPS` naive WMMA `16x16x16`.
  - Measured NVFP4 LM-head GEMV (`200058x2304`): **Wave64 + 256-entry LDS `f16vec2` LUT wins at `562.1 GB/s` (`0.463 ms`)** vs Wave64 Reg-LUT16 (`557.5`), Wave32 LDS-LUT256 (`549.3`), Wave32 Reg-LUT16 (`486.9`).
  - **Verdict:** KEEP Wave64 + LDS-LUT256 as primary NVFP4 decode GEMV architecture; KEEP zero-copy DDR5 pinned embeddings (`0.86 GiB` saved); KEEP ReBAR direct upload.

- **Exp 1 — Full End-to-End NVFP4 Target Engine (`mtp=0` baseline):**
  - All 998 NVFP4 tensors loaded in `~7.2s`, `13.63 GiB` VRAM (`13.70 GiB` reserved across 18 slabs) + `0.86 GiB` pinned DDR5 embeddings + `98 MiB` DDR5 `markov_w1`.
  - Baseline greedy decode (`mtp=0`): **`65.5 tok/s`** (`15.27 ms/step`).
  - **Verdict:** KEEP (already 2.6x faster than 24–25 tok/s GGUF baseline).

- **Exp 2 — DSpark MTP Speculative Decoding (`draft_vocab=49152` + adaptive `{2,4,8}` + `PromptLookupIndex`):**
  - Added 3-layer DSpark draft head, Markov confidence gating (`--spec-threshold 0.80`), and O(1) prompt n-gram lookup (`min_match_len=6`).
  - Measured `mtp=3`: `67.4 tok/s`; `mtp=7`: **`83.1 tok/s`**.
  - **Verdict:** KEEP; profiled per-stage bottlenecks (`router` taking `10.90 ms` at B=8, `acc[8]` scratch spills).

- **Exp 3 — `SPEC_B` Specialization Constants + Parallel Wave64 Sorted `route_topk`:**
  - Replaced dynamic-indexed `float acc[8]` arrays (which spilled to RDNA 3 scratch VRAM on AMDGPU LLVM) with Vulkan `SPEC_B` specialization constants + scalar accumulators (`a0..a7`) in `linear_nvfp4`, `moe_up`, `moe_down`, `lm_head_argmax`.
  - Rewrote `route_topk.comp` with parallel Wave64 sorted expert compaction (also established bit-exact greedy parity between `mtp=0` and `mtp=7`).
  - Results: `router` dropped **`10.90 ms -> 1.26 ms` (8.7x faster at B=8)**; `mtp=0` jumped **`65.5 -> 78.57 tok/s` (+20%)**; `mtp=3` jumped **`67.4 -> 96.57 tok/s` (+43%)**; `mtp=7` reached **`97.03 tok/s`**.
  - **Verdict:** KEEP.

- **Exp 4 — Persistent 240-WG Zero-Barrier `subgroupShuffleXor(v,16)` SwiGLU `moe_up` + `Uvec4Buf` Experiment:**
  - `moe_up` B=1 dropped `3.55 -> 3.18 ms`; DSpark draft dropped `1.74 -> 1.54 ms`; `qkvrg` B=8 dropped `7.35 -> 5.32 ms`.
  - However, `Uvec4Buf` + `unpackHalf2x16` added FP32<->FP16 conversion overhead and regressed `router` 2.5x.
  - **Verdict:** KEEP persistent zero-barrier `moe_up`; REVERT `Uvec4Buf` (keep native `F16Vec2Buf`).

- **Exp 5 — Batched 3-Stream `controller_nvfp4` (7.6 KB `CtrlDesc` Table in ReBAR) + `post_depth_norm` / `F16Vec4Buf` Experiment:**
  - Batched 3-stream `controller_nvfp4` cut controller latency **`1.48 -> 0.18 ms` (8.2x faster)**.
  - Fusing 18.5 KB LDS 1-WG `post_depth_norm` after `w_o`/`moe_down` caused pipeline drain stalls (`+1.8 ms` regression), and `F16Vec4Buf` under `scalar` layout lowered to 16-bit scalar loads (`router` 4x slower).
  - **Verdict:** KEEP 3-stream `controller_nvfp4` (in ReBAR VRAM) + `uvec2` `sconv_w` in `post_res`; REVERT `post_depth_norm` and `F16Vec4Buf`.

- **Exp 6–8 — Combined Proven Winners (512-Token Code Generation Benchmark):**
  - Combined `SPEC_B`, parallel Wave64 `route_topk`, persistent zero-barrier `moe_up`, 3-stream ReBAR `controller_nvfp4`, native `F16Vec2Buf` `linear_nvfp4`.
  - **`mtp=7 --spec-threshold 0.80` reached `100.58 tok/s` (`acc=61.0%`, `dl=2.65`)** — **4.1x faster** than the 24–25 tok/s GGUF baseline; `mtp=0` measured `76.4–80.55 tok/s` (greedy/sampled).
  - **Verdict:** KEEP.

- **Exp 9 — Production Engine (`B=64` RDNA 3 WMMA Prefill, 3-Slot DDR5 Prefix KV Cache, Split-K Attention Spill Fix, GPU Gumbel-Max Sampling, Zero-Malloc Winsock2 OpenAI HTTP/SSE Server):**
  - Added `linear_wmma_nvfp4.comp` (`VK_KHR_cooperative_matrix`, Wave32) + `moe_prefill_b64.comp`: prefill jumped `185.9 -> 301.3 tok/s` on 198-tok prompt (`240–290 tok/s` sustained up to 5.5K tokens).
  - Eliminated 72-float `m_val[24]/d_val[24]/o_val[24]` scratch spills in `attn_splitk.comp` & `dspark_attn.comp` via `Grid=(8, 64, B)` 9-VGPR scalar accumulators (`m0..2, d0..2, o0..2`):
    - At `pos=65,000` (`ctx=65536`, Q4 KV): **`B=1` dropped `97.71 ms -> 43.43 ms` (2.25x faster)** and **`B=4` dropped `179.32 ms -> 101.35 ms` (1.77x faster)**.
    - At `pos=0`: `B=1 11.99 ms (83.4 tok/s)`, `B=4 24.28 ms (MTP ~109.1 tok/s)`.
  - Added contiguous `local_state_buf` + 3-slot DDR5 multi-checkpoint prefix KV cache (`3x190 MiB`): Turn 2 hit **`667/688` cached tokens**, slashing prefill **`2284.2 ms -> 207.3 ms` (11.0x faster)**.
  - Verified all quality gates end-to-end over `http://127.0.0.1:8017/v1`:
    - `GET /health` & `GET /v1/models` -> 200 OK
    - SSE Turn 1 (`17 * 23`) -> `'391'`, Turn 2 (`+ 9`, cached `667/688`) -> `'400'`
    - **5,481-token Needle-in-a-Haystack** -> exact `'78492-ZETA'` (`decode=56.8 tok/s` at 5.5K ctx)
    - GPU Gumbel-Max sampling (`--temperature 0.7 --seed 123`) -> coherent output at `80.55 tok/s` (`mtp=0`).
  - **Verdict:** KEEP ALL. Engine production-ready.

- **Exp 10 — Tiled Split Attention (`attn_tile.comp` + `attn_merge_gate.comp`), Head-Major Q4 KV, Row-Owner Softmax:**
  - Diagnosis (baseline sweep, `ctx=131072`): `pos=0 11.69 ms` → `pos=4096 24.42 ms` → `pos=65000 43.25 ms` → `pos=130000 63.18 ms`. Global attention read ~570 MB of Q4 KV at 65K (≈1.1 ms at roofline) but cost ~31 ms; the 1024-token local/tail stage (48 visits, only 24 workgroups, one token per iteration with 2 barriers + 3 wave reductions) added ~12.7 ms as soon as `pos > 1K`; MTP verify re-read the whole KV once per batch row.
  - New stage 1 (`attn_tile`): one Wave64 per split, 16-token tiles (lane = token × 32-dim slice, one 16 B `uvec4` per K/V per lane), next tile prefetched into registers, rel-position bias computed inline (no separate 1024-entry bias table), **all `B` rows × 3 GQA heads share every KV tile** (`SPEC_B` = 1..8). Handles target global, target local SWA, and DSpark ctx attention (replaces `attn_splitk`, the serial tail loop of `attn_reduce_gate`, and DSpark mode 0 / local loop).
  - Split layout is a pure function of the visible range (`attn_split_layout`, min 4 tiles/split, ≤256 splits for B≤4, ≤128 for B=8) so producer, merge and DSpark agree exactly.
  - Q4 KV is now **head-major** (`[kvh][cap][64 B]` + `[kvh][cap][2×f16]` scales next to each other) so a split streams one contiguous run instead of 64 B pieces at a 512 B stride plus 2 B scale gathers.
  - Row-owner softmax: scores go through LDS and lane `r` updates row `r` serially, replacing 2 full-wave reductions per row per tile (48 per tile at B=8). B=64 prefill at 130K: `131 → 153 tok/s`; decode unchanged.
  - **Bug found & fixed during validation:** the recorded dispatch count was derived from the *maximum* span, but split count is not monotonic in range length, so at ~100K one split was never computed → garbage/NaN logits (empty output on a 100K-token prompt). Dispatch is now `min(split_max, ceil(max_tiles / min_tiles))`; `attncheck` exhaustively verifies layout ≤ dispatch for every B and every range length ≤ 1M.
  - Validation: new `attncheck` subcommand compares GPU output to an f64 CPU reference (Q4/BF16, global/local/DSpark visibility, B=1..8, rel bias, sink, gate, ring wrap, positions up to 131K): worst relative error `7.7e-4` (f16 output rounding). Needle tests: 2.1K (CLI, mtp 0/7), 30K and **101,782-token** (server) all retrieve `78492-ZETA`.
  - Tried & reverted: fewer min tiles per split (1/2/4: within noise); batching prefill global attention across the 8 sub-chunks in one dispatch (slower: 64 rows leave only 16 splits per KV head); `vec4` probability reads in PV (no gain, noise ±7%).
  - **Verdict:** KEEP.

- **Exp 11 — VRAM Budget Probe + DDR5 Expert Tier (`--expert-tier auto|N`) + Arena Fixes:**
  - Finding: on this Windows box the desktop compositor holds **~1.6–1.8 GiB of the 7800 XT's VRAM** (monitor attached), leaving ~14.1–14.4 GiB. With 13.70 GiB of weights, part of the global KV landed in pinned DDR5 *even at `ctx=32768`* (read over PCIe at ~25 GB/s every token), and runs at 81K/128K could OOM. `VK_EXT_memory_budget` over-reported (15.05 GiB "free" when allocations failed near 14.5 GiB).
  - Every token reads the **whole** KV cache but only 8/128 experts per layer, so experts are the cheapest bytes to evict. Expert banks now split into a VRAM part (experts `[0, n_lo)`) and a pinned-DDR5 tail with a pre-biased second base address (`moe_up`, `moe_down`, `moe_prefill_b64`, MXFP8 packer: one `select` per site). Spilling 16 experts/layer gives **bit-identical output**; cost ≈0.46 ms/token per spilled expert-per-layer, ~11× cheaper per GB than spilled KV at full context.
  - `--expert-tier auto` (default) probes allocatable VRAM (allocate-until-fail, min with the WDDM budget), estimates runtime needs for `--ctx`, and moves just enough tail experts to keep **all KV in VRAM**. The engine arena now (a) never lets a DDR5 fallback slab attract allocations that still fit a new VRAM slab, and (b) reuses the weight arena's unused slab tails (~0.2 GiB) before reserving VRAM.
  - Result: `DDR5_KV=0.00 GiB` at 32K/64K/128K (auto tier = 1 / 4 / 9–10 experts per layer).
  - **Verdict:** KEEP. (Driving the display from the Ryzen iGPU should free the desktop's ~1.7 GiB and make the tier unnecessary up to ~128K; not measured.)

- **Exp 10+11 combined — long-context sweep (`longctx`, MTP=7, desktop holding ~1.7 GiB VRAM):**

  | pos | before B=1 | after B=1 (`ctx=131072`, tier 9) | before MTP est. | after MTP est. |
  |---:|---:|---:|---:|---:|
  | 0 | 11.69 ms (85.5 tok/s) | 16.02 ms (62.4 tok/s) | 110.9 tok/s | 71.6 tok/s |
  | 4,096 | 24.42 ms (41.0) | 18.99 ms (52.7) | 74.1 | 73.6 |
  | 16,384 | 27.19 ms (36.8) | 19.26 ms (51.9) | 56.7 | 68.3 |
  | 32,768 | 32.59 ms (30.7) | 19.45 ms (51.4) | 43.0 | 60.9 |
  | 65,000 | 43.25 ms (23.1) | 22.03 ms (45.4) | 29.2 | 56.4 |
  | 130,000 | 63.18 ms (15.8) | 22.39 ms (44.7) | 15.1 | 47.9–51.9 |

  With `ctx=65536` (tier 4): `14.01 / 14.75 / 15.87 / 16.31 / 18.30 ms` at `0 / 4K / 16K / 32K / 65K` (vs `11.63 / 23.37 / 26.48 / 30.98 / 41.54 ms` before). The `pos=0` cost of the tier is the price of keeping the full window's KV in VRAM; `--expert-tier 0` restores it when long contexts are not needed.

- **Exp 12 — Output-quality eval vs. `UD-IQ4_XS` + BF16 router:**
  - New `score` subcommand (teacher-forced B=8 forward + full-vocab LM head into host memory) and `eval/quality.py`, which scores the same token ids with llama-server (forced-token `logit_bias`, raw full-vocab probabilities). 24×512 WikiText-2 windows + 12×512 HumanEval windows, first 32 positions of each skipped.
  - Baseline (NVFP4 router): ppl `7.172` vs GGUF `7.067` on WikiText-2 (**+1.49%**, CI excludes 0), `2.338` vs `2.343` on code (tied). Top-1 agreement 93.1% / 96.4%.
  - Router kept in BF16 (`linear_bf16.comp`, used in decode and the B=64 prefill; +10 MiB VRAM): WikiText-2 gap **+1.49% → +1.25%** (paired Δ −0.0023 nats/token, window-bootstrap CI [−0.0049, +0.0004]: borderline), code unchanged. Decode `mtp=0` 82.7 → 82.1 tok/s (−0.7%), MTP within noise.
  - **Verdict:** KEEP (cheap, slightly better). Most of the remaining gap is elsewhere (attention / LM head / scale choice).
