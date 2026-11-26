# THEORY.md — Measured RX 7800 XT Ceilings & Inkling-2-Mini Roofline

Measured directly via `squid7800xt.exe` (Vulkan 1.4.359, Adrenalin 26.11.2 LLPC, Windows 11, AMD Ryzen 7 7800X3D + 32 GiB DDR5).

## 1. Measured Hardware Ceilings (AMD Radeon RX 7800 XT, `gfx1101`)

| Metric | Measured Value | Spec / Notes |
|---|---:|---|
| GPU | AMD Radeon RX 7800 XT (`0x747e`) | RDNA 3 (`gfx1101`), 60 CUs, 64 MiB Infinity Cache |
| VRAM | `15.98 GiB` GDDR6 (256-bit, 19.5 Gbps) | Spec `624.0 GB/s`; full ReBAR (`DEVICE_LOCAL \| HOST_VISIBLE`) |
| Host-Pinned DDR5 | `15.54 GiB` GPU-accessible | Zero-copy over PCIe 4.0 x16 |
| **Pure Read BW (`1 GiB`, `DEVICE_LOCAL`)** | **`532.1 GB/s`** (Wave32) / `525.9 GB/s` (Wave64) | Real decode baseline ceiling |
| **Pure Read BW (`1 GiB`, ReBAR)** | **`524.9 GB/s`** | `<0.2%` delta vs non-visible VRAM → zero penalty |
| **D2D Copy BW (`512 MiB` R+W)** | **`502.3 GB/s`** | Read + write sustained |
| **Zero-Copy DDR5 Read BW (PCIe 4.0 x16)** | **`25.1 GB/s`** | Explains 24–25 tok/s bottleneck on VRAM spill! |
| **NVFP4 LM-Head GEMV (`[200058 × 2304]`)** | **`562.1 GB/s`** (`0.463 ms`) | **Wave64 + 256-entry LDS `f16vec2` LUT** (beats Wave32 `549.3` & Reg-LUT `557.5`) |
| **RDNA 3 WMMA FP16 GEMM (`4096³`, naive)** | **`16.06 TFLOPS`** | `VK_KHR_cooperative_matrix` `16×16×16` FP16→FP32 verified; tunable to 45–60+ TFLOPS |

## 2. Inkling-2-Mini Memory Footprint (`--mode nvfp4`, zero weight spill)

| Component | Bytes | Placement |
|---|---:|---|
| MoE Experts L0–19 (native NVFP4) | `11.466 GB` | VRAM (`VRAMArena` multi-slab `<=1.75 GiB`) |
| MoE Experts L20–23 (MXFP8 → NVFP4) | `2.293 GB` | VRAM |
| Dense Attention + Router + Controllers + Norms + SConv | `0.426 GB` | VRAM |
| LM Head (`unembed`, 200,058 × 2304 NVFP4) | `0.261 GB` | VRAM |
| DSpark MTP (3 layers + Markov + 32K/full draft head) | `~0.240 GB` | VRAM |
| BF16 Embeddings (`201,024 × 2304 × 2`) | `0.926 GB` | **Zero-copy host-pinned DDR5** (`4.6 KB/tok` = `0.18 µs`) |
| **Total Resident VRAM Weights** | **`~13.70 GiB`** | **0.0 GiB weight spill inside 15.98 GiB VRAM!** |
| Local SWA KV (`40 visits × 1280 slots BF16`) + scratch | `~0.35 GiB` | VRAM |
| Global Q4 KV (`8 global visits × 1,088 B/tok/visit`, head-major) | `~0.57 GiB` @64K, `~1.14 GiB` @128K | VRAM (kept there by the DDR5 expert tier, see §4) |

## 3. Decode & Speculative Roofline (`--mode nvfp4`, `2,800.3 MB/tok` at `B=1`)

| Regime | Verify `B` | Est. MB / Accepted Tok | Raw BW Ceiling (`562 GB/s`) | Practical Target (`squid7800xt`) |
|---|---:|---:|---:|---:|
| Baseline (`unsloth` GGUF `UD-IQ4_XS` w/ spill) | 1 | — | — | **24–25 tok/s** *(baseline)* |
| Autoregressive (`mtp=0`, 100% VRAM NVFP4) | 1 | `2,800 MB` | **200.7 tok/s** | **85–110 tok/s** |
| DSpark Chat (`mtp=3..7`) | 2..4 | `~2,100 MB` | **268 tok/s** | **115–150 tok/s** |
| Adaptive `{2,4,8}` DSpark Code | 2..8 | `~1,857 MB` | **303 tok/s** | **165–220 tok/s** |
| Hybrid Prompt-Lookup RAG/Copy | 8 | `~1,450 MB` | **388 tok/s** | **220–280 tok/s** |

## 4. Real VRAM Budget on a Windows Desktop & the DDR5 Expert Tier

- With a monitor attached to the 7800 XT, the Windows desktop compositor holds **~1.6–1.8 GiB** of VRAM. Measured allocatable VRAM for this process: **~14.1–14.4 GiB** (`VK_EXT_memory_budget` reported up to 15.05 GiB, so the engine probes by allocate-until-fail).
- Weights need 13.70 GiB, so at `--ctx ≥ 32K` something must live in pinned DDR5 (PCIe 4.0 x16, ~25 GB/s peak, ~13–20 GB/s achieved by the MoE kernels).
- Per token, the engine reads **all** of the KV cache but only **8/128 experts per layer**:
  - 1 GiB of KV in DDR5 → ~1 GiB over PCIe per token at full context (≈40+ ms).
  - 1 GiB of experts in DDR5 (≈9–10 experts/layer) → ≈`24 × 8 × (9.4/128) × 4.48 MB ≈ 63 MB` per token (≈4–5 ms measured).
- `--expert-tier auto` therefore spills the *tail* experts of every bank (uniform usage is expected thanks to the router's aux-free balancing bias) just far enough to keep the KV cache in VRAM: 1 / 4 / 9–10 experts per layer at 32K / 64K / 128K.

## 5. Long-Context Attention Roofline (after Exp 10)

| pos | Global Q4 KV read / token | Roofline @532 GB/s | Measured attention growth vs `pos=4096` |
|---:|---:|---:|---:|
| 65,000 | 566 MB | 1.06 ms | ~3.0 ms |
| 130,000 | 1.13 GB | 2.13 ms | ~3.4 ms |

Remaining long-context cost is dominated by the DDR5 expert tier (≈0.46 ms per spilled expert-per-layer at B=1, more under MTP verify where more distinct experts are touched), not by attention.
