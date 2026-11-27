// Shared GLSL 460 helpers for squid7800xt (RDNA 3 gfx1101, Vulkan 1.4, BDA).

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(buffer_reference, scalar, buffer_reference_align = 1) buffer U8Buf { uint8_t data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer U16Buf { uint16_t data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer F16Buf { float16_t data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F16Vec2Buf { f16vec2 data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer F16Vec4Buf { f16vec4 data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32Buf { uint data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer I32Buf { int data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32Buf { float data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer Uvec2Buf { uvec2 data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer U64Buf { uint64_t data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer Uvec4Buf { uvec4 data[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer Vec4Buf { vec4 data[]; };

// Architecture constants (mirrored from src/constants.zig)
const uint HIDDEN = 2304u;
const uint NUM_HEADS = 24u;
const uint NUM_KV_HEADS = 8u;
const uint HEAD_DIM = 128u;
const uint Q_DIM = 3072u;
const uint KV_DIM = 1024u;
const uint GQA_RATIO = 3u;
const uint D_REL = 16u;
const uint R_DIM = 384u;
const uint QRGS_DIM = 6552u;
const uint REL_EXTENT = 1024u;
const uint SWA_WINDOW = 1024u;
const uint NUM_EXPERTS = 128u;
const uint TOP_K = 8u;
const uint INTERMEDIATE = 1152u;
const uint GATE_RANK = 64u;
const uint VOCAB_UNPADDED = 200058u;
const uint MTP_INTERMEDIATE = 9216u;
const uint MTP_MARKOV_RANK = 256u;
// Tiled split attention (attn_tile.comp / attn_merge_gate.comp / dspark_attn.comp).
// One Wave64 per split; each split walks 16-token tiles. The split layout is a pure
// function of the visible token range so producer and merge kernels agree exactly.
const uint ATTN_TILE = 16u;
const uint ATTN_MIN_TILES = 4u;     // >= 64 tokens per split (amortizes Q staging)
const uint ATTN_SPLIT_MAX = 128u;   // decode / verify / DSpark split budget per KV head
const uint KV_Q4_SCALES_PER_TOK = 64u;

const float RMS_EPS = 1e-6;
const float QK_CLIP = 22.627416997969522;
const float REL_CLIP = 8.0;
const int   LOG_SCALE_FLOOR = 8192;
const float LOG_SCALE_ALPHA = 0.1;
const float ROUTE_SCALE = 8.0;
const float GATE_SOFTCAP = 4.0;
const float UP_SOFTCAP = 25.0;
const float FINAL_SOFTCAP = 30.0;
const float INV_SQRT_HEAD_DIM = 0.08838834764831845;
const float INV_SQRT_HIDDEN = 0.020833333333333332;
const float INV_SQRT_5 = 0.4472135954999579;
const float INV_SQRT_9 = 0.3333333333333333;

const float16_t E2M1_LUT[16] = float16_t[16](
    float16_t(0.0),  float16_t(0.5),  float16_t(1.0),  float16_t(1.5),
    float16_t(2.0),  float16_t(3.0),  float16_t(4.0),  float16_t(6.0),
    float16_t(-0.0), float16_t(-0.5), float16_t(-1.0), float16_t(-1.5),
    float16_t(-2.0), float16_t(-3.0), float16_t(-4.0), float16_t(-6.0)
);

float bf16_to_f32(uint16_t bits) {
    return uintBitsToFloat(uint(bits) << 16u);
}

uint16_t f32_to_bf16(float v) {
    uint u = floatBitsToUint(v);
    uint lsb = (u >> 16u) & 1u;
    u += 0x7FFFu + lsb;
    return uint16_t(u >> 16u);
}

float round_bf16(float v) {
    return bf16_to_f32(f32_to_bf16(v));
}

float decode_e4m3(uint code) {
    uint sign = (code >> 7u) & 1u;
    uint exp = (code >> 3u) & 0xFu;
    uint man = code & 0x7u;
    float val;
    if (exp == 0u) {
        val = float(man) * (1.0 / 512.0);
    } else if (exp == 15u && man == 7u) {
        val = 0.0;
    } else {
        uint f32_bits = ((exp + 120u) << 23u) | (man << 20u);
        val = uintBitsToFloat(f32_bits);
    }
    return (sign != 0u) ? -val : val;
}

uint encode_e4m3(float x) {
    uint sign = (x < 0.0) ? 0x80u : 0u;
    float a = min(abs(x), 448.0);
    if (a < 0.015625) { // < 2^-6 -> subnormal
        uint man = clamp(uint(round(a * 512.0)), 0u, 8u);
        if (man == 8u) return sign | 0x08u;
        return sign | man;
    }
    uint bits = floatBitsToUint(a);
    int f_exp = int((bits >> 23u) & 0xFFu) - 127;
    uint f_man = bits & 0x7FFFFFu;
    uint rounded = (f_man + 0x80000u) >> 20u;
    if (rounded >= 8u) {
        rounded = 0u;
        f_exp += 1;
    }
    int e4_exp = f_exp + 7;
    if (e4_exp >= 15) {
        e4_exp = 15;
        if (rounded > 6u) rounded = 6u;
    }
    return sign | (uint(e4_exp) << 3u) | rounded;
}

float decode_ue8m0(uint e) {
    uint bits = (e == 0u) ? 0x00400000u : ((e == 255u) ? 0x7fc00000u : (e << 23u));
    return uintBitsToFloat(bits);
}

uint quantize_e2m1(float v) {
    uint sign = (v < 0.0) ? 8u : 0u;
    float a = abs(v);
    uint code;
    if      (a < 0.25) code = 0u;
    else if (a < 0.75) code = 1u;
    else if (a < 1.25) code = 2u;
    else if (a < 1.75) code = 3u;
    else if (a < 2.50) code = 4u;
    else if (a < 3.50) code = 5u;
    else if (a < 5.00) code = 6u;
    else               code = 7u;
    return sign | code;
}

float fast_sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Returns (tiles_per_split, n_splits) for visible tokens [lo, hi).
uvec2 attn_split_layout(int lo, int hi, uint n_split_max) {
    if (hi <= lo) return uvec2(1u, 0u);
    uint ntiles = (uint(hi - lo) + ATTN_TILE - 1u) / ATTN_TILE;
    uint tps = max(ATTN_MIN_TILES, (ntiles + n_split_max - 1u) / n_split_max);
    return uvec2(tps, (ntiles + tps - 1u) / tps);
}

// Q4_1 KV cache is HEAD-MAJOR: data [kvh][cap][64 B], scales [kvh][cap][4 x f16vec2(lo, sc) = 16 B].
// Group-32 asymmetric Q4_1: lane j in attn_tile owns 32-dim slice j and reads 1 u32 at + j*4!
uint64_t q4_data_addr(uint64_t base, uint cap, uint slot, uint kvh) {
    return base + (uint64_t(kvh) * uint64_t(cap) + uint64_t(slot)) * 64ul;
}
uint64_t q4_scale_addr(uint64_t base, uint cap, uint slot, uint kvh) {
    return base + (uint64_t(kvh) * uint64_t(cap) + uint64_t(slot)) * 16ul;
}
