// NVFP4 block-scale search, shared by the load-time packers (pack_bf16_nvfp4, pack_mxfp8_nvfp4).
// Include after common.glsl.
//
// The naive scale (block max / 6) puts the largest value on E2M1's +-6, leaving a wide [4, 6]
// gap where few values land. Trying smaller divisors down to 4 (larger scales, mapping block max
// toward +-4 for a denser grid over [0, 4]) plus neighbouring E4M3 codes (including dc=-1 on 6.0,
// which clips the outlier slightly) cuts squared reconstruction error by ~26.5% on real weights,
// within ~0.2% of exhaustive E4M3 search. Ties keep the naive scale.
// Same format and bytes; only the chosen scale code changes, so runtime kernels are untouched.

const uint SCALE_SEARCH_N_DIVS = 5u;
const float SCALE_SEARCH_DIVS[5] = float[5](6.0, 5.5, 5.0, 4.5, 4.0);

float nvfp4_block_sq_err(float vals[16], float s_eff) {
    float err = 0.0;
    if (s_eff <= 1e-30) {
        for (uint i = 0u; i < 16u; ++i) err = fma(vals[i], vals[i], err);
        return err;
    }
    float inv_s = 1.0 / s_eff;
    for (uint i = 0u; i < 16u; ++i) {
        float d = float(E2M1_LUT[quantize_e2m1(vals[i] * inv_s)]) * s_eff - vals[i];
        err = fma(d, d, err);
    }
    return err;
}

// Returns the E4M3 scale code for a block; s_eff = decode_e4m3(code) * gscale.
uint nvfp4_search_block_scale(float vals[16], float bmax, float gscale) {
    uint best = encode_e4m3(min(bmax / (6.0 * gscale), 448.0));
    float best_err = nvfp4_block_sq_err(vals, decode_e4m3(best) * gscale);
    if (bmax <= 0.0) return best;
    for (uint di = 0u; di < SCALE_SEARCH_N_DIVS; ++di) {
        int c0 = int(encode_e4m3(min(bmax / (SCALE_SEARCH_DIVS[di] * gscale), 448.0)));
        for (int dc = -1; dc <= 1; ++dc) {
            uint c = uint(clamp(c0 + dc, 1, 0x7E)); // positive, finite E4M3 codes
            float err = nvfp4_block_sq_err(vals, decode_e4m3(c) * gscale);
            if (err < best_err) {
                best_err = err;
                best = c;
            }
        }
    }
    return best;
}
