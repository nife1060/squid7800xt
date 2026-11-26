// TigerStyle Inkling-2-Mini Vulkan 1.4 Inference Engine for AMD Radeon RX 7800 XT.
// Pre-records reusable VkCommandBuffers (Vulkan Graphs) for B=1, B=2, B=4, B=8, and DSpark MTP,
// with Adaptive {2,4,8} Markov Confidence Verify + Hybrid O(1) Prompt-Lookup Drafter!

const std = @import("std");
const constants = @import("constants.zig");
const kernels = @import("kernels.zig");
const tiger = @import("tiger.zig");
const vk = @import("vulkan.zig");
const vram_arena = @import("vram_arena.zig");
const weights_mod = @import("weights.zig");

pub const TokenCallbackFn = *const fn (context: *anyopaque, token_id: i32) bool;

pub const SampleParams = struct {
    temperature: f32 = 0.0,
    seed: u32 = 42,
};

pub const GenerateStats = struct {
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    prefill_ms: f64 = 0.0,
    decode_ms: f64 = 0.0,
    spec_steps: u32 = 0,
    lookup_steps: u32 = 0,
    total_drafts: u32 = 0,
    total_accepted: u32 = 0,
    last_token_id: i32 = -1,
};

/// Zero-malloc O(1) n-gram prompt-lookup drafter (ported from squid5090).
const PromptLookupIndex = struct {
    const hist_cap: usize = 8_192;
    const hist_mask: usize = hist_cap - 1;
    const table_cap: usize = 16_384;
    const table_mask: usize = table_cap - 1;
    const min_match_len: u32 = 6;

    hist: [hist_cap]i32 = @splat(0),
    count: u32 = 0,
    gen: u16 = 0,
    table_hash: [table_cap]u64 = @splat(0),
    table_pos0: [table_cap]u32 = @splat(0),
    table_pos1: [table_cap]u32 = @splat(0),
    table_gen: [table_cap]u16 = @splat(0),

    fn reset(self: *PromptLookupIndex) void {
        self.count = 0;
        self.gen +%= 1;
        if (self.gen == 0) {
            @memset(&self.table_gen, 0);
            self.gen = 1;
        }
    }

    inline fn hash4(a: i32, b: i32, c: i32, d: i32) u64 {
        var h: u64 = @as(u64, @bitCast(@as(i64, a))) *% 0x9E3779B97F4A7C15;
        h = (h ^ @as(u64, @bitCast(@as(i64, b)))) *% 0xBF58476D1CE4E5B9;
        h = (h ^ @as(u64, @bitCast(@as(i64, c)))) *% 0x94D049BB133111EB;
        h = (h ^ @as(u64, @bitCast(@as(i64, d)))) *% 0x9E3779B97F4A7C15;
        return h ^ (h >> 31);
    }

    inline fn at(self: *const PromptLookupIndex, abs_idx: u32) i32 {
        return self.hist[abs_idx & hist_mask];
    }

    fn push(self: *PromptLookupIndex, tok: i32) void {
        const idx = self.count;
        self.hist[idx & hist_mask] = tok;
        self.count = idx + 1;
        if (self.count >= 4) {
            const h = hash4(self.at(self.count - 4), self.at(self.count - 3), self.at(self.count - 2), self.at(self.count - 1));
            const slot: usize = @intCast(h & table_mask);
            if (self.table_gen[slot] == self.gen and self.table_hash[slot] == h) {
                self.table_pos1[slot] = self.table_pos0[slot];
                self.table_pos0[slot] = self.count;
            } else {
                self.table_hash[slot] = h;
                self.table_pos0[slot] = self.count;
                self.table_pos1[slot] = 0;
                self.table_gen[slot] = self.gen;
            }
        }
    }

    fn match(self: *const PromptLookupIndex, out_drafts: *[constants.mtp_draft_count_max]i32) bool {
        const n = self.count;
        if (n < min_match_len + constants.mtp_draft_count_max + 1) return false;
        const h = hash4(self.at(n - 4), self.at(n - 3), self.at(n - 2), self.at(n - 1));
        const slot: usize = @intCast(h & table_mask);
        if (self.table_gen[slot] != self.gen or self.table_hash[slot] != h) return false;

        const cand: u32 = if (self.table_pos0[slot] == n) self.table_pos1[slot] else self.table_pos0[slot];
        if (cand < min_match_len) return false;
        if (cand + constants.mtp_draft_count_max >= n) return false;
        const win_lo: u32 = if (n > hist_cap) (n - @as(u32, @intCast(hist_cap))) else 0;
        if (cand - min_match_len < win_lo) return false;

        var k: u32 = 1;
        while (k <= min_match_len) : (k += 1) {
            if (self.at(cand - k) != self.at(n - k)) return false;
        }
        var j: u32 = 0;
        while (j < constants.mtp_draft_count_max) : (j += 1) {
            out_drafts[j] = self.at(cand + j);
        }
        return true;
    }
};

// Host-visible coherent control block (zero-copy CPU<->GPU token, confidence, sampling & position handoff!)
const ControlBlock = extern struct {
    pos: i32 = 0,
    num_acc: i32 = 0,
    anchor: i32 = 0,
    pad0: i32 = 0,
    tokens: [8]i32 = @splat(0),
    out_tokens: [8]i32 = @splat(0),
    drafts: [8]i32 = @splat(0),
    accepted: [8]i32 = @splat(0),
    draft_conf: [8]f32 = @splat(0.0),
    sample_inv_temp: f32 = 0.0,
    sample_seed: u32 = 42,
    sample_pad0: u32 = 0,
    sample_pad1: u32 = 0,
    pos_sub: [8]i32 = @splat(0),
    tokens_b64: [64]i32 = @splat(0),
};

const PrefixCheckpoint = struct {
    valid: bool = false,
    pos: u32 = 0,
    host_ram: []u8 = &.{},
};

const PrefixCache = struct {
    enabled: bool = true,
    session_tokens: []i32 = &.{},
    session_len: u32 = 0,
    session_use_mtp: bool = false,
    gpu_live_at_pos: i32 = -1,
    stage_buf: vk.GpuBuffer = .{},
    checkpoints: [3]PrefixCheckpoint = .{ .{}, .{}, .{} },
};

pub const Engine = struct {
    ctx: *const vk.Context = undefined,
    weights: *const weights_mod.ModelWeights = undefined,
    pipes: kernels.Pipelines = .{},
    arena: vram_arena.VRAMArena = .{},
    ctrl_buf: vk.GpuBuffer = .{},
    ctrl: *volatile ControlBlock = undefined,
    ctrl_desc_buf: vk.GpuBuffer = .{},
    local_state_buf: vk.GpuBuffer = .{},
    p64_buf: vk.GpuBuffer = .{},
    prefix_cache: PrefixCache = .{},
    sample_params: SampleParams = .{},

    context_tokens_max: u32 = 65536,
    kv_format: constants.KVFormat = .q4,
    current_position: i32 = 0,
    spec_threshold: f32 = 0.80,
    use_prompt_lookup: bool = true,
    debug_mtp: bool = false,
    schedule: [constants.visit_count]constants.VisitInfo = undefined,
    prompt_lookup: PromptLookupIndex = .{},

    // Pre-recorded reusable VkCommandBuffers (Vulkan Graphs!)
    cmd_b1_decode: vk.VkCommandBuffer = null,
    cmd_b1_prefill: vk.VkCommandBuffer = null,
    cmd_b8_prefill: vk.VkCommandBuffer = null,
    cmd_b8_prefill_lm: vk.VkCommandBuffer = null,
    cmd_b64_prefill: vk.VkCommandBuffer = null,
    // MTP-enabled prefill & speculative graphs
    cmd_b1_prefill_mtp: vk.VkCommandBuffer = null,
    cmd_b1_decode_mtp: vk.VkCommandBuffer = null,
    cmd_b8_prefill_mtp: vk.VkCommandBuffer = null,
    cmd_b8_prefill_lm_mtp: vk.VkCommandBuffer = null,
    cmd_b64_prefill_mtp: vk.VkCommandBuffer = null,
    cmd_dspark_draft: [constants.mtp_draft_count_max + 1]vk.VkCommandBuffer = @splat(null),
    cmd_verify: [constants.mtp_block_size_max + 1]vk.VkCommandBuffer = @splat(null),

    // B=64 WMMA prefill scratch addresses (suballocated inside p64_buf)
    p_hidden: u64 = 0,
    p_prelude: u64 = 0,
    p_partial: u64 = 0,
    p_blocks: u64 = 0,
    p_saved: u64 = 0,
    p_prior_last: u64 = 0,
    p_attn_in: [3]u64 = @splat(0),
    p_attn_in_h: [3]u64 = @splat(0),
    p_qrgs: u64 = 0,
    p_k_proj: u64 = 0,
    p_v_proj: u64 = 0,
    p_attn_gated_h: u64 = 0,
    p_o_proj: u64 = 0,
    p_residual: u64 = 0,
    p_mlp_in: u64 = 0,
    p_mlp_normed: u64 = 0,
    p_mlp_normed_h: u64 = 0,
    p_router_logits: u64 = 0,
    p_exp_table64: u64 = 0,
    p_moe_act_h: u64 = 0,
    p_moe_down: u64 = 0,
    p_taps: u64 = 0,

    // Per-visit KV caches and sconv rings
    kv_cap: [constants.visit_count]u32 = @splat(0),
    k_cache: [constants.visit_count]u64 = @splat(0),
    v_cache: [constants.visit_count]u64 = @splat(0),
    k_scale: [constants.visit_count]u64 = @splat(0),
    v_scale: [constants.visit_count]u64 = @splat(0),
    k_ring: [constants.visit_count]u64 = @splat(0),
    v_ring: [constants.visit_count]u64 = @splat(0),
    attn_ring: [constants.visit_count]u64 = @splat(0),
    mlp_ring: [constants.visit_count]u64 = @splat(0),

    // DSpark KV caches & workspace
    ds_kv_cap: [constants.mtp_layer_count]u32 = .{ 0, 0, 0 },
    ds_k_cache: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 },
    ds_v_cache: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 },
    ds_k_scale: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 },
    ds_v_scale: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 },
    ds_f_ring: u64 = 0,
    ds_mix_h: u64 = 0,
    ds_kv: u64 = 0,
    ds_ws: u64 = 0,

    // Activation scratch (sized for B = 8)
    split_buf: u64 = 0,
    hidden: u64 = 0,
    prelude: u64 = 0,
    partial: u64 = 0,
    blocks: u64 = 0,
    saved: u64 = 0,
    prior_last: u64 = 0,
    attn_in: [3]u64 = @splat(0),
    attn_in_h: [3]u64 = @splat(0),
    qrgs: u64 = 0,
    k_proj: u64 = 0,
    v_proj: u64 = 0,
    attn_gated: u64 = 0,
    attn_gated_h: u64 = 0,
    o_proj: u64 = 0,
    residual: u64 = 0,
    mlp_in: u64 = 0,
    mlp_normed: u64 = 0,
    mlp_normed_h: u64 = 0,
    router_logits: u64 = 0,
    route_ids: u64 = 0,
    route_w: u64 = 0,
    exp_groups: u64 = 0,
    moe_act_h: u64 = 0,
    moe_down: u64 = 0,
    taps: u64 = 0,
    lm_blk_val: u64 = 0,
    lm_blk_idx: u64 = 0,

    pub fn init(
        self: *Engine,
        ctx: *const vk.Context,
        weights: *const weights_mod.ModelWeights,
        max_ctx: u32,
        kv_format: constants.KVFormat,
    ) void {
        self.ctx = ctx;
        self.weights = weights;
        self.context_tokens_max = max_ctx;
        self.kv_format = kv_format;
        self.schedule = constants.build_visit_schedule();
        self.pipes = kernels.Pipelines.init(ctx);

        const host_mt = if (ctx.mem_type_host_wc != std.math.maxInt(u32)) ctx.mem_type_host_wc else ctx.mem_type_host_cached;
        self.ctrl_buf = ctx.create_host_buffer(4096, true, true);
        @memset(self.ctrl_buf.mapped.?[0..4096], 0);
        self.ctrl = @ptrCast(@alignCast(self.ctrl_buf.mapped.?));

        // Build static 96-entry ControllerDesc table (24 layers x 4 streams = 7680 B) in ReBAR VRAM (with host fallback)!
        const desc_bytes: u64 = @as(u64, constants.layer_count) * 4 * @sizeOf(kernels.CtrlDesc);
        self.ctrl_desc_buf = (if (ctx.mem_type_vram_rebar != std.math.maxInt(u32)) ctx.try_create_buffer(desc_bytes, ctx.mem_type_vram_rebar, true) else null) orelse
            ctx.create_host_buffer(desc_bytes, false, true);
        const descs: [*]kernels.CtrlDesc = @ptrCast(@alignCast(self.ctrl_desc_buf.mapped.?));
        @memset(descs[0 .. constants.layer_count * 4], .{});
        var cl: u32 = constants.loop_start_layer;
        while (cl < constants.loop_end_layer) : (cl += 1) {
            const lw = &weights.layers[cl];
            const loc: u64 = cl - constants.loop_start_layer;
            const cp_qs = [4]u64{
                weights.cross_pass_queries + (2 * loc) * constants.hidden_dim * 4,
                weights.cross_pass_kv_queries + (2 * loc) * constants.hidden_dim * 4,
                weights.cross_pass_kv_queries + (2 * loc + 1) * constants.hidden_dim * 4,
                weights.cross_pass_queries + (2 * loc + 1) * constants.hidden_dim * 4,
            };
            const pnorms = [4]u64{
                if (lw.is_global) lw.q_in_norm else lw.attn_norm,
                if (lw.is_global) lw.k_in_norm else lw.attn_norm,
                if (lw.is_global) lw.v_in_norm else lw.attn_norm,
                lw.mlp_norm,
            };
            for (0..4) |cs| {
                const cw = &lw.ctrl[cs];
                descs[cl * 4 + cs] = .{
                    .cp_q_addr = cp_qs[cs],
                    .down_w_addr = cw.down_nvfp4.data,
                    .down_s_addr = cw.down_nvfp4.scales,
                    .forget_w_addr = cw.forget_nvfp4.data,
                    .forget_s_addr = cw.forget_nvfp4.scales,
                    .gain_w_addr = cw.gain_nvfp4.data,
                    .gain_s_addr = cw.gain_nvfp4.scales,
                    .post_norm_addr = pnorms[cs],
                    .down_gs = cw.down_nvfp4.global_scale,
                    .forget_gs = cw.forget_nvfp4.global_scale,
                    .gain_gs = cw.gain_nvfp4.global_scale,
                    .pad = 0,
                };
            }
        }

        self.arena = vram_arena.VRAMArena.init(ctx, ctx.mem_type_vram_only, false);
        self.arena.slab_size_max = 64 * 1024 * 1024;
        self.arena.fallback_mem_type_idx = host_mt;
        // Reuse the weight arena's unused slab tails (~0.2 GiB) before reserving new VRAM.
        self.arena.donor = @constCast(&weights.arena);

        // 1. Hot activation scratch (always in VRAM)
        const Bmax: u64 = constants.mtp_block_size_max;
        self.split_buf = self.arena.alloc_addr(constants.split_buffer_floats * 4);
        self.hidden = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.prelude = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.partial = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.blocks = self.arena.alloc_addr(Bmax * 4 * constants.hidden_dim * 4);
        self.saved = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.prior_last = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        var s: usize = 0;
        while (s < 3) : (s += 1) {
            self.attn_in[s] = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
            self.attn_in_h[s] = self.arena.alloc_addr(Bmax * constants.hidden_dim * 2);
        }
        self.qrgs = self.arena.alloc_addr(Bmax * constants.qrgs_out_dim * 4);
        self.k_proj = self.arena.alloc_addr(Bmax * constants.kv_dim * 4);
        self.v_proj = self.arena.alloc_addr(Bmax * constants.kv_dim * 4);
        self.attn_gated = self.arena.alloc_addr(Bmax * constants.query_dim * 4);
        self.attn_gated_h = self.arena.alloc_addr(Bmax * constants.query_dim * 2);
        self.o_proj = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.residual = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.mlp_in = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.mlp_normed = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.mlp_normed_h = self.arena.alloc_addr(Bmax * constants.hidden_dim * 2);
        self.router_logits = self.arena.alloc_addr(Bmax * constants.expert_count * 4);
        self.route_ids = self.arena.alloc_addr(Bmax * constants.top_k_experts * 4);
        self.route_w = self.arena.alloc_addr(Bmax * constants.top_k_experts * 4);
        self.exp_groups = self.arena.alloc_addr(8192);
        self.moe_act_h = self.arena.alloc_addr(Bmax * constants.top_k_experts * constants.intermediate_dim * 2);
        self.moe_down = self.arena.alloc_addr(Bmax * constants.hidden_dim * 4);
        self.taps = self.arena.alloc_addr(4 * Bmax * constants.hidden_dim * 4);
        self.lm_blk_val = self.arena.alloc_addr(Bmax * 512 * 4);
        self.lm_blk_idx = self.arena.alloc_addr(Bmax * 512 * 4);

        // 2. DSpark scratch (hot in VRAM!)
        if (weights.dspark.loaded) {
            self.ds_mix_h = self.arena.alloc_addr(3 * Bmax * constants.hidden_dim * 2);
            self.ds_kv = self.arena.alloc_addr(Bmax * 2 * constants.kv_dim * 4);
            const base_logits_bytes: u64 = 7 * @as(u64, weights.dspark.draft_vocab_size) * 4;
            self.ds_ws = self.arena.alloc_addr(4 * 1024 * 1024 + base_logits_bytes);
        }

        // 2b. B=64 WMMA Prefill scratch (~18 MiB contiguous buffer in VRAM with DDR5 fallback)
        if (ctx.has_coop_matrix) {
            const B64: u64 = 64;
            var p_off: u64 = 0;
            const sub = struct {
                fn f(cur: *u64, bytes: u64) u64 {
                    const r = cur.*;
                    cur.* += tiger.align_up_u64(bytes, 256);
                    return r;
                }
            }.f;
            const o_hidden = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_prelude = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_partial = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_blocks = sub(&p_off, B64 * 4 * constants.hidden_dim * 4);
            const o_saved = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_prior_last = sub(&p_off, B64 * constants.hidden_dim * 4);
            var o_ai: [3]u64 = undefined;
            var o_aih: [3]u64 = undefined;
            for (0..3) |si| {
                o_ai[si] = sub(&p_off, B64 * constants.hidden_dim * 4);
                o_aih[si] = sub(&p_off, B64 * constants.hidden_dim * 2);
            }
            const o_qrgs = sub(&p_off, B64 * constants.qrgs_out_dim * 4);
            const o_k = sub(&p_off, B64 * constants.kv_dim * 4);
            const o_v = sub(&p_off, B64 * constants.kv_dim * 4);
            const o_ag_h = sub(&p_off, B64 * constants.query_dim * 2);
            const o_oproj = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_res = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_mlpin = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_mlpn = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_mlpnh = sub(&p_off, B64 * constants.hidden_dim * 2);
            const o_rlog = sub(&p_off, B64 * constants.expert_count * 4);
            const o_etab = sub(&p_off, 128 * 1024);
            const o_mact = sub(&p_off, B64 * constants.top_k_experts * constants.intermediate_dim * 2);
            const o_mdown = sub(&p_off, B64 * constants.hidden_dim * 4);
            const o_taps = sub(&p_off, 8 * 4 * 8 * constants.hidden_dim * 4);

            self.p64_buf = ctx.try_create_buffer(p_off, ctx.mem_type_vram_only, false) orelse
                (ctx.try_create_host_buffer(p_off, false, false) orelse .{});
            if (self.p64_buf.buffer != vk.VK_NULL_HANDLE) {
                const pb = self.p64_buf.address;
                self.p_hidden = pb + o_hidden;
                self.p_prelude = pb + o_prelude;
                self.p_partial = pb + o_partial;
                self.p_blocks = pb + o_blocks;
                self.p_saved = pb + o_saved;
                self.p_prior_last = pb + o_prior_last;
                for (0..3) |si| {
                    self.p_attn_in[si] = pb + o_ai[si];
                    self.p_attn_in_h[si] = pb + o_aih[si];
                }
                self.p_qrgs = pb + o_qrgs;
                self.p_k_proj = pb + o_k;
                self.p_v_proj = pb + o_v;
                self.p_attn_gated_h = pb + o_ag_h;
                self.p_o_proj = pb + o_oproj;
                self.p_residual = pb + o_res;
                self.p_mlp_in = pb + o_mlpin;
                self.p_mlp_normed = pb + o_mlpn;
                self.p_mlp_normed_h = pb + o_mlpnh;
                self.p_router_logits = pb + o_rlog;
                self.p_exp_table64 = pb + o_etab;
                self.p_moe_act_h = pb + o_mact;
                self.p_moe_down = pb + o_mdown;
                self.p_taps = pb + o_taps;
            }
        }

        // 3. Contiguous Local State Buffer (ALL local SWA KV + sconv rings + DSpark local KV + ds_f_ring!)
        // Enables 1-copy (~8ms) DDR5 Multi-Checkpoint Prefix KV Cache save/restore!
        {
            var l_off: u64 = 0;
            const sub = struct {
                fn f(cur: *u64, bytes: u64) u64 {
                    const r = cur.*;
                    cur.* += tiger.align_up_u64(bytes, 256);
                    return r;
                }
            }.f;

            var o_ds_f: u64 = 0;
            var o_ds_k: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 };
            var o_ds_v: [constants.mtp_layer_count]u64 = .{ 0, 0, 0 };
            if (weights.dspark.loaded) {
                o_ds_f = sub(&l_off, constants.sconv_ring_slots * constants.hidden_dim * 4);
                var ml: u32 = 0;
                while (ml < constants.mtp_layer_count) : (ml += 1) {
                    if (constants.is_mtp_local_layer(ml)) {
                        const cap: u32 = 1040;
                        self.ds_kv_cap[ml] = cap;
                        const bytes: u64 = @as(u64, cap) * constants.kv_dim * 2;
                        o_ds_k[ml] = sub(&l_off, bytes);
                        o_ds_v[ml] = sub(&l_off, bytes);
                    }
                }
            }

            var o_kr: [constants.visit_count]u64 = @splat(0);
            var o_vr: [constants.visit_count]u64 = @splat(0);
            var o_ar: [constants.visit_count]u64 = @splat(0);
            var o_mr: [constants.visit_count]u64 = @splat(0);
            var o_lk: [constants.visit_count]u64 = @splat(0);
            var o_lv: [constants.visit_count]u64 = @splat(0);
            var vi: u32 = 0;
            while (vi < constants.visit_count) : (vi += 1) {
                const kv_ring_b: u64 = constants.sconv_ring_slots * constants.kv_dim * 4;
                const h_ring_b: u64 = constants.sconv_ring_slots * constants.hidden_dim * 4;
                o_kr[vi] = sub(&l_off, kv_ring_b);
                o_vr[vi] = sub(&l_off, kv_ring_b);
                o_ar[vi] = sub(&l_off, h_ring_b);
                o_mr[vi] = sub(&l_off, h_ring_b);
                if (!self.schedule[vi].is_global) {
                    const cap: u32 = 1040;
                    self.kv_cap[vi] = cap;
                    const bytes: u64 = @as(u64, cap) * constants.kv_dim * 2;
                    o_lk[vi] = sub(&l_off, bytes);
                    o_lv[vi] = sub(&l_off, bytes);
                }
            }

            self.local_state_buf = ctx.try_create_buffer(l_off, ctx.mem_type_vram_only, false) orelse ctx.create_host_buffer(l_off, false, false);
            const lb = self.local_state_buf.address;
            if (weights.dspark.loaded) {
                self.ds_f_ring = lb + o_ds_f;
                var ml: u32 = 0;
                while (ml < constants.mtp_layer_count) : (ml += 1) {
                    if (constants.is_mtp_local_layer(ml)) {
                        self.ds_k_cache[ml] = lb + o_ds_k[ml];
                        self.ds_v_cache[ml] = lb + o_ds_v[ml];
                    }
                }
            }
            vi = 0;
            while (vi < constants.visit_count) : (vi += 1) {
                self.k_ring[vi] = lb + o_kr[vi];
                self.v_ring[vi] = lb + o_vr[vi];
                self.attn_ring[vi] = lb + o_ar[vi];
                self.mlp_ring[vi] = lb + o_mr[vi];
                if (!self.schedule[vi].is_global) {
                    self.k_cache[vi] = lb + o_lk[vi];
                    self.v_cache[vi] = lb + o_lv[vi];
                }
            }

            // Allocate 3 prefix-cache checkpoints in plain CPU DDR5 (ZERO Vulkan WDDM commit!) + 1x 16 MiB DMA staging buffer
            self.prefix_cache.session_tokens = tiger.alloc_host_zeroed(i32, max_ctx);
            self.prefix_cache.stage_buf = ctx.try_create_host_buffer(16 * 1024 * 1024, true, true) orelse .{};
            for (&self.prefix_cache.checkpoints) |*ckpt| {
                ckpt.host_ram = tiger.alloc_host_zeroed(u8, @intCast(l_off));
                ckpt.valid = false;
                ckpt.pos = 0;
            }
        }

        // 4. Global KV caches (target 8 global visits + DSpark global layer 1)
        if (weights.dspark.loaded) {
            const cap: u32 = @min(max_ctx, 32768);
            self.ds_kv_cap[1] = cap;
            const data_b: u64 = @as(u64, cap) * (constants.kv_dim / 2);
            const scale_b: u64 = @as(u64, cap) * constants.kv_q4_scales_per_token * 2;
            self.ds_k_cache[1] = self.arena.alloc_addr(data_b);
            self.ds_v_cache[1] = self.arena.alloc_addr(data_b);
            self.ds_k_scale[1] = self.arena.alloc_addr(scale_b);
            self.ds_v_scale[1] = self.arena.alloc_addr(scale_b);
        }
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            if (!self.schedule[vi].is_global) continue;
            const cap: u32 = max_ctx;
            self.kv_cap[vi] = cap;
            switch (kv_format) {
                .bf16 => {
                    const bytes: u64 = @as(u64, cap) * constants.kv_dim * 2;
                    self.k_cache[vi] = self.arena.alloc_addr(bytes);
                    self.v_cache[vi] = self.arena.alloc_addr(bytes);
                },
                .q4 => {
                    const data_bytes: u64 = @as(u64, cap) * (constants.kv_dim / 2);
                    const scale_bytes: u64 = @as(u64, cap) * constants.kv_q4_scales_per_token * 2;
                    self.k_cache[vi] = self.arena.alloc_addr(data_bytes);
                    self.v_cache[vi] = self.arena.alloc_addr(data_bytes);
                    self.k_scale[vi] = self.arena.alloc_addr(scale_bytes);
                    self.v_scale[vi] = self.arena.alloc_addr(scale_bytes);
                },
                .fp8 => unreachable,
            }
        }

        const local_in_vram: u64 = if (self.local_state_buf.memory_type_index == ctx.mem_type_vram_only) self.local_state_buf.size else 0;
        const p64_in_vram: u64 = if (self.p64_buf.buffer != vk.VK_NULL_HANDLE and self.p64_buf.memory_type_index == ctx.mem_type_vram_only) self.p64_buf.size else 0;
        const local_in_ddr5: u64 = if (local_in_vram == 0) self.local_state_buf.size else 0;
        const rt_vram_bytes = self.arena.total_reserved_bytes + local_in_vram + p64_in_vram;
        const rt_ddr5_bytes = self.arena.total_fallback_bytes + local_in_ddr5;
        const rt_vram_gib = @as(f64, @floatFromInt(rt_vram_bytes)) / (1024.0 * 1024.0 * 1024.0);
        const rt_ddr5_gib = @as(f64, @floatFromInt(rt_ddr5_bytes)) / (1024.0 * 1024.0 * 1024.0);
        const tot_vram_gib = @as(f64, @floatFromInt(rt_vram_bytes + weights.arena.total_reserved_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[engine] Runtime VRAM={d:.2} GiB + DDR5_KV={d:.2} GiB (ctx={d}, kv={s}, wmma_pf={s}, pfx_cache=3x{d}MiB) | Total VRAM={d:.2} / 15.98 GiB\n",
            .{
                rt_vram_gib,
                rt_ddr5_gib,
                max_ctx,
                kv_format.name(),
                if (ctx.has_coop_matrix) "B64" else "off",
                self.local_state_buf.size / (1024 * 1024),
                tot_vram_gib,
            },
        );

        // Pre-record reusable Vulkan Graphs!
        self.cmd_b1_decode = self.build_forward_cmd(1, false, true, true, false);
        self.cmd_b1_prefill = self.build_forward_cmd(1, false, false, false, false);
        self.cmd_b8_prefill = self.build_forward_cmd(8, false, false, false, false);
        self.cmd_b8_prefill_lm = self.build_forward_cmd(8, false, true, false, false);
        if (ctx.has_coop_matrix and self.p64_buf.buffer != vk.VK_NULL_HANDLE) {
            self.cmd_b64_prefill = self.build_prefill_b64_cmd(false);
        }

        if (weights.dspark.loaded) {
            self.cmd_b1_prefill_mtp = self.build_forward_cmd(1, true, false, false, false);
            self.cmd_b1_decode_mtp = self.build_forward_cmd(1, true, true, true, false);
            self.cmd_b8_prefill_mtp = self.build_forward_cmd(8, true, false, false, false);
            self.cmd_b8_prefill_lm_mtp = self.build_forward_cmd(8, true, true, false, false);
            if (ctx.has_coop_matrix and self.p64_buf.buffer != vk.VK_NULL_HANDLE) {
                self.cmd_b64_prefill_mtp = self.build_prefill_b64_cmd(true);
            }

            // Draft graphs for K = 3 and K = 7
            self.cmd_dspark_draft[3] = self.build_dspark_draft_cmd(3);
            self.cmd_dspark_draft[7] = self.build_dspark_draft_cmd(7);

            // Verify graphs for B = 2, 4, 8 (Adaptive {2, 4, 8} + Fixed B=4/B=8!)
            self.cmd_verify[2] = self.build_forward_cmd(2, true, true, false, true);
            self.cmd_verify[4] = self.build_forward_cmd(4, true, true, false, true);
            self.cmd_verify[8] = self.build_forward_cmd(8, true, true, false, true);
        }
    }

    pub fn deinit(self: *Engine) void {
        _ = self.ctx.vkDeviceWaitIdle(self.ctx.device);
        const singles = [_]vk.VkCommandBuffer{
            self.cmd_b1_decode,
            self.cmd_b1_prefill,
            self.cmd_b8_prefill,
            self.cmd_b8_prefill_lm,
            self.cmd_b64_prefill,
            self.cmd_b1_prefill_mtp,
            self.cmd_b1_decode_mtp,
            self.cmd_b8_prefill_mtp,
            self.cmd_b8_prefill_lm_mtp,
            self.cmd_b64_prefill_mtp,
        };
        for (singles) |c| {
            if (c != null) self.ctx.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &.{c});
        }
        for (self.cmd_dspark_draft) |c| {
            if (c != null) self.ctx.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &.{c});
        }
        for (self.cmd_verify) |c| {
            if (c != null) self.ctx.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &.{c});
        }
        if (self.prefix_cache.stage_buf.buffer != vk.VK_NULL_HANDLE) {
            self.ctx.destroy_buffer(&self.prefix_cache.stage_buf);
        }
        for (&self.prefix_cache.checkpoints) |*ckpt| {
            if (ckpt.host_ram.len > 0) {
                tiger.free_host_slice(u8, ckpt.host_ram);
                ckpt.host_ram = &.{};
            }
        }
        if (self.prefix_cache.session_tokens.len > 0) {
            tiger.free_host_slice(i32, self.prefix_cache.session_tokens);
            self.prefix_cache.session_tokens = &.{};
        }
        self.ctx.destroy_buffer(&self.local_state_buf);
        self.ctx.destroy_buffer(&self.p64_buf);
        self.arena.deinit();
        self.ctx.destroy_buffer(&self.ctrl_desc_buf);
        self.ctx.destroy_buffer(&self.ctrl_buf);
        self.pipes.deinit();
    }

    fn ctrl_addr(self: *const Engine, comptime field: []const u8) u64 {
        return self.ctrl_buf.address + @offsetOf(ControlBlock, field);
    }

    pub fn reset(self: *Engine) void {
        self.current_position = 0;
        self.ctrl.pos = 0;
        self.prompt_lookup.reset();
    }

    const StageId = enum(u8) {
        embed = 0,
        depth_norm = 1,
        controller = 2,
        qkvrg = 3,
        kv_sconv = 4,
        attn_prep = 5,
        attn_core = 6,
        w_o = 7,
        post_res = 8,
        router = 9,
        moe_up = 10,
        moe_down = 11,
        lm_head = 12,
        ds_commit = 13,
        ds_layers = 14,
        ds_unembed = 15,
        ds_markov = 16,
        count = 17,
    };

    const StageProfiler = struct {
        ctx: *const vk.Context,
        pool: vk.VkQueryPool,
        n_queries: u32 = 0,
        stage_ids: [2048]u8 = @splat(0),

        fn mark(self: *StageProfiler, cmd: vk.VkCommandBuffer, stage: StageId) void {
            if (self.n_queries >= 2046) return;
            self.stage_ids[self.n_queries] = @intFromEnum(stage);
            self.ctx.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.pool, self.n_queries);
            self.n_queries += 1;
        }

        fn begin(self: *StageProfiler, cmd: vk.VkCommandBuffer) void {
            self.n_queries = 0;
            self.ctx.vkCmdResetQueryPool(cmd, self.pool, 0, 2048);
            self.ctx.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.pool, 0);
            self.n_queries = 1;
        }
    };

    fn prof_mark(prof: ?*StageProfiler, cmd: vk.VkCommandBuffer, stage: StageId) void {
        if (prof) |pr| pr.mark(cmd, stage);
    }

    fn build_forward_cmd(self: *const Engine, B: u32, commit_dspark: bool, compute_lm_head: bool, advance_b1: bool, do_verify: bool) vk.VkCommandBuffer {
        const cmd = self.ctx.alloc_command_buffer();
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
        tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
        self.record_target_forward(cmd, B, commit_dspark, compute_lm_head, advance_b1, do_verify, null);
        tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
        return cmd;
    }

    fn build_dspark_draft_cmd(self: *const Engine, num_drafts: u32) vk.VkCommandBuffer {
        const cmd = self.ctx.alloc_command_buffer();
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
        tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
        self.record_dspark_draft(cmd, num_drafts, null);
        tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
        return cmd;
    }

    fn build_prefill_b64_cmd(self: *const Engine, commit_dspark: bool) vk.VkCommandBuffer {
        const cmd = self.ctx.alloc_command_buffer();
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
        tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
        self.record_prefill_b64(cmd, commit_dspark);
        tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
        return cmd;
    }

    fn record_dspark_commit_at(self: *const Engine, cmd: vk.VkCommandBuffer, taps_addr: u64, d_pos_addr: u64, B: u32) void {
        const dw = &self.weights.dspark;
        const p = &self.pipes;

        p.dispatch(cmd, p.dspark_commit, kernels.DsparkCommitPC{
            .a0 = taps_addr,
            .a1 = dw.fuse_alpha,
            .a2 = dw.layers[0].dflare_norm,
            .a3 = dw.layers[1].dflare_norm,
            .a4 = dw.layers[2].dflare_norm,
            .a5 = dw.markov_dflare_norm,
            .a6 = self.ds_mix_h,
            .a7 = self.ds_f_ring,
            .d_pos_addr = d_pos_addr,
            .B = B,
            .cap = 0,
            .kv_fmt = 0,
            .mode = 0,
        }, B, 4, 1);
        self.ctx.cmd_compute_barrier(cmd);

        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            const dl = &dw.layers[ml];
            const mix_l = self.ds_mix_h + @as(u64, ml) * @as(u64, B) * constants.hidden_dim * 2;
            p.cmd_linear(cmd, &dl.w_kv_ctx, mix_l, self.ds_kv, B, false);
            self.ctx.cmd_compute_barrier(cmd);

            const lfmt: u32 = if (dl.is_local) @intFromEnum(constants.KVFormat.bf16) else @intFromEnum(self.kv_format);
            p.dispatch(cmd, p.dspark_commit, kernels.DsparkCommitPC{
                .a0 = self.ds_kv,
                .a1 = dl.k_norm_ctx,
                .a2 = self.ds_k_cache[ml],
                .a3 = self.ds_v_cache[ml],
                .a4 = self.ds_k_scale[ml],
                .a5 = self.ds_v_scale[ml],
                .a6 = 0,
                .a7 = 0,
                .d_pos_addr = d_pos_addr,
                .B = B,
                .cap = self.ds_kv_cap[ml],
                .kv_fmt = lfmt,
                .mode = 1,
            }, B, constants.kv_head_count, 1);
            self.ctx.cmd_compute_barrier(cmd);
        }
    }

    /// Dispatch count (per KV head) that covers attn_split_layout for ANY visible range of up to
    /// `span + B - 1` tokens. n_splits is NOT monotonic in the range length (a longer range can round
    /// tiles-per-split up and yield fewer splits), so bound it by min(smax, ceil(ntiles / min_tiles)),
    /// which dominates every shorter range. Surplus workgroups exit immediately.
    fn attn_splits_for_span(span: u32, B: u32, rows_total: u32) u32 {
        const smax = attn_split_max_for_rows(rows_total);
        const tokens: u32 = span + B - 1;
        const ntiles: u32 = (tokens + constants.attn_tile_tokens - 1) / constants.attn_tile_tokens;
        return @min(smax, (ntiles + constants.attn_min_tiles - 1) / constants.attn_min_tiles);
    }

    /// Split budget per KV head for `rows_total` query rows: keeps heads * rows * splits (the
    /// split-buffer footprint) constant -> 256 (B<=4), 128 (B=8).
    fn attn_split_max_for_rows(rows_total: u32) u32 {
        return @min(constants.attn_split_max_small_b, (constants.attn_split_max * constants.mtp_block_size_max) / rows_total);
    }

    const AttnRaw = struct {
        qrgs: u64,
        rel_proj: u64,
        k: u64,
        v: u64,
        ks: u64,
        vs: u64,
        cap: u32,
        fmt: u32,
        span: u32,
        last_off: i32 = 0,
        row_step: u32 = 1,
        d_pos: u64,
        out_h: u64,
    };

    /// Tiled split attention (attn_tile) + merge/sink/gate (attn_merge_gate) for B rows.
    fn record_attn_raw(self: *const Engine, cmd: vk.VkCommandBuffer, a: AttnRaw, B: u32) void {
        const p = &self.pipes;
        const rows_total: u32 = B;
        const smax = attn_split_max_for_rows(rows_total);
        p.dispatch(cmd, p.attn_tile.pair(B), kernels.AttnTilePC{
            .q_addr = a.qrgs,
            .r_addr = if (a.rel_proj != 0) a.qrgs + @as(u64, constants.query_dim) * 4 else 0,
            .rel_proj_addr = a.rel_proj,
            .k_cache_addr = a.k,
            .v_cache_addr = a.v,
            .k_scale_addr = a.ks,
            .v_scale_addr = a.vs,
            .split_buf_addr = self.split_buf,
            .d_pos_addr = a.d_pos,
            .q_row_stride = constants.qrgs_out_dim,
            .cap = a.cap,
            .kv_fmt = a.fmt,
            .span = a.span,
            .last_off = a.last_off,
            .row_step = a.row_step,
            .rows_total = rows_total,
            .n_split_max = smax,
        }, constants.kv_head_count, attn_splits_for_span(a.span, B, rows_total), 1);
        self.ctx.cmd_compute_barrier(cmd);
        p.dispatch(cmd, p.attn_merge, kernels.AttnMergePC{
            .qrgs_addr = a.qrgs,
            .split_buf_addr = self.split_buf,
            .out_gated_addr = 0,
            .out_gated_h_addr = a.out_h,
            .d_pos_addr = a.d_pos,
            .span = a.span,
            .last_off = a.last_off,
            .row_step = a.row_step,
            .group_rows = B,
            .rows_total = rows_total,
            .n_split_max = smax,
        }, constants.head_count, rows_total, 1);
        self.ctx.cmd_compute_barrier(cmd);
    }

    /// Target attention for B rows at d_pos..d_pos+B-1.
    /// Global visits attend [0, pos] (rel-bias on the last 1024), local visits the 1024 SWA window.
    fn record_target_attn(self: *const Engine, cmd: vk.VkCommandBuffer, vi: u32, rel_proj: u64, qrgs: u64, out_h: u64, d_pos: u64, B: u32) void {
        const v = self.schedule[vi];
        self.record_attn_raw(cmd, .{
            .qrgs = qrgs,
            .rel_proj = rel_proj,
            .k = self.k_cache[vi],
            .v = self.v_cache[vi],
            .ks = self.k_scale[vi],
            .vs = self.v_scale[vi],
            .cap = self.kv_cap[vi],
            .fmt = if (v.is_global) @intFromEnum(self.kv_format) else @intFromEnum(constants.KVFormat.bf16),
            .span = if (v.is_global) self.kv_cap[vi] else constants.swa_window_tokens,
            .d_pos = d_pos,
            .out_h = out_h,
        }, B);
    }

    fn record_dspark_commit(self: *const Engine, cmd: vk.VkCommandBuffer, B: u32) void {
        self.record_dspark_commit_at(cmd, self.taps, self.ctrl_addr("pos"), B);
    }

    fn record_prefill_b64(self: *const Engine, cmd: vk.VkCommandBuffer, commit_dspark: bool) void {
        const w = self.weights;
        const p = &self.pipes;
        const B64: u32 = 64;
        const pos_sub_base = self.ctrl_addr("pos_sub");

        p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
            .embed_addr = w.embed,
            .norm_addr = w.embed_norm,
            .tokens_addr = self.ctrl_addr("tokens_b64"),
            .out_addr = self.p_hidden,
            .B = B64,
            .mode = 0,
            .src_stride_elems = 0,
            .slot_idx = 0,
        }, B64, 1, 1);
        self.ctx.cmd_compute_barrier(cmd);

        var block_count: u32 = 0;
        var has_prior: bool = false;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const v = self.schedule[vi];
            const l = v.physical_layer;
            const lw = &w.layers[l];

            if (v.is_loop and l == constants.loop_start_layer and v.pass_index > 0) {
                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.p_blocks,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.p_saved,
                    .B = B64,
                    .mode = 2,
                    .src_stride_elems = 4 * constants.hidden_dim,
                    .slot_idx = 2,
                }, B64, 1, 1);
                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.p_hidden,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.p_prior_last,
                    .B = B64,
                    .mode = 1,
                    .src_stride_elems = 0,
                    .slot_idx = 0,
                }, B64, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);

                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.p_prelude,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.p_hidden,
                    .B = B64,
                    .mode = 2,
                    .src_stride_elems = constants.hidden_dim,
                    .slot_idx = 0,
                }, B64, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);

                block_count = 1;
                has_prior = true;
            }
            const use_prior = v.is_loop and has_prior;
            const distinct_inputs = v.is_global or use_prior;
            const partial_dst: u64 = if (l % constants.block_layer_count == 0) self.p_partial else 0;

            var norm0: u64 = 0;
            var norm1: u64 = 0;
            var norm2: u64 = 0;
            var out0: u64 = 0;
            var out1: u64 = 0;
            var out2: u64 = 0;
            var out_h0: u64 = 0;
            var out_h1: u64 = 0;
            var out_h2: u64 = 0;
            var n_out: u32 = 1;
            if (!distinct_inputs) {
                norm0 = lw.attn_norm;
                out0 = self.p_attn_in[0];
                out_h0 = self.p_attn_in_h[0];
                n_out = 1;
            } else if (v.is_global and !use_prior) {
                norm0 = lw.q_in_norm;
                norm1 = lw.k_in_norm;
                norm2 = lw.v_in_norm;
                out0 = self.p_attn_in[0];
                out1 = self.p_attn_in[1];
                out2 = self.p_attn_in[2];
                out_h0 = self.p_attn_in_h[0];
                out_h1 = self.p_attn_in_h[1];
                out_h2 = self.p_attn_in_h[2];
                n_out = 3;
            } else if (v.is_global) {
                out0 = self.p_attn_in[0];
                out1 = self.p_attn_in[1];
                out2 = self.p_attn_in[2];
                n_out = 3;
            } else {
                out0 = self.p_mlp_in;
                n_out = 1;
            }

            p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
                .hidden_addr = self.p_hidden,
                .partial_out_addr = partial_dst,
                .blocks_addr = self.p_blocks,
                .queries_addr = if (l > 0) w.attn_queries[l - 1] else 0,
                .norm0_addr = norm0,
                .norm1_addr = norm1,
                .norm2_addr = norm2,
                .out0_addr = out0,
                .out1_addr = out1,
                .out2_addr = out2,
                .out_h0_addr = out_h0,
                .out_h1_addr = out_h1,
                .out_h2_addr = out_h2,
                .n_blocks = block_count,
                .B = B64,
            }, B64, n_out, 1);
            self.ctx.cmd_compute_barrier(cmd);

            if (use_prior) {
                const desc_base = self.ctrl_desc_buf.address + @as(u64, l) * 4 * @sizeOf(kernels.CtrlDesc);
                p.dispatch(cmd, p.controller, kernels.ControllerPC{
                    .desc_base_addr = desc_base,
                    .x0_addr = if (v.is_global) self.p_attn_in[0] else self.p_mlp_in,
                    .x1_addr = if (v.is_global) self.p_attn_in[1] else self.p_mlp_in,
                    .x2_addr = if (v.is_global) self.p_attn_in[2] else self.p_mlp_in,
                    .saved_addr = self.p_saved,
                    .prior_last_addr = self.p_prior_last,
                    .out0_addr = self.p_attn_in[0],
                    .out1_addr = self.p_attn_in[1],
                    .out2_addr = self.p_attn_in[2],
                    .out_h0_addr = self.p_attn_in_h[0],
                    .out_h1_addr = self.p_attn_in_h[1],
                    .out_h2_addr = self.p_attn_in_h[2],
                    .B = B64,
                }, B64, 3, 1);
                self.ctx.cmd_compute_barrier(cmd);
            }

            // RDNA 3 WMMA Q/K/V/R/G/S projections over ALL 64 tokens in ONE shot!
            const in_qh = self.p_attn_in_h[0];
            const in_kh = if (distinct_inputs) self.p_attn_in_h[1] else self.p_attn_in_h[0];
            const in_vh = if (distinct_inputs) self.p_attn_in_h[2] else self.p_attn_in_h[0];
            const in_k = if (distinct_inputs) self.p_attn_in[1] else self.p_attn_in[0];
            const in_v = if (distinct_inputs) self.p_attn_in[2] else self.p_attn_in[0];

            p.cmd_linear_wmma(cmd, &lw.w_qrgs, in_qh, self.p_qrgs, false);
            p.cmd_linear_wmma(cmd, &lw.w_k, in_kh, self.p_k_proj, false);
            p.cmd_linear_wmma(cmd, &lw.w_v, in_vh, self.p_v_proj, false);
            self.ctx.cmd_compute_barrier(cmd);

            // Causal KV sconv + Q/K/V store + FlashDecoding in 8 sub-chunks of 8 tokens
            const kv_fmt_u32: u32 = if (v.is_global) @intFromEnum(self.kv_format) else @intFromEnum(constants.KVFormat.bf16);
            var sub: u32 = 0;
            while (sub < 8) : (sub += 1) {
                const sub_u64: u64 = sub;
                const d_pos_sub = pos_sub_base + sub_u64 * 4;
                const qrgs_s = self.p_qrgs + sub_u64 * 8 * constants.qrgs_out_dim * 4;
                const kp_s = self.p_k_proj + sub_u64 * 8 * constants.kv_dim * 4;
                const vp_s = self.p_v_proj + sub_u64 * 8 * constants.kv_dim * 4;
                const ink_s = in_k + sub_u64 * 8 * constants.hidden_dim * 4;
                const inv_s = in_v + sub_u64 * 8 * constants.hidden_dim * 4;
                const ag_s = self.p_attn_gated_h + sub_u64 * 8 * constants.query_dim * 2;

                p.dispatch(cmd, p.kv_sconv, kernels.KvSconvPC{
                    .k_proj_addr = kp_s,
                    .v_proj_addr = vp_s,
                    .in_k_addr = ink_s,
                    .in_v_addr = inv_s,
                    .k_sconv_w_addr = lw.k_sconv_w,
                    .v_sconv_w_addr = lw.v_sconv_w,
                    .k_sconv_proj_addr = lw.k_sconv_proj,
                    .v_sconv_proj_addr = lw.v_sconv_proj,
                    .k_ring_addr = self.k_ring[vi],
                    .v_ring_addr = self.v_ring[vi],
                    .d_pos_addr = d_pos_sub,
                    .B = 8,
                    .is_global = if (v.is_global) 1 else 0,
                }, if (v.is_global) 2 else 4, if (v.is_global) 1 else 2, 1);
                self.ctx.cmd_compute_barrier(cmd);

                p.dispatch(cmd, p.attn_prep, kernels.AttnPrepPC{
                    .qrgs_addr = qrgs_s,
                    .ck_addr = kp_s,
                    .cv_addr = vp_s,
                    .q_clip_scale_addr = lw.q_clip_scale,
                    .rel_clip_scale_addr = lw.rel_clip_scale,
                    .k_cache_addr = self.k_cache[vi],
                    .v_cache_addr = self.v_cache[vi],
                    .k_scale_addr = self.k_scale[vi],
                    .v_scale_addr = self.v_scale[vi],
                    .d_pos_addr = d_pos_sub,
                    .kv_cap = self.kv_cap[vi],
                    .is_global = if (v.is_global) 1 else 0,
                    .kv_fmt = kv_fmt_u32,
                }, 8, 64, 1);
                self.ctx.cmd_compute_barrier(cmd);

                self.record_target_attn(cmd, vi, lw.rel_logits_proj, qrgs_s, ag_s, d_pos_sub, 8);
            }

            // RDNA 3 WMMA w_o over ALL 64 tokens!
            p.cmd_linear_wmma(cmd, &lw.w_o, self.p_attn_gated_h, self.p_o_proj, false);
            self.ctx.cmd_compute_barrier(cmd);

            // Causal Attn post_res in 8 sub-chunks of 8
            const is_boundary = (l % constants.block_layer_count == 0);
            sub = 0;
            while (sub < 8) : (sub += 1) {
                const sub_u64: u64 = sub;
                const h_off = sub_u64 * 8 * constants.hidden_dim * 4;
                p.dispatch(cmd, p.post_res, kernels.PostResPC{
                    .src_addr = self.p_o_proj + h_off,
                    .sconv_w_addr = lw.attn_sconv_w,
                    .ring_addr = self.attn_ring[vi],
                    .residual_addr = self.p_hidden + h_off,
                    .out_addr = self.p_residual + h_off,
                    .blocks_addr = if (is_boundary) (self.p_blocks + sub_u64 * 8 * 4 * constants.hidden_dim * 4) else 0,
                    .partial_addr = if (is_boundary) (self.p_partial + h_off) else 0,
                    .tap_out_addr = 0,
                    .d_pos_addr = pos_sub_base + sub_u64 * 4,
                    .B = 8,
                    .block_idx = block_count,
                }, (constants.hidden_dim + 255) / 256, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);
            }
            if (is_boundary) block_count += 1;

            p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
                .hidden_addr = self.p_residual,
                .partial_out_addr = 0,
                .blocks_addr = self.p_blocks,
                .queries_addr = w.mlp_queries[l],
                .norm0_addr = if (use_prior) 0 else lw.mlp_norm,
                .norm1_addr = 0,
                .norm2_addr = 0,
                .out0_addr = if (use_prior) self.p_mlp_in else self.p_mlp_normed,
                .out1_addr = 0,
                .out2_addr = 0,
                .out_h0_addr = if (use_prior) 0 else self.p_mlp_normed_h,
                .out_h1_addr = 0,
                .out_h2_addr = 0,
                .n_blocks = block_count,
                .B = B64,
            }, B64, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            if (use_prior) {
                const desc_mlp = self.ctrl_desc_buf.address + (@as(u64, l) * 4 + 3) * @sizeOf(kernels.CtrlDesc);
                p.dispatch(cmd, p.controller, kernels.ControllerPC{
                    .desc_base_addr = desc_mlp,
                    .x0_addr = self.p_mlp_in,
                    .x1_addr = 0,
                    .x2_addr = 0,
                    .saved_addr = self.p_saved,
                    .prior_last_addr = self.p_prior_last,
                    .out0_addr = self.p_mlp_normed,
                    .out1_addr = 0,
                    .out2_addr = 0,
                    .out_h0_addr = self.p_mlp_normed_h,
                    .out_h1_addr = 0,
                    .out_h2_addr = 0,
                    .B = B64,
                }, B64, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);
            }

            // RDNA 3 WMMA router + Single-Pass B=64 MoE (reads each expert ONCE across all 64 tokens!)
            p.cmd_linear_wmma(cmd, &lw.router_nvfp4, self.p_mlp_normed_h, self.p_router_logits, false);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.moe_prefill_b64, kernels.MoePrefillB64PC{
                .a0 = self.p_router_logits,
                .a1 = lw.router_bias,
                .a2 = self.p_exp_table64,
                .a3 = 0,
                .a4 = 0,
                .a5 = 0,
                .mode = 0,
            }, 1, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.moe_prefill_b64, kernels.MoePrefillB64PC{
                .a0 = lw.w13.data,
                .a1 = lw.w13.scales,
                .n_lo = lw.w13.n_lo,
                .a6 = lw.w13.data_hi,
                .a7 = lw.w13.scales_hi,
                .a2 = lw.w13.expert_scale,
                .a3 = self.p_mlp_normed_h,
                .a4 = self.p_exp_table64,
                .a5 = self.p_moe_act_h,
                .mode = 1,
            }, 240, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.moe_prefill_b64, kernels.MoePrefillB64PC{
                .a0 = lw.w2.data,
                .a1 = lw.w2.scales,
                .n_lo = lw.w2.n_lo,
                .a6 = lw.w2.data_hi,
                .a7 = lw.w2.scales_hi,
                .a2 = lw.w2.expert_scale,
                .a3 = self.p_moe_act_h,
                .a4 = self.p_exp_table64,
                .a5 = self.p_moe_down,
                .mode = 2,
            }, constants.hidden_dim / 16, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            // Causal MLP post_res in 8 sub-chunks of 8 (writing per-subchunk DSpark taps directly!)
            sub = 0;
            while (sub < 8) : (sub += 1) {
                const sub_u64: u64 = sub;
                const h_off = sub_u64 * 8 * constants.hidden_dim * 4;
                var tap_dst: u64 = 0;
                if (vi == 5) {
                    tap_dst = self.p_prelude + h_off;
                } else if (commit_dspark and vi == 4) {
                    tap_dst = self.p_taps + (sub_u64 * 4 + 0) * 8 * constants.hidden_dim * 4;
                } else if (commit_dspark and vi == 22) {
                    tap_dst = self.p_taps + (sub_u64 * 4 + 1) * 8 * constants.hidden_dim * 4;
                } else if (commit_dspark and vi == 40) {
                    tap_dst = self.p_taps + (sub_u64 * 4 + 2) * 8 * constants.hidden_dim * 4;
                }
                p.dispatch(cmd, p.post_res, kernels.PostResPC{
                    .src_addr = self.p_moe_down + h_off,
                    .sconv_w_addr = lw.mlp_sconv_w,
                    .ring_addr = self.mlp_ring[vi],
                    .residual_addr = self.p_residual + h_off,
                    .out_addr = self.p_hidden + h_off,
                    .blocks_addr = 0,
                    .partial_addr = 0,
                    .tap_out_addr = tap_dst,
                    .d_pos_addr = pos_sub_base + sub_u64 * 4,
                    .B = 8,
                    .block_idx = 0,
                }, (constants.hidden_dim + 255) / 256, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);
            }
        }

        if (commit_dspark) {
            p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
                .hidden_addr = self.p_hidden,
                .partial_out_addr = 0,
                .blocks_addr = self.p_blocks,
                .queries_addr = w.output_query,
                .norm0_addr = w.final_norm,
                .norm1_addr = 0,
                .norm2_addr = 0,
                .out0_addr = self.p_mlp_normed,
                .out1_addr = 0,
                .out2_addr = 0,
                .out_h0_addr = 0,
                .out_h1_addr = 0,
                .out_h2_addr = 0,
                .n_blocks = block_count,
                .B = B64,
            }, B64, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            var sub: u32 = 0;
            while (sub < 8) : (sub += 1) {
                const sub_u64: u64 = sub;
                const taps_s = self.p_taps + sub_u64 * 4 * 8 * constants.hidden_dim * 4;
                const tap3_dst = taps_s + 3 * 8 * constants.hidden_dim * 4;
                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.p_mlp_normed + sub_u64 * 8 * constants.hidden_dim * 4,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = tap3_dst,
                    .B = 8,
                    .mode = 2,
                    .src_stride_elems = constants.hidden_dim,
                    .slot_idx = 0,
                }, 8, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);
                self.record_dspark_commit_at(cmd, taps_s, pos_sub_base + sub_u64 * 4, 8);
            }
        }
    }

    fn record_dspark_draft(self: *const Engine, cmd: vk.VkCommandBuffer, num_drafts: u32, prof: ?*StageProfiler) void {
        const w = self.weights;
        const dw = &w.dspark;
        const p = &self.pipes;
        const B8: u32 = constants.mtp_block_size_max;
        const Vd: u32 = dw.draft_vocab_size;

        // Partition ds_ws cleanly (256-byte aligned)
        var off: u64 = self.ds_ws;
        const alloc_ws = struct {
            fn f(cur: *u64, bytes: u64) u64 {
                const res = cur.*;
                cur.* += tiger.align_up_u64(bytes, 256);
                return res;
            }
        }.f;

        const block_x = alloc_ws(&off, B8 * constants.hidden_dim * 4);
        const normed = alloc_ws(&off, B8 * constants.hidden_dim * 4);
        const normed_h = alloc_ws(&off, B8 * constants.hidden_dim * 2);
        const qkv = alloc_ws(&off, B8 * constants.dspark_qkv_out_dim * 4);
        const attn_out_h = alloc_ws(&off, B8 * constants.query_dim * 2);
        const gu = alloc_ws(&off, B8 * 2 * constants.mtp_intermediate_dim * 4);
        const swiglu_h = alloc_ws(&off, B8 * constants.mtp_intermediate_dim * 2);
        const f_prev = alloc_ws(&off, constants.hidden_dim * 4);
        const c_f = alloc_ws(&off, 3 * constants.mtp_markov_rank * 4);
        const c_h = alloc_ws(&off, constants.mtp_draft_count_max * 3 * constants.mtp_markov_rank * 4);
        const state = alloc_ws(&off, constants.mtp_markov_rank * 4);
        const raw_rnn = alloc_ws(&off, 3 * constants.mtp_markov_rank * 4);
        const blk_val = alloc_ws(&off, 128 * 4);
        const blk_idx = alloc_ws(&off, 128 * 4);
        const blk_cm = alloc_ws(&off, 128 * 4);
        const blk_csum = alloc_ws(&off, 128 * 4);
        const base_logits = alloc_ws(&off, @as(u64, num_drafts) * Vd * 4);

        // 1. Build 8-row block [anchor, mask x7]
        p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
            .a0 = w.embed,
            .a1 = w.embed_norm,
            .a2 = dw.mask_embed,
            .a3 = self.ctrl_addr("anchor"),
            .a4 = block_x,
            .a5 = 0,
            .a6 = 0,
            .param0 = 0,
            .mode = 0,
        }, B8, 1, 1);
        self.ctx.cmd_compute_barrier(cmd);

        // 2. Run 3 dense DSpark layers over all 8 rows
        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            const dl = &dw.layers[ml];

            p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
                .a0 = block_x,
                .a1 = dl.input_norm,
                .a2 = 0,
                .a3 = normed_h,
                .a4 = 0,
                .a5 = 0,
                .a6 = 0,
                .param0 = B8,
                .mode = 1,
            }, B8, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.cmd_linear(cmd, &dl.w_qkv, normed_h, qkv, B8, false);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
                .a0 = qkv,
                .a1 = dl.q_norm,
                .a2 = dl.k_norm,
                .a3 = self.ctrl_addr("pos"),
                .a4 = 0,
                .a5 = 0,
                .a6 = 0,
                .param0 = 0,
                .mode = 2,
            }, B8, constants.head_count + constants.kv_head_count, 1);
            self.ctx.cmd_compute_barrier(cmd);

            const lfmt: u32 = if (dl.is_local) @intFromEnum(constants.KVFormat.bf16) else @intFromEnum(self.kv_format);
            {
                // ctx attention for all 8 block rows (all see [p - span, p)) via the tiled split kernel
                const span: u32 = if (dl.is_local) constants.swa_window_tokens else self.ds_kv_cap[ml];
                p.dispatch(cmd, p.attn_tile.pair(8), kernels.AttnTilePC{
                    .q_addr = qkv,
                    .r_addr = 0,
                    .rel_proj_addr = 0,
                    .k_cache_addr = self.ds_k_cache[ml],
                    .v_cache_addr = self.ds_v_cache[ml],
                    .k_scale_addr = self.ds_k_scale[ml],
                    .v_scale_addr = self.ds_v_scale[ml],
                    .split_buf_addr = self.split_buf,
                    .d_pos_addr = self.ctrl_addr("pos"),
                    .q_row_stride = constants.dspark_qkv_out_dim,
                    .cap = self.ds_kv_cap[ml],
                    .kv_fmt = lfmt,
                    .span = span,
                    .last_off = -1,
                    .row_step = 0,
                    .rows_total = 8,
                    .n_split_max = constants.attn_split_max,
                }, constants.kv_head_count, attn_splits_for_span(span, 8, 8), 1);
                self.ctx.cmd_compute_barrier(cmd);
            }
            p.dispatch(cmd, p.dspark_attn, kernels.DsparkAttnPC{
                .qkv_addr = qkv,
                .ck_addr = self.ds_k_cache[ml],
                .cv_addr = self.ds_v_cache[ml],
                .cks_addr = self.ds_k_scale[ml],
                .cvs_addr = self.ds_v_scale[ml],
                .split_buf_addr = self.split_buf,
                .out_h_addr = attn_out_h,
                .d_pos_addr = self.ctrl_addr("pos"),
                .cap = self.ds_kv_cap[ml],
                .window = if (dl.is_local) constants.swa_window_tokens else 0,
                .kv_fmt = lfmt,
                .mode = 1,
            }, constants.head_count, B8, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.cmd_linear(cmd, &dl.w_o, attn_out_h, block_x, B8, true);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
                .a0 = block_x,
                .a1 = dl.post_norm,
                .a2 = 0,
                .a3 = normed_h,
                .a4 = 0,
                .a5 = 0,
                .a6 = 0,
                .param0 = B8,
                .mode = 1,
            }, B8, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.cmd_linear(cmd, &dl.w_gu, normed_h, gu, B8, false);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
                .a0 = gu,
                .a1 = swiglu_h,
                .a2 = 0,
                .a3 = 0,
                .a4 = 0,
                .a5 = 0,
                .a6 = 0,
                .param0 = B8,
                .mode = 3,
            }, (B8 * constants.mtp_intermediate_dim + 127) / 128, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.cmd_linear(cmd, &dl.w_down, swiglu_h, block_x, B8, true);
            self.ctx.cmd_compute_barrier(cmd);
        }
        prof_mark(prof, cmd, .ds_layers);

        // 3. Final norm on rows 1..num_drafts -> normed (F32 for rnn_w_h) + normed_h (F16 for unembed_draft)
        p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
            .a0 = block_x + constants.hidden_dim * 4,
            .a1 = dw.norm_w,
            .a2 = normed,
            .a3 = normed_h,
            .a4 = 0,
            .a5 = 0,
            .a6 = 0,
            .param0 = num_drafts,
            .mode = 1,
        }, num_drafts, 1, 1);
        p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
            .a0 = self.ds_f_ring,
            .a1 = self.ctrl_addr("pos"),
            .a2 = f_prev,
            .a3 = state,
            .a4 = 0,
            .a5 = 0,
            .a6 = 0,
            .param0 = 0,
            .mode = 4,
        }, (constants.hidden_dim + 127) / 128, 1, 1);
        self.ctx.cmd_compute_barrier(cmd);

        // 4. Precompute c_f, c_h, and base_logits[0..num_drafts-1, 0..Vd-1] in parallel!
        p.dispatch(cmd, p.dspark_markov, kernels.DsparkMarkovPC{
            .a0 = dw.rnn_w_f,
            .a1 = f_prev,
            .a2 = c_f,
            .a3 = 0, .a4 = 0, .a5 = 0, .a6 = 0, .a7 = 0, .a8 = 0, .a9 = 0, .a10 = 0,
            .p0 = 3 * constants.mtp_markov_rank,
            .p1 = constants.hidden_dim,
            .p2 = 1,
            .gscale = 1.0,
            .mode = 0,
        }, (3 * constants.mtp_markov_rank + 15) / 16, 1, 1);

        p.dispatch(cmd, p.dspark_markov, kernels.DsparkMarkovPC{
            .a0 = dw.rnn_w_h,
            .a1 = normed,
            .a2 = c_h,
            .a3 = 0, .a4 = 0, .a5 = 0, .a6 = 0, .a7 = 0, .a8 = 0, .a9 = 0, .a10 = 0,
            .p0 = 3 * constants.mtp_markov_rank,
            .p1 = constants.hidden_dim,
            .p2 = num_drafts,
            .gscale = 1.0,
            .mode = 0,
        }, (3 * constants.mtp_markov_rank + 15) / 16, 1, 1);

        p.cmd_linear(cmd, &dw.unembed_draft, normed_h, base_logits, num_drafts, false);
        self.ctx.cmd_compute_barrier(cmd);
        prof_mark(prof, cmd, .ds_unembed);

        // 5. Run num_drafts sequential Markov steps (ultra-fast on pruned 49K vocab!)
        const drafts_base = self.ctrl_addr("drafts");
        const conf_base = self.ctrl_addr("draft_conf");
        var i: u32 = 0;
        while (i < num_drafts) : (i += 1) {
            const prev_tok_a = if (i == 0) self.ctrl_addr("anchor") else (drafts_base + @as(u64, i - 1) * 4);
            p.dispatch(cmd, p.dspark_markov, kernels.DsparkMarkovPC{
                .a0 = dw.rnn_w_sp,
                .a1 = dw.rnn_bias,
                .a2 = dw.markov_w1,
                .a3 = c_h + @as(u64, i) * 3 * constants.mtp_markov_rank * 4,
                .a4 = c_f,
                .a5 = prev_tok_a,
                .a6 = state,
                .a7 = raw_rnn,
                .a8 = 0, .a9 = 0, .a10 = 0,
                .p0 = 0, .p1 = 0, .p2 = 0,
                .gscale = 1.0,
                .mode = 1,
            }, 48, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.dspark_markov, kernels.DsparkMarkovPC{
                .a0 = dw.fold_w.data,
                .a1 = dw.fold_w.scales,
                .a2 = base_logits + @as(u64, i) * Vd * 4,
                .a3 = dw.fold_b,
                .a4 = raw_rnn,
                .a5 = state,
                .a6 = blk_val,
                .a7 = blk_idx,
                .a8 = blk_cm,
                .a9 = blk_csum,
                .a10 = 0,
                .p0 = Vd,
                .p1 = 0,
                .p2 = 0,
                .gscale = dw.fold_w.global_scale,
                .mode = 2,
            }, 128, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.dspark_markov, kernels.DsparkMarkovPC{
                .a0 = blk_val,
                .a1 = blk_idx,
                .a2 = blk_cm,
                .a3 = blk_csum,
                .a4 = dw.draft_vocab_ids,
                .a5 = drafts_base + @as(u64, i) * 4,
                .a6 = conf_base + @as(u64, i) * 4,
                .a7 = 0, .a8 = 0, .a9 = 0, .a10 = 0,
                .p0 = Vd,
                .p1 = 0,
                .p2 = 0,
                .gscale = 1.0,
                .mode = 3,
            }, 1, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
        }
        prof_mark(prof, cmd, .ds_markov);
    }

    fn record_target_forward(self: *const Engine, cmd: vk.VkCommandBuffer, B: u32, commit_dspark: bool, compute_lm_head: bool, advance_b1: bool, do_verify: bool, prof: ?*StageProfiler) void {
        const w = self.weights;
        const p = &self.pipes;
        const b_u64: u64 = B;

        p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
            .embed_addr = w.embed,
            .norm_addr = w.embed_norm,
            .tokens_addr = self.ctrl_addr("tokens"),
            .out_addr = self.hidden,
            .B = B,
            .mode = 0,
            .src_stride_elems = 0,
            .slot_idx = 0,
        }, B, 1, 1);
        self.ctx.cmd_compute_barrier(cmd);
        prof_mark(prof, cmd, .embed);

        var block_count: u32 = 0;
        var has_prior: bool = false;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const v = self.schedule[vi];
            const l = v.physical_layer;
            const lw = &w.layers[l];

            if (v.is_loop and l == constants.loop_start_layer and v.pass_index > 0) {
                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.blocks,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.saved,
                    .B = B,
                    .mode = 2,
                    .src_stride_elems = 4 * constants.hidden_dim,
                    .slot_idx = 2,
                }, B, 1, 1);
                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.hidden,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.prior_last,
                    .B = B,
                    .mode = 1,
                    .src_stride_elems = 0,
                    .slot_idx = 0,
                }, B, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);

                p.dispatch(cmd, p.embed_norm, kernels.EmbedNormPC{
                    .embed_addr = self.prelude,
                    .norm_addr = 0,
                    .tokens_addr = 0,
                    .out_addr = self.hidden,
                    .B = B,
                    .mode = 2,
                    .src_stride_elems = constants.hidden_dim,
                    .slot_idx = 0,
                }, B, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);

                block_count = 1;
                has_prior = true;
            }
            const use_prior = v.is_loop and has_prior;
            const distinct_inputs = v.is_global or use_prior;
            const partial_dst: u64 = if (l % constants.block_layer_count == 0) self.partial else 0;

            var norm0: u64 = 0;
            var norm1: u64 = 0;
            var norm2: u64 = 0;
            var out0: u64 = 0;
            var out1: u64 = 0;
            var out2: u64 = 0;
            var out_h0: u64 = 0;
            var out_h1: u64 = 0;
            var out_h2: u64 = 0;
            var n_out: u32 = 1;
            if (!distinct_inputs) {
                norm0 = lw.attn_norm;
                out0 = self.attn_in[0];
                out_h0 = self.attn_in_h[0];
                n_out = 1;
            } else if (v.is_global and !use_prior) {
                norm0 = lw.q_in_norm;
                norm1 = lw.k_in_norm;
                norm2 = lw.v_in_norm;
                out0 = self.attn_in[0];
                out1 = self.attn_in[1];
                out2 = self.attn_in[2];
                out_h0 = self.attn_in_h[0];
                out_h1 = self.attn_in_h[1];
                out_h2 = self.attn_in_h[2];
                n_out = 3;
            } else if (v.is_global) {
                out0 = self.attn_in[0];
                out1 = self.attn_in[1];
                out2 = self.attn_in[2];
                n_out = 3;
            } else {
                out0 = self.mlp_in;
                n_out = 1;
            }

            p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
                .hidden_addr = self.hidden,
                .partial_out_addr = partial_dst,
                .blocks_addr = self.blocks,
                .queries_addr = if (l > 0) w.attn_queries[l - 1] else 0,
                .norm0_addr = norm0,
                .norm1_addr = norm1,
                .norm2_addr = norm2,
                .out0_addr = out0,
                .out1_addr = out1,
                .out2_addr = out2,
                .out_h0_addr = out_h0,
                .out_h1_addr = out_h1,
                .out_h2_addr = out_h2,
                .n_blocks = block_count,
                .B = B,
            }, B, n_out, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .depth_norm);

            if (use_prior) {
                const desc_base = self.ctrl_desc_buf.address + @as(u64, l) * 4 * @sizeOf(kernels.CtrlDesc);
                p.dispatch(cmd, p.controller, kernels.ControllerPC{
                    .desc_base_addr = desc_base,
                    .x0_addr = if (v.is_global) self.attn_in[0] else self.mlp_in,
                    .x1_addr = if (v.is_global) self.attn_in[1] else self.mlp_in,
                    .x2_addr = if (v.is_global) self.attn_in[2] else self.mlp_in,
                    .saved_addr = self.saved,
                    .prior_last_addr = self.prior_last,
                    .out0_addr = self.attn_in[0],
                    .out1_addr = self.attn_in[1],
                    .out2_addr = self.attn_in[2],
                    .out_h0_addr = self.attn_in_h[0],
                    .out_h1_addr = self.attn_in_h[1],
                    .out_h2_addr = self.attn_in_h[2],
                    .B = B,
                }, B, 3, 1);
                self.ctx.cmd_compute_barrier(cmd);
                prof_mark(prof, cmd, .controller);
            }

            const in_qh = self.attn_in_h[0];
            const in_kh = if (distinct_inputs) self.attn_in_h[1] else self.attn_in_h[0];
            const in_vh = if (distinct_inputs) self.attn_in_h[2] else self.attn_in_h[0];
            const in_k = if (distinct_inputs) self.attn_in[1] else self.attn_in[0];
            const in_v = if (distinct_inputs) self.attn_in[2] else self.attn_in[0];

            p.cmd_linear(cmd, &lw.w_qrgs, in_qh, self.qrgs, B, false);
            p.cmd_linear(cmd, &lw.w_k, in_kh, self.k_proj, B, false);
            p.cmd_linear(cmd, &lw.w_v, in_vh, self.v_proj, B, false);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .qkvrg);

            p.dispatch(cmd, p.kv_sconv, kernels.KvSconvPC{
                .k_proj_addr = self.k_proj,
                .v_proj_addr = self.v_proj,
                .in_k_addr = in_k,
                .in_v_addr = in_v,
                .k_sconv_w_addr = lw.k_sconv_w,
                .v_sconv_w_addr = lw.v_sconv_w,
                .k_sconv_proj_addr = lw.k_sconv_proj,
                .v_sconv_proj_addr = lw.v_sconv_proj,
                .k_ring_addr = self.k_ring[vi],
                .v_ring_addr = self.v_ring[vi],
                .d_pos_addr = self.ctrl_addr("pos"),
                .B = B,
                .is_global = if (v.is_global) 1 else 0,
            }, if (v.is_global) 2 else 4, if (v.is_global) 1 else 2, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .kv_sconv);

            const kv_fmt_u32: u32 = if (v.is_global) @intFromEnum(self.kv_format) else @intFromEnum(constants.KVFormat.bf16);
            p.dispatch(cmd, p.attn_prep, kernels.AttnPrepPC{
                .qrgs_addr = self.qrgs,
                .ck_addr = self.k_proj,
                .cv_addr = self.v_proj,
                .q_clip_scale_addr = lw.q_clip_scale,
                .rel_clip_scale_addr = lw.rel_clip_scale,
                .k_cache_addr = self.k_cache[vi],
                .v_cache_addr = self.v_cache[vi],
                .k_scale_addr = self.k_scale[vi],
                .v_scale_addr = self.v_scale[vi],
                .d_pos_addr = self.ctrl_addr("pos"),
                .kv_cap = self.kv_cap[vi],
                .is_global = if (v.is_global) 1 else 0,
                .kv_fmt = kv_fmt_u32,
            }, B, 64, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .attn_prep);

            self.record_target_attn(cmd, vi, lw.rel_logits_proj, self.qrgs, self.attn_gated_h, self.ctrl_addr("pos"), B);
            prof_mark(prof, cmd, .attn_core);

            p.cmd_linear(cmd, &lw.w_o, self.attn_gated_h, self.o_proj, B, false);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .w_o);

            const is_boundary = (l % constants.block_layer_count == 0);
            p.dispatch(cmd, p.post_res, kernels.PostResPC{
                .src_addr = self.o_proj,
                .sconv_w_addr = lw.attn_sconv_w,
                .ring_addr = self.attn_ring[vi],
                .residual_addr = self.hidden,
                .out_addr = self.residual,
                .blocks_addr = if (is_boundary) self.blocks else 0,
                .partial_addr = if (is_boundary) self.partial else 0,
                .tap_out_addr = 0,
                .d_pos_addr = self.ctrl_addr("pos"),
                .B = B,
                .block_idx = block_count,
            }, (constants.hidden_dim + 255) / 256, 1, 1);
            if (is_boundary) block_count += 1;
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .post_res);

            p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
                .hidden_addr = self.residual,
                .partial_out_addr = 0,
                .blocks_addr = self.blocks,
                .queries_addr = w.mlp_queries[l],
                .norm0_addr = if (use_prior) 0 else lw.mlp_norm,
                .norm1_addr = 0,
                .norm2_addr = 0,
                .out0_addr = if (use_prior) self.mlp_in else self.mlp_normed,
                .out1_addr = 0,
                .out2_addr = 0,
                .out_h0_addr = if (use_prior) 0 else self.mlp_normed_h,
                .out_h1_addr = 0,
                .out_h2_addr = 0,
                .n_blocks = block_count,
                .B = B,
            }, B, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .depth_norm);

            if (use_prior) {
                const desc_mlp = self.ctrl_desc_buf.address + (@as(u64, l) * 4 + 3) * @sizeOf(kernels.CtrlDesc);
                p.dispatch(cmd, p.controller, kernels.ControllerPC{
                    .desc_base_addr = desc_mlp,
                    .x0_addr = self.mlp_in,
                    .x1_addr = 0,
                    .x2_addr = 0,
                    .saved_addr = self.saved,
                    .prior_last_addr = self.prior_last,
                    .out0_addr = self.mlp_normed,
                    .out1_addr = 0,
                    .out2_addr = 0,
                    .out_h0_addr = self.mlp_normed_h,
                    .out_h1_addr = 0,
                    .out_h2_addr = 0,
                    .B = B,
                }, B, 1, 1);
                self.ctx.cmd_compute_barrier(cmd);
                prof_mark(prof, cmd, .controller);
            }

            p.cmd_linear(cmd, &lw.router_nvfp4, self.mlp_normed_h, self.router_logits, B, false);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.route_topk, kernels.RouteTopkPC{
                .logits_addr = self.router_logits,
                .bias_addr = lw.router_bias,
                .out_ids_addr = self.route_ids,
                .out_w_addr = self.route_w,
                .groups_addr = self.exp_groups,
                .B = B,
            }, 1, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .router);

            p.dispatch(cmd, p.moe_up.pair(B), kernels.MoeUpPC{
                .w_addr = lw.w13.data,
                .s_addr = lw.w13.scales,
                .n_lo = lw.w13.n_lo,
                .w_hi_addr = lw.w13.data_hi,
                .s_hi_addr = lw.w13.scales_hi,
                .exp_sc_addr = lw.w13.expert_scale,
                .x_h_addr = self.mlp_normed_h,
                .groups_addr = self.exp_groups,
                .out_act_h_addr = self.moe_act_h,
                .B = B,
            }, 240, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .moe_up);

            p.dispatch(cmd, p.moe_down.pair(B), kernels.MoeDownPC{
                .w_addr = lw.w2.data,
                .s_addr = lw.w2.scales,
                .n_lo = lw.w2.n_lo,
                .w_hi_addr = lw.w2.data_hi,
                .s_hi_addr = lw.w2.scales_hi,
                .exp_sc_addr = lw.w2.expert_scale,
                .act_h_addr = self.moe_act_h,
                .groups_addr = self.exp_groups,
                .out_down_addr = self.moe_down,
                .B = B,
            }, constants.hidden_dim / 16, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .moe_down);

            var tap_dst: u64 = 0;
            if (vi == 5) {
                tap_dst = self.prelude;
            } else if (commit_dspark and vi == 4) {
                tap_dst = self.taps;
            } else if (commit_dspark and vi == 22) {
                tap_dst = self.taps + b_u64 * constants.hidden_dim * 4;
            } else if (commit_dspark and vi == 40) {
                tap_dst = self.taps + 2 * b_u64 * constants.hidden_dim * 4;
            }

            p.dispatch(cmd, p.post_res, kernels.PostResPC{
                .src_addr = self.moe_down,
                .sconv_w_addr = lw.mlp_sconv_w,
                .ring_addr = self.mlp_ring[vi],
                .residual_addr = self.residual,
                .out_addr = self.hidden,
                .blocks_addr = 0,
                .partial_addr = 0,
                .tap_out_addr = tap_dst,
                .d_pos_addr = self.ctrl_addr("pos"),
                .B = B,
                .block_idx = 0,
            }, (constants.hidden_dim + 255) / 256, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);
            prof_mark(prof, cmd, .post_res);
        }

        // Final Output Depth-Attn + Final RMSNorm (writes tap[3] directly when commit_dspark!)
        const final_f32_dst = if (commit_dspark)
            (self.taps + 3 * b_u64 * constants.hidden_dim * 4)
        else
            self.mlp_normed;
        p.dispatch(cmd, p.depth_norm, kernels.DepthNormPC{
            .hidden_addr = self.hidden,
            .partial_out_addr = 0,
            .blocks_addr = self.blocks,
            .queries_addr = w.output_query,
            .norm0_addr = w.final_norm,
            .norm1_addr = 0,
            .norm2_addr = 0,
            .out0_addr = final_f32_dst,
            .out1_addr = 0,
            .out2_addr = 0,
            .out_h0_addr = self.mlp_normed_h,
            .out_h1_addr = 0,
            .out_h2_addr = 0,
            .n_blocks = block_count,
            .B = B,
        }, B, 1, 1);
        self.ctx.cmd_compute_barrier(cmd);
        prof_mark(prof, cmd, .depth_norm);

        if (commit_dspark) {
            self.record_dspark_commit(cmd, B);
            prof_mark(prof, cmd, .ds_commit);
        }

        if (compute_lm_head) {
            p.dispatch(cmd, p.lm_head.pair(B), kernels.LmHeadPC{
                .w_addr = w.unembed.data,
                .s_addr = w.unembed.scales,
                .x_addr = self.mlp_normed_h,
                .blk_val_addr = self.lm_blk_val,
                .blk_idx_addr = self.lm_blk_idx,
                .out_tokens_addr = self.ctrl_addr("out_tokens"),
                .d_pos_addr = self.ctrl_addr("pos"),
                .d_tokens_addr = self.ctrl_addr("tokens"),
                .sample_cfg_addr = self.ctrl_addr("sample_inv_temp"),
                .V = constants.vocab_unpadded_count,
                .global_scale = w.unembed.global_scale,
                .B = B,
                .stage = 0,
                .advance_b1 = 0,
            }, 512, 1, 1);
            self.ctx.cmd_compute_barrier(cmd);

            p.dispatch(cmd, p.lm_head.pair(B), kernels.LmHeadPC{
                .w_addr = w.unembed.data,
                .s_addr = w.unembed.scales,
                .x_addr = self.mlp_normed_h,
                .blk_val_addr = self.lm_blk_val,
                .blk_idx_addr = self.lm_blk_idx,
                .out_tokens_addr = self.ctrl_addr("out_tokens"),
                .d_pos_addr = self.ctrl_addr("pos"),
                .d_tokens_addr = self.ctrl_addr("tokens"),
                .sample_cfg_addr = self.ctrl_addr("sample_inv_temp"),
                .V = constants.vocab_unpadded_count,
                .global_scale = w.unembed.global_scale,
                .B = B,
                .stage = 1,
                .advance_b1 = if (advance_b1) 1 else 0,
            }, B, 1, 1);

            if (do_verify) {
                self.ctx.cmd_compute_barrier(cmd);
                p.dispatch(cmd, p.dspark_ops, kernels.DsparkOpsPC{
                    .a0 = self.ctrl_addr("out_tokens"),
                    .a1 = self.ctrl_addr("drafts"),
                    .a2 = self.ctrl_addr("accepted"),
                    .a3 = self.ctrl_addr("num_acc"),
                    .a4 = self.ctrl_addr("pos"),
                    .a5 = self.ctrl_addr("anchor"),
                    .a6 = self.ctrl_addr("tokens"),
                    .param0 = B,
                    .mode = 5,
                }, 1, 1, 1);
            }
            prof_mark(prof, cmd, .lm_head);
        }
    }

    fn save_local_checkpoint(self: *Engine, slot: usize, pos: u32) void {
        tiger.assert(slot < self.prefix_cache.checkpoints.len);
        const stg = &self.prefix_cache.stage_buf;
        const dst = self.prefix_cache.checkpoints[slot].host_ram;
        if (stg.buffer == vk.VK_NULL_HANDLE or stg.mapped == null or dst.len < self.local_state_buf.size) return;
        const total: u64 = self.local_state_buf.size;
        var off: u64 = 0;
        const cmd = self.ctx.imm_cmd;
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        while (off < total) {
            const chunk: u64 = @min(stg.size, total - off);
            tiger.assert_vk(self.ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
            const region = [_]vk.VkBufferCopy{.{ .srcOffset = off, .dstOffset = 0, .size = chunk }};
            self.ctx.vkCmdCopyBuffer(cmd, self.local_state_buf.buffer, stg.buffer, 1, &region);
            tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
            self.ctx.submit_and_wait(cmd);
            @memcpy(dst[@intCast(off)..@intCast(off + chunk)], stg.mapped.?[0..@intCast(chunk)]);
            off += chunk;
        }
        self.prefix_cache.checkpoints[slot].valid = true;
        self.prefix_cache.checkpoints[slot].pos = pos;
    }

    fn restore_local_checkpoint(self: *Engine, slot: usize) void {
        tiger.assert(slot < self.prefix_cache.checkpoints.len and self.prefix_cache.checkpoints[slot].valid);
        const stg = &self.prefix_cache.stage_buf;
        const src = self.prefix_cache.checkpoints[slot].host_ram;
        if (stg.buffer == vk.VK_NULL_HANDLE or stg.mapped == null or src.len < self.local_state_buf.size) return;
        const total: u64 = self.local_state_buf.size;
        var off: u64 = 0;
        const cmd = self.ctx.imm_cmd;
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        while (off < total) {
            const chunk: u64 = @min(stg.size, total - off);
            @memcpy(stg.mapped.?[0..@intCast(chunk)], src[@intCast(off)..@intCast(off + chunk)]);
            tiger.assert_vk(self.ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
            const region = [_]vk.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = off, .size = chunk }};
            self.ctx.vkCmdCopyBuffer(cmd, stg.buffer, self.local_state_buf.buffer, 1, &region);
            tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
            self.ctx.submit_and_wait(cmd);
            off += chunk;
        }
    }

    fn prefill_range(self: *Engine, prompt: []const i32, start: usize, end: usize, use_mtp: bool, compute_final_lm: bool) void {
        var pos = start;
        const has_b64 = (self.cmd_b64_prefill != null);
        const cmd64 = if (use_mtp) self.cmd_b64_prefill_mtp else self.cmd_b64_prefill;

        // 1. RDNA 3 WMMA B=64 batched prefill (strictly before the final token if compute_final_lm!)
        const limit64 = if (compute_final_lm and end > 0) (end - 1) else end;
        if (has_b64 and cmd64 != null) {
            while (pos + 64 <= limit64) : (pos += 64) {
                for (0..8) |s| self.ctrl.pos_sub[s] = @intCast(pos + s * 8);
                for (0..64) |j| self.ctrl.tokens_b64[j] = prompt[pos + j];
                self.ctx.submit_and_wait(cmd64);
            }
        }

        // 2. B=8 tail chunks
        while (pos + 8 <= end) : (pos += 8) {
            const is_last = compute_final_lm and (pos + 8 == end);
            self.ctrl.pos = @intCast(pos);
            for (0..8) |j| self.ctrl.tokens[j] = prompt[pos + j];
            const cmd = if (use_mtp)
                (if (is_last) self.cmd_b8_prefill_lm_mtp else self.cmd_b8_prefill_mtp)
            else
                (if (is_last) self.cmd_b8_prefill_lm else self.cmd_b8_prefill);
            self.ctx.submit_and_wait(cmd);
            if (is_last) {
                self.ctrl.pos = @intCast(end);
                self.ctrl.tokens[0] = self.ctrl.out_tokens[7];
                self.ctrl.anchor = self.ctrl.out_tokens[7];
            }
        }

        // 3. B=1 remainder (< 8 tokens)
        while (pos < end) : (pos += 1) {
            const is_last = compute_final_lm and (pos + 1 == end);
            self.ctrl.pos = @intCast(pos);
            self.ctrl.tokens[0] = prompt[pos];
            if (is_last) {
                self.ctx.submit_and_wait(if (use_mtp) self.cmd_b1_decode_mtp else self.cmd_b1_decode);
                self.ctrl.anchor = self.ctrl.tokens[0];
            } else {
                self.ctx.submit_and_wait(if (use_mtp) self.cmd_b1_prefill_mtp else self.cmd_b1_prefill);
            }
        }
    }

    pub fn generate(
        self: *Engine,
        prompt: []const i32,
        max_new_tokens: u32,
        num_drafts_in: u32,
        out_tokens: []i32,
        on_token: ?TokenCallbackFn,
        cb_ctx: ?*anyopaque,
    ) GenerateStats {
        return self.generate_with_hint(prompt, 0, max_new_tokens, num_drafts_in, out_tokens, on_token, cb_ctx);
    }

    pub fn generate_with_hint(
        self: *Engine,
        prompt: []const i32,
        system_prefix_hint: u32,
        max_new_tokens_in: u32,
        num_drafts_in: u32,
        out_tokens: []i32,
        on_token: ?TokenCallbackFn,
        cb_ctx: ?*anyopaque,
    ) GenerateStats {
        var stats: GenerateStats = .{ .prompt_tokens = @intCast(prompt.len) };
        if (prompt.len == 0 or max_new_tokens_in == 0) return stats;
        if (prompt.len >= self.context_tokens_max) return stats;

        const room: u32 = self.context_tokens_max - @as(u32, @intCast(prompt.len));
        const max_new_tokens: u32 = @min(max_new_tokens_in, room);
        const sampling_on = (self.sample_params.temperature >= 1e-4);
        const use_mtp = (!sampling_on and num_drafts_in > 0 and self.weights.dspark.loaded);

        // Configure GPU sampling in ControlBlock (zero command-buffer rebuild!)
        self.ctrl.sample_inv_temp = if (sampling_on) (1.0 / self.sample_params.temperature) else 0.0;
        self.ctrl.sample_seed = self.sample_params.seed;

        const t_pf0 = tiger.clock_monotonic_ns();
        const prompt_len: u32 = @intCast(prompt.len);

        // --- DDR5 MULTI-CHECKPOINT PREFIX KV CACHE MATCHING ---
        var start_pos: u32 = 0;
        if (self.prefix_cache.enabled and self.prefix_cache.session_tokens.len >= prompt_len) {
            const max_match: u32 = @min(self.prefix_cache.session_len, if (prompt_len > 1) prompt_len - 1 else 0);
            var common: u32 = 0;
            while (common < max_match and self.prefix_cache.session_tokens[common] == prompt[common]) : (common += 1) {}

            if (common > 0 and use_mtp == self.prefix_cache.session_use_mtp) {
                var matched_live = false;
                if (self.prefix_cache.gpu_live_at_pos > 0 and @as(u32, @intCast(self.prefix_cache.gpu_live_at_pos)) <= common) {
                    start_pos = @intCast(self.prefix_cache.gpu_live_at_pos);
                    matched_live = true;
                }
                var best_slot: ?usize = null;
                for (self.prefix_cache.checkpoints, 0..) |ckpt, sidx| {
                    if (ckpt.valid and ckpt.pos <= common and ckpt.pos > start_pos) {
                        start_pos = ckpt.pos;
                        best_slot = sidx;
                        matched_live = false;
                    }
                }
                if (start_pos > 0) {
                    if (!matched_live) {
                        self.restore_local_checkpoint(best_slot.?);
                    }
                    self.ctrl.pos = @intCast(start_pos);
                    self.current_position = @intCast(start_pos);
                    self.prompt_lookup.reset();
                    stats.cached_tokens = start_pos;
                } else {
                    self.reset();
                    for (&self.prefix_cache.checkpoints) |*ckpt| ckpt.valid = false;
                }
            } else {
                self.reset();
                for (&self.prefix_cache.checkpoints) |*ckpt| ckpt.valid = false;
            }
        } else {
            self.reset();
        }

        if (self.use_prompt_lookup) {
            for (prompt) |pt| self.prompt_lookup.push(pt);
        }

        // --- PREFILL (ONLY prompt[start_pos .. prompt_len]) with intermediate checkpoints! ---
        var cur: usize = start_pos;
        if (self.prefix_cache.enabled and system_prefix_hint >= 128 and system_prefix_hint > cur and system_prefix_hint < prompt_len) {
            self.prefill_range(prompt, cur, system_prefix_hint, use_mtp, false);
            self.save_local_checkpoint(0, system_prefix_hint);
            cur = system_prefix_hint;
        }
        const p2k: u32 = ((prompt_len - 1) / 2048) * 2048;
        if (self.prefix_cache.enabled and p2k >= 2048 and p2k > cur and p2k < prompt_len) {
            self.prefill_range(prompt, cur, p2k, use_mtp, false);
            self.save_local_checkpoint(1, p2k);
            cur = p2k;
        }
        self.prefill_range(prompt, cur, prompt_len, use_mtp, true);

        // Save prefill_end checkpoint (Slot 2) BEFORE decode mutates rings & local SWA KV!
        if (self.prefix_cache.enabled and self.prefix_cache.session_tokens.len >= prompt_len) {
            @memcpy(self.prefix_cache.session_tokens[start_pos..prompt_len], prompt[start_pos..prompt_len]);
            self.prefix_cache.session_len = prompt_len;
            self.prefix_cache.session_use_mtp = use_mtp;
            self.prefix_cache.gpu_live_at_pos = @intCast(prompt_len);
            for (&self.prefix_cache.checkpoints) |*ckpt| {
                if (ckpt.valid and ckpt.pos > prompt_len) ckpt.valid = false;
            }
            if (prompt_len >= 64) {
                self.save_local_checkpoint(2, prompt_len);
            }
        }

        const t_pf1 = tiger.clock_monotonic_ns();
        stats.prefill_ms = @as(f64, @floatFromInt(t_pf1 - t_pf0)) / 1e6;

        // Emit first token (produced by final prefill step)
        var emitted: u32 = 0;
        const first_tok: i32 = self.ctrl.tokens[0];
        if (emitted < out_tokens.len) out_tokens[emitted] = first_tok;
        emitted += 1;
        stats.last_token_id = first_tok;
        if (self.use_prompt_lookup) self.prompt_lookup.push(first_tok);
        if (on_token) |cb| {
            if (!cb(cb_ctx.?, first_tok)) {
                stats.generated_tokens = emitted;
                return stats;
            }
        }
        if (first_tok == constants.eos_token_id or emitted >= max_new_tokens) {
            stats.generated_tokens = emitted;
            return stats;
        }

        // Decode mutates local SWA KV & rings past prompt_len, so invalidate gpu_live_at_pos
        self.prefix_cache.gpu_live_at_pos = -1;

        const t_dec0 = tiger.clock_monotonic_ns();

        if (!use_mtp) {
            // Pure autoregressive B=1 Vulkan Graph loop
            while (emitted < max_new_tokens) {
                if (@as(u32, @intCast(self.ctrl.pos)) + 2 >= self.context_tokens_max) break;
                self.ctx.submit_and_wait(self.cmd_b1_decode);
                const tok = self.ctrl.tokens[0];
                if (emitted < out_tokens.len) out_tokens[emitted] = tok;
                emitted += 1;
                stats.last_token_id = tok;
                if (on_token) |cb| {
                    if (!cb(cb_ctx.?, tok)) break;
                }
                if (tok == constants.eos_token_id) break;
            }
        } else {
            // Speculative DSpark MTP + Adaptive {2, 4, 8} + Hybrid O(1) Prompt-Lookup loop!
            const adaptive_mode = (num_drafts_in >= 7 and self.spec_threshold > 0.0);
            const fixed_draft_k: u32 = if (num_drafts_in <= 3) 3 else 7;
            var lookup_drafts: [constants.mtp_draft_count_max]i32 = undefined;

            while (emitted < max_new_tokens) {
                if (@as(u32, @intCast(self.ctrl.pos)) + 16 >= self.context_tokens_max) break;

                var verify_b: u32 = fixed_draft_k + 1;

                if (self.use_prompt_lookup and self.prompt_lookup.match(&lookup_drafts)) {
                    // O(1) Prompt-Lookup hit! Bypass DSpark neural drafter completely -> verify B=8!
                    self.ctrl.tokens[0] = self.ctrl.anchor;
                    for (0..7) |j| {
                        self.ctrl.drafts[j] = lookup_drafts[j];
                        self.ctrl.tokens[1 + j] = lookup_drafts[j];
                    }
                    verify_b = 8;
                    stats.lookup_steps += 1;
                } else {
                    // Run DSpark neural draft graph (3 layers + Markov RNN)
                    self.ctx.submit_and_wait(self.cmd_dspark_draft[fixed_draft_k]);
                    if (adaptive_mode) {
                        var n_conf: u32 = 0;
                        while (n_conf < 7 and self.ctrl.draft_conf[n_conf] >= self.spec_threshold) : (n_conf += 1) {}
                        verify_b = if (n_conf <= 1) 2 else if (n_conf <= 3) 4 else 8;
                    }
                    self.ctrl.tokens[0] = self.ctrl.anchor;
                    var j: usize = 0;
                    while (j + 1 < verify_b) : (j += 1) {
                        self.ctrl.tokens[1 + j] = self.ctrl.drafts[j];
                    }
                }

                // Target model verifies [anchor, draft_0 .. draft_{B-2}] in ONE Vulkan Graph submit!
                self.ctx.submit_and_wait(self.cmd_verify[verify_b]);

                const n_acc: u32 = @intCast(self.ctrl.num_acc);
                tiger.assert(n_acc >= 1 and n_acc <= verify_b);
                if (self.debug_mtp and stats.spec_steps < 16) {
                    tiger.print_stderr("[mtp step={d} B={d} n_acc={d}] drafts=[", .{ stats.spec_steps, verify_b, n_acc });
                    var dj: usize = 0;
                    while (dj + 1 < verify_b) : (dj += 1) {
                        tiger.print_stderr("{d}({d:.2}) ", .{ self.ctrl.drafts[dj], self.ctrl.draft_conf[dj] });
                    }
                    tiger.print_stderr("] target=[", .{});
                    dj = 0;
                    while (dj < verify_b) : (dj += 1) {
                        tiger.print_stderr("{d} ", .{self.ctrl.out_tokens[dj]});
                    }
                    tiger.print_stderr("]\n", .{});
                }
                stats.spec_steps += 1;
                stats.total_drafts += verify_b - 1;
                stats.total_accepted += @min(n_acc - 1, verify_b - 1);

                var stop = false;
                var a: u32 = 0;
                while (a < n_acc and emitted < max_new_tokens) : (a += 1) {
                    const tok = self.ctrl.accepted[a];
                    if (emitted < out_tokens.len) out_tokens[emitted] = tok;
                    emitted += 1;
                    stats.last_token_id = tok;
                    if (self.use_prompt_lookup) self.prompt_lookup.push(tok);
                    if (on_token) |cb| {
                        if (!cb(cb_ctx.?, tok)) {
                            stop = true;
                            break;
                        }
                    }
                    if (tok == constants.eos_token_id) {
                        stop = true;
                        break;
                    }
                }
                if (stop) break;
            }
        }

        const t_dec1 = tiger.clock_monotonic_ns();
        stats.generated_tokens = emitted;
        stats.decode_ms = @as(f64, @floatFromInt(t_dec1 - t_dec0)) / 1e6;
        return stats;
    }

    pub fn profile_stages(self: *Engine, prompt: []const i32) void {
        const names = [17][]const u8{
            "embed", "depth_norm", "controller", "qkvrg", "kv_sconv", "attn_prep",
            "attn_core", "w_o", "post_res", "router", "moe_up", "moe_down",
            "lm_head", "ds_commit", "ds_layers", "ds_unembed", "ds_markov",
        };
        const qp_ci: vk.VkQueryPoolCreateInfo = .{
            .queryType = vk.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2048,
        };
        var pool: vk.VkQueryPool = vk.VK_NULL_HANDLE;
        tiger.assert_vk(self.ctx.vkCreateQueryPool(self.ctx.device, &qp_ci, null, &pool));
        defer self.ctx.vkDestroyQueryPool(self.ctx.device, pool, null);

        // Prefill prompt so KV/rings have real data
        self.reset();
        const use_mtp = self.weights.dspark.loaded;
        var pos: usize = 0;
        while (pos < prompt.len) : (pos += 1) {
            self.ctrl.pos = @intCast(pos);
            self.ctrl.tokens[0] = prompt[pos];
            self.ctx.submit_and_wait(if (use_mtp) self.cmd_b1_prefill_mtp else self.cmd_b1_prefill);
        }
        const saved_pos: i32 = @intCast(prompt.len);

        const configs = [_]struct { label: []const u8, is_draft: bool, B: u32, mtp: bool }{
            .{ .label = "Target B=1 (mtp=0)", .is_draft = false, .B = 1, .mtp = false },
            .{ .label = "Target B=2 (verify)", .is_draft = false, .B = 2, .mtp = use_mtp },
            .{ .label = "Target B=4 (verify)", .is_draft = false, .B = 4, .mtp = use_mtp },
            .{ .label = "Target B=8 (verify)", .is_draft = false, .B = 8, .mtp = use_mtp },
            .{ .label = "DSpark Draft K=3", .is_draft = true, .B = 3, .mtp = use_mtp },
            .{ .label = "DSpark Draft K=7", .is_draft = true, .B = 7, .mtp = use_mtp },
        };

        var ts: [2048]u64 = @splat(0);
        const cmd = self.ctx.alloc_command_buffer();
        defer self.ctx.vkFreeCommandBuffers(self.ctx.device, self.ctx.cmd_pool, 1, &.{cmd});

        for (configs) |cfg| {
            if (cfg.is_draft and !use_mtp) continue;
            var prof: StageProfiler = .{ .ctx = self.ctx, .pool = pool };
            tiger.assert_vk(self.ctx.vkResetCommandBuffer(cmd, 0));
            const bi: vk.VkCommandBufferBeginInfo = .{};
            tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
            prof.begin(cmd);
            if (cfg.is_draft) {
                self.record_dspark_draft(cmd, cfg.B, &prof);
            } else {
                self.record_target_forward(cmd, cfg.B, cfg.mtp, true, false, false, &prof);
            }
            tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));

            // Warmup 3 + Measure 8 (calibrated to exact wall-clock ms!)
            var sums: [17]f64 = @splat(0.0);
            var raw_total: f64 = 0.0;
            var wall_ms_sum: f64 = 0.0;
            const n_iters: u32 = 8;
            var it: u32 = 0;
            while (it < 3 + n_iters) : (it += 1) {
                self.ctrl.pos = saved_pos;
                for (0..8) |j| self.ctrl.tokens[j] = prompt[j % prompt.len];
                self.ctrl.anchor = prompt[prompt.len - 1];
                const t0 = tiger.clock_monotonic_ns();
                self.ctx.submit_and_wait(cmd);
                const t1 = tiger.clock_monotonic_ns();
                if (it >= 3) {
                    wall_ms_sum += @as(f64, @floatFromInt(t1 - t0)) / 1e6;
                    tiger.assert_vk(self.ctx.vkGetQueryPoolResults(
                        self.ctx.device,
                        pool,
                        0,
                        prof.n_queries,
                        @as(usize, prof.n_queries) * @sizeOf(u64),
                        @ptrCast(&ts),
                        @sizeOf(u64),
                        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
                    ));
                    var qi: u32 = 1;
                    while (qi < prof.n_queries) : (qi += 1) {
                        sums[prof.stage_ids[qi]] += @floatFromInt(ts[qi] - ts[qi - 1]);
                    }
                    raw_total += @floatFromInt(ts[prof.n_queries - 1] - ts[0]);
                }
            }
            const avg_tot = wall_ms_sum / @as(f64, @floatFromInt(n_iters));
            tiger.print_stdout("\n=== PROFILE: {s} | total={d:.2} ms ({d:.1} steps/s) ===\n", .{
                cfg.label, avg_tot, 1000.0 / @max(avg_tot, 0.001),
            });
            var si: usize = 0;
            while (si < 17) : (si += 1) {
                const frac = if (raw_total > 0.0) (sums[si] / raw_total) else 0.0;
                const ms = avg_tot * frac;
                if (ms >= 0.01) {
                    tiger.print_stdout("  {s:<12}: {d:6.2} ms ({d:5.1}%)\n", .{
                        names[si], ms, 100.0 * frac,
                    });
                }
            }
        }
    }

    /// Long-context latency scaling sweep from pos = 0 up to context_tokens_max (Q4 global KV!).
    pub fn bench_longctx(self: *Engine) void {
        // Zero-fill all KV slabs so synthetic long-context reads are strictly finite (zero NaN)
        const cmd = self.ctx.imm_cmd;
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        tiger.assert_vk(self.ctx.vkResetCommandBuffer(cmd, 0));
        tiger.assert_vk(self.ctx.vkBeginCommandBuffer(cmd, &bi));
        self.ctx.vkCmdFillBuffer(cmd, self.local_state_buf.buffer, 0, self.local_state_buf.size, 0);
        var si: usize = 0;
        while (si < self.arena.slab_count) : (si += 1) {
            self.ctx.vkCmdFillBuffer(cmd, self.arena.slabs[si].buffer, 0, self.arena.slabs[si].size, 0);
        }
        for (self.arena.donated[0..self.arena.donated_count]) |dr| {
            self.ctx.vkCmdFillBuffer(cmd, dr.buffer, dr.offset, dr.size, 0);
        }
        tiger.assert_vk(self.ctx.vkEndCommandBuffer(cmd));
        self.ctx.submit_and_wait(cmd);

        const positions = [_]u32{ 0, 4096, 16384, 32768, 65000, 130000 };
        tiger.print_stdout("\n=== LONG-CONTEXT SCALING SWEEP (ctx={d}, kv={s}) ===\n", .{
            self.context_tokens_max, self.kv_format.name(),
        });
        tiger.print_stdout("  {s:>8} | {s:>10} | {s:>10} | {s:>10} | {s:>12}\n", .{
            "pos", "B=1 (ms)", "B=1 tok/s", "B=4 (ms)", "MTP~2.6x",
        });
        const use_mtp = self.weights.dspark.loaded;
        for (positions) |p| {
            if (p + 64 >= self.context_tokens_max) continue;
            self.ctrl.tokens[0] = 150_000;
            self.ctrl.anchor = 150_000;
            self.ctrl.sample_inv_temp = 0.0;
            // Warmup + measure B=1
            var it: u32 = 0;
            while (it < 3) : (it += 1) {
                self.ctrl.pos = @intCast(p);
                self.ctx.submit_and_wait(self.cmd_b1_decode);
            }
            const t0 = tiger.clock_monotonic_ns();
            const n_it: u32 = 10;
            it = 0;
            while (it < n_it) : (it += 1) {
                self.ctrl.pos = @intCast(p);
                self.ctx.submit_and_wait(self.cmd_b1_decode);
            }
            const b1_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e6 / @as(f64, @floatFromInt(n_it));

            var b4_ms: f64 = 0.0;
            if (use_mtp) {
                it = 0;
                while (it < 3) : (it += 1) {
                    self.ctrl.pos = @intCast(p);
                    self.ctx.submit_and_wait(self.cmd_dspark_draft[3]);
                    self.ctx.submit_and_wait(self.cmd_verify[4]);
                }
                const t2 = tiger.clock_monotonic_ns();
                it = 0;
                while (it < n_it) : (it += 1) {
                    self.ctrl.pos = @intCast(p);
                    self.ctx.submit_and_wait(self.cmd_dspark_draft[3]);
                    self.ctx.submit_and_wait(self.cmd_verify[4]);
                }
                b4_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t2)) / 1e6 / @as(f64, @floatFromInt(n_it));
            }
            // B=64 WMMA prefill chunk at this position (time per 64 prompt tokens)
            var pf_ms: f64 = 0.0;
            if (self.cmd_b64_prefill) |cmd64| {
                for (0..8) |s| self.ctrl.pos_sub[s] = @intCast(p + s * 8);
                for (0..64) |j| self.ctrl.tokens_b64[j] = 150_000;
                self.ctx.submit_and_wait(cmd64);
                const t4 = tiger.clock_monotonic_ns();
                it = 0;
                while (it < 3) : (it += 1) self.ctx.submit_and_wait(cmd64);
                pf_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t4)) / 1e6 / 3.0;
            }
            const mtp_est = if (b4_ms > 0.0) (2.65 * 1000.0 / b4_ms) else 0.0;
            tiger.print_stdout("  {d:8} | {d:10.2} | {d:10.1} | {d:10.2} | {d:10.1} tok/s | prefill {d:6.0} tok/s\n", .{
                p, b1_ms, 1000.0 / b1_ms, b4_ms, mtp_est, if (pf_ms > 0.0) 64000.0 / pf_ms else 0.0,
            });
        }
    }

    /// GPU vs f64 CPU reference for attn_tile + attn_merge_gate (Q4 head-major + BF16,
    /// global/local/DSpark-style visibility, B = 1..8, rel-bias, sink, gate, ring wrap).
    pub fn check_attention(self: *Engine) void {
        const ctx = self.ctx;
        const cap_q4: u32 = 131072;
        const cap_bf: u32 = 1040;
        const qrgs_bytes: u64 = 8 * constants.qrgs_out_dim * 4;
        var b_q = ctx.create_host_buffer(qrgs_bytes, true, true);
        var b_rel = ctx.create_host_buffer(constants.rel_dim * constants.rel_extent * 2, true, true);
        var b_kq = ctx.create_host_buffer(@as(u64, cap_q4) * 512, true, true);
        var b_vq = ctx.create_host_buffer(@as(u64, cap_q4) * 512, true, true);
        var b_ks = ctx.create_host_buffer(@as(u64, cap_q4) * 32, true, true);
        var b_vs = ctx.create_host_buffer(@as(u64, cap_q4) * 32, true, true);
        var b_kb = ctx.create_host_buffer(@as(u64, cap_bf) * constants.kv_dim * 2, true, true);
        var b_vb = ctx.create_host_buffer(@as(u64, cap_bf) * constants.kv_dim * 2, true, true);
        var b_out = ctx.create_host_buffer(8 * constants.query_dim * 2, true, true);
        var b_pos = ctx.create_host_buffer(256, true, true);
        defer {
            for ([_]*vk.GpuBuffer{ &b_q, &b_rel, &b_kq, &b_vq, &b_ks, &b_vs, &b_kb, &b_vb, &b_out, &b_pos }) |bp| ctx.destroy_buffer(bp);
        }
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const rnd = prng.random();

        const f32s = struct {
            fn get(buf: *const vk.GpuBuffer, comptime T: type) []T {
                const n = buf.size / @sizeOf(T);
                const ptr: [*]T = @ptrCast(@alignCast(buf.mapped.?));
                return ptr[0..@intCast(n)];
            }
        }.get;
        const q = f32s(&b_q, f32);
        for (q) |*x| x.* = (rnd.float(f32) * 2.0 - 1.0) * 0.6;
        const rel = f32s(&b_rel, u16);
        for (rel) |*x| x.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2.0 - 1.0) * 0.5)) >> 16);
        for ([_]*vk.GpuBuffer{ &b_kq, &b_vq }) |bp| {
            for (f32s(bp, u8)) |*x| x.* = @intCast(rnd.intRangeAtMost(u8, 0, 14) | (@as(u8, rnd.intRangeAtMost(u8, 0, 14)) << 4));
        }
        for ([_]*vk.GpuBuffer{ &b_ks, &b_vs }) |bp| {
            for (f32s(bp, f16)) |*x| x.* = @floatCast(0.05 + rnd.float(f32) * 0.4);
        }
        for ([_]*vk.GpuBuffer{ &b_kb, &b_vb }) |bp| {
            for (f32s(bp, u16)) |*x| x.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2.0 - 1.0) * 2.0)) >> 16);
        }

        const Case = struct { fmt: u32, pos: i32, B: u32, span: u32, last_off: i32, row_step: u32, bias: bool };
        const cases = [_]Case{
            .{ .fmt = 2, .pos = 0, .B = 1, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 37, .B = 1, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 1500, .B = 1, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 1500, .B = 4, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 19980, .B = 8, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 12345, .B = 3, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 9000, .B = 8, .span = cap_q4, .last_off = -1, .row_step = 0, .bias = false },
            .{ .fmt = 2, .pos = 70000, .B = 1, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 101000, .B = 8, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 131000, .B = 4, .span = cap_q4, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 2, .pos = 100000, .B = 8, .span = 32768, .last_off = -1, .row_step = 0, .bias = false },
            .{ .fmt = 0, .pos = 5, .B = 1, .span = 1024, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 0, .pos = 700, .B = 2, .span = 1024, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 0, .pos = 5000, .B = 1, .span = 1024, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 0, .pos = 5000, .B = 8, .span = 1024, .last_off = 0, .row_step = 1, .bias = true },
            .{ .fmt = 0, .pos = 3001, .B = 8, .span = 1024, .last_off = -1, .row_step = 0, .bias = false },
            .{ .fmt = 0, .pos = 3001, .B = 7, .span = 1024, .last_off = 0, .row_step = 1, .bias = true },
        };

        const ks16 = f32s(&b_ks, f16);
        const vs16 = f32s(&b_vs, f16);
        const kq8 = f32s(&b_kq, u8);
        const vq8 = f32s(&b_vq, u8);
        const kb16 = f32s(&b_kb, u16);
        const vb16 = f32s(&b_vb, u16);
        const out16 = f32s(&b_out, f16);
        const posp = f32s(&b_pos, i32);

        // Exhaustive: for every B and every visible-range length up to the largest span we dispatch,
        // the shader's split count (attn_split_layout) must fit inside the recorded dispatch.
        {
            var layout_ok = true;
            for ([_][2]u32{ .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 4 }, .{ 5, 5 }, .{ 6, 6 }, .{ 7, 7 }, .{ 8, 8 } }) |cfg| {
                const bb = cfg[0];
                for ([_]u32{ constants.swa_window_tokens, 32768, 131072, 1048576 }) |span| {
                    const disp = attn_splits_for_span(span, bb, cfg[1]);
                    const smax = attn_split_max_for_rows(cfg[1]);
                    var len: u32 = 1;
                    while (len <= span + bb - 1) : (len += 1) {
                        const ntiles = (len + constants.attn_tile_tokens - 1) / constants.attn_tile_tokens;
                        const tps = @max(constants.attn_min_tiles, (ntiles + smax - 1) / smax);
                        const nsp = (ntiles + tps - 1) / tps;
                        if (nsp > disp or nsp > smax) {
                            layout_ok = false;
                            tiger.print_stdout("[attncheck] layout FAIL B={d} span={d} len={d}: n_splits={d} > dispatch={d}\n", .{ bb, span, len, nsp, disp });
                            break;
                        }
                    }
                }
            }
            tiger.print_stdout("[attncheck] split layout vs dispatch (all B, all lengths <= 1M): {s}\n", .{if (layout_ok) "OK" else "FAIL"});
        }

        const scores_heap = tiger.alloc_host_zeroed(f64, cap_q4 + 16);
        defer tiger.free_host_slice(f64, scores_heap);
        var worst: f64 = 0.0;
        var all_ok = true;
        for (cases) |c| {
            posp[0] = c.pos;
            const is_q4 = c.fmt == 2;
            const cap: u32 = if (is_q4) cap_q4 else cap_bf;
            const cmd = ctx.imm_cmd;
            const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
            self.record_attn_raw(cmd, .{
                .qrgs = b_q.address,
                .rel_proj = if (c.bias) b_rel.address else 0,
                .k = if (is_q4) b_kq.address else b_kb.address,
                .v = if (is_q4) b_vq.address else b_vb.address,
                .ks = b_ks.address,
                .vs = b_vs.address,
                .cap = cap,
                .fmt = c.fmt,
                .span = c.span,
                .last_off = c.last_off,
                .row_step = c.row_step,
                .d_pos = b_pos.address,
                .out_h = b_out.address,
            }, c.B);
            tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
            ctx.submit_and_wait(cmd);

            var case_err: f64 = 0.0;
            var case_mag: f64 = 0.0;
            var row: u32 = 0;
            while (row < c.B) : (row += 1) {
                const last: i32 = c.pos + c.last_off + @as(i32, @intCast(row * c.row_step));
                const first: i32 = @max(0, last + 1 - @as(i32, @intCast(c.span)));
                var h: u32 = 0;
                while (h < constants.head_count) : (h += 1) {
                    const kvh = h / constants.gqa_ratio;
                    const qb = row * constants.qrgs_out_dim;
                    const qh = q[qb + h * constants.head_dim ..][0..constants.head_dim];
                    const rq = q[qb + constants.query_dim + h * constants.rel_dim ..][0..constants.rel_dim];
                    const gate = q[qb + constants.query_dim + constants.r_dim + h * constants.head_dim ..][0..constants.head_dim];
                    const sink: f64 = q[qb + constants.query_dim + constants.r_dim + constants.query_dim + h];
                    var mx: f64 = sink;
                    var t: i32 = first;
                    // two passes: max, then accumulate
                    const scores_buf = scores_heap;
                    while (t <= last) : (t += 1) {
                        const slot: u32 = @intCast(@mod(t, @as(i32, @intCast(cap))));
                        var dot: f64 = 0.0;
                        for (0..constants.head_dim) |d| {
                            const kval: f64 = if (is_q4) blk: {
                                const byte = kq8[(kvh * cap + slot) * 64 + d / 2];
                                const nib: i32 = if (d & 1 == 0) byte & 0xF else byte >> 4;
                                break :blk @as(f64, @floatFromInt(nib - 7)) * @as(f64, ks16[(kvh * cap + slot) * 2 + d / 64]);
                            } else @as(f64, @as(f32, @bitCast(@as(u32, kb16[slot * constants.kv_dim + kvh * constants.head_dim + d]) << 16)));
                            dot += kval * qh[d];
                        }
                        var sc = dot * constants.inv_sqrt_head_dim;
                        const dist = last - t;
                        if (c.bias and dist < constants.rel_extent) {
                            for (0..constants.rel_dim) |k| sc += @as(f64, rq[k]) * @as(f64, @as(f32, @bitCast(@as(u32, rel[k * constants.rel_extent + @as(usize, @intCast(dist))]) << 16)));
                        }
                        scores_buf[@intCast(t - first)] = sc;
                        mx = @max(mx, sc);
                    }
                    var den: f64 = @exp(sink - mx);
                    var acc: [constants.head_dim]f64 = @splat(0.0);
                    t = first;
                    while (t <= last) : (t += 1) {
                        const slot: u32 = @intCast(@mod(t, @as(i32, @intCast(cap))));
                        const pw = @exp(scores_buf[@intCast(t - first)] - mx);
                        den += pw;
                        for (0..constants.head_dim) |d| {
                            const vval: f64 = if (is_q4) blk: {
                                const byte = vq8[(kvh * cap + slot) * 64 + d / 2];
                                const nib: i32 = if (d & 1 == 0) byte & 0xF else byte >> 4;
                                break :blk @as(f64, @floatFromInt(nib - 7)) * @as(f64, vs16[(kvh * cap + slot) * 2 + d / 64]);
                            } else @as(f64, @as(f32, @bitCast(@as(u32, vb16[slot * constants.kv_dim + kvh * constants.head_dim + d]) << 16)));
                            acc[d] += pw * vval;
                        }
                    }
                    for (0..constants.head_dim) |d| {
                        const ref = acc[d] / den * (1.0 / (1.0 + @exp(-@as(f64, gate[d]))));
                        const got: f64 = out16[row * constants.query_dim + h * constants.head_dim + d];
                        case_err = @max(case_err, @abs(got - ref));
                        case_mag = @max(case_mag, @abs(ref));
                    }
                }
            }
            const rel_err = case_err / @max(case_mag, 1e-6);
            worst = @max(worst, rel_err);
            const ok = rel_err < 4e-3;
            if (!ok) all_ok = false;
            tiger.print_stdout("[attncheck] fmt={s} pos={d:6} B={d} span={d:5} last_off={d:2} step={d} bias={} | max_abs_err={e:.3} max|ref|={d:.4} rel={e:.3} {s}\n", .{
                if (is_q4) "q4  " else "bf16", c.pos, c.B, c.span, c.last_off, c.row_step, c.bias, case_err, case_mag, rel_err, if (ok) "OK" else "FAIL",
            });
        }
        tiger.print_stdout("[attncheck] {s} (worst rel err {e:.3})\n", .{ if (all_ok) "ALL PASS" else "FAILURES", worst });
    }

    /// Bytes of VRAM the engine's runtime buffers need at `ctx_tokens` (KV caches, local state,
    /// scratch), mirroring Engine.init. Used to size the DDR5 expert tier before weights load.
    pub fn runtime_vram_estimate(ctx_tokens: u32, has_mtp: bool) u64 {
        const q4_tok: u64 = constants.kv_dim + 2 * constants.kv_q4_scales_per_token * 2; // K+V data + scales
        var total: u64 = 0;
        var n_global: u64 = 0;
        var n_local: u64 = 0;
        for (build_schedule_static()) |v| {
            if (v.is_global) n_global += 1 else n_local += 1;
        }
        total += n_global * @as(u64, ctx_tokens) * q4_tok;
        if (has_mtp) total += @as(u64, @min(ctx_tokens, 32768)) * q4_tok;
        const local_kv: u64 = 1040 * constants.kv_dim * 2 * 2;
        const rings: u64 = constants.sconv_ring_slots * (constants.kv_dim * 2 + constants.hidden_dim * 2) * 4;
        total += n_local * local_kv + constants.visit_count * rings;
        if (has_mtp) total += 2 * local_kv;
        total += constants.split_buffer_floats * 4;
        total += 48 * 1024 * 1024; // activation scratch + B=64 prefill scratch + DSpark workspace
        return total;
    }

    fn build_schedule_static() [constants.visit_count]constants.VisitInfo {
        return constants.build_visit_schedule();
    }

    /// Experts per bank to place in pinned DDR5 so that weights + runtime fit the WDDM VRAM budget.
    /// Returns 0 when everything fits or the budget cannot be queried.
    pub fn plan_expert_tier(ctx: *const vk.Context, ctx_tokens: u32, has_mtp: bool) u32 {
        const b = ctx.vram_budget() orelse return 0;
        const probed = ctx.probe_vram_allocatable();
        const avail: u64 = @min(b.budget -| b.usage, probed);
        const per_expert_layer: u64 = 2 * constants.intermediate_dim * (constants.hidden_dim / 2 + constants.hidden_dim / 16) +
            constants.hidden_dim * (constants.intermediate_dim / 2 + constants.intermediate_dim / 16);
        const per_expert_all_layers: u64 = per_expert_layer * constants.layer_count;
        const experts_total: u64 = per_expert_all_layers * constants.expert_count;
        // Dense attention/router/controllers/norms/LM head (+ DSpark) measured at ~0.61 / ~0.82 GiB.
        const non_expert: u64 = if (has_mtp) 880 * 1024 * 1024 else 655 * 1024 * 1024;
        const margin: u64 = 96 * 1024 * 1024; // desktop VRAM jitter + arena slab rounding (overflow degrades gracefully to DDR5)
        const need: u64 = non_expert + experts_total + runtime_vram_estimate(ctx_tokens, has_mtp) + margin;
        const gib = 1024.0 * 1024.0 * 1024.0;
        if (need <= avail) {
            tiger.print_stdout("[tier] VRAM budget {d:.2} GiB free | need {d:.2} GiB -> all experts in VRAM\n", .{
                @as(f64, @floatFromInt(avail)) / gib, @as(f64, @floatFromInt(need)) / gib,
            });
            return 0;
        }
        const deficit = need - avail;
        const n: u64 = @min((deficit + per_expert_all_layers - 1) / per_expert_all_layers, constants.expert_count / 2);
        tiger.print_stdout("[tier] VRAM budget {d:.2} GiB free | need {d:.2} GiB -> {d}/128 experts per layer to DDR5 ({d:.2} GiB) so the KV cache stays in VRAM\n", .{
            @as(f64, @floatFromInt(avail)) / gib,  @as(f64, @floatFromInt(need)) / gib,
            n,                                      @as(f64, @floatFromInt(n * per_expert_all_layers)) / gib,
        });
        return @intCast(n);
    }
};
