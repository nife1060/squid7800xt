// Vulkan 1.4 Compute Pipeline Registry & BDA Push-Constant Dispatchers for RX 7800 XT (Wave64).

const std = @import("std");
const constants = @import("constants.zig");
const tiger = @import("tiger.zig");
const vk = @import("vulkan.zig");
const weights_mod = @import("weights.zig");

const spv_embed_norm align(4) = @embedFile("spv_embed_norm");
const spv_depth_norm align(4) = @embedFile("spv_depth_norm");
const spv_controller_nvfp4 align(4) = @embedFile("spv_controller_nvfp4");
const spv_linear_nvfp4 align(4) = @embedFile("spv_linear_nvfp4");
const spv_kv_sconv align(4) = @embedFile("spv_kv_sconv");
const spv_attn_prep_store align(4) = @embedFile("spv_attn_prep_store");
const spv_attn_tile align(4) = @embedFile("spv_attn_tile");
const spv_attn_merge_gate align(4) = @embedFile("spv_attn_merge_gate");
const spv_post_res align(4) = @embedFile("spv_post_res");
const spv_post_depth_norm align(4) = @embedFile("spv_post_depth_norm");
const spv_route_topk align(4) = @embedFile("spv_route_topk");
const spv_moe_up align(4) = @embedFile("spv_moe_up");
const spv_moe_down align(4) = @embedFile("spv_moe_down");
const spv_lm_head_argmax align(4) = @embedFile("spv_lm_head_argmax");
const spv_dspark_commit align(4) = @embedFile("spv_dspark_commit");
const spv_dspark_ops align(4) = @embedFile("spv_dspark_ops");
const spv_dspark_attn align(4) = @embedFile("spv_dspark_attn");
const spv_dspark_markov align(4) = @embedFile("spv_dspark_markov");
const spv_linear_wmma_nvfp4 align(4) = @embedFile("spv_linear_wmma_nvfp4");
const spv_moe_prefill_b64 align(4) = @embedFile("spv_moe_prefill_b64");

pub const EmbedNormPC = extern struct {
    embed_addr: u64,
    norm_addr: u64,
    tokens_addr: u64,
    out_addr: u64,
    B: u32,
    mode: u32,
    src_stride_elems: u32,
    slot_idx: u32,
};

pub const DepthNormPC = extern struct {
    hidden_addr: u64,
    partial_out_addr: u64,
    blocks_addr: u64,
    queries_addr: u64,
    norm0_addr: u64,
    norm1_addr: u64,
    norm2_addr: u64,
    out0_addr: u64,
    out1_addr: u64,
    out2_addr: u64,
    out_h0_addr: u64,
    out_h1_addr: u64,
    out_h2_addr: u64,
    n_blocks: u32,
    B: u32,
};

pub const CtrlDesc = extern struct {
    cp_q_addr: u64 = 0,
    down_w_addr: u64 = 0,
    down_s_addr: u64 = 0,
    forget_w_addr: u64 = 0,
    forget_s_addr: u64 = 0,
    gain_w_addr: u64 = 0,
    gain_s_addr: u64 = 0,
    post_norm_addr: u64 = 0,
    down_gs: f32 = 1.0,
    forget_gs: f32 = 1.0,
    gain_gs: f32 = 1.0,
    pad: u32 = 0,
};

comptime {
    if (@sizeOf(CtrlDesc) != 80) @compileError("CtrlDesc size mismatch");
}

pub const ControllerPC = extern struct {
    desc_base_addr: u64,
    x0_addr: u64,
    x1_addr: u64,
    x2_addr: u64,
    saved_addr: u64,
    prior_last_addr: u64,
    out0_addr: u64,
    out1_addr: u64,
    out2_addr: u64,
    out_h0_addr: u64,
    out_h1_addr: u64,
    out_h2_addr: u64,
    B: u32,
};

pub const PostDepthNormPC = extern struct {
    src_addr: u64,
    sconv_w_addr: u64,
    ring_addr: u64,
    residual_addr: u64,
    out_res_addr: u64,
    blocks_addr: u64,
    partial_in_addr: u64,
    tap_out_addr: u64,
    d_pos_addr: u64,
    partial_out_addr: u64,
    queries_addr: u64,
    norm0_addr: u64,
    norm1_addr: u64,
    norm2_addr: u64,
    out0_addr: u64,
    out1_addr: u64,
    out2_addr: u64,
    out_h0_addr: u64,
    out_h1_addr: u64,
    out_h2_addr: u64,
    n_blocks: u32,
    B: u32,
    commit_block: u32,
    block_idx: u32,
};

pub const LinearPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    x_h_addr: u64,
    y_addr: u64,
    rows: u32,
    cols: u32,
    global_scale: f32,
    B: u32,
    add_residual: u32,
};

pub const KvSconvPC = extern struct {
    k_proj_addr: u64,
    v_proj_addr: u64,
    in_k_addr: u64,
    in_v_addr: u64,
    k_sconv_w_addr: u64,
    v_sconv_w_addr: u64,
    k_sconv_proj_addr: u64,
    v_sconv_proj_addr: u64,
    k_ring_addr: u64,
    v_ring_addr: u64,
    d_pos_addr: u64,
    B: u32,
    is_global: u32,
};

pub const AttnPrepPC = extern struct {
    qrgs_addr: u64,
    ck_addr: u64,
    cv_addr: u64,
    q_clip_scale_addr: u64,
    rel_clip_scale_addr: u64,
    k_cache_addr: u64,
    v_cache_addr: u64,
    k_scale_addr: u64,
    v_scale_addr: u64,
    d_pos_addr: u64,
    kv_cap: u32,
    is_global: u32,
    kv_fmt: u32,
};

/// attn_tile.comp: tiled split attention (one Wave64 per split, all rows x GQA share KV tiles).
pub const AttnTilePC = extern struct {
    q_addr: u64,
    r_addr: u64,
    rel_proj_addr: u64,
    k_cache_addr: u64,
    v_cache_addr: u64,
    k_scale_addr: u64,
    v_scale_addr: u64,
    split_buf_addr: u64,
    d_pos_addr: u64,
    q_row_stride: u32,
    cap: u32,
    kv_fmt: u32,
    span: u32,
    last_off: i32,
    row_step: u32,
    rows_total: u32,
    n_split_max: u32,
};

/// attn_merge_gate.comp: merge split partials + sink + sigmoid gate.
pub const AttnMergePC = extern struct {
    qrgs_addr: u64,
    split_buf_addr: u64,
    out_gated_addr: u64,
    out_gated_h_addr: u64,
    d_pos_addr: u64,
    span: u32,
    last_off: i32,
    row_step: u32,
    group_rows: u32,
    rows_total: u32,
    n_split_max: u32,
};

pub const PostResPC = extern struct {
    src_addr: u64,
    sconv_w_addr: u64,
    ring_addr: u64,
    residual_addr: u64,
    out_addr: u64,
    blocks_addr: u64,
    partial_addr: u64,
    tap_out_addr: u64,
    d_pos_addr: u64,
    B: u32,
    block_idx: u32,
};

pub const RouteTopkPC = extern struct {
    logits_addr: u64,
    bias_addr: u64,
    out_ids_addr: u64,
    out_w_addr: u64,
    groups_addr: u64,
    B: u32,
};

pub const MoeUpPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    exp_sc_addr: u64,
    x_h_addr: u64,
    groups_addr: u64,
    out_act_h_addr: u64,
    B: u32,
    n_lo: u32 = constants.expert_count,
    w_hi_addr: u64 = 0,
    s_hi_addr: u64 = 0,
};

pub const MoeDownPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    exp_sc_addr: u64,
    act_h_addr: u64,
    groups_addr: u64,
    out_down_addr: u64,
    B: u32,
    n_lo: u32 = constants.expert_count,
    w_hi_addr: u64 = 0,
    s_hi_addr: u64 = 0,
};

pub const LinearWmmaPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    x_h_addr: u64,
    y_addr: u64,
    N: u32,
    K: u32,
    global_scale: f32,
    add_residual: u32,
};

pub const MoePrefillB64PC = extern struct {
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    mode: u32,
    n_lo: u32 = constants.expert_count,
    a6: u64 = 0,
    a7: u64 = 0,
};

pub const LmHeadPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    x_addr: u64,
    blk_val_addr: u64,
    blk_idx_addr: u64,
    out_tokens_addr: u64,
    d_pos_addr: u64,
    d_tokens_addr: u64,
    sample_cfg_addr: u64,
    V: u32,
    global_scale: f32,
    B: u32,
    stage: u32,
    advance_b1: u32,
};

pub const DsparkCommitPC = extern struct {
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    d_pos_addr: u64,
    B: u32,
    cap: u32,
    kv_fmt: u32,
    mode: u32,
};

pub const DsparkOpsPC = extern struct {
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    param0: u32,
    mode: u32,
};

pub const DsparkAttnPC = extern struct {
    qkv_addr: u64,
    ck_addr: u64,
    cv_addr: u64,
    cks_addr: u64,
    cvs_addr: u64,
    split_buf_addr: u64,
    out_h_addr: u64,
    d_pos_addr: u64,
    cap: u32,
    window: u32,
    kv_fmt: u32,
    mode: u32,
};

pub const DsparkMarkovPC = extern struct {
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
    a10: u64,
    p0: u32,
    p1: u32,
    p2: u32,
    gscale: f32,
    mode: u32,
};

const PipePair = struct {
    layout: vk.VkPipelineLayout = vk.VK_NULL_HANDLE,
    pipe: vk.VkPipeline = vk.VK_NULL_HANDLE,

    fn create(ctx: *const vk.Context, pc_size: u32, spv: []const u8, sg_size: u32) PipePair {
        tiger.assert(pc_size <= 256);
        const l = ctx.create_pipeline_layout(pc_size);
        const p = ctx.create_compute_pipeline(l, spv, sg_size);
        return .{ .layout = l, .pipe = p };
    }

    fn destroy(self: *PipePair, ctx: *const vk.Context) void {
        if (self.pipe != vk.VK_NULL_HANDLE) ctx.vkDestroyPipeline(ctx.device, self.pipe, null);
        if (self.layout != vk.VK_NULL_HANDLE) ctx.vkDestroyPipelineLayout(ctx.device, self.layout, null);
        self.* = .{};
    }
};

const SpecPipeSet = struct {
    layout: vk.VkPipelineLayout = vk.VK_NULL_HANDLE,
    pipes: [9]vk.VkPipeline = @splat(vk.VK_NULL_HANDLE),

    fn create(ctx: *const vk.Context, pc_size: u32, spv: []const u8, sg_size: u32, bs: []const u32) SpecPipeSet {
        tiger.assert(pc_size <= 256);
        var set: SpecPipeSet = .{ .layout = ctx.create_pipeline_layout(pc_size) };
        for (bs) |b| {
            tiger.assert(b >= 1 and b <= 8);
            set.pipes[b] = ctx.create_compute_pipeline_spec(set.layout, spv, sg_size, b);
        }
        return set;
    }

    pub fn pair(self: *const SpecPipeSet, b: u32) PipePair {
        tiger.assert(b >= 1 and b <= 8 and self.pipes[b] != vk.VK_NULL_HANDLE);
        return .{ .layout = self.layout, .pipe = self.pipes[b] };
    }

    fn destroy(self: *SpecPipeSet, ctx: *const vk.Context) void {
        for (&self.pipes) |*p| {
            if (p.* != vk.VK_NULL_HANDLE) {
                ctx.vkDestroyPipeline(ctx.device, p.*, null);
                p.* = vk.VK_NULL_HANDLE;
            }
        }
        if (self.layout != vk.VK_NULL_HANDLE) ctx.vkDestroyPipelineLayout(ctx.device, self.layout, null);
        self.layout = vk.VK_NULL_HANDLE;
    }
};

pub const Pipelines = struct {
    ctx: *const vk.Context = undefined,
    embed_norm: PipePair = .{},
    depth_norm: PipePair = .{},
    controller: PipePair = .{},
    linear: SpecPipeSet = .{},
    kv_sconv: PipePair = .{},
    attn_prep: PipePair = .{},
    attn_tile: SpecPipeSet = .{},
    attn_merge: PipePair = .{},
    post_res: PipePair = .{},
    post_depth_norm: PipePair = .{},
    route_topk: PipePair = .{},
    moe_up: SpecPipeSet = .{},
    moe_down: SpecPipeSet = .{},
    lm_head: SpecPipeSet = .{},
    dspark_commit: PipePair = .{},
    dspark_ops: PipePair = .{},
    dspark_attn: PipePair = .{},
    dspark_markov: PipePair = .{},
    linear_wmma: PipePair = .{},
    moe_prefill_b64: PipePair = .{},

    pub fn init(ctx: *const vk.Context) Pipelines {
        const linear_bs = [_]u32{ 1, 2, 3, 4, 7, 8 };
        const verify_bs = [_]u32{ 1, 2, 4, 8 };
        const attn_bs = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        return .{
            .ctx = ctx,
            .embed_norm = PipePair.create(ctx, @sizeOf(EmbedNormPC), spv_embed_norm, 64),
            .depth_norm = PipePair.create(ctx, @sizeOf(DepthNormPC), spv_depth_norm, 64),
            .controller = PipePair.create(ctx, @sizeOf(ControllerPC), spv_controller_nvfp4, 64),
            .linear = SpecPipeSet.create(ctx, @sizeOf(LinearPC), spv_linear_nvfp4, 64, &linear_bs),
            .kv_sconv = PipePair.create(ctx, @sizeOf(KvSconvPC), spv_kv_sconv, 64),
            .attn_prep = PipePair.create(ctx, @sizeOf(AttnPrepPC), spv_attn_prep_store, 64),
            .attn_tile = SpecPipeSet.create(ctx, @sizeOf(AttnTilePC), spv_attn_tile, 64, &attn_bs),
            .attn_merge = PipePair.create(ctx, @sizeOf(AttnMergePC), spv_attn_merge_gate, 64),
            .post_res = PipePair.create(ctx, @sizeOf(PostResPC), spv_post_res, 64),
            .post_depth_norm = PipePair.create(ctx, @sizeOf(PostDepthNormPC), spv_post_depth_norm, 64),
            .route_topk = PipePair.create(ctx, @sizeOf(RouteTopkPC), spv_route_topk, 64),
            .moe_up = SpecPipeSet.create(ctx, @sizeOf(MoeUpPC), spv_moe_up, 64, &verify_bs),
            .moe_down = SpecPipeSet.create(ctx, @sizeOf(MoeDownPC), spv_moe_down, 64, &verify_bs),
            .lm_head = SpecPipeSet.create(ctx, @sizeOf(LmHeadPC), spv_lm_head_argmax, 64, &verify_bs),
            .dspark_commit = PipePair.create(ctx, @sizeOf(DsparkCommitPC), spv_dspark_commit, 64),
            .dspark_ops = PipePair.create(ctx, @sizeOf(DsparkOpsPC), spv_dspark_ops, 64),
            .dspark_attn = PipePair.create(ctx, @sizeOf(DsparkAttnPC), spv_dspark_attn, 64),
            .dspark_markov = PipePair.create(ctx, @sizeOf(DsparkMarkovPC), spv_dspark_markov, 64),
            .linear_wmma = if (ctx.has_coop_matrix) PipePair.create(ctx, @sizeOf(LinearWmmaPC), spv_linear_wmma_nvfp4, 32) else .{},
            .moe_prefill_b64 = PipePair.create(ctx, @sizeOf(MoePrefillB64PC), spv_moe_prefill_b64, 64),
        };
    }

    pub fn deinit(self: *Pipelines) void {
        self.embed_norm.destroy(self.ctx);
        self.depth_norm.destroy(self.ctx);
        self.controller.destroy(self.ctx);
        self.linear.destroy(self.ctx);
        self.kv_sconv.destroy(self.ctx);
        self.attn_prep.destroy(self.ctx);
        self.attn_tile.destroy(self.ctx);
        self.attn_merge.destroy(self.ctx);
        self.post_res.destroy(self.ctx);
        self.post_depth_norm.destroy(self.ctx);
        self.route_topk.destroy(self.ctx);
        self.moe_up.destroy(self.ctx);
        self.moe_down.destroy(self.ctx);
        self.lm_head.destroy(self.ctx);
        self.dspark_commit.destroy(self.ctx);
        self.dspark_ops.destroy(self.ctx);
        self.dspark_attn.destroy(self.ctx);
        self.dspark_markov.destroy(self.ctx);
        self.linear_wmma.destroy(self.ctx);
        self.moe_prefill_b64.destroy(self.ctx);
    }

    pub fn cmd_linear_wmma(self: *const Pipelines, cmd: vk.VkCommandBuffer, lw: *const weights_mod.LinearWeight, x_h: u64, y: u64, add_res: bool) void {
        const pc: LinearWmmaPC = .{
            .w_addr = lw.data,
            .s_addr = lw.scales,
            .x_h_addr = x_h,
            .y_addr = y,
            .N = lw.rows,
            .K = lw.cols,
            .global_scale = lw.global_scale,
            .add_residual = if (add_res) 1 else 0,
        };
        self.dispatch(cmd, self.linear_wmma, pc, (lw.rows + 63) / 64, 1, 1);
    }

    pub fn dispatch(self: *const Pipelines, cmd: vk.VkCommandBuffer, pair: PipePair, pc: anytype, gx: u32, gy: u32, gz: u32) void {
        self.ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pair.pipe);
        self.ctx.vkCmdPushConstants(cmd, pair.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(pc)), &pc);
        self.ctx.vkCmdDispatch(cmd, gx, gy, gz);
    }

    pub fn cmd_linear(self: *const Pipelines, cmd: vk.VkCommandBuffer, lw: *const weights_mod.LinearWeight, x_h: u64, y: u64, B: u32, add_res: bool) void {
        const pc: LinearPC = .{
            .w_addr = lw.data,
            .s_addr = lw.scales,
            .x_h_addr = x_h,
            .y_addr = y,
            .rows = lw.rows,
            .cols = lw.cols,
            .global_scale = lw.global_scale,
            .B = B,
            .add_residual = if (add_res) 1 else 0,
        };
        const gx: u32 = (lw.rows + 15) / 16;
        self.dispatch(cmd, self.linear.pair(B), pc, gx, 1, 1);
    }
};
