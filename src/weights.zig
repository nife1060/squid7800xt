// TigerStyle Inkling-2-Mini + DSpark MTP weight loader for RX 7800 XT (Vulkan 1.4 BDA).
// Packs all dense layers, L20..23 MXFP8 experts, LM head, and DSpark into NVFP4
// inside multi-slab VRAM arenas (~13.70 GiB total -> 0.0 GiB weight spill on 16 GiB RX 7800 XT!).
// BF16 embeddings live in zero-copy host-pinned DDR5 (saves 0.86 GiB VRAM; 4.6 KB/tok read).

const std = @import("std");
const constants = @import("constants.zig");
const safetensors = @import("safetensors.zig");
const tiger = @import("tiger.zig");
const vk = @import("vulkan.zig");
const vram_arena = @import("vram_arena.zig");

const spv_pack_bf16 align(4) = @embedFile("spv_pack_bf16_nvfp4");
const spv_pack_mxfp8 align(4) = @embedFile("spv_pack_mxfp8_nvfp4");
const spv_pack_dspark align(4) = @embedFile("spv_pack_dspark");

pub const LinearWeight = struct {
    kind: constants.WeightKind = .nvfp4,
    rows: u32 = 0,
    cols: u32 = 0,
    data: u64 = 0, // VkDeviceAddress: NVFP4 U8 [rows, cols/2] or BF16 [rows, cols]
    scales: u64 = 0, // VkDeviceAddress: NVFP4 F8_E4M3 [rows, cols/16]
    global_scale: f32 = 1.0,
};

pub const ExpertBank = struct {
    n_experts: u32 = constants.expert_count,
    rows: u32 = 0,
    cols: u32 = 0,
    data: u64 = 0, // U8 [E, rows, cols/2]
    scales: u64 = 0, // F8_E4M3 [E, rows, cols/16]
    expert_scale: u64 = 0, // F32 [E]
    // Expert tiering: experts [0, n_lo) at data/scales (VRAM); [n_lo, E) in pinned DDR5 at
    // data_hi/scales_hi, which are PRE-BIASED by -n_lo * stride so shaders index by expert id.
    n_lo: u32 = constants.expert_count,
    data_hi: u64 = 0,
    scales_hi: u64 = 0,
};

pub const ControllerWeights = struct {
    down_nvfp4: LinearWeight = .{},
    forget_nvfp4: LinearWeight = .{},
    gain_nvfp4: LinearWeight = .{},
};

pub const LayerWeights = struct {
    is_global: bool = false,
    has_controllers: bool = false,

    attn_norm: u64 = 0,
    q_in_norm: u64 = 0,
    k_in_norm: u64 = 0,
    v_in_norm: u64 = 0,
    mlp_norm: u64 = 0,

    w_qrgs: LinearWeight = .{},
    w_kv: LinearWeight = .{},
    w_k: LinearWeight = .{},
    w_v: LinearWeight = .{},
    w_o: LinearWeight = .{},

    k_sconv_w: u64 = 0,
    v_sconv_w: u64 = 0,
    k_sconv_proj: u64 = 0,
    v_sconv_proj: u64 = 0,
    attn_sconv_w: u64 = 0,
    mlp_sconv_w: u64 = 0,

    q_clip_scale: u64 = 0,
    rel_clip_scale: u64 = 0,
    rel_logits_proj: u64 = 0,

    ctrl: [4]ControllerWeights = @splat(.{}),

    router: LinearWeight = .{}, // BF16 [expert_count, hidden_dim]: kept exact, routing flips are discrete errors
    router_bias: u64 = 0,
    w13: ExpertBank = .{},
    w2: ExpertBank = .{},
};

pub const DsparkLayerWeights = struct {
    is_local: bool = false,
    input_norm: u64 = 0,
    post_norm: u64 = 0,
    dflare_norm: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    k_norm_ctx: u64 = 0,

    w_qkv: LinearWeight = .{},
    w_o: LinearWeight = .{},
    w_gu: LinearWeight = .{},
    w_down: LinearWeight = .{},
    w_kv_ctx: LinearWeight = .{},
};

pub const DsparkWeights = struct {
    loaded: bool = false,
    fuse_alpha: u64 = 0, // [4, 4] F32
    mask_embed: u64 = 0, // [2304] BF16
    norm_w: u64 = 0, // [2304] BF16
    markov_dflare_norm: u64 = 0, // [2304] BF16
    markov_w1: u64 = 0, // [201024, 256] BF16 (102.9 MB)
    rnn_w_sp: u64 = 0, // [768, 512] BF16
    rnn_w_h: u64 = 0, // [768, 2304] BF16
    rnn_w_f: u64 = 0, // [768, 2304] BF16
    rnn_bias: u64 = 0, // [768] BF16
    hidden_out_w: u64 = 0, // [2304, 256] BF16
    hidden_out_b: u64 = 0, // [2304] BF16

    draft_vocab_size: u32 = 49152,
    draft_vocab_ids: u64 = 0, // optional [draft_vocab_size] I32 remap to real vocab IDs
    unembed_draft: LinearWeight = .{}, // [draft_vocab_size, 2304] NVFP4
    fold_w: LinearWeight = .{}, // [draft_vocab_size, 256] NVFP4
    fold_b: u64 = 0, // [draft_vocab_size] F32

    layers: [constants.mtp_layer_count]DsparkLayerWeights = @splat(.{}),
};

const PackBf16PC = extern struct {
    src_addr: u64,
    dst_w_addr: u64,
    dst_s_addr: u64,
    amax_addr: u64,
    total_elems: u32,
    mode: u32,
};

const PackMxfp8PC = extern struct {
    src_w_addr: u64,
    src_s_addr: u64,
    dst_w_addr: u64,
    dst_s_addr: u64,
    exp_scale_addr: u64,
    n_experts: u32,
    rows: u32,
    cols: u32,
    mode: u32,
    n_lo: u32,
    dst_w_hi_addr: u64,
    dst_s_hi_addr: u64,
};

const PackDsparkPC = extern struct {
    addr0: u64,
    addr1: u64,
    addr2: u64,
    addr3: u64,
    V: u32,
    mode: u32,
};

const PackContext = struct {
    ctx: *const vk.Context,
    layout_bf16: vk.VkPipelineLayout,
    pipe_bf16: vk.VkPipeline,
    layout_mxfp8: vk.VkPipelineLayout,
    pipe_mxfp8: vk.VkPipeline,
    layout_ds: vk.VkPipelineLayout,
    pipe_ds: vk.VkPipeline,
    stage: vk.GpuBuffer, // 1 GiB in host-pinned DDR5 (0 VRAM!)
    amax_buf: vk.GpuBuffer, // 16 bytes in host-cached DDR5

    fn init(ctx: *const vk.Context, stage_bytes: u64) PackContext {
        const l_bf16 = ctx.create_pipeline_layout(@sizeOf(PackBf16PC));
        const p_bf16 = ctx.create_compute_pipeline(l_bf16, spv_pack_bf16, 64);
        const l_mx = ctx.create_pipeline_layout(@sizeOf(PackMxfp8PC));
        const p_mx = ctx.create_compute_pipeline(l_mx, spv_pack_mxfp8, 64);
        const l_ds = ctx.create_pipeline_layout(@sizeOf(PackDsparkPC));
        const p_ds = ctx.create_compute_pipeline(l_ds, spv_pack_dspark, 64);

        return .{
            .ctx = ctx,
            .layout_bf16 = l_bf16,
            .pipe_bf16 = p_bf16,
            .layout_mxfp8 = l_mx,
            .pipe_mxfp8 = p_mx,
            .layout_ds = l_ds,
            .pipe_ds = p_ds,
            .stage = ctx.create_host_buffer(stage_bytes, false, true),
            .amax_buf = ctx.create_host_buffer(64, true, true),
        };
    }

    fn resize_stage(self: *PackContext, new_bytes: u64) void {
        if (self.stage.size == new_bytes) return;
        _ = self.ctx.vkDeviceWaitIdle(self.ctx.device);
        self.ctx.destroy_buffer(&self.stage);
        self.stage = self.ctx.create_host_buffer(new_bytes, false, true);
    }

    fn deinit(self: *PackContext) void {
        self.ctx.destroy_buffer(&self.stage);
        self.ctx.destroy_buffer(&self.amax_buf);
        self.ctx.vkDestroyPipeline(self.ctx.device, self.pipe_bf16, null);
        self.ctx.vkDestroyPipelineLayout(self.ctx.device, self.layout_bf16, null);
        self.ctx.vkDestroyPipeline(self.ctx.device, self.pipe_mxfp8, null);
        self.ctx.vkDestroyPipelineLayout(self.ctx.device, self.layout_mxfp8, null);
        self.ctx.vkDestroyPipeline(self.ctx.device, self.pipe_ds, null);
        self.ctx.vkDestroyPipelineLayout(self.ctx.device, self.layout_ds, null);
    }
};

pub const ModelWeights = struct {
    ctx: ?*const vk.Context = null,
    arena: vram_arena.VRAMArena = .{},
    host_embed_buf: vk.GpuBuffer = .{},
    host_markov_w1_buf: vk.GpuBuffer = .{},
    /// Pinned-DDR5 arena for the cold expert tier (see `expert_hi_count`).
    host_expert_arena: vram_arena.VRAMArena = .{},
    /// Experts per bank (per layer, w13 and w2) placed in DDR5 instead of VRAM. Set before load().
    /// Every token reads all of the KV cache but only 8/128 experts per layer, so under VRAM
    /// pressure the tail experts are the cheapest bytes to evict.
    expert_hi_count: u32 = 0,

    embed: u64 = 0,
    embed_norm: u64 = 0,
    final_norm: u64 = 0,
    unembed: LinearWeight = .{},

    attn_queries: [constants.layer_count - 1]u64 = @splat(0),
    mlp_queries: [constants.layer_count]u64 = @splat(0),
    output_query: u64 = 0,
    cross_pass_queries: u64 = 0,
    cross_pass_kv_queries: u64 = 0,

    layers: [constants.layer_count]LayerWeights = @splat(.{}),
    dspark: DsparkWeights = .{},

    pub fn deinit(self: *ModelWeights) void {
        self.arena.deinit();
        self.host_expert_arena.deinit();
        if (self.ctx) |ctx| {
            ctx.destroy_buffer(&self.host_embed_buf);
            ctx.destroy_buffer(&self.host_markov_w1_buf);
        }
        self.* = .{};
    }

    pub fn load(
        self: *ModelWeights,
        ctx: *const vk.Context,
        model_directory: []const u8,
        load_mtp: bool,
        draft_vocab_limit: u32,
    ) bool {
        tiger.assert(model_directory.len > 0);
        const start_ns = tiger.clock_monotonic_ns();
        self.ctx = ctx;

        var main_path_buf: [1024]u8 = undefined;
        const main_path_z = tiger.format_path_z(&main_path_buf, model_directory, "/model-00001-of-00001.safetensors");
        var sf_main: safetensors.SafetensorsFile = .{};
        if (!sf_main.open_file(main_path_z)) {
            tiger.print_stderr("[weights] Failed to open {s}/model-00001-of-00001.safetensors\n", .{model_directory});
            return false;
        }
        defer sf_main.close_file();

        var mtp_path_buf: [1024]u8 = undefined;
        const mtp_path_z = tiger.format_path_z(&mtp_path_buf, model_directory, "/mtp.safetensors");
        var sf_mtp: safetensors.SafetensorsFile = .{};
        const has_mtp: bool = load_mtp and sf_mtp.open_file(mtp_path_z);
        defer if (has_mtp) sf_mtp.close_file();

        // Use pure DEVICE_LOCAL VRAM arena (uploaded via 1 GiB host-pinned staging buffer)
        // with graceful DDR5 fallback if Windows WDDM desktop apps borrow VRAM!
        self.arena = vram_arena.VRAMArena.init(ctx, ctx.mem_type_vram_only, false);
        self.arena.fallback_mem_type_idx = if (ctx.mem_type_host_wc != std.math.maxInt(u32)) ctx.mem_type_host_wc else ctx.mem_type_host_cached;
        tiger.assert(self.expert_hi_count < constants.expert_count);
        if (self.expert_hi_count > 0) {
            const host_mt = if (ctx.mem_type_host_wc != std.math.maxInt(u32)) ctx.mem_type_host_wc else ctx.mem_type_host_cached;
            self.host_expert_arena = vram_arena.VRAMArena.init(ctx, host_mt, true);
            self.host_expert_arena.slab_size_max = 512 * 1024 * 1024;
        }

        var pk = PackContext.init(ctx, 900 * 1024 * 1024);
        defer pk.deinit();

        // 1. Pack Unembed (LM Head, 879 MiB BF16 -> 440 MiB per-row FP8 E4M3) FIRST while host budget is empty, then shrink stage to 704 MiB!
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const usable_bytes: usize = @as(usize, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
            tiger.assert(ui.byte_length >= usable_bytes);
            @memcpy(pk.stage.mapped.?[0..usable_bytes], sf_main.data_slice(ui)[0..usable_bytes]);
            self.unembed = self.pack_bf16_to_fp8(&pk, pk.stage.address, constants.vocab_unpadded_count, constants.hidden_dim);
        }
        pk.resize_stage(704 * 1024 * 1024);

        // 2. Embeddings -> zero-copy host-pinned DDR5 (GPU-read-only -> prefer host_wc!)
        {
            const ei = sf_main.lookup("model.llm.embed.weight");
            const embed_bytes: u64 = @as(u64, constants.vocab_padded_count) * constants.hidden_dim * 2;
            tiger.assert(ei.byte_length == embed_bytes);
            self.host_embed_buf = ctx.create_host_buffer(embed_bytes, false, true);
            @memcpy(self.host_embed_buf.mapped.?[0..@intCast(embed_bytes)], sf_main.data_slice(ei));
            self.embed = self.host_embed_buf.address;
            tiger.print_stdout("[weights] Embed table pinned in zero-copy DDR5 ({d:.2} GiB saved from VRAM)\n", .{
                @as(f64, @floatFromInt(embed_bytes)) / (1024.0 * 1024.0 * 1024.0),
            });
        }

        self.embed_norm = self.upload_tensor(&sf_main, "model.llm.embed_norm.weight", constants.hidden_dim * 2, &pk.stage);
        self.final_norm = self.upload_tensor(&sf_main, "model.llm.norm.weight", constants.hidden_dim * 2, &pk.stage);

        // 3. AttnRes & Cross-Pass Queries (F32)
        var nbuf: [128]u8 = undefined;
        var layer_idx: u32 = 1;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const elems: u64 = if (constants.is_global_layer(layer_idx)) 3 * constants.hidden_dim else constants.hidden_dim;
            const tname = std.fmt.bufPrint(&nbuf, "model.llm.attn_res.attn_queries.{d}", .{layer_idx - 1}) catch unreachable;
            self.attn_queries[layer_idx - 1] = self.upload_tensor(&sf_main, tname, elems * 4, &pk.stage);
        }
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const tname = std.fmt.bufPrint(&nbuf, "model.llm.attn_res.mlp_queries.{d}", .{layer_idx}) catch unreachable;
            self.mlp_queries[layer_idx] = self.upload_tensor(&sf_main, tname, @as(u64, constants.hidden_dim) * 4, &pk.stage);
        }
        self.output_query = self.upload_tensor(&sf_main, "model.llm.attn_res.output_query", @as(u64, constants.hidden_dim) * 4, &pk.stage);
        self.cross_pass_queries = self.upload_tensor(&sf_main, "model.llm.middle_cycle_cross_pass_queries", 24 * @as(u64, constants.hidden_dim) * 4, &pk.stage);
        self.cross_pass_kv_queries = self.upload_tensor(&sf_main, "model.llm.middle_cycle_cross_pass_kv_queries", 24 * @as(u64, constants.hidden_dim) * 4, &pk.stage);

        // 4. 24 Physical Layers
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const lw = &self.layers[layer_idx];
            lw.is_global = constants.is_global_layer(layer_idx);

            if (lw.is_global) {
                lw.q_in_norm = self.upload_layer_tensor(&sf_main, layer_idx, "q_input_norm.weight", constants.hidden_dim * 2, &pk.stage);
                lw.k_in_norm = self.upload_layer_tensor(&sf_main, layer_idx, "k_input_norm.weight", constants.hidden_dim * 2, &pk.stage);
                lw.v_in_norm = self.upload_layer_tensor(&sf_main, layer_idx, "v_input_norm.weight", constants.hidden_dim * 2, &pk.stage);
            } else {
                lw.attn_norm = self.upload_layer_tensor(&sf_main, layer_idx, "attn_norm.weight", constants.hidden_dim * 2, &pk.stage);
            }
            lw.mlp_norm = self.upload_layer_tensor(&sf_main, layer_idx, "mlp_norm.weight", constants.hidden_dim * 2, &pk.stage);

            // Fused w_qrgs [6552, 2304] -> NVFP4
            {
                const dst = pk.stage.mapped.?;
                var off: usize = 0;
                off += copy_layer_tensor(&sf_main, layer_idx, "attn.wq_du.weight", dst + off, @as(usize, constants.query_dim) * constants.hidden_dim * 2);
                off += copy_layer_tensor(&sf_main, layer_idx, "attn.wr_du.weight", dst + off, @as(usize, constants.r_dim) * constants.hidden_dim * 2);
                off += copy_layer_tensor(&sf_main, layer_idx, "attn.gate_attn.weight", dst + off, @as(usize, constants.query_dim) * constants.hidden_dim * 2);
                off += copy_layer_tensor(&sf_main, layer_idx, "attn.attn_sink_proj.weight", dst + off, @as(usize, constants.head_count) * constants.hidden_dim * 2);
                tiger.assert(off == @as(usize, constants.qrgs_out_dim) * constants.hidden_dim * 2);
                lw.w_qrgs = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, constants.qrgs_out_dim, constants.hidden_dim);
            }

            // Fused w_kv [2048, 2304] -> NVFP4, sliced into w_k and w_v
            {
                const dst = pk.stage.mapped.?;
                const kv_b: usize = @as(usize, constants.kv_dim) * constants.hidden_dim * 2;
                _ = copy_layer_tensor(&sf_main, layer_idx, "attn.wk_dv.weight", dst, kv_b);
                _ = copy_layer_tensor(&sf_main, layer_idx, "attn.wv_dv.weight", dst + kv_b, kv_b);
                lw.w_kv = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, 2 * constants.kv_dim, constants.hidden_dim);

                lw.w_k = lw.w_kv;
                lw.w_k.rows = constants.kv_dim;
                lw.w_v = lw.w_kv;
                lw.w_v.rows = constants.kv_dim;
                lw.w_v.data += @as(u64, constants.kv_dim) * (constants.hidden_dim / 2);
                lw.w_v.scales += @as(u64, constants.kv_dim) * (constants.hidden_dim / 16);
            }

            // w_o [2304, 3072] -> NVFP4
            {
                _ = copy_layer_tensor(&sf_main, layer_idx, "attn.wo_ud.weight", pk.stage.mapped.?, @as(usize, constants.hidden_dim) * constants.query_dim * 2);
                lw.w_o = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, constants.hidden_dim, constants.query_dim);
            }

            // Short convs & clip/rel projections
            if (lw.is_global) {
                lw.k_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "attn.k_sconv.static_w", constants.conv_tap_count * constants.kv_dim * 2, &pk.stage);
                lw.v_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "attn.v_sconv.static_w", constants.conv_tap_count * constants.kv_dim * 2, &pk.stage);
                lw.k_sconv_proj = self.upload_layer_tensor(&sf_main, layer_idx, "attn.k_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim * 2, &pk.stage);
                lw.v_sconv_proj = self.upload_layer_tensor(&sf_main, layer_idx, "attn.v_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim * 2, &pk.stage);
            } else {
                lw.k_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "attn.k_sconv.weight", constants.kv_dim * constants.conv_tap_count * 2, &pk.stage);
                lw.v_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "attn.v_sconv.weight", constants.kv_dim * constants.conv_tap_count * 2, &pk.stage);
            }
            lw.attn_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "attn_sconv.weight", constants.hidden_dim * constants.conv_tap_count * 2, &pk.stage);
            lw.mlp_sconv_w = self.upload_layer_tensor(&sf_main, layer_idx, "mlp_sconv.weight", constants.hidden_dim * constants.conv_tap_count * 2, &pk.stage);
            lw.q_clip_scale = self.upload_layer_tensor(&sf_main, layer_idx, "attn.q_l2_clip.scale", constants.head_count * constants.head_dim * 2, &pk.stage);
            lw.rel_clip_scale = self.upload_layer_tensor(&sf_main, layer_idx, "attn.rel_l2_clip.scale", constants.head_count * constants.rel_dim * 2, &pk.stage);
            lw.rel_logits_proj = self.upload_layer_tensor(&sf_main, layer_idx, "attn.rel_logits_proj.proj", constants.rel_dim * constants.rel_extent * 2, &pk.stage);

            // Cross-pass controllers (layers 6..17) -> NVFP4
            if (layer_idx >= constants.loop_start_layer and layer_idx < constants.loop_end_layer) {
                lw.has_controllers = true;
                const stems: [4][]const u8 = .{ "attn", "k", "v", "mlp" };
                var s_idx: u32 = 0;
                while (s_idx < 4) : (s_idx += 1) {
                    const dn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.controllers.{s}.down_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    copy_named_tensor(&sf_main, dn, pk.stage.mapped.?, @as(usize, constants.gate_rank) * 2 * constants.hidden_dim * 2);
                    lw.ctrl[s_idx].down_nvfp4 = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, constants.gate_rank, 2 * constants.hidden_dim);

                    const fn_name = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    copy_named_tensor(&sf_main, fn_name, pk.stage.mapped.?, @as(usize, constants.hidden_dim) * constants.gate_rank * 2);
                    lw.ctrl[s_idx].forget_nvfp4 = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, constants.hidden_dim, constants.gate_rank);

                    const gn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    copy_named_tensor(&sf_main, gn, pk.stage.mapped.?, @as(usize, constants.hidden_dim) * constants.gate_rank * 2);
                    lw.ctrl[s_idx].gain_nvfp4 = self.pack_bf16_to_nvfp4(&pk, pk.stage.address, constants.hidden_dim, constants.gate_rank);
                }
            }

            // Router -> BF16 (exact, ~0.6 MB/layer) + F32 bias
            {
                lw.router = .{
                    .kind = .bf16,
                    .rows = constants.expert_count,
                    .cols = constants.hidden_dim,
                    .data = self.upload_layer_tensor(&sf_main, layer_idx, "mlp.gate.weight", @as(u64, constants.expert_count) * constants.hidden_dim * 2, &pk.stage),
                };
                const rb_name = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.gate.bias", .{layer_idx}) catch unreachable;
                lw.router_bias = self.upload_tensor(&sf_main, rb_name, constants.expert_count * 4, &pk.stage);
            }

            // MoE Expert Banks (w13 & w2)
            lw.w13 = self.load_expert_bank(&sf_main, &pk, layer_idx, "w13", 2 * constants.intermediate_dim, constants.hidden_dim);
            lw.w2 = self.load_expert_bank(&sf_main, &pk, layer_idx, "w2", constants.hidden_dim, constants.intermediate_dim);
        }

        // 5. DSpark MTP weights (shrink staging buffer from 704 MiB -> 256 MiB first to free WDDM budget!)
        if (has_mtp) {
            const ds_stage_bytes: u64 = if (draft_vocab_limit > 0 and draft_vocab_limit <= 49152)
                256 * 1024 * 1024
            else
                1024 * 1024 * 1024;
            pk.resize_stage(ds_stage_bytes);
            self.load_dspark(model_directory, &sf_main, &sf_mtp, &pk, draft_vocab_limit);
        }

        const elapsed_sec = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - start_ns)) / 1e9;
        const vram_alloc_gib = @as(f64, @floatFromInt(self.arena.total_allocated_bytes)) / (1024.0 * 1024.0 * 1024.0);
        const vram_res_gib = @as(f64, @floatFromInt(self.arena.total_reserved_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[weights] Loaded NVFP4 in {d:.2}s | VRAM used={d:.2} GiB (reserved={d:.2} GiB across {d} slabs, {d} tensors) | MTP={s} (draft_vocab={d}) | DDR5 expert tier={d}/128 per layer ({d:.2} GiB)\n",
            .{
                elapsed_sec,
                vram_alloc_gib,
                vram_res_gib,
                self.arena.slab_count,
                self.arena.allocation_count,
                if (self.dspark.loaded) "on" else "off",
                self.dspark.draft_vocab_size,
                self.expert_hi_count,
                @as(f64, @floatFromInt(self.host_expert_arena.total_reserved_bytes + self.host_expert_arena.total_fallback_bytes)) / (1024.0 * 1024.0 * 1024.0),
            },
        );
        return true;
    }

    fn upload_tensor(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, expected_bytes: u64, stage: *vk.GpuBuffer) u64 {
        const info = sf.lookup(name);
        tiger.assert(info.byte_length == expected_bytes);
        return self.arena.upload_bytes(sf.data_slice(info), stage).addr;
    }

    /// Upload bytes [0, lo_bytes) to VRAM and [lo_bytes, total) to the pinned-DDR5 expert arena.
    /// Returns { vram_addr, hi_addr pre-biased by -lo_bytes } (hi == vram when nothing spills).
    fn upload_tensor_tiered(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, expected_bytes: u64, lo_bytes: u64, stage: *vk.GpuBuffer) [2]u64 {
        const info = sf.lookup(name);
        tiger.assert(info.byte_length == expected_bytes);
        tiger.assert(lo_bytes > 0 and lo_bytes <= expected_bytes);
        const bytes = sf.data_slice(info);
        const lo_n: usize = @intCast(lo_bytes);
        const lo_addr = self.arena.upload_bytes(bytes[0..lo_n], stage).addr;
        if (lo_bytes == expected_bytes) return .{ lo_addr, lo_addr };
        const hi = self.host_expert_arena.alloc(expected_bytes - lo_bytes);
        @memcpy(hi.mapped.?[0 .. bytes.len - lo_n], bytes[lo_n..]);
        return .{ lo_addr, hi.addr -% lo_bytes };
    }

    fn upload_layer_tensor(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, expected_bytes: u64, stage: *vk.GpuBuffer) u64 {
        var nbuf: [128]u8 = undefined;
        const tname = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        return self.upload_tensor(sf, tname, expected_bytes, stage);
    }

    fn pack_bf16_to_fp8(self: *ModelWeights, pk: *PackContext, src_addr: u64, rows: u32, cols: u32) LinearWeight {
        tiger.assert(rows > 0 and cols > 0 and cols % 16 == 0);
        const w_bytes: u64 = @as(u64, rows) * cols;
        const s_bytes: u64 = @as(u64, rows) * 4;
        const dst_w = self.arena.alloc_addr(w_bytes);
        const dst_s = self.arena.alloc_addr(s_bytes);

        const ctx = pk.ctx;
        const cmd = ctx.imm_cmd;
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
        tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
        ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pk.pipe_bf16);

        const pc: PackBf16PC = .{
            .src_addr = src_addr,
            .dst_w_addr = dst_w,
            .dst_s_addr = dst_s,
            .amax_addr = 0,
            .total_elems = cols,
            .mode = 2,
        };
        ctx.vkCmdPushConstants(cmd, pk.layout_bf16, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackBf16PC), &pc);
        ctx.vkCmdDispatch(cmd, rows, 1, 1);

        tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
        ctx.submit_and_wait(cmd);

        return .{
            .kind = .fp8,
            .rows = rows,
            .cols = cols,
            .data = dst_w,
            .scales = dst_s,
            .global_scale = 1.0,
        };
    }

    fn pack_bf16_to_nvfp4(self: *ModelWeights, pk: *PackContext, src_addr: u64, rows: u32, cols: u32) LinearWeight {
        tiger.assert(rows > 0 and cols > 0 and cols % 16 == 0);
        const w_bytes: u64 = @as(u64, rows) * (cols / 2);
        const s_bytes: u64 = @as(u64, rows) * (cols / 16);
        const dst_w = self.arena.alloc_addr(w_bytes);
        const dst_s = self.arena.alloc_addr(s_bytes);

        // Zero amax_buf[0]
        const amax_ptr: *u32 = @ptrCast(@alignCast(pk.amax_buf.mapped.?));
        amax_ptr.* = 0;

        const ctx = pk.ctx;
        const cmd = ctx.imm_cmd;
        const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
        tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
        ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pk.pipe_bf16);

        // Pass 1: amax
        const pc0: PackBf16PC = .{
            .src_addr = src_addr,
            .dst_w_addr = dst_w,
            .dst_s_addr = dst_s,
            .amax_addr = pk.amax_buf.address,
            .total_elems = rows * cols,
            .mode = 0,
        };
        ctx.vkCmdPushConstants(cmd, pk.layout_bf16, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackBf16PC), &pc0);
        const g0: u32 = @min(4096, (rows * cols + 255) / 256);
        ctx.vkCmdDispatch(cmd, g0, 1, 1);

        ctx.cmd_compute_barrier(cmd);

        // Pass 2: quantize blocks of 16
        const n_blocks: u32 = rows * (cols / 16);
        const pc1: PackBf16PC = .{
            .src_addr = src_addr,
            .dst_w_addr = dst_w,
            .dst_s_addr = dst_s,
            .amax_addr = pk.amax_buf.address,
            .total_elems = n_blocks,
            .mode = 1,
        };
        ctx.vkCmdPushConstants(cmd, pk.layout_bf16, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackBf16PC), &pc1);
        ctx.vkCmdDispatch(cmd, (n_blocks + 255) / 256, 1, 1);

        tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
        ctx.submit_and_wait(cmd);

        const amax_f32: f32 = @bitCast(amax_ptr.*);
        const gscale: f32 = @max(amax_f32 / (448.0 * 6.0), 1e-12);
        return .{
            .kind = .nvfp4,
            .rows = rows,
            .cols = cols,
            .data = dst_w,
            .scales = dst_s,
            .global_scale = gscale,
        };
    }

    fn load_expert_bank(
        self: *ModelWeights,
        sf: *const safetensors.SafetensorsFile,
        pk: *PackContext,
        layer_idx: u32,
        wname: []const u8,
        rows: u32,
        cols: u32,
    ) ExpertBank {
        const n_lo: u32 = constants.expert_count - self.expert_hi_count;
        var eb: ExpertBank = .{
            .n_experts = constants.expert_count,
            .rows = rows,
            .cols = cols,
            .n_lo = n_lo,
        };
        var nbuf: [128]u8 = undefined;
        const w_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 2);
        const s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 16);
        const w_lo: u64 = @as(u64, n_lo) * rows * (cols / 2);
        const s_lo: u64 = @as(u64, n_lo) * rows * (cols / 16);

        if (constants.is_nvfp4_expert_layer(layer_idx)) {
            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            const wt = self.upload_tensor_tiered(sf, wn, w_bytes, w_lo, &pk.stage);
            eb.data = wt[0];
            eb.data_hi = wt[1];
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            const st = self.upload_tensor_tiered(sf, sn, s_bytes, s_lo, &pk.stage);
            eb.scales = st[0];
            eb.scales_hi = st[1];

            const an = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_amax", .{ layer_idx, wname }) catch unreachable;
            const amax_slice = sf.data_slice(sf.lookup(an));
            const dn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_outer_scale_denominator", .{ layer_idx, wname }) catch unreachable;
            const denom_slice = sf.data_slice(sf.lookup(dn));
            tiger.assert(amax_slice.len == constants.expert_count * 4 and denom_slice.len >= 4);

            const denom: f32 = @bitCast(std.mem.readInt(u32, denom_slice[0..4], .little));
            const inv_denom: f32 = 1.0 / denom;
            var h_scales: [constants.expert_count]f32 = undefined;
            var e: u32 = 0;
            while (e < constants.expert_count) : (e += 1) {
                const amax_val: f32 = @bitCast(std.mem.readInt(u32, amax_slice[e * 4 ..][0..4], .little));
                h_scales[e] = amax_val * inv_denom;
            }
            eb.expert_scale = self.arena.upload_bytes(std.mem.sliceAsBytes(&h_scales), &pk.stage).addr;
        } else {
            // Layers 20..23: MXFP8 -> NVFP4 via GPU shader (reads from host-pinned stage, 0 VRAM spike!)
            const mx_w_bytes: usize = @as(usize, constants.expert_count) * rows * cols;
            const mx_s_bytes: usize = @as(usize, constants.expert_count) * rows * (cols / 32);
            const mx_s_off: usize = @intCast(tiger.align_up_u64(mx_w_bytes, 256));
            tiger.assert(mx_s_off + mx_s_bytes <= pk.stage.size);

            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            copy_named_tensor(sf, wn, pk.stage.mapped.?, mx_w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            copy_named_tensor(sf, sn, pk.stage.mapped.? + mx_s_off, mx_s_bytes);

            eb.expert_scale = self.arena.alloc_addr(@as(u64, constants.expert_count) * 4);
            eb.data = self.arena.alloc_addr(w_lo);
            eb.scales = self.arena.alloc_addr(s_lo);
            eb.data_hi = eb.data;
            eb.scales_hi = eb.scales;
            if (n_lo < constants.expert_count) {
                eb.data_hi = self.host_expert_arena.alloc_addr(w_bytes - w_lo) -% w_lo;
                eb.scales_hi = self.host_expert_arena.alloc_addr(s_bytes - s_lo) -% s_lo;
            }

            const ctx = pk.ctx;
            const cmd = ctx.imm_cmd;
            const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
            ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pk.pipe_mxfp8);

            const pc0: PackMxfp8PC = .{
                .src_w_addr = pk.stage.address,
                .src_s_addr = pk.stage.address + mx_s_off,
                .dst_w_addr = eb.data,
                .dst_s_addr = eb.scales,
                .exp_scale_addr = eb.expert_scale,
                .n_experts = constants.expert_count,
                .rows = rows,
                .cols = cols,
                .mode = 0,
                .n_lo = n_lo,
                .dst_w_hi_addr = eb.data_hi,
                .dst_s_hi_addr = eb.scales_hi,
            };
            ctx.vkCmdPushConstants(cmd, pk.layout_mxfp8, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackMxfp8PC), &pc0);
            ctx.vkCmdDispatch(cmd, constants.expert_count, 1, 1);

            ctx.cmd_compute_barrier(cmd);

            const total_blocks: u32 = constants.expert_count * rows * (cols / 16);
            var pc1 = pc0;
            pc1.mode = 1;
            ctx.vkCmdPushConstants(cmd, pk.layout_mxfp8, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackMxfp8PC), &pc1);
            ctx.vkCmdDispatch(cmd, (total_blocks + 255) / 256, 1, 1);

            tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
            ctx.submit_and_wait(cmd);
        }
        return eb;
    }

    fn load_dspark(
        self: *ModelWeights,
        model_directory: []const u8,
        sf_main: *const safetensors.SafetensorsFile,
        sf_mtp: *const safetensors.SafetensorsFile,
        pk: *PackContext,
        draft_vocab_limit: u32,
    ) void {
        const ctx = pk.ctx;
        // fuse_weight [4, 4] BF16 -> row-wise softmax F32
        const fw_slice = sf_mtp.data_slice(sf_mtp.lookup("fuse_weight"));
        tiger.assert(fw_slice.len == 32);
        var alpha: [16]f32 = undefined;
        var row: u32 = 0;
        while (row < 4) : (row += 1) {
            var vals: [4]f64 = undefined;
            var mx: f64 = -1e30;
            var col: u32 = 0;
            while (col < 4) : (col += 1) {
                const b16 = std.mem.readInt(u16, fw_slice[(row * 4 + col) * 2 ..][0..2], .little);
                const f: f32 = @bitCast(@as(u32, b16) << 16);
                vals[col] = @floatCast(f);
                if (vals[col] > mx) mx = vals[col];
            }
            var sum: f64 = 0.0;
            col = 0;
            while (col < 4) : (col += 1) {
                vals[col] = @exp(vals[col] - mx);
                sum += vals[col];
            }
            col = 0;
            while (col < 4) : (col += 1) {
                alpha[row * 4 + col] = @floatCast(vals[col] / sum);
            }
        }
        self.dspark.fuse_alpha = self.arena.upload_bytes(std.mem.sliceAsBytes(&alpha), &pk.stage).addr;
        self.dspark.mask_embed = self.upload_tensor(sf_mtp, "mask_embed", constants.hidden_dim * 2, &pk.stage);
        self.dspark.norm_w = self.upload_tensor(sf_mtp, "norm.weight", constants.hidden_dim * 2, &pk.stage);
        self.dspark.markov_dflare_norm = self.upload_tensor(sf_mtp, "markov_head.dflare_norm.weight", constants.hidden_dim * 2, &pk.stage);
        {
            const mw1_info = sf_mtp.lookup("markov_head.markov_w1.weight");
            const mw1_bytes: u64 = @as(u64, constants.vocab_padded_count) * constants.mtp_markov_rank * 2;
            tiger.assert(mw1_info.byte_length == mw1_bytes);
            self.host_markov_w1_buf = ctx.create_host_buffer(mw1_bytes, false, true);
            @memcpy(self.host_markov_w1_buf.mapped.?[0..@intCast(mw1_bytes)], sf_mtp.data_slice(mw1_info));
            self.dspark.markov_w1 = self.host_markov_w1_buf.address;
        }
        self.dspark.rnn_bias = self.upload_tensor(sf_mtp, "markov_head.rnn_proj.bias", 3 * constants.mtp_markov_rank * 2, &pk.stage);
        self.dspark.hidden_out_w = self.upload_tensor(sf_mtp, "markov_head.hidden_out.weight", @as(u64, constants.hidden_dim) * constants.mtp_markov_rank * 2, &pk.stage);
        self.dspark.hidden_out_b = self.upload_tensor(sf_mtp, "markov_head.hidden_out.bias", constants.hidden_dim * 2, &pk.stage);

        // Split rnn_proj.weight [768, 5120] -> w_sp, w_h, w_f
        {
            const rnn_bytes: usize = 768 * (512 + 2 * constants.hidden_dim) * 2;
            copy_named_tensor(sf_mtp, "markov_head.rnn_proj.weight", pk.stage.mapped.?, rnn_bytes);
            self.dspark.rnn_w_sp = self.arena.alloc_addr(768 * 512 * 2);
            self.dspark.rnn_w_h = self.arena.alloc_addr(768 * @as(u64, constants.hidden_dim) * 2);
            self.dspark.rnn_w_f = self.arena.alloc_addr(768 * @as(u64, constants.hidden_dim) * 2);

            const cmd = ctx.imm_cmd;
            const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
            ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pk.pipe_ds);
            const pc: PackDsparkPC = .{
                .addr0 = pk.stage.address,
                .addr1 = self.dspark.rnn_w_sp,
                .addr2 = self.dspark.rnn_w_h,
                .addr3 = self.dspark.rnn_w_f,
                .V = 0,
                .mode = 0,
            };
            ctx.vkCmdPushConstants(cmd, pk.layout_ds, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackDsparkPC), &pc);
            ctx.vkCmdDispatch(cmd, 768, 1, 1);
            tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
            ctx.submit_and_wait(cmd);
        }

        // Folded Markov draft head (uses draft_vocab_49k.bin / draft_vocab_32k.bin if present, else full 200K)
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const h_unembed = sf_main.data_slice(ui);
            const row_bytes: usize = constants.hidden_dim * 2;

            var dv_host: ?[]i32 = null;
            if (draft_vocab_limit > 0) {
                dv_host = try_load_draft_vocab_bin(model_directory, draft_vocab_limit);
            }
            defer if (dv_host) |sl| tiger.free_host_slice(i32, sl);

            const dv: u32 = if (dv_host) |sl| @intCast(sl.len) else constants.vocab_unpadded_count;
            self.dspark.draft_vocab_size = dv;

            const dv_bytes: usize = @as(usize, dv) * row_bytes;
            const fold_off: usize = @intCast(tiger.align_up_u64(dv_bytes, 256));
            const fold_bytes: usize = @as(usize, dv) * constants.mtp_markov_rank * 2;
            tiger.assert(fold_off + fold_bytes <= pk.stage.size);

            if (dv_host) |sl| {
                self.dspark.draft_vocab_ids = self.arena.upload_bytes(std.mem.sliceAsBytes(sl), &pk.stage).addr;
                for (sl, 0..) |tok_id, idx| {
                    tiger.assert(tok_id >= 0 and tok_id < constants.vocab_unpadded_count);
                    const src_off: usize = @as(usize, @intCast(tok_id)) * row_bytes;
                    @memcpy(pk.stage.mapped.?[idx * row_bytes ..][0..row_bytes], h_unembed[src_off..][0..row_bytes]);
                }
                self.dspark.unembed_draft = self.pack_bf16_to_nvfp4(pk, pk.stage.address, dv, constants.hidden_dim);
            } else {
                self.dspark.draft_vocab_ids = 0;
                @memcpy(pk.stage.mapped.?[0..dv_bytes], h_unembed[0..dv_bytes]);
                self.dspark.unembed_draft = if (self.unembed.kind == .nvfp4) self.unembed else self.pack_bf16_to_nvfp4(pk, pk.stage.address, dv, constants.hidden_dim);
            }

            self.dspark.fold_b = self.arena.alloc_addr(@as(u64, dv) * 4);
            const cmd = ctx.imm_cmd;
            const bi: vk.VkCommandBufferBeginInfo = .{ .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            tiger.assert_vk(ctx.vkResetCommandBuffer(cmd, 0));
            tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
            ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pk.pipe_ds);

            const pc1: PackDsparkPC = .{
                .addr0 = pk.stage.address,
                .addr1 = self.dspark.hidden_out_b,
                .addr2 = self.dspark.fold_b,
                .addr3 = 0,
                .V = dv,
                .mode = 1,
            };
            ctx.vkCmdPushConstants(cmd, pk.layout_ds, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackDsparkPC), &pc1);
            ctx.vkCmdDispatch(cmd, dv, 1, 1);

            const pc2: PackDsparkPC = .{
                .addr0 = pk.stage.address,
                .addr1 = self.dspark.hidden_out_w,
                .addr2 = pk.stage.address + fold_off,
                .addr3 = 0,
                .V = dv,
                .mode = 2,
            };
            ctx.vkCmdPushConstants(cmd, pk.layout_ds, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PackDsparkPC), &pc2);
            ctx.vkCmdDispatch(cmd, dv, 1, 1);
            tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));
            ctx.submit_and_wait(cmd);

            self.dspark.fold_w = self.pack_bf16_to_nvfp4(pk, pk.stage.address + fold_off, dv, constants.mtp_markov_rank);
        }

        // 3 DSpark layers -> NVFP4
        var l: u32 = 0;
        while (l < constants.mtp_layer_count) : (l += 1) {
            const dl = &self.dspark.layers[l];
            dl.is_local = constants.is_mtp_local_layer(l);
            var nbuf: [128]u8 = undefined;

            dl.input_norm = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim * 2, &pk.stage);
            dl.post_norm = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim * 2, &pk.stage);
            dl.dflare_norm = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.dflare_norm.weight", .{l}) catch unreachable, constants.hidden_dim * 2, &pk.stage);
            dl.q_norm = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, constants.head_dim * 2, &pk.stage);
            dl.k_norm = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, constants.head_dim * 2, &pk.stage);
            dl.k_norm_ctx = self.upload_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm_ctx.weight", .{l}) catch unreachable, constants.head_dim * 2, &pk.stage);

            const qb: usize = @as(usize, constants.query_dim) * constants.hidden_dim * 2;
            const kb: usize = @as(usize, constants.kv_dim) * constants.hidden_dim * 2;
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, pk.stage.mapped.?, qb);
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, pk.stage.mapped.? + qb, kb);
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, pk.stage.mapped.? + qb + kb, kb);
            dl.w_qkv = self.pack_bf16_to_nvfp4(pk, pk.stage.address, constants.dspark_qkv_out_dim, constants.hidden_dim);

            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, pk.stage.mapped.?, @as(usize, constants.hidden_dim) * constants.query_dim * 2);
            dl.w_o = self.pack_bf16_to_nvfp4(pk, pk.stage.address, constants.hidden_dim, constants.query_dim);

            const gb: usize = @as(usize, constants.mtp_intermediate_dim) * constants.hidden_dim * 2;
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, pk.stage.mapped.?, gb);
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, pk.stage.mapped.? + gb, gb);
            dl.w_gu = self.pack_bf16_to_nvfp4(pk, pk.stage.address, 2 * constants.mtp_intermediate_dim, constants.hidden_dim);

            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, pk.stage.mapped.?, @as(usize, constants.hidden_dim) * constants.mtp_intermediate_dim * 2);
            dl.w_down = self.pack_bf16_to_nvfp4(pk, pk.stage.address, constants.hidden_dim, constants.mtp_intermediate_dim);

            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}) catch unreachable, pk.stage.mapped.?, kb);
            copy_named_tensor(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}) catch unreachable, pk.stage.mapped.? + kb, kb);
            dl.w_kv_ctx = self.pack_bf16_to_nvfp4(pk, pk.stage.address, 2 * constants.kv_dim, constants.hidden_dim);
        }
        self.dspark.loaded = true;
    }
};

fn try_load_draft_vocab_bin(model_directory: []const u8, prefer_limit: u32) ?[]i32 {
    const names = if (prefer_limit <= 32768)
        [2][]const u8{ "/draft_vocab_32k.bin", "/draft_vocab_49k.bin" }
    else
        [2][]const u8{ "/draft_vocab_49k.bin", "/draft_vocab_32k.bin" };
    var path_buf: [1024]u8 = undefined;
    for (names) |suffix| {
        const pz = tiger.format_path_z(&path_buf, model_directory, suffix);
        const fh = tiger.CreateFileA(pz, tiger.GENERIC_READ, tiger.FILE_SHARE_READ, null, tiger.OPEN_EXISTING, tiger.FILE_ATTRIBUTE_NORMAL, null);
        if (fh == null or fh == tiger.INVALID_HANDLE_VALUE) continue;
        defer _ = tiger.CloseHandle(fh);
        var sz: i64 = 0;
        if (tiger.GetFileSizeEx(fh, &sz) == 0 or sz < 1024 or @mod(sz, 4) != 0) continue;
        const count: usize = @intCast(@divExact(sz, 4));
        const out = tiger.alloc_host_zeroed(i32, count);
        var got: u32 = 0;
        const ok = tiger.ReadFile(fh, @ptrCast(out.ptr), @intCast(sz), &got, null);
        if (ok != 0 and got == @as(u32, @intCast(sz))) return out;
        tiger.free_host_slice(i32, out);
    }
    return null;
}

fn copy_named_tensor(sf: *const safetensors.SafetensorsFile, name: []const u8, dst: [*]u8, expected_bytes: usize) void {
    const info = sf.lookup(name);
    tiger.assert(info.byte_length == expected_bytes);
    @memcpy(dst[0..expected_bytes], sf.data_slice(info));
}

fn copy_layer_tensor(sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, dst: [*]u8, expected_bytes: usize) usize {
    var nbuf: [128]u8 = undefined;
    const tname = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
    copy_named_tensor(sf, tname, dst, expected_bytes);
    return expected_bytes;
}
