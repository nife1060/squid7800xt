// RX 7800 XT Hardware Ceiling Microbenchmark (`squid7800xt ceilings`).
// Per sybil-solutions/inference-optimisation-skill §1:
// "Measure, don't trust spec sheets: real read BW, real copy BW, real PCIe BW, real NVFP4 GEMV, real WMMA TFLOPS."

const std = @import("std");
const tiger = @import("tiger.zig");
const vk = @import("vulkan.zig");

const spv_read_bw align(4) = @embedFile("spv_bench_read_bw");
const spv_nvfp4_gemv align(4) = @embedFile("spv_bench_nvfp4_gemv");
const spv_wmma align(4) = @embedFile("spv_bench_wmma");

const ReadBwPC = extern struct {
    src_addr: u64,
    dst_addr: u64,
    n_vec4: u32,
    stride_vec4: u32,
};

const Nvfp4GemvPC = extern struct {
    w_addr: u64,
    s_addr: u64,
    x_addr: u64,
    y_addr: u64,
    rows: u32,
    cols: u32,
    global_scale: f32,
    mode: u32,
};

const WmmaPC = extern struct {
    a_addr: u64,
    b_addr: u64,
    c_addr: u64,
    M: u32,
    N: u32,
    K: u32,
};

fn median_f64(vals: []f64) f64 {
    tiger.assert(vals.len > 0);
    // Insertion sort (small N)
    var i: usize = 1;
    while (i < vals.len) : (i += 1) {
        const key = vals[i];
        var j: usize = i;
        while (j > 0 and vals[j - 1] > key) : (j -= 1) {
            vals[j] = vals[j - 1];
        }
        vals[j] = key;
    }
    return vals[vals.len / 2];
}

pub fn run(ctx: *vk.Context) void {
    tiger.print_stdout("=== squid7800xt Hardware Ceiling Probe ===\n", .{});
    tiger.print_stdout("GPU: {s}\n", .{ctx.device_name_slice()});
    tiger.print_stdout("VRAM Heap: {d:.2} GiB | Host-Pinned Heap: {d:.2} GiB\n", .{
        @as(f64, @floatFromInt(ctx.vram_heap_bytes)) / (1024.0 * 1024.0 * 1024.0),
        @as(f64, @floatFromInt(ctx.host_heap_bytes)) / (1024.0 * 1024.0 * 1024.0),
    });
    tiger.print_stdout("MemTypes: vram_only={d} vram_rebar={d} host_cached={d} host_wc={d} | CoopMatrix(WMMA)={s}\n", .{
        ctx.mem_type_vram_only,
        ctx.mem_type_vram_rebar,
        ctx.mem_type_host_cached,
        ctx.mem_type_host_wc,
        if (ctx.has_coop_matrix) "YES" else "NO",
    });

    // Query supported cooperative matrix shapes on RX 7800 XT
    if (ctx.has_coop_matrix and ctx.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR != null) {
        var count: u32 = 0;
        const fn_ptr = ctx.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.?;
        _ = fn_ptr(ctx.phys_dev, &count, null);
        if (count > 0 and count <= 64) {
            var props: [64]vk.VkCooperativeMatrixPropertiesKHR = @splat(.{});
            _ = fn_ptr(ctx.phys_dev, &count, &props);
            tiger.print_stdout("Supported WMMA Cooperative Matrix configs ({d}):\n", .{count});
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const p = props[i];
                tiger.print_stdout("  [{d}] {d}x{d}x{d} A={d} B={d} C={d} R={d} scope={d}\n", .{
                    i, p.MSize, p.NSize, p.KSize, p.AType, p.BType, p.CType, p.ResultType, p.scope,
                });
            }
        }
    }

    // 1. Pure VRAM Read Bandwidth (1 GiB): DEVICE_LOCAL vs ReBAR vs Host-Pinned DDR5
    const layout_rbw = ctx.create_pipeline_layout(@sizeOf(ReadBwPC));
    defer ctx.vkDestroyPipelineLayout(ctx.device, layout_rbw, null);
    const pipe_rbw_w64 = ctx.create_compute_pipeline(layout_rbw, spv_read_bw, 64);
    defer ctx.vkDestroyPipeline(ctx.device, pipe_rbw_w64, null);
    const pipe_rbw_w32 = ctx.create_compute_pipeline(layout_rbw, spv_read_bw, 32);
    defer ctx.vkDestroyPipeline(ctx.device, pipe_rbw_w32, null);

    var sink_buf = ctx.create_buffer(4096, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&sink_buf);

    const bytes_1g: u64 = 1024 * 1024 * 1024;
    const bw_vram_w64 = bench_read_bw(ctx, layout_rbw, pipe_rbw_w64, ctx.mem_type_vram_only, bytes_1g, sink_buf.address);
    const bw_vram_w32 = bench_read_bw(ctx, layout_rbw, pipe_rbw_w32, ctx.mem_type_vram_only, bytes_1g, sink_buf.address);
    tiger.print_stdout("\n[1] Pure Read BW (1 GiB, DEVICE_LOCAL):       Wave64 = {d:.1} GB/s | Wave32 = {d:.1} GB/s\n", .{ bw_vram_w64, bw_vram_w32 });

    if (ctx.mem_type_vram_rebar != std.math.maxInt(u32)) {
        const bw_rebar = bench_read_bw(ctx, layout_rbw, pipe_rbw_w64, ctx.mem_type_vram_rebar, bytes_1g, sink_buf.address);
        tiger.print_stdout("[2] Pure Read BW (1 GiB, ReBAR HOST_VISIBLE): Wave64 = {d:.1} GB/s\n", .{bw_rebar});
    }

    const host_mt = if (ctx.mem_type_host_cached != std.math.maxInt(u32)) ctx.mem_type_host_cached else ctx.mem_type_host_wc;
    const bw_pcie = bench_read_bw(ctx, layout_rbw, pipe_rbw_w64, host_mt, 256 * 1024 * 1024, sink_buf.address);
    tiger.print_stdout("[3] Zero-Copy DDR5 Read BW over PCIe (256MB): Wave64 = {d:.1} GB/s\n", .{bw_pcie});

    // 2. D2D Copy BW (512 MiB -> 512 MiB)
    const bw_d2d = bench_d2d_copy(ctx, 512 * 1024 * 1024);
    tiger.print_stdout("[4] D2D Copy BW (512 MiB R+W):                {d:.1} GB/s\n", .{bw_d2d});

    // 3. NVFP4 LM-Head GEMV (200,058 x 2304 = 259.3 MB weights+scales) — Reg LUT16 vs LDS LUT256, Wave32 vs Wave64
    const layout_gemv = ctx.create_pipeline_layout(@sizeOf(Nvfp4GemvPC));
    defer ctx.vkDestroyPipelineLayout(ctx.device, layout_gemv, null);
    const pipe_gemv_w32 = ctx.create_compute_pipeline(layout_gemv, spv_nvfp4_gemv, 32);
    defer ctx.vkDestroyPipeline(ctx.device, pipe_gemv_w32, null);
    const pipe_gemv_w64 = ctx.create_compute_pipeline(layout_gemv, spv_nvfp4_gemv, 64);
    defer ctx.vkDestroyPipeline(ctx.device, pipe_gemv_w64, null);

    const g_w32_reg = bench_nvfp4_gemv(ctx, layout_gemv, pipe_gemv_w32, 200058, 2304, 0);
    const g_w32_lds = bench_nvfp4_gemv(ctx, layout_gemv, pipe_gemv_w32, 200058, 2304, 1);
    const g_w64_reg = bench_nvfp4_gemv(ctx, layout_gemv, pipe_gemv_w64, 200058, 2304, 0);
    const g_w64_lds = bench_nvfp4_gemv(ctx, layout_gemv, pipe_gemv_w64, 200058, 2304, 1);
    tiger.print_stdout("\n[5] NVFP4 LM-Head GEMV [200058 x 2304] (259.3 MB):\n", .{});
    tiger.print_stdout("    Wave32 Reg-LUT16: {d:.1} GB/s ({d:.3} ms)\n", .{ g_w32_reg.gbps, g_w32_reg.ms });
    tiger.print_stdout("    Wave32 LDS-LUT256:{d:.1} GB/s ({d:.3} ms)\n", .{ g_w32_lds.gbps, g_w32_lds.ms });
    tiger.print_stdout("    Wave64 Reg-LUT16: {d:.1} GB/s ({d:.3} ms)\n", .{ g_w64_reg.gbps, g_w64_reg.ms });
    tiger.print_stdout("    Wave64 LDS-LUT256:{d:.1} GB/s ({d:.3} ms)\n", .{ g_w64_lds.gbps, g_w64_lds.ms });

    // 4. RDNA 3 WMMA FP16 GEMM (4096 x 4096 x 4096)
    if (ctx.has_coop_matrix) {
        const layout_wmma = ctx.create_pipeline_layout(@sizeOf(WmmaPC));
        defer ctx.vkDestroyPipelineLayout(ctx.device, layout_wmma, null);
        const pipe_wmma = ctx.create_compute_pipeline(layout_wmma, spv_wmma, 32);
        defer ctx.vkDestroyPipeline(ctx.device, pipe_wmma, null);
        const tflops = bench_wmma(ctx, layout_wmma, pipe_wmma, 4096, 4096, 4096);
        tiger.print_stdout("\n[6] RDNA 3 WMMA FP16 GEMM (4096^3, Wave32):   {d:.2} TFLOPS\n", .{tflops});
    }

    // 5. Roofline summary for Inkling-2-Mini NVFP4 (2,800.3 MB/tok B=1)
    const peak_bw = @max(bw_vram_w64, bw_vram_w32);
    const best_gemv_bw = @max(@max(g_w32_reg.gbps, g_w32_lds.gbps), @max(g_w64_reg.gbps, g_w64_lds.gbps));
    tiger.print_stdout("\n=== Inkling-2-Mini (NVFP4, 2.800 GB/tok) RX 7800 XT Roofline ===\n", .{});
    tiger.print_stdout("  Raw Measured Read Ceiling:  {d:.1} tok/s (at {d:.1} GB/s)\n", .{ peak_bw / 2.8003, peak_bw });
    tiger.print_stdout("  NVFP4 GEMV Kernel Ceiling:  {d:.1} tok/s (at {d:.1} GB/s = {d:.1}% of peak BW)\n", .{
        best_gemv_bw / 2.8003,
        best_gemv_bw,
        100.0 * best_gemv_bw / peak_bw,
    });
}

fn bench_read_bw(ctx: *vk.Context, layout: vk.VkPipelineLayout, pipe: vk.VkPipeline, mem_type: u32, size_bytes: u64, sink_addr: u64) f64 {
    var buf = ctx.create_buffer(size_bytes, mem_type, false);
    defer ctx.destroy_buffer(&buf);

    const n_vec4: u32 = @intCast(size_bytes / 16);
    const pc: ReadBwPC = .{
        .src_addr = buf.address,
        .dst_addr = sink_addr,
        .n_vec4 = n_vec4,
        .stride_vec4 = 1,
    };

    const cmd = ctx.alloc_command_buffer();
    defer ctx.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &.{cmd});

    const reps: u32 = 16;
    const bi: vk.VkCommandBufferBeginInfo = .{};
    tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
    ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    ctx.vkCmdPushConstants(cmd, layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ReadBwPC), &pc);
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        ctx.vkCmdDispatch(cmd, 4096, 1, 1);
    }
    tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));

    // Warmup
    ctx.submit_and_wait(cmd);
    ctx.submit_and_wait(cmd);

    var samples: [7]f64 = undefined;
    var s: usize = 0;
    while (s < samples.len) : (s += 1) {
        const t0 = tiger.clock_monotonic_ns();
        ctx.submit_and_wait(cmd);
        const dt_s = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e9;
        samples[s] = (@as(f64, @floatFromInt(size_bytes)) * @as(f64, @floatFromInt(reps))) / (dt_s * 1e9);
    }
    return median_f64(&samples);
}

fn bench_d2d_copy(ctx: *vk.Context, size_bytes: u64) f64 {
    var src = ctx.create_buffer(size_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&src);
    var dst = ctx.create_buffer(size_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&dst);

    const cmd = ctx.alloc_command_buffer();
    defer ctx.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &.{cmd});

    const reps: u32 = 16;
    const bi: vk.VkCommandBufferBeginInfo = .{};
    tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
    const reg = [_]vk.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = size_bytes }};
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        ctx.vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &reg);
    }
    tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));

    ctx.submit_and_wait(cmd);
    var samples: [7]f64 = undefined;
    var s: usize = 0;
    while (s < samples.len) : (s += 1) {
        const t0 = tiger.clock_monotonic_ns();
        ctx.submit_and_wait(cmd);
        const dt_s = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e9;
        // Read + write = 2 * size_bytes
        samples[s] = (2.0 * @as(f64, @floatFromInt(size_bytes)) * @as(f64, @floatFromInt(reps))) / (dt_s * 1e9);
    }
    return median_f64(&samples);
}

const GemvResult = struct { gbps: f64, ms: f64 };

fn bench_nvfp4_gemv(ctx: *vk.Context, layout: vk.VkPipelineLayout, pipe: vk.VkPipeline, rows: u32, cols: u32, mode: u32) GemvResult {
    const w_bytes: u64 = @as(u64, rows) * (cols / 2);
    const s_bytes: u64 = @as(u64, rows) * (cols / 16);
    const x_bytes: u64 = @as(u64, cols) * 2;
    const y_bytes: u64 = @as(u64, rows) * 4;

    var w_buf = ctx.create_buffer(w_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&w_buf);
    var s_buf = ctx.create_buffer(s_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&s_buf);
    var x_buf = ctx.create_buffer(x_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&x_buf);
    var y_buf = ctx.create_buffer(y_bytes, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&y_buf);

    const pc: Nvfp4GemvPC = .{
        .w_addr = w_buf.address,
        .s_addr = s_buf.address,
        .x_addr = x_buf.address,
        .y_addr = y_buf.address,
        .rows = rows,
        .cols = cols,
        .global_scale = 1.0,
        .mode = mode,
    };

    const cmd = ctx.alloc_command_buffer();
    defer ctx.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &.{cmd});

    const reps: u32 = 32;
    const groups_x: u32 = (rows + 15) / 16;
    const bi: vk.VkCommandBufferBeginInfo = .{};
    tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
    ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    ctx.vkCmdPushConstants(cmd, layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(Nvfp4GemvPC), &pc);
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        ctx.vkCmdDispatch(cmd, groups_x, 1, 1);
    }
    tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));

    ctx.submit_and_wait(cmd);
    ctx.submit_and_wait(cmd);

    var samples: [7]f64 = undefined;
    var s: usize = 0;
    while (s < samples.len) : (s += 1) {
        const t0 = tiger.clock_monotonic_ns();
        ctx.submit_and_wait(cmd);
        const dt_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e6;
        samples[s] = dt_ms / @as(f64, @floatFromInt(reps));
    }
    const med_ms = median_f64(&samples);
    const total_bytes = @as(f64, @floatFromInt(w_bytes + s_bytes + y_bytes));
    return .{ .gbps = total_bytes / (med_ms * 1e6), .ms = med_ms };
}

fn bench_wmma(ctx: *vk.Context, layout: vk.VkPipelineLayout, pipe: vk.VkPipeline, M: u32, N: u32, K: u32) f64 {
    var a_buf = ctx.create_buffer(@as(u64, M) * K * 2, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&a_buf);
    var b_buf = ctx.create_buffer(@as(u64, N) * K * 2, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&b_buf);
    var c_buf = ctx.create_buffer(@as(u64, M) * N * 4, ctx.mem_type_vram_only, false);
    defer ctx.destroy_buffer(&c_buf);

    const pc: WmmaPC = .{
        .a_addr = a_buf.address,
        .b_addr = b_buf.address,
        .c_addr = c_buf.address,
        .M = M,
        .N = N,
        .K = K,
    };

    const cmd = ctx.alloc_command_buffer();
    defer ctx.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &.{cmd});

    const reps: u32 = 8;
    const bi: vk.VkCommandBufferBeginInfo = .{};
    tiger.assert_vk(ctx.vkBeginCommandBuffer(cmd, &bi));
    ctx.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    ctx.vkCmdPushConstants(cmd, layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(WmmaPC), &pc);
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        ctx.vkCmdDispatch(cmd, N / 64, M / 64, 1);
    }
    tiger.assert_vk(ctx.vkEndCommandBuffer(cmd));

    ctx.submit_and_wait(cmd);
    ctx.submit_and_wait(cmd);

    var samples: [7]f64 = undefined;
    var s: usize = 0;
    while (s < samples.len) : (s += 1) {
        const t0 = tiger.clock_monotonic_ns();
        ctx.submit_and_wait(cmd);
        const dt_s = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e9;
        const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(reps));
        samples[s] = flops / (dt_s * 1e12);
    }
    return median_f64(&samples);
}
