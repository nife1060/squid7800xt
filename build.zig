const std = @import("std");

const ShaderSpec = struct {
    name: []const u8,
    src: []const u8,
    out: []const u8,
};

const shaders = [_]ShaderSpec{
    .{ .name = "spv_bench_read_bw", .src = "shaders/bench_read_bw.comp", .out = "bench_read_bw.spv" },
    .{ .name = "spv_bench_nvfp4_gemv", .src = "shaders/bench_nvfp4_gemv.comp", .out = "bench_nvfp4_gemv.spv" },
    .{ .name = "spv_bench_wmma", .src = "shaders/bench_wmma.comp", .out = "bench_wmma.spv" },
    .{ .name = "spv_pack_bf16_nvfp4", .src = "shaders/pack_bf16_nvfp4.comp", .out = "pack_bf16_nvfp4.spv" },
    .{ .name = "spv_pack_mxfp8_nvfp4", .src = "shaders/pack_mxfp8_nvfp4.comp", .out = "pack_mxfp8_nvfp4.spv" },
    .{ .name = "spv_pack_dspark", .src = "shaders/pack_dspark.comp", .out = "pack_dspark.spv" },
    .{ .name = "spv_embed_norm", .src = "shaders/embed_norm.comp", .out = "embed_norm.spv" },
    .{ .name = "spv_depth_norm", .src = "shaders/depth_norm.comp", .out = "depth_norm.spv" },
    .{ .name = "spv_controller_nvfp4", .src = "shaders/controller_nvfp4.comp", .out = "controller_nvfp4.spv" },
    .{ .name = "spv_linear_nvfp4", .src = "shaders/linear_nvfp4.comp", .out = "linear_nvfp4.spv" },
    .{ .name = "spv_linear_bf16", .src = "shaders/linear_bf16.comp", .out = "linear_bf16.spv" },
    .{ .name = "spv_kv_sconv", .src = "shaders/kv_sconv.comp", .out = "kv_sconv.spv" },
    .{ .name = "spv_attn_prep_store", .src = "shaders/attn_prep_store.comp", .out = "attn_prep_store.spv" },
    .{ .name = "spv_attn_tile", .src = "shaders/attn_tile.comp", .out = "attn_tile.spv" },
    .{ .name = "spv_attn_merge_gate", .src = "shaders/attn_merge_gate.comp", .out = "attn_merge_gate.spv" },
    .{ .name = "spv_post_res", .src = "shaders/post_res.comp", .out = "post_res.spv" },
    .{ .name = "spv_post_depth_norm", .src = "shaders/post_depth_norm.comp", .out = "post_depth_norm.spv" },
    .{ .name = "spv_route_topk", .src = "shaders/route_topk.comp", .out = "route_topk.spv" },
    .{ .name = "spv_moe_up", .src = "shaders/moe_up.comp", .out = "moe_up.spv" },
    .{ .name = "spv_moe_down", .src = "shaders/moe_down.comp", .out = "moe_down.spv" },
    .{ .name = "spv_lm_head_argmax", .src = "shaders/lm_head_argmax.comp", .out = "lm_head_argmax.spv" },
    .{ .name = "spv_dspark_commit", .src = "shaders/dspark_commit.comp", .out = "dspark_commit.spv" },
    .{ .name = "spv_dspark_ops", .src = "shaders/dspark_ops.comp", .out = "dspark_ops.spv" },
    .{ .name = "spv_dspark_attn", .src = "shaders/dspark_attn.comp", .out = "dspark_attn.spv" },
    .{ .name = "spv_dspark_markov", .src = "shaders/dspark_markov.comp", .out = "dspark_markov.spv" },
    .{ .name = "spv_linear_wmma_nvfp4", .src = "shaders/linear_wmma_nvfp4.comp", .out = "linear_wmma_nvfp4.spv" },
    .{ .name = "spv_moe_prefill_b64", .src = "shaders/moe_prefill_b64.comp", .out = "moe_prefill_b64.spv" },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize: std.builtin.Optimize = b.option(
        std.builtin.Optimize,
        "optimize",
        "Optimization mode: fast (default), safe, small, debug",
    ) orelse switch (b.graph.release_mode) {
        .off, .any, .fast => .fast,
        .safe => .safe,
        .small => .small,
    };

    const glslang_bin = b.option(
        []const u8,
        "glslang",
        "Path to glslang shader compiler",
    ) orelse "tools/glslang/bin/glslang.exe";

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    for (shaders) |sh| {
        const cmd = b.addSystemCommand(&.{
            glslang_bin,
            "--quiet",
            "-V",
            "--target-env",
            "vulkan1.4",
            "-Ishaders",
            "-o",
        });
        const spv = cmd.addOutputFileArg(sh.out);
        cmd.addFileArg(b.path(sh.src));
        cmd.addFileInput(b.path("shaders/common.glsl"));
        cmd.addFileInput(b.path("shaders/nvfp4_scale_search.glsl"));
        root_module.addAnonymousImport(sh.name, .{ .root_source_file = spv });
    }

    root_module.linkSystemLibrary("kernel32", .{});
    root_module.linkSystemLibrary("ws2_32", .{});

    const exe = b.addExecutable(.{
        .name = "squid7800xt",
        .root_module = root_module,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run squid7800xt");
    run_step.dependOn(&run_cmd.step);
}
