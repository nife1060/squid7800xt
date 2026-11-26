// `score` subcommand: teacher-forced log-probabilities for quality evaluation (eval/quality.py).
//
// Input  (--tokens): raw little-endian i32 token ids, a whole number of windows of --window tokens.
// Output (--out):    for each window w and position i in [0, window-1), one record predicting
//                    tokens[w*window + i + 1]:
//                      f32 target_logprob, f32 entropy (nats), i32 top_ids[K], f32 top_logprobs[K]
//                    Logprobs use the model's final softcap (30 * tanh(x / 30)) and a softmax over
//                    the unpadded vocab.

const std = @import("std");
const constants = @import("constants.zig");
const engine_mod = @import("engine.zig");
const tiger = @import("tiger.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const top_k_max: u32 = 64;
const final_softcap: f32 = 30.0;

pub const Options = struct {
    tokens_path: []const u8,
    out_path: []const u8,
    window: u32,
    top_k: u32,
};

const RowCtx = struct {
    window_tokens: []const i32,
    top_k: u32,
    out: []u8,
    out_len: usize = 0,
    capped: []f32,
    nll_sum: f64 = 0.0,
    n_scored: u64 = 0,
};

fn on_row(cb_ctx: ?*anyopaque, pos: usize, logits: []const f32) void {
    const rc: *RowCtx = @ptrCast(@alignCast(cb_ctx.?));
    if (pos + 1 >= rc.window_tokens.len) return;
    const target: usize = @intCast(rc.window_tokens[pos + 1]);
    tiger.assert(target < logits.len);

    var top_ids: [top_k_max]i32 = undefined;
    var top_vals: [top_k_max]f32 = undefined;
    var n_top: u32 = 0;
    var vmax: f32 = -std.math.inf(f32);
    for (logits, 0..) |raw, i| {
        const v = std.math.tanh(raw / final_softcap) * final_softcap;
        rc.capped[i] = v;
        if (v > vmax) vmax = v;
        if (n_top < rc.top_k or v > top_vals[n_top - 1]) {
            var j: u32 = if (n_top < rc.top_k) n_top else rc.top_k - 1;
            if (n_top < rc.top_k) n_top += 1;
            while (j > 0 and top_vals[j - 1] < v) : (j -= 1) {
                top_vals[j] = top_vals[j - 1];
                top_ids[j] = top_ids[j - 1];
            }
            top_vals[j] = v;
            top_ids[j] = @intCast(i);
        }
    }
    var sum: f64 = 0.0;
    var sum_xe: f64 = 0.0;
    for (rc.capped[0..logits.len]) |v| {
        const d: f64 = v - vmax;
        const e = @exp(d);
        sum += e;
        sum_xe += e * d;
    }
    const log_z: f64 = @log(sum);
    const entropy: f64 = log_z - sum_xe / sum;
    const target_lp: f64 = @as(f64, rc.capped[target] - vmax) - log_z;
    rc.nll_sum -= target_lp;
    rc.n_scored += 1;

    const rec = rc.out[rc.out_len..];
    var o: usize = 0;
    std.mem.writeInt(u32, rec[o..][0..4], @bitCast(@as(f32, @floatCast(target_lp))), .little);
    o += 4;
    std.mem.writeInt(u32, rec[o..][0..4], @bitCast(@as(f32, @floatCast(entropy))), .little);
    o += 4;
    for (0..rc.top_k) |k| {
        std.mem.writeInt(u32, rec[o..][0..4], @bitCast(top_ids[k]), .little);
        o += 4;
    }
    for (0..rc.top_k) |k| {
        const lp: f32 = @floatCast(@as(f64, top_vals[k] - vmax) - log_z);
        std.mem.writeInt(u32, rec[o..][0..4], @bitCast(lp), .little);
        o += 4;
    }
    rc.out_len += o;
}

fn read_tokens(path: []const u8) []i32 {
    var path_buf: [1024]u8 = undefined;
    const path_z = tiger.format_path_z(&path_buf, path, "");
    const fh = tiger.CreateFileA(path_z, tiger.GENERIC_READ, tiger.FILE_SHARE_READ, null, tiger.OPEN_EXISTING, tiger.FILE_ATTRIBUTE_NORMAL, null);
    if (fh == null or fh == tiger.INVALID_HANDLE_VALUE) {
        tiger.print_stderr("[score] cannot open {s}\n", .{path});
        std.process.exit(1);
    }
    defer _ = tiger.CloseHandle(fh);
    var size_i64: i64 = 0;
    tiger.assert(tiger.GetFileSizeEx(fh, &size_i64) != 0 and size_i64 > 0 and @mod(size_i64, 4) == 0);
    const n: usize = @intCast(@divExact(size_i64, 4));
    const toks = tiger.alloc_host_zeroed(i32, n);
    const bytes: []u8 = std.mem.sliceAsBytes(toks);
    var got: usize = 0;
    while (got < bytes.len) {
        const chunk: u32 = @intCast(@min(bytes.len - got, 1 << 24));
        var nread: u32 = 0;
        tiger.assert(tiger.ReadFile(fh, bytes[got..].ptr, chunk, &nread, null) != 0 and nread > 0);
        got += nread;
    }
    return toks;
}

pub fn run(eng: *engine_mod.Engine, tok: *const tokenizer_mod.Tokenizer, opt: Options) void {
    tiger.assert(opt.window >= 16 and opt.window % 8 == 0);
    tiger.assert(opt.top_k >= 1 and opt.top_k <= top_k_max);
    const toks = read_tokens(opt.tokens_path);
    defer tiger.free_host_slice(i32, toks);
    if (toks.len % opt.window != 0) {
        tiger.print_stderr("[score] {d} tokens is not a multiple of --window {d}\n", .{ toks.len, opt.window });
        std.process.exit(1);
    }
    for (toks) |t| tiger.assert(t >= 0 and t < constants.vocab_unpadded_count);
    const n_windows = toks.len / opt.window;

    // Tokenization cross-check: the harness prints the GGUF detokenization of the same ids.
    tiger.print_stdout("[score] first ids decode as: \"", .{});
    for (toks[0..@min(toks.len, 24)]) |id| tiger.print_stdout("{s}", .{tok.decode_token_bytes(id, false)});
    tiger.print_stdout("\"\n", .{});

    const rec_bytes: usize = 8 + 8 * @as(usize, opt.top_k);
    const out = tiger.alloc_host_zeroed(u8, n_windows * (opt.window - 1) * rec_bytes);
    defer tiger.free_host_slice(u8, out);
    const capped = tiger.alloc_host_zeroed(f32, constants.vocab_unpadded_count);
    defer tiger.free_host_slice(f32, capped);

    var rc: RowCtx = .{ .window_tokens = &.{}, .top_k = opt.top_k, .out = out, .capped = capped };
    const t0 = tiger.clock_monotonic_ns();
    for (0..n_windows) |w| {
        rc.window_tokens = toks[w * opt.window ..][0..opt.window];
        eng.score_tokens(rc.window_tokens, on_row, @ptrCast(&rc));
        const nll = rc.nll_sum / @as(f64, @floatFromInt(@max(rc.n_scored, 1)));
        tiger.print_stdout("[score] window {d}/{d}  running ppl={d:.4}\n", .{ w + 1, n_windows, @exp(nll) });
    }
    tiger.assert(rc.out_len == out.len);
    const secs = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e9;

    var path_buf: [1024]u8 = undefined;
    const path_z = tiger.format_path_z(&path_buf, opt.out_path, "");
    const fh = tiger.CreateFileA(path_z, tiger.GENERIC_WRITE, 0, null, tiger.CREATE_ALWAYS, tiger.FILE_ATTRIBUTE_NORMAL, null);
    if (fh == null or fh == tiger.INVALID_HANDLE_VALUE) {
        tiger.print_stderr("[score] cannot create {s}\n", .{opt.out_path});
        std.process.exit(1);
    }
    tiger.write_handle_all(fh, out);
    _ = tiger.CloseHandle(fh);
    tiger.print_stdout("[score] {d} positions in {d:.1}s, ppl={d:.4} -> {s}\n", .{
        rc.n_scored, secs, @exp(rc.nll_sum / @as(f64, @floatFromInt(@max(rc.n_scored, 1)))), opt.out_path,
    });
}
