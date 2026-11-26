const std = @import("std");
const constants = @import("constants.zig");
const engine_mod = @import("engine.zig");
const tiger = @import("tiger.zig");
const tokenizer_mod = @import("tokenizer.zig");
const vk = @import("vulkan.zig");
const weights_mod = @import("weights.zig");
const bench_ceilings = @import("bench_ceilings.zig");
const server_mod = @import("server.zig");
const score_mod = @import("score.zig");

extern "kernel32" fn GetCommandLineW() callconv(.winapi) [*:0]const u16;

/// Command-line arguments decoded from the UTF-16 command line into UTF-8.
///
/// The argument buffer is sized from the actual command line once at startup, so long
/// `--prompt` values are never truncated. (The previous fixed 8 KiB buffer silently cut
/// prompts at a point that depended on the length of the preceding arguments, e.g. the
/// exe/model paths, which looked like run-to-run tokenizer nondeterminism.)
/// Quoting follows the MSVC CRT rules so arguments with embedded quotes and backslashes
/// round-trip exactly as shells (PowerShell, Git Bash) encode them.
const CliArgs = struct {
    buf: []u8 = &.{},
    slices: [64][]const u8 = undefined,
    count: usize = 0,

    fn parse(self: *CliArgs) void {
        const raw = std.mem.span(GetCommandLineW());
        // Each UTF-16 code unit becomes at most 3 UTF-8 bytes (surrogate pairs: 2 units -> 4 bytes).
        self.buf = tiger.alloc_host_zeroed(u8, raw.len * 3 + 1);
        var out_len: usize = 0;
        var i: usize = 0;
        while (i < raw.len) {
            while (i < raw.len and (raw[i] == ' ' or raw[i] == '\t')) : (i += 1) {}
            if (i >= raw.len) break;
            if (self.count == self.slices.len) {
                tiger.print_stderr("[args] Too many arguments (max {d})\n", .{self.slices.len});
                std.process.exit(2);
            }
            const start = out_len;
            var in_quotes = false;
            if (self.count == 0) {
                // argv[0]: quotes delimit, backslashes are literal (path separators).
                while (i < raw.len) : (i += 1) {
                    const c = raw[i];
                    if (c == '"') {
                        in_quotes = !in_quotes;
                        continue;
                    }
                    if (!in_quotes and (c == ' ' or c == '\t')) break;
                    i += append_utf16_as_utf8(self.buf, &out_len, raw, i);
                }
            } else {
                while (i < raw.len) {
                    const c = raw[i];
                    if (c == '\\') {
                        var n_bs: usize = 0;
                        while (i < raw.len and raw[i] == '\\') : (i += 1) n_bs += 1;
                        const before_quote = i < raw.len and raw[i] == '"';
                        // 2n backslashes + quote -> n backslashes (quote handled next iteration);
                        // 2n+1 backslashes + quote -> n backslashes + literal quote.
                        const emit_bs = if (before_quote) n_bs / 2 else n_bs;
                        var k: usize = 0;
                        while (k < emit_bs) : (k += 1) {
                            self.buf[out_len] = '\\';
                            out_len += 1;
                        }
                        if (before_quote and (n_bs & 1) == 1) {
                            self.buf[out_len] = '"';
                            out_len += 1;
                            i += 1;
                        }
                        continue;
                    }
                    if (c == '"') {
                        if (in_quotes and i + 1 < raw.len and raw[i + 1] == '"') {
                            // "" inside a quoted span is a literal quote.
                            self.buf[out_len] = '"';
                            out_len += 1;
                            i += 2;
                            continue;
                        }
                        in_quotes = !in_quotes;
                        i += 1;
                        continue;
                    }
                    if (!in_quotes and (c == ' ' or c == '\t')) break;
                    i += 1 + append_utf16_as_utf8(self.buf, &out_len, raw, i);
                }
            }
            self.slices[self.count] = self.buf[start..out_len];
            self.count += 1;
        }
        tiger.assert(out_len <= self.buf.len);
    }

    /// Appends the code point starting at raw[i] as UTF-8. Returns the number of EXTRA
    /// code units consumed (1 for a surrogate pair, else 0). Lone surrogates become U+FFFD.
    fn append_utf16_as_utf8(out: []u8, out_len: *usize, raw: []const u16, i: usize) usize {
        var cp: u32 = raw[i];
        var extra: usize = 0;
        if (cp >= 0xD800 and cp <= 0xDBFF and i + 1 < raw.len and raw[i + 1] >= 0xDC00 and raw[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (@as(u32, raw[i + 1]) - 0xDC00);
            extra = 1;
        } else if (cp >= 0xD800 and cp <= 0xDFFF) {
            cp = 0xFFFD;
        }
        const n = std.unicode.utf8Encode(@intCast(cp), out[out_len.*..]) catch unreachable;
        out_len.* += n;
        return extra;
    }

    fn get(self: *const CliArgs, idx: usize) ?[]const u8 {
        if (idx >= self.count) return null;
        return self.slices[idx];
    }
};

fn stream_print_cb(ctx_raw: *anyopaque, token_id: i32) bool {
    const tok: *const tokenizer_mod.Tokenizer = @ptrCast(@alignCast(ctx_raw));
    const bytes = tok.decode_token_bytes(token_id, true);
    if (bytes.len > 0) {
        tiger.write_handle_all(tiger.GetStdHandle(tiger.STD_OUTPUT_HANDLE), bytes);
    }
    return true;
}

pub fn main() void {
    var args: CliArgs = .{};
    args.parse();

    const subcmd = args.get(1) orelse "ceilings";

    if (std.mem.eql(u8, subcmd, "ceilings")) {
        var ctx: vk.Context = .{};
        ctx.init();
        defer ctx.deinit();
        bench_ceilings.run(&ctx);
        return;
    }

    if (std.mem.eql(u8, subcmd, "load")) {
        const model_dir = args.get(2) orelse "models/inkling-2-mini";
        var tok: tokenizer_mod.Tokenizer = .{};
        tiger.assert(tok.load(model_dir));
        defer tok.deinit();

        var ctx: vk.Context = .{};
        ctx.init();
        defer ctx.deinit();

        var w: weights_mod.ModelWeights = .{};
        tiger.assert(w.load(&ctx, model_dir, true, 32768));
        defer w.deinit();
        tiger.print_stdout("[load] SUCCESS!\n", .{});
        return;
    }

    if (std.mem.eql(u8, subcmd, "generate") or std.mem.eql(u8, subcmd, "bench") or std.mem.eql(u8, subcmd, "profile") or std.mem.eql(u8, subcmd, "serve") or std.mem.eql(u8, subcmd, "longctx") or std.mem.eql(u8, subcmd, "attncheck") or std.mem.eql(u8, subcmd, "score")) {
        const is_score = std.mem.eql(u8, subcmd, "score");
        const model_dir = args.get(2) orelse "models/inkling-2-mini";
        var prompt_text: []const u8 = "Write a Python LRU cache with tests.";
        var max_tokens: u32 = 256;
        var ctx_len: u32 = 65536;
        var runs: u32 = 3;
        var mtp: u32 = 7;
        var draft_vocab: u32 = 49152;
        var spec_threshold: f32 = 0.80;
        var temperature: f32 = 0.0;
        var seed: u32 = 42;
        var debug_mtp: bool = false;
        var effort: []const u8 = "none";
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8000;
        var api_key: []const u8 = "";
        var model_name: []const u8 = "inkling-2-mini";
        var no_prefix_cache: bool = false;
        var expert_tier: ?u32 = null; // null = auto (size from VRAM budget and --ctx)
        var score_tokens_path: []const u8 = "";
        var score_out_path: []const u8 = "";
        var score_window: u32 = 1024;
        var score_top_k: u32 = 20;

        var ai: usize = 3;
        while (ai < args.count) : (ai += 1) {
            const flag = args.slices[ai];
            if (std.mem.eql(u8, flag, "--prompt") and ai + 1 < args.count) {
                ai += 1;
                prompt_text = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--max-tokens") and ai + 1 < args.count) {
                ai += 1;
                max_tokens = std.fmt.parseInt(u32, args.slices[ai], 10) catch 256;
            } else if (std.mem.eql(u8, flag, "--ctx") and ai + 1 < args.count) {
                ai += 1;
                ctx_len = std.fmt.parseInt(u32, args.slices[ai], 10) catch 65536;
            } else if (std.mem.eql(u8, flag, "--runs") and ai + 1 < args.count) {
                ai += 1;
                runs = std.fmt.parseInt(u32, args.slices[ai], 10) catch 3;
            } else if (std.mem.eql(u8, flag, "--mtp") and ai + 1 < args.count) {
                ai += 1;
                mtp = std.fmt.parseInt(u32, args.slices[ai], 10) catch 7;
            } else if (std.mem.eql(u8, flag, "--draft-vocab") and ai + 1 < args.count) {
                ai += 1;
                draft_vocab = std.fmt.parseInt(u32, args.slices[ai], 10) catch 49152;
            } else if (std.mem.eql(u8, flag, "--spec-threshold") and ai + 1 < args.count) {
                ai += 1;
                spec_threshold = std.fmt.parseFloat(f32, args.slices[ai]) catch 0.80;
            } else if (std.mem.eql(u8, flag, "--temperature") and ai + 1 < args.count) {
                ai += 1;
                temperature = std.fmt.parseFloat(f32, args.slices[ai]) catch 0.0;
            } else if (std.mem.eql(u8, flag, "--seed") and ai + 1 < args.count) {
                ai += 1;
                seed = std.fmt.parseInt(u32, args.slices[ai], 10) catch 42;
            } else if (std.mem.eql(u8, flag, "--host") and ai + 1 < args.count) {
                ai += 1;
                host = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--port") and ai + 1 < args.count) {
                ai += 1;
                port = std.fmt.parseInt(u16, args.slices[ai], 10) catch 8000;
            } else if (std.mem.eql(u8, flag, "--api-key") and ai + 1 < args.count) {
                ai += 1;
                api_key = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--model-name") and ai + 1 < args.count) {
                ai += 1;
                model_name = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--expert-tier") and ai + 1 < args.count) {
                ai += 1;
                expert_tier = if (std.mem.eql(u8, args.slices[ai], "auto")) null else (std.fmt.parseInt(u32, args.slices[ai], 10) catch null);
            } else if (std.mem.eql(u8, flag, "--tokens") and ai + 1 < args.count) {
                ai += 1;
                score_tokens_path = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--out") and ai + 1 < args.count) {
                ai += 1;
                score_out_path = args.slices[ai];
            } else if (std.mem.eql(u8, flag, "--window") and ai + 1 < args.count) {
                ai += 1;
                score_window = std.fmt.parseInt(u32, args.slices[ai], 10) catch 1024;
            } else if (std.mem.eql(u8, flag, "--topk") and ai + 1 < args.count) {
                ai += 1;
                score_top_k = std.fmt.parseInt(u32, args.slices[ai], 10) catch 20;
            } else if (std.mem.eql(u8, flag, "--no-prefix-cache")) {
                no_prefix_cache = true;
            } else if (std.mem.eql(u8, flag, "--debug-mtp")) {
                debug_mtp = true;
            } else if (std.mem.eql(u8, flag, "--effort") and ai + 1 < args.count) {
                ai += 1;
                effort = args.slices[ai];
            }
        }

        if (is_score) {
            if (score_tokens_path.len == 0 or score_out_path.len == 0) {
                tiger.print_stderr("Usage: squid7800xt score <model_dir> --tokens in.i32 --out out.bin [--window 1024 --topk 20 --ctx N]\n", .{});
                std.process.exit(2);
            }
            mtp = 0; // scoring never uses the DSpark head; skip loading it
            ctx_len = @max(ctx_len, score_window + 64);
        }

        var tok: tokenizer_mod.Tokenizer = .{};
        tiger.assert(tok.load(model_dir));
        defer tok.deinit();

        var ctx: vk.Context = .{};
        ctx.init();
        defer ctx.deinit();

        var w: weights_mod.ModelWeights = .{};
        w.expert_hi_count = @min(expert_tier orelse engine_mod.Engine.plan_expert_tier(&ctx, ctx_len, mtp > 0), constants.expert_count - 1);
        tiger.assert(w.load(&ctx, model_dir, mtp > 0, draft_vocab));
        defer w.deinit();

        var eng: engine_mod.Engine = .{};
        eng.init(&ctx, &w, ctx_len, .q4);
        eng.spec_threshold = spec_threshold;
        eng.debug_mtp = debug_mtp;
        eng.sample_params = .{ .temperature = temperature, .seed = seed };
        eng.prefix_cache.enabled = !no_prefix_cache;
        defer eng.deinit();

        if (std.mem.eql(u8, subcmd, "serve")) {
            server_mod.run(&eng, &tok, .{
                .host = host,
                .port = port,
                .model_name = model_name,
                .api_key = api_key,
                .default_effort = effort,
                .default_mtp = mtp,
                .default_temperature = temperature,
                .context_tokens_max = ctx_len,
            });
            return;
        }

        if (std.mem.eql(u8, subcmd, "longctx")) {
            eng.bench_longctx();
            return;
        }
        if (std.mem.eql(u8, subcmd, "attncheck")) {
            eng.check_attention();
            return;
        }
        if (is_score) {
            score_mod.run(&eng, &tok, .{
                .tokens_path = score_tokens_path,
                .out_path = score_out_path,
                .window = score_window,
                .top_k = score_top_k,
            });
            return;
        }

        var scratch = tokenizer_mod.TokenizerScratch.init();
        defer scratch.deinit();

        var tmpl_buf: [65536]u8 = undefined;
        var writer = tiger.ByteWriter.init(&tmpl_buf);
        tokenizer_mod.Tokenizer.render_single_user_prompt(&writer, prompt_text, effort);

        const prompt_ids = tiger.alloc_host_zeroed(i32, 32768);
        defer tiger.free_host_slice(i32, prompt_ids);
        const n_prompt = tok.encode_into(writer.slice(), true, &scratch, prompt_ids);
        tiger.assert(n_prompt > 0);
        tiger.print_stdout("[prompt] len={d} ids:", .{n_prompt});
        for (prompt_ids[0..@min(n_prompt, 32)]) |id| tiger.print_stdout(" {d}", .{id});
        tiger.print_stdout("\n", .{});

        if (std.mem.eql(u8, subcmd, "profile")) {
            eng.profile_stages(prompt_ids[0..n_prompt]);
            return;
        }

        const out_buf = tiger.alloc_host_zeroed(i32, @max(max_tokens, 64));
        defer tiger.free_host_slice(i32, out_buf);

        if (std.mem.eql(u8, subcmd, "generate")) {
            tiger.print_stdout("\n--- Output ---\n", .{});
            const st = eng.generate(prompt_ids[0..n_prompt], max_tokens, mtp, out_buf, stream_print_cb, @ptrCast(&tok));
            const dec_toks: f64 = @floatFromInt(if (st.generated_tokens > 0) st.generated_tokens - 1 else 0);
            const tps: f64 = if (st.decode_ms > 0.0) (dec_toks * 1000.0 / st.decode_ms) else 0.0;
            const acc_pct: f64 = if (st.total_drafts > 0) (100.0 * @as(f64, @floatFromInt(st.total_accepted)) / @as(f64, @floatFromInt(st.total_drafts))) else 0.0;
            const dl: f64 = if (st.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(st.spec_steps))) else 1.0;
            tiger.print_stdout(
                "\n--- Stats ---\n[gen] prompt={d} gen={d} prefill={d:.2}ms ({d:.1} tok/s) decode={d:.2}ms speed={d:.2} tok/s | mtp={d} acc={d:.1}% dl={d:.2} lookup={d}/{d}\n",
                .{
                    st.prompt_tokens,
                    st.generated_tokens,
                    st.prefill_ms,
                    @as(f64, @floatFromInt(st.prompt_tokens)) * 1000.0 / @max(st.prefill_ms, 0.01),
                    st.decode_ms,
                    tps,
                    mtp,
                    acc_pct,
                    dl,
                    st.lookup_steps,
                    st.spec_steps,
                },
            );
            return;
        }

        // bench: 1 warmup + N runs median
        _ = eng.generate(prompt_ids[0..n_prompt], @min(32, max_tokens), mtp, out_buf, null, null);
        var speeds: [64]f64 = undefined;
        const n_runs: usize = @min(@max(runs, 1), 64);
        var r: usize = 0;
        while (r < n_runs) : (r += 1) {
            const st = eng.generate(prompt_ids[0..n_prompt], max_tokens, mtp, out_buf, null, null);
            const dec_toks: f64 = @floatFromInt(if (st.generated_tokens > 0) st.generated_tokens - 1 else 0);
            const tps: f64 = if (st.decode_ms > 0.0) (dec_toks * 1000.0 / st.decode_ms) else 0.0;
            const acc_pct: f64 = if (st.total_drafts > 0) (100.0 * @as(f64, @floatFromInt(st.total_accepted)) / @as(f64, @floatFromInt(st.total_drafts))) else 0.0;
            const dl: f64 = if (st.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(st.spec_steps))) else 1.0;
            speeds[r] = tps;
            tiger.print_stdout("  run {d}: gen={d} prefill={d:.2}ms (cached={d}) decode={d:.2}ms -> {d:.2} tok/s (acc={d:.1}% dl={d:.2} lookup={d}/{d})\n", .{
                r + 1, st.generated_tokens, st.prefill_ms, st.cached_tokens, st.decode_ms, tps, acc_pct, dl, st.lookup_steps, st.spec_steps,
            });
        }
        var i: usize = 1;
        while (i < n_runs) : (i += 1) {
            const k = speeds[i];
            var j = i;
            while (j > 0 and speeds[j - 1] > k) : (j -= 1) speeds[j] = speeds[j - 1];
            speeds[j] = k;
        }
        tiger.print_stdout("[bench] MEDIAN decode (mtp={d}): {d:.2} tok/s\n", .{ mtp, speeds[n_runs / 2] });
        return;
    }

    tiger.print_stderr("Usage: squid7800xt <ceilings|load|generate|bench|profile|longctx|attncheck|score|serve> [model_dir] [--prompt ... --max-tokens N --ctx N --mtp N --host H --port P --expert-tier auto|N]\n", .{});
}
