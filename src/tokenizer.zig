// TigerStyle zero-malloc-per-request BPE Tokenizer + o200k_base UTF-8 Pre-Tokenizer
// + Native Inkling-2-Mini Chat Template Renderer.
// All tables are flat open-addressed arrays allocated once at init() — zero dynamic
// allocation during encode(), decode(), or chat template rendering.

const std = @import("std");
const tiger = @import("tiger.zig");

pub const vocab_id_capacity: u32 = 204_800;
pub const vocab_hash_slots: u32 = 524_288; // 2^19 -> ~38% load factor for 200k vocab
pub const merge_hash_slots: u32 = 1_048_576; // 2^20 -> ~19% load factor for ~200k merges
pub const byte_arena_capacity: u32 = 8 * 1024 * 1024; // 8 MiB flat contiguous token string storage
pub const special_tokens_max: u32 = 256;
pub const encode_piece_max_bytes: u32 = 65_536;
pub const scratch_codepoints_max: u32 = 2 * 1024 * 1024;

comptime {
    if ((vocab_hash_slots & (vocab_hash_slots - 1)) != 0) @compileError("vocab_hash_slots must be power of 2");
    if ((merge_hash_slots & (merge_hash_slots - 1)) != 0) @compileError("merge_hash_slots must be power of 2");
}

pub const TokenSlice = struct {
    offset: u32 = 0,
    length: u16 = 0,
    is_special: u8 = 0,
};

pub const VocabHashSlot = struct {
    hash: u64 = 0, // 0 == empty
    token_id: i32 = -1,
};

pub const MergeHashSlot = struct {
    pair_key: u64 = empty_pair_key,
    rank: u32 = 0,
    merged_id: i32 = -1,

    pub const empty_pair_key: u64 = std.math.maxInt(u64);
};

pub const SpecialTokenEntry = struct {
    offset: u32 = 0,
    length: u16 = 0,
    token_id: i32 = -1,
};

const ByteLevelTable = struct {
    cp_to_byte: [512]i16,

    fn build() ByteLevelTable {
        var table: ByteLevelTable = .{ .cp_to_byte = @splat(-1) };
        var direct: [256]bool = @splat(false);
        var b: u32 = 33;
        while (b <= 126) : (b += 1) direct[b] = true;
        b = 161;
        while (b <= 172) : (b += 1) direct[b] = true;
        b = 174;
        while (b <= 255) : (b += 1) direct[b] = true;

        var next_cp: u32 = 256;
        b = 0;
        while (b < 256) : (b += 1) {
            const cp: u32 = if (direct[b]) b else blk: {
                const assigned = next_cp;
                next_cp += 1;
                break :blk assigned;
            };
            table.cp_to_byte[cp] = @intCast(b);
        }
        return table;
    }
};

const byte_level_table: ByteLevelTable = ByteLevelTable.build();

/// Reusable preallocated scratch workspace for zero-malloc BPE encoding.
pub const TokenizerScratch = struct {
    codepoints: []u32,
    byte_offsets: []u32,
    bpe_ids: []i32,

    pub fn init() TokenizerScratch {
        return .{
            .codepoints = tiger.alloc_host_zeroed(u32, scratch_codepoints_max),
            .byte_offsets = tiger.alloc_host_zeroed(u32, scratch_codepoints_max + 1),
            .bpe_ids = tiger.alloc_host_zeroed(i32, encode_piece_max_bytes),
        };
    }

    pub fn deinit(self: *TokenizerScratch) void {
        tiger.free_host_slice(u32, self.codepoints);
        tiger.free_host_slice(u32, self.byte_offsets);
        tiger.free_host_slice(i32, self.bpe_ids);
    }
};

pub const Tokenizer = struct {
    byte_arena: []u8 = &.{},
    byte_arena_used: u32 = 0,
    id_to_slice: []TokenSlice = &.{},
    vocab_table: []VocabHashSlot = &.{},
    merge_table: []MergeHashSlot = &.{},
    byte_to_id: [256]i32 = @splat(0),
    special_tokens: [special_tokens_max]SpecialTokenEntry = @splat(.{}),
    special_count: u32 = 0,
    vocab_count: u32 = 0,
    merge_count: u32 = 0,
    max_token_id: u32 = 0,

    // Cached Inkling-2-Mini special control token IDs
    id_message_model: i32 = 200001,
    id_thinking: i32 = 200008,
    id_text: i32 = 200004,
    id_tool_json: i32 = -1,
    id_end_message: i32 = 200010,
    id_eos: i32 = 200006,

    pub fn load(self: *Tokenizer, model_directory: []const u8) bool {
        tiger.assert(model_directory.len > 0);
        tiger.assert(self.byte_arena.len == 0);

        const start_ns: u64 = tiger.clock_monotonic_ns();
        var path_buffer: [1024]u8 = undefined;
        const path_z = tiger.format_path_z(&path_buffer, model_directory, "/tokenizer.json");

        const fh = tiger.CreateFileA(
            path_z,
            tiger.GENERIC_READ,
            tiger.FILE_SHARE_READ,
            null,
            tiger.OPEN_EXISTING,
            tiger.FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (fh == null or fh == tiger.INVALID_HANDLE_VALUE) {
            tiger.print_stderr("[tokenizer] Failed to open {s}/tokenizer.json\n", .{model_directory});
            return false;
        }
        defer _ = tiger.CloseHandle(fh);

        var size_i64: i64 = 0;
        if (tiger.GetFileSizeEx(fh, &size_i64) == 0 or size_i64 <= 2) return false;
        const file_bytes: usize = @intCast(size_i64);
        const mh = tiger.CreateFileMappingA(fh, null, tiger.PAGE_READONLY, 0, 0, null);
        if (mh == null) return false;
        defer _ = tiger.CloseHandle(mh);

        const mapped_raw = tiger.MapViewOfFile(mh, tiger.FILE_MAP_READ, 0, 0, 0);
        if (mapped_raw == null) return false;
        defer _ = tiger.UnmapViewOfFile(mapped_raw.?);

        const data: []const u8 = @as([*]const u8, @ptrCast(mapped_raw.?))[0..file_bytes];

        // One-time static allocations at startup (zero dynamic allocations afterward!)
        self.byte_arena = tiger.alloc_host_zeroed(u8, byte_arena_capacity);
        self.id_to_slice = tiger.alloc_host_zeroed(TokenSlice, vocab_id_capacity);
        self.vocab_table = tiger.alloc_host_zeroed(VocabHashSlot, vocab_hash_slots);
        self.merge_table = tiger.alloc_host_zeroed(MergeHashSlot, merge_hash_slots);
        for (self.merge_table) |*slot| {
            slot.pair_key = MergeHashSlot.empty_pair_key;
        }

        var unescape_buf_a: [4096]u8 = undefined;
        var unescape_buf_b: [4096]u8 = undefined;
        var raw_buf_a: [4096]u8 = undefined;
        var raw_buf_b: [4096]u8 = undefined;
        var concat_buf: [8192]u8 = undefined;

        var merges_offset: usize = std.math.maxInt(usize);
        var pos: usize = 0;
        skip_ws(data, &pos);
        if (pos >= data.len or data[pos] != '{') return false;
        pos += 1;

        var top_iter: u32 = 0;
        while (pos < data.len and top_iter < 64) : (top_iter += 1) {
            skip_ws(data, &pos);
            if (pos < data.len and data[pos] == '}') {
                pos += 1;
                break;
            }
            const key = parse_json_string(data, &pos, &unescape_buf_a) orelse return false;
            skip_ws(data, &pos);
            if (pos >= data.len or data[pos] != ':') return false;
            pos += 1;
            skip_ws(data, &pos);

            if (std.mem.eql(u8, key, "added_tokens") and pos < data.len and data[pos] == '[') {
                pos += 1;
                var added_iter: u32 = 0;
                while (pos < data.len and added_iter < special_tokens_max * 4) : (added_iter += 1) {
                    skip_ws(data, &pos);
                    if (pos < data.len and data[pos] == ']') {
                        pos += 1;
                        break;
                    }
                    if (data[pos] == '{') {
                        pos += 1;
                        var aid: i32 = -1;
                        var content_len: usize = 0;
                        var field_iter: u32 = 0;
                        while (pos < data.len and field_iter < 32) : (field_iter += 1) {
                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == '}') {
                                pos += 1;
                                break;
                            }
                            const fk = parse_json_string(data, &pos, &unescape_buf_a) orelse return false;
                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == ':') pos += 1;
                            skip_ws(data, &pos);
                            if (std.mem.eql(u8, fk, "id")) {
                                aid = @intCast(parse_i64(data, &pos));
                            } else if (std.mem.eql(u8, fk, "content")) {
                                const content = parse_json_string(data, &pos, &unescape_buf_b) orelse return false;
                                content_len = content.len;
                            } else {
                                skip_json_value(data, &pos);
                            }
                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == ',') pos += 1;
                        }
                        if (aid >= 0 and content_len > 0) {
                            const uid: u32 = @intCast(aid);
                            tiger.assert(uid < vocab_id_capacity);
                            const off = self.store_arena_bytes(unescape_buf_b[0..content_len]);
                            self.id_to_slice[uid] = .{
                                .offset = off,
                                .length = @intCast(content_len),
                                .is_special = 1,
                            };
                            if (uid > self.max_token_id) self.max_token_id = uid;
                            tiger.assert(self.special_count < special_tokens_max);
                            self.special_tokens[self.special_count] = .{
                                .offset = off,
                                .length = @intCast(content_len),
                                .token_id = aid,
                            };
                            self.special_count += 1;
                        }
                    } else {
                        skip_json_value(data, &pos);
                    }
                    skip_ws(data, &pos);
                    if (pos < data.len and data[pos] == ',') pos += 1;
                }
            } else if (std.mem.eql(u8, key, "model") and pos < data.len and data[pos] == '{') {
                pos += 1;
                var model_iter: u32 = 0;
                while (pos < data.len and model_iter < 64) : (model_iter += 1) {
                    skip_ws(data, &pos);
                    if (pos < data.len and data[pos] == '}') {
                        pos += 1;
                        break;
                    }
                    const mk = parse_json_string(data, &pos, &unescape_buf_a) orelse return false;
                    skip_ws(data, &pos);
                    if (pos >= data.len or data[pos] != ':') return false;
                    pos += 1;
                    skip_ws(data, &pos);

                    if (std.mem.eql(u8, mk, "vocab") and pos < data.len and data[pos] == '{') {
                        pos += 1;
                        var vocab_iter: u32 = 0;
                        while (pos < data.len and vocab_iter < vocab_id_capacity) : (vocab_iter += 1) {
                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == '}') {
                                pos += 1;
                                break;
                            }
                            const tok_str = parse_json_string(data, &pos, &unescape_buf_a) orelse return false;
                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == ':') pos += 1;
                            const id: i32 = @intCast(parse_i64(data, &pos));
                            tiger.assert(id >= 0);
                            const uid: u32 = @intCast(id);
                            tiger.assert(uid < vocab_id_capacity);

                            const raw = decode_byte_level(tok_str, &raw_buf_a);
                            const off = self.store_arena_bytes(raw);
                            self.id_to_slice[uid] = .{
                                .offset = off,
                                .length = @intCast(raw.len),
                                .is_special = 0,
                            };
                            if (uid > self.max_token_id) self.max_token_id = uid;
                            self.insert_vocab(raw, id);
                            self.vocab_count += 1;

                            skip_ws(data, &pos);
                            if (pos < data.len and data[pos] == ',') pos += 1;
                        }
                    } else if (std.mem.eql(u8, mk, "merges") and pos < data.len and data[pos] == '[') {
                        merges_offset = pos;
                        skip_json_value(data, &pos);
                    } else {
                        skip_json_value(data, &pos);
                    }
                    skip_ws(data, &pos);
                    if (pos < data.len and data[pos] == ',') pos += 1;
                }
            } else {
                skip_json_value(data, &pos);
            }
            skip_ws(data, &pos);
            if (pos < data.len and data[pos] == ',') pos += 1;
        }

        // Pass 2: parse BPE merges now that vocab table is populated
        if (merges_offset != std.math.maxInt(usize)) {
            var mp: usize = merges_offset + 1;
            var rank: u32 = 0;
            var merge_iter: u32 = 0;
            while (mp < data.len and merge_iter < 600_000) : (merge_iter += 1) {
                skip_ws(data, &mp);
                if (mp < data.len and data[mp] == ']') break;

                var a_slice: []const u8 = &.{};
                var b_slice: []const u8 = &.{};
                if (data[mp] == '[') {
                    mp += 1;
                    skip_ws(data, &mp);
                    a_slice = parse_json_string(data, &mp, &unescape_buf_a) orelse return false;
                    skip_ws(data, &mp);
                    if (mp < data.len and data[mp] == ',') mp += 1;
                    skip_ws(data, &mp);
                    b_slice = parse_json_string(data, &mp, &unescape_buf_b) orelse return false;
                    skip_ws(data, &mp);
                    if (mp < data.len and data[mp] == ']') mp += 1;
                } else if (data[mp] == '"') {
                    const pair = parse_json_string(data, &mp, &unescape_buf_a) orelse return false;
                    if (std.mem.indexOfScalar(u8, pair, ' ')) |sp| {
                        a_slice = pair[0..sp];
                        b_slice = pair[sp + 1 ..];
                    } else {
                        rank += 1;
                        skip_ws(data, &mp);
                        if (mp < data.len and data[mp] == ',') mp += 1;
                        continue;
                    }
                } else {
                    skip_json_value(data, &mp);
                    skip_ws(data, &mp);
                    if (mp < data.len and data[mp] == ',') mp += 1;
                    continue;
                }

                const ra = decode_byte_level(a_slice, &raw_buf_a);
                const rb = decode_byte_level(b_slice, &raw_buf_b);
                const id_a = self.lookup_vocab(ra);
                const id_b = self.lookup_vocab(rb);
                if (id_a >= 0 and id_b >= 0 and ra.len + rb.len <= concat_buf.len) {
                    @memcpy(concat_buf[0..ra.len], ra);
                    @memcpy(concat_buf[ra.len .. ra.len + rb.len], rb);
                    const id_ab = self.lookup_vocab(concat_buf[0 .. ra.len + rb.len]);
                    if (id_ab >= 0) {
                        const pair_key: u64 = (@as(u64, @as(u32, @intCast(id_a))) << 32) | @as(u64, @as(u32, @intCast(id_b)));
                        self.insert_merge(pair_key, rank, id_ab);
                        self.merge_count += 1;
                    }
                }
                rank += 1;
                skip_ws(data, &mp);
                if (mp < data.len and data[mp] == ',') mp += 1;
            }
        }

        // Single-byte token lookup
        var byte_val: u32 = 0;
        while (byte_val < 256) : (byte_val += 1) {
            const single: [1]u8 = .{@intCast(byte_val)};
            const found = self.lookup_vocab(&single);
            self.byte_to_id[byte_val] = if (found >= 0) found else 0;
        }

        // Sort special tokens longest-first (insertion sort; special_count <= 256)
        var si: u32 = 1;
        while (si < self.special_count) : (si += 1) {
            const cur = self.special_tokens[si];
            var sj: u32 = si;
            while (sj > 0 and self.special_tokens[sj - 1].length < cur.length) : (sj -= 1) {
                self.special_tokens[sj] = self.special_tokens[sj - 1];
            }
            self.special_tokens[sj] = cur;
        }

        self.id_message_model = self.lookup_special("<|message_model|>", 200001);
        self.id_thinking = self.lookup_special("<|content_thinking|>", 200008);
        self.id_text = self.lookup_special("<|content_text|>", 200004);
        self.id_tool_json = self.lookup_special("<|content_invoke_tool_json|>", -1);
        self.id_end_message = self.lookup_special("<|end_message|>", 200010);
        self.id_eos = self.lookup_special("<|content_model_end_sampling|>", 200006);

        const elapsed_ms: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - start_ns)) / 1_000_000.0;
        tiger.print_stdout(
            "[tokenizer] Loaded {d} vocab + {d} merges + {d} special tokens in {d:.1}ms (arena={d:.2} MiB)\n",
            .{ self.vocab_count, self.merge_count, self.special_count, elapsed_ms, @as(f64, @floatFromInt(self.byte_arena_used)) / (1024.0 * 1024.0) },
        );
        tiger.assert(self.vocab_count > 0);
        return true;
    }

    pub fn deinit(self: *Tokenizer) void {
        if (self.byte_arena.len > 0) tiger.free_host_slice(u8, self.byte_arena);
        if (self.id_to_slice.len > 0) tiger.free_host_slice(TokenSlice, self.id_to_slice);
        if (self.vocab_table.len > 0) tiger.free_host_slice(VocabHashSlot, self.vocab_table);
        if (self.merge_table.len > 0) tiger.free_host_slice(MergeHashSlot, self.merge_table);
        self.* = .{};
    }

    pub fn decode_token_bytes(self: *const Tokenizer, token_id: i32, skip_special: bool) []const u8 {
        tiger.assert(self.byte_arena.len > 0);
        if (token_id < 0) return &.{};
        const uid: u32 = @intCast(token_id);
        if (uid > self.max_token_id) return &.{};
        const entry = self.id_to_slice[uid];
        if (skip_special and entry.is_special != 0) return &.{};
        return self.byte_arena[entry.offset .. entry.offset + entry.length];
    }

    /// Encode `text` into caller-provided `output_ids` using `scratch`. ZERO dynamic allocation!
    pub fn encode_into(
        self: *const Tokenizer,
        text: []const u8,
        parse_special: bool,
        scratch: *TokenizerScratch,
        output_ids: []i32,
    ) usize {
        tiger.assert(self.byte_arena.len > 0);
        tiger.assert(output_ids.len > 0);
        var out_len: usize = 0;
        if (!parse_special or self.special_count == 0) {
            self.pretokenize_and_encode(text, scratch, output_ids, &out_len);
            return out_len;
        }

        var pos: usize = 0;
        var seg_start: usize = 0;
        var iter: usize = 0;
        const iter_max: usize = text.len + 1;
        while (pos < text.len and iter < iter_max) : (iter += 1) {
            const cand_rel = std.mem.indexOf(u8, text[pos..], "<|") orelse break;
            const cand = pos + cand_rel;
            var matched_id: i32 = -1;
            var matched_len: usize = 0;
            var si: u32 = 0;
            while (si < self.special_count) : (si += 1) {
                const sp = self.special_tokens[si];
                const sp_bytes = self.byte_arena[sp.offset .. sp.offset + sp.length];
                if (cand + sp_bytes.len <= text.len and std.mem.eql(u8, text[cand .. cand + sp_bytes.len], sp_bytes)) {
                    matched_id = sp.token_id;
                    matched_len = sp_bytes.len;
                    break;
                }
            }
            if (matched_id >= 0) {
                if (cand > seg_start) {
                    self.pretokenize_and_encode(text[seg_start..cand], scratch, output_ids, &out_len);
                }
                tiger.assert(out_len < output_ids.len);
                output_ids[out_len] = matched_id;
                out_len += 1;
                pos = cand + matched_len;
                seg_start = pos;
            } else {
                pos = cand + 2;
            }
        }
        if (seg_start < text.len) {
            self.pretokenize_and_encode(text[seg_start..], scratch, output_ids, &out_len);
        }
        return out_len;
    }

    /// Render a single user prompt with Inkling-2-Mini chat template into `writer`.
    pub fn render_single_user_prompt(
        writer: *tiger.ByteWriter,
        prompt_text: []const u8,
        effort: []const u8,
    ) void {
        tiger.assert(writer.buffer.len > 0);
        emit_thinking_effort(writer, effort);
        writer.append_slice("<|message_user|><|content_text|>");
        writer.append_slice(prompt_text);
        writer.append_slice("<|end_message|>");
    }

    pub fn emit_thinking_effort(writer: *tiger.ByteWriter, effort: []const u8) void {
        var num: f64 = 0.0;
        if (effort.len > 0) {
            if (std.mem.eql(u8, effort, "none")) {
                num = 0.0;
            } else if (std.mem.eql(u8, effort, "minimal")) {
                num = 0.1;
            } else if (std.mem.eql(u8, effort, "low")) {
                num = 0.2;
            } else if (std.mem.eql(u8, effort, "medium")) {
                num = 0.7;
            } else if (std.mem.eql(u8, effort, "high")) {
                num = 0.9;
            } else if (std.mem.eql(u8, effort, "xhigh") or std.mem.eql(u8, effort, "max")) {
                num = 0.99;
            } else {
                num = std.fmt.parseFloat(f64, effort) catch 0.0;
            }
        }
        if (num < 0.0) num = 0.0;
        if (num > 0.99) num = 0.99;
        writer.append_slice("<|message_system|><|content_text|>Thinking effort level: ");
        if (num == 0.0) {
            writer.append_slice("0");
        } else {
            // Trim trailing zeros to match C++ `%.6g` formatting
            var tmp: [32]u8 = undefined;
            const formatted = std.fmt.bufPrint(&tmp, "{d:.6}", .{num}) catch "0";
            var end: usize = formatted.len;
            if (std.mem.indexOfScalar(u8, formatted, '.') != null) {
                while (end > 1 and formatted[end - 1] == '0') end -= 1;
                if (end > 1 and formatted[end - 1] == '.') end -= 1;
            }
            writer.append_slice(formatted[0..end]);
        }
        writer.append_slice("<|end_message|>");
    }

    fn store_arena_bytes(self: *Tokenizer, bytes: []const u8) u32 {
        tiger.assert(bytes.len <= std.math.maxInt(u16));
        tiger.assert(self.byte_arena_used + bytes.len <= self.byte_arena.len);
        const off = self.byte_arena_used;
        @memcpy(self.byte_arena[off .. off + bytes.len], bytes);
        self.byte_arena_used += @intCast(bytes.len);
        return off;
    }

    fn insert_vocab(self: *Tokenizer, raw: []const u8, token_id: i32) void {
        tiger.assert(token_id >= 0);
        const hash = tiger.hash_bytes_u64(raw);
        const mask: u32 = vocab_hash_slots - 1;
        var slot: u32 = @intCast(hash & mask);
        var probe: u32 = 0;
        while (probe < vocab_hash_slots) : (probe += 1) {
            const entry = &self.vocab_table[slot];
            if (entry.hash == 0) {
                entry.hash = hash;
                entry.token_id = token_id;
                return;
            }
            if (entry.hash == hash) {
                const existing = self.id_to_slice[@intCast(entry.token_id)];
                const existing_bytes = self.byte_arena[existing.offset .. existing.offset + existing.length];
                if (std.mem.eql(u8, existing_bytes, raw)) {
                    entry.token_id = token_id;
                    return;
                }
            }
            slot = (slot + 1) & mask;
        }
        @panic("Vocab hash table overflow");
    }

    fn lookup_vocab(self: *const Tokenizer, raw: []const u8) i32 {
        const hash = tiger.hash_bytes_u64(raw);
        const mask: u32 = vocab_hash_slots - 1;
        var slot: u32 = @intCast(hash & mask);
        var probe: u32 = 0;
        while (probe < vocab_hash_slots) : (probe += 1) {
            const entry = self.vocab_table[slot];
            if (entry.hash == 0) return -1;
            if (entry.hash == hash) {
                const existing = self.id_to_slice[@intCast(entry.token_id)];
                const existing_bytes = self.byte_arena[existing.offset .. existing.offset + existing.length];
                if (std.mem.eql(u8, existing_bytes, raw)) {
                    return entry.token_id;
                }
            }
            slot = (slot + 1) & mask;
        }
        return -1;
    }

    fn insert_merge(self: *Tokenizer, pair_key: u64, rank: u32, merged_id: i32) void {
        tiger.assert(pair_key != MergeHashSlot.empty_pair_key);
        tiger.assert(merged_id >= 0);
        const mask: u32 = merge_hash_slots - 1;
        var slot: u32 = @intCast(tiger.hash_pair_u64(pair_key) & mask);
        var probe: u32 = 0;
        while (probe < merge_hash_slots) : (probe += 1) {
            const entry = &self.merge_table[slot];
            if (entry.pair_key == MergeHashSlot.empty_pair_key) {
                entry.pair_key = pair_key;
                entry.rank = rank;
                entry.merged_id = merged_id;
                return;
            }
            if (entry.pair_key == pair_key) return; // keep earliest (lowest rank)
            slot = (slot + 1) & mask;
        }
        @panic("Merge hash table overflow");
    }

    inline fn lookup_merge(self: *const Tokenizer, pair_key: u64) ?MergeHashSlot {
        const mask: u32 = merge_hash_slots - 1;
        var slot: u32 = @intCast(tiger.hash_pair_u64(pair_key) & mask);
        var probe: u32 = 0;
        while (probe < merge_hash_slots) : (probe += 1) {
            const entry = self.merge_table[slot];
            if (entry.pair_key == MergeHashSlot.empty_pair_key) return null;
            if (entry.pair_key == pair_key) return entry;
            slot = (slot + 1) & mask;
        }
        return null;
    }

    fn lookup_special(self: *const Tokenizer, name: []const u8, fallback: i32) i32 {
        var si: u32 = 0;
        while (si < self.special_count) : (si += 1) {
            const sp = self.special_tokens[si];
            const bytes = self.byte_arena[sp.offset .. sp.offset + sp.length];
            if (std.mem.eql(u8, bytes, name)) return sp.token_id;
        }
        return fallback;
    }

    fn encode_bpe_piece(
        self: *const Tokenizer,
        piece: []const u8,
        scratch: *TokenizerScratch,
        output_ids: []i32,
        out_len: *usize,
    ) void {
        if (piece.len == 0) return;
        if (piece.len == 1) {
            tiger.assert(out_len.* < output_ids.len);
            output_ids[out_len.*] = self.byte_to_id[piece[0]];
            out_len.* += 1;
            return;
        }
        // Fast path: check if entire piece is directly in vocab (Wait: BPE merges can differ from direct lookup
        // only if an intermediate merge didn't exist; in o200k_base we run exact merge loop for 100% parity!)
        tiger.assert(piece.len <= scratch.bpe_ids.len);
        var ids_len: usize = piece.len;
        for (piece, 0..) |byte, idx| {
            scratch.bpe_ids[idx] = self.byte_to_id[byte];
        }

        const ids = scratch.bpe_ids;
        var step: usize = 0;
        while (ids_len >= 2 and step < piece.len) : (step += 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_merged: i32 = -1;
            var best_idx: usize = 0;

            var i: usize = 0;
            while (i + 1 < ids_len) : (i += 1) {
                const pair_key: u64 = (@as(u64, @as(u32, @intCast(ids[i]))) << 32) | @as(u64, @as(u32, @intCast(ids[i + 1])));
                if (self.lookup_merge(pair_key)) |entry| {
                    if (entry.rank < best_rank) {
                        best_rank = entry.rank;
                        best_merged = entry.merged_id;
                        best_idx = i;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;

            const target_a: i32 = ids[best_idx];
            const target_b: i32 = ids[best_idx + 1];
            var write_idx: usize = 0;
            var read_idx: usize = 0;
            while (read_idx < ids_len) {
                if (read_idx + 1 < ids_len and ids[read_idx] == target_a and ids[read_idx + 1] == target_b) {
                    ids[write_idx] = best_merged;
                    write_idx += 1;
                    read_idx += 2;
                } else {
                    ids[write_idx] = ids[read_idx];
                    write_idx += 1;
                    read_idx += 1;
                }
            }
            tiger.assert(write_idx < ids_len);
            ids_len = write_idx;
        }

        tiger.assert(out_len.* + ids_len <= output_ids.len);
        @memcpy(output_ids[out_len.* .. out_len.* + ids_len], ids[0..ids_len]);
        out_len.* += ids_len;
    }

    fn pretokenize_and_encode(
        self: *const Tokenizer,
        text: []const u8,
        scratch: *TokenizerScratch,
        output_ids: []i32,
        out_len: *usize,
    ) void {
        if (text.len == 0) return;
        tiger.assert(text.len <= scratch.codepoints.len);

        var cp_count: usize = 0;
        var bp: usize = 0;
        while (bp < text.len) {
            scratch.byte_offsets[cp_count] = @intCast(bp);
            scratch.codepoints[cp_count] = next_utf8(text, &bp);
            cp_count += 1;
        }
        scratch.byte_offsets[cp_count] = @intCast(text.len);

        const cps = scratch.codepoints[0..cp_count];
        const offs = scratch.byte_offsets[0 .. cp_count + 1];
        var i: usize = 0;
        var iter: usize = 0;
        while (i < cp_count and iter <= cp_count) : (iter += 1) {
            var end: usize = i;
            const m1 = try_branch1(cps, i);
            if (m1 > i) {
                end = m1;
            } else {
                const m2 = try_branch2(cps, i);
                if (m2 > i) {
                    end = m2;
                } else if (cp_is_digit(cps[i])) {
                    end = i + 1;
                    if (end < cp_count and cp_is_digit(cps[end])) end += 1;
                    if (end < cp_count and cp_is_digit(cps[end])) end += 1;
                } else {
                    var j: usize = i;
                    if (cps[j] == ' ' and j + 1 < cp_count and !cp_is_space(cps[j + 1]) and !cp_is_letter(cps[j + 1]) and !cp_is_digit(cps[j + 1])) {
                        j += 1;
                    }
                    if (!cp_is_space(cps[j]) and !cp_is_letter(cps[j]) and !cp_is_digit(cps[j])) {
                        while (j < cp_count and !cp_is_space(cps[j]) and !cp_is_letter(cps[j]) and !cp_is_digit(cps[j])) : (j += 1) {}
                        while (j < cp_count and (cps[j] == '\r' or cps[j] == '\n' or cps[j] == '/')) : (j += 1) {}
                        end = j;
                    } else if (cp_is_space(cps[i])) {
                        var ws_end: usize = i;
                        var last_nl: usize = std.math.maxInt(usize);
                        while (ws_end < cp_count and cp_is_space(cps[ws_end])) : (ws_end += 1) {
                            if (cp_is_newline(cps[ws_end])) last_nl = ws_end;
                        }
                        if (last_nl != std.math.maxInt(usize)) {
                            end = last_nl + 1;
                        } else if (ws_end == cp_count) {
                            end = ws_end;
                        } else if (ws_end - i > 1) {
                            end = ws_end - 1;
                        } else {
                            end = i + 1;
                        }
                    } else {
                        end = i + 1;
                    }
                }
            }
            tiger.assert(end > i and end <= cp_count);
            self.encode_bpe_piece(text[offs[i]..offs[end]], scratch, output_ids, out_len);
            i = end;
        }
    }
};

// --- Exact o200k_base UTF-8 Lexer Helpers ---

inline fn next_utf8(s: []const u8, pos: *usize) u32 {
    const c0: u32 = s[pos.*];
    pos.* += 1;
    if (c0 < 0x80) return c0;
    if ((c0 & 0xE0) == 0xC0 and pos.* < s.len) {
        const c1: u32 = s[pos.*];
        pos.* += 1;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 and pos.* + 1 < s.len) {
        const c1: u32 = s[pos.*];
        const c2: u32 = s[pos.* + 1];
        pos.* += 2;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0 and pos.* + 2 < s.len) {
        const c1: u32 = s[pos.*];
        const c2: u32 = s[pos.* + 1];
        const c3: u32 = s[pos.* + 2];
        pos.* += 3;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }
    return c0;
}

fn decode_byte_level(input: []const u8, out_buffer: []u8) []const u8 {
    var out_len: usize = 0;
    var p: usize = 0;
    while (p < input.len) {
        const prev = p;
        const cp = next_utf8(input, &p);
        if (cp < 512 and byte_level_table.cp_to_byte[cp] >= 0) {
            tiger.assert(out_len < out_buffer.len);
            out_buffer[out_len] = @intCast(byte_level_table.cp_to_byte[cp]);
            out_len += 1;
        } else {
            const span_len = p - prev;
            tiger.assert(out_len + span_len <= out_buffer.len);
            @memcpy(out_buffer[out_len .. out_len + span_len], input[prev..p]);
            out_len += span_len;
        }
    }
    return out_buffer[0..out_len];
}

inline fn cp_is_newline(cp: u32) bool {
    return cp == '\r' or cp == '\n';
}

inline fn cp_is_space(cp: u32) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C or
        cp == 0x0085 or cp == 0x00A0 or cp == 0x1680 or
        (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029 or
        cp == 0x202F or cp == 0x205F or cp == 0x3000;
}

inline fn cp_is_digit(cp: u32) bool {
    if (cp >= '0' and cp <= '9') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x0660 and cp <= 0x0669) or (cp >= 0x06F0 and cp <= 0x06F9) or
        (cp >= 0x0966 and cp <= 0x096F) or (cp >= 0xFF10 and cp <= 0xFF19);
}

inline fn cp_is_upper(cp: u32) bool {
    if (cp >= 'A' and cp <= 'Z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00C0 and cp <= 0x00D6) or (cp >= 0x00D8 and cp <= 0x00DE) or
        (cp >= 0x0391 and cp <= 0x03A9) or (cp >= 0x0410 and cp <= 0x042F) or
        (cp >= 0xFF21 and cp <= 0xFF3A);
}

inline fn cp_is_lower(cp: u32) bool {
    if (cp >= 'a' and cp <= 'z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00DF and cp <= 0x00F6) or (cp >= 0x00F8 and cp <= 0x00FF) or
        (cp >= 0x03B1 and cp <= 0x03C9) or (cp >= 0x0430 and cp <= 0x044F) or
        (cp >= 0xFF41 and cp <= 0xFF5A);
}

inline fn cp_is_other_letter_or_mark(cp: u32) bool {
    if (cp < 0x80) return false;
    return (cp >= 0x0100 and cp <= 0x024F) or (cp >= 0x0300 and cp <= 0x036F) or
        (cp >= 0x0590 and cp <= 0x06FF) or (cp >= 0x0900 and cp <= 0x097F) or
        (cp >= 0x3040 and cp <= 0x30FF) or (cp >= 0x3400 and cp <= 0x9FFF) or
        (cp >= 0xAC00 and cp <= 0xD7AF) or (cp >= 0xF900 and cp <= 0xFAFF);
}

inline fn cp_is_letter(cp: u32) bool {
    return cp_is_upper(cp) or cp_is_lower(cp) or cp_is_other_letter_or_mark(cp);
}

inline fn cp_is_cat_U(cp: u32) bool {
    return cp_is_upper(cp) or cp_is_other_letter_or_mark(cp);
}

inline fn cp_is_cat_L(cp: u32) bool {
    return cp_is_lower(cp) or cp_is_other_letter_or_mark(cp);
}

fn match_contraction(cps: []const u32, i: usize) usize {
    if (i + 1 >= cps.len or cps[i] != '\'') return 0;
    const c1: u32 = if (cps[i + 1] >= 'A' and cps[i + 1] <= 'Z') cps[i + 1] + 32 else cps[i + 1];
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return 2;
    if (i + 2 < cps.len) {
        const c2: u32 = if (cps[i + 2] >= 'A' and cps[i + 2] <= 'Z') cps[i + 2] + 32 else cps[i + 2];
        if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return 3;
    }
    return 0;
}

fn try_branch1(cps: []const u32, start: usize) usize {
    const n = cps.len;
    var j: usize = start;
    if (j < n and !cp_is_newline(cps[j]) and !cp_is_letter(cps[j]) and !cp_is_digit(cps[j])) {
        j += 1;
    }
    const run_start = j;
    while (j < n and cp_is_letter(cps[j])) : (j += 1) {}
    var u_end: usize = run_start;
    while (u_end < j and cp_is_cat_U(cps[u_end])) : (u_end += 1) {}
    var l_start: usize = u_end;
    if (l_start == j and u_end > run_start and cp_is_cat_L(cps[u_end - 1])) {
        l_start = u_end - 1;
    }
    var l_end: usize = l_start;
    while (l_end < j and cp_is_cat_L(cps[l_end])) : (l_end += 1) {}
    if (l_end > l_start) {
        return l_end + match_contraction(cps, l_end);
    }
    return 0;
}

fn try_branch2(cps: []const u32, start: usize) usize {
    const n = cps.len;
    var j: usize = start;
    if (j < n and !cp_is_newline(cps[j]) and !cp_is_letter(cps[j]) and !cp_is_digit(cps[j])) {
        j += 1;
    }
    const u_start = j;
    while (j < n and cp_is_cat_U(cps[j])) : (j += 1) {}
    if (j == u_start) return 0;
    while (j < n and cp_is_cat_L(cps[j])) : (j += 1) {}
    return j + match_contraction(cps, j);
}

// --- Iterative Zero-Allocation JSON Scanner Helpers (shared with server.zig) ---

pub inline fn skip_ws(data: []const u8, pos: *usize) void {
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
            pos.* += 1;
        } else {
            break;
        }
    }
}

fn hex_nibble(c: u8) ?u32 {
    if (c >= '0' and c <= '9') return c - '0';
    if (c >= 'a' and c <= 'f') return c - 'a' + 10;
    if (c >= 'A' and c <= 'F') return c - 'A' + 10;
    return null;
}

fn parse_hex4(data: []const u8, pos: usize) ?u32 {
    if (pos + 4 > data.len) return null;
    const h0 = hex_nibble(data[pos]) orelse return null;
    const h1 = hex_nibble(data[pos + 1]) orelse return null;
    const h2 = hex_nibble(data[pos + 2]) orelse return null;
    const h3 = hex_nibble(data[pos + 3]) orelse return null;
    return (h0 << 12) | (h1 << 8) | (h2 << 4) | h3;
}

fn append_utf8_codepoint(out: []u8, out_len: *usize, cp: u32) bool {
    if (cp < 0x80) {
        if (out_len.* + 1 > out.len) return false;
        out[out_len.*] = @intCast(cp);
        out_len.* += 1;
    } else if (cp < 0x800) {
        if (out_len.* + 2 > out.len) return false;
        out[out_len.*] = @intCast(0xC0 | (cp >> 6));
        out[out_len.* + 1] = @intCast(0x80 | (cp & 0x3F));
        out_len.* += 2;
    } else if (cp < 0x10000) {
        if (out_len.* + 3 > out.len) return false;
        out[out_len.*] = @intCast(0xE0 | (cp >> 12));
        out[out_len.* + 1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        out[out_len.* + 2] = @intCast(0x80 | (cp & 0x3F));
        out_len.* += 3;
    } else {
        if (out_len.* + 4 > out.len) return false;
        out[out_len.*] = @intCast(0xF0 | (cp >> 18));
        out[out_len.* + 1] = @intCast(0x80 | ((cp >> 12) & 0x3F));
        out[out_len.* + 2] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        out[out_len.* + 3] = @intCast(0x80 | (cp & 0x3F));
        out_len.* += 4;
    }
    return true;
}

pub fn parse_json_string(data: []const u8, pos: *usize, out_buf: []u8) ?[]const u8 {
    skip_ws(data, pos);
    if (pos.* >= data.len or data[pos.*] != '"') return null;
    pos.* += 1;

    // Fast path: check if string has no escapes before closing quote -> zero copy?
    // Copy into out_buf so callers always own a contiguous decoded slice.
    var out_len: usize = 0;
    while (pos.* < data.len) {
        const c = data[pos.*];
        pos.* += 1;
        if (c == '"') return out_buf[0..out_len];
        if (c == '\\') {
            if (pos.* >= data.len) return null;
            const esc = data[pos.*];
            pos.* += 1;
            switch (esc) {
                '"' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '"';
                    out_len += 1;
                },
                '\\' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '\\';
                    out_len += 1;
                },
                '/' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '/';
                    out_len += 1;
                },
                'b' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = 0x08;
                    out_len += 1;
                },
                'f' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = 0x0C;
                    out_len += 1;
                },
                'n' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '\n';
                    out_len += 1;
                },
                'r' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '\r';
                    out_len += 1;
                },
                't' => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = '\t';
                    out_len += 1;
                },
                'u' => {
                    var cp = parse_hex4(data, pos.*) orelse return null;
                    pos.* += 4;
                    if (cp >= 0xD800 and cp <= 0xDBFF and pos.* + 6 <= data.len and data[pos.*] == '\\' and data[pos.* + 1] == 'u') {
                        const lo = parse_hex4(data, pos.* + 2) orelse return null;
                        if (lo >= 0xDC00 and lo <= 0xDFFF) {
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                            pos.* += 6;
                        }
                    }
                    if (!append_utf8_codepoint(out_buf, &out_len, cp)) return null;
                },
                else => {
                    if (out_len >= out_buf.len) return null;
                    out_buf[out_len] = esc;
                    out_len += 1;
                },
            }
        } else {
            if (out_len >= out_buf.len) return null;
            out_buf[out_len] = c;
            out_len += 1;
        }
    }
    return null;
}

pub fn parse_i64(data: []const u8, pos: *usize) i64 {
    skip_ws(data, pos);
    var neg: bool = false;
    if (pos.* < data.len and data[pos.*] == '-') {
        neg = true;
        pos.* += 1;
    }
    var val: i64 = 0;
    var digits: u32 = 0;
    while (pos.* < data.len and digits < 20) : (digits += 1) {
        const c = data[pos.*];
        if (c < '0' or c > '9') break;
        val = val * 10 + @as(i64, c - '0');
        pos.* += 1;
    }
    return if (neg) -val else val;
}

pub fn skip_json_value(data: []const u8, pos: *usize) void {
    skip_ws(data, pos);
    if (pos.* >= data.len) return;
    const c0 = data[pos.*];
    if (c0 == '"') {
        pos.* += 1;
        while (pos.* < data.len) {
            const c = data[pos.*];
            if (c == '\\') {
                pos.* += 2;
                continue;
            }
            pos.* += 1;
            if (c == '"') return;
        }
        return;
    }
    if (c0 == '{' or c0 == '[') {
        var depth: i32 = 0;
        while (pos.* < data.len) {
            const c = data[pos.*];
            if (c == '"') {
                pos.* += 1;
                while (pos.* < data.len) {
                    const sc = data[pos.*];
                    if (sc == '\\') {
                        pos.* += 2;
                        continue;
                    }
                    pos.* += 1;
                    if (sc == '"') break;
                }
                continue;
            }
            if (c == '{' or c == '[') {
                depth += 1;
            } else if (c == '}' or c == ']') {
                depth -= 1;
                pos.* += 1;
                if (depth == 0) return;
                continue;
            }
            pos.* += 1;
        }
        return;
    }
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ',' or c == '}' or c == ']' or c == ' ' or c == '\t' or c == '\n' or c == '\r') break;
        pos.* += 1;
    }
}
