// TigerStyle Win32 mmap Safetensors loader.
// Static fixed-capacity tensor directory (2048 tensors, 4096-slot FNV-1a open-addressed table).
// Zero recursion, zero heap allocations, bounded loops, paired offset & size assertions.

const std = @import("std");
const tiger = @import("tiger.zig");

pub const tensor_count_max: u32 = 2048;
pub const hash_slot_count: u32 = 4096;
pub const header_bytes_max: u64 = 16 * 1024 * 1024;

comptime {
    if ((hash_slot_count & (hash_slot_count - 1)) != 0) {
        @compileError("hash_slot_count must be a power of two");
    }
    if (hash_slot_count < tensor_count_max * 2) {
        @compileError("hash_slot_count must provide <= 50% load factor");
    }
}

pub const TensorInfo = struct {
    name_offset: u32 = 0,
    name_length: u16 = 0,
    data_offset: u64 = 0,
    byte_length: u64 = 0,
};

pub const SafetensorsFile = struct {
    file_handle: tiger.HANDLE = null,
    mapping_handle: tiger.HANDLE = null,
    mapped_ptr: ?[*]const u8 = null,
    file_bytes: u64 = 0,
    data_base_offset: u64 = 0,
    header_bytes: u64 = 0,
    tensor_count: u32 = 0,
    tensors: [tensor_count_max]TensorInfo = @splat(.{}),
    hash_table: [hash_slot_count]u16 = @splat(empty_slot),

    const empty_slot: u16 = 0xffff;

    pub fn open_file(self: *SafetensorsFile, path_z: [*:0]const u8) bool {
        tiger.assert(self.mapped_ptr == null);
        tiger.assert(self.file_handle == null);

        const fh = tiger.CreateFileA(
            path_z,
            tiger.GENERIC_READ,
            tiger.FILE_SHARE_READ,
            null,
            tiger.OPEN_EXISTING,
            tiger.FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (fh == null or fh == tiger.INVALID_HANDLE_VALUE) return false;

        var size_i64: i64 = 0;
        if (tiger.GetFileSizeEx(fh, &size_i64) == 0 or size_i64 <= 8) {
            _ = tiger.CloseHandle(fh);
            return false;
        }
        const file_bytes: u64 = @intCast(size_i64);

        const mh = tiger.CreateFileMappingA(fh, null, tiger.PAGE_READONLY, 0, 0, null);
        if (mh == null) {
            _ = tiger.CloseHandle(fh);
            return false;
        }

        const mapped_raw = tiger.MapViewOfFile(mh, tiger.FILE_MAP_READ, 0, 0, 0);
        if (mapped_raw == null) {
            _ = tiger.CloseHandle(mh);
            _ = tiger.CloseHandle(fh);
            return false;
        }

        const base: [*]const u8 = @ptrCast(mapped_raw.?);
        const header_len: u64 = std.mem.readInt(u64, base[0..8], .little);
        tiger.assert(header_len > 0);
        tiger.assert(header_len <= header_bytes_max);
        tiger.assert(8 + header_len <= file_bytes);

        self.file_handle = fh;
        self.mapping_handle = mh;
        self.mapped_ptr = base;
        self.file_bytes = file_bytes;
        self.header_bytes = header_len;
        self.data_base_offset = 8 + header_len;
        self.tensor_count = 0;
        @memset(&self.hash_table, empty_slot);

        self.parse_header(base[8 .. 8 + @as(usize, @intCast(header_len))]);
        tiger.assert(self.tensor_count > 0);
        return true;
    }

    pub fn close_file(self: *SafetensorsFile) void {
        if (self.mapped_ptr) |base| {
            _ = tiger.UnmapViewOfFile(@ptrCast(base));
            self.mapped_ptr = null;
        }
        if (self.mapping_handle) |mh| {
            _ = tiger.CloseHandle(mh);
            self.mapping_handle = null;
        }
        if (self.file_handle) |fh| {
            _ = tiger.CloseHandle(fh);
            self.file_handle = null;
        }
        self.file_bytes = 0;
        self.tensor_count = 0;
    }

    pub fn lookup_optional(self: *const SafetensorsFile, name: []const u8) ?*const TensorInfo {
        tiger.assert(self.mapped_ptr != null);
        tiger.assert(name.len > 0);
        const header_slice: []const u8 = self.mapped_ptr.?[8 .. 8 + @as(usize, @intCast(self.header_bytes))];
        const hash: u64 = tiger.hash_bytes_u64(name);
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(hash & mask);

        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            const index: u16 = self.hash_table[slot];
            if (index == empty_slot) break;
            tiger.assert(index < self.tensor_count);
            const info: *const TensorInfo = &self.tensors[index];
            const entry_name = header_slice[info.name_offset .. info.name_offset + info.name_length];
            if (std.mem.eql(u8, entry_name, name)) {
                return info;
            }
            slot = (slot + 1) & mask;
        }
        return null;
    }

    pub fn lookup(self: *const SafetensorsFile, name: []const u8) *const TensorInfo {
        if (self.lookup_optional(name)) |info| return info;
        tiger.print_stderr("[safetensors] Missing tensor: {s}\n", .{name});
        @panic("Safetensors tensor not found");
    }

    pub fn data_slice(self: *const SafetensorsFile, info: *const TensorInfo) []const u8 {
        tiger.assert(self.mapped_ptr != null);
        const start: u64 = self.data_base_offset + info.data_offset;
        const end: u64 = start + info.byte_length;
        tiger.assert(end >= start);
        tiger.assert(end <= self.file_bytes);
        return self.mapped_ptr.?[@intCast(start)..@intCast(end)];
    }

    pub fn data_ptr(self: *const SafetensorsFile, info: *const TensorInfo) *const anyopaque {
        const bytes = self.data_slice(info);
        tiger.assert(bytes.len == info.byte_length);
        return @ptrCast(bytes.ptr);
    }

    fn insert_tensor(self: *SafetensorsFile, header: []const u8, info: TensorInfo) void {
        tiger.assert(self.tensor_count < tensor_count_max);
        tiger.assert(info.name_length > 0);
        const index: u16 = @intCast(self.tensor_count);
        self.tensors[index] = info;
        self.tensor_count += 1;

        const name = header[info.name_offset .. info.name_offset + info.name_length];
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(tiger.hash_bytes_u64(name) & mask);
        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            if (self.hash_table[slot] == empty_slot) {
                self.hash_table[slot] = index;
                return;
            }
            slot = (slot + 1) & mask;
        }
        @panic("Safetensors hash table full");
    }

    fn parse_header(self: *SafetensorsFile, header: []const u8) void {
        tiger.assert(header.len >= 2);
        var pos: usize = 0;
        skip_ws(header, &pos);
        tiger.assert(pos < header.len and header[pos] == '{');
        pos += 1;

        var entry_iteration: u32 = 0;
        const max_entries: u32 = tensor_count_max + 16;
        while (pos < header.len and entry_iteration < max_entries) : (entry_iteration += 1) {
            skip_ws(header, &pos);
            if (pos < header.len and header[pos] == '}') break;

            const key_span = parse_raw_json_string(header, &pos);
            skip_ws(header, &pos);
            tiger.assert(pos < header.len and header[pos] == ':');
            pos += 1;
            skip_ws(header, &pos);

            const key_name = header[key_span.start .. key_span.start + key_span.length];
            if (std.mem.eql(u8, key_name, "__metadata__")) {
                skip_json_value(header, &pos);
            } else {
                tiger.assert(pos < header.len and header[pos] == '{');
                pos += 1;
                var start_off: u64 = 0;
                var end_off: u64 = 0;
                var saw_offsets: bool = false;

                var field_iter: u32 = 0;
                while (pos < header.len and field_iter < 32) : (field_iter += 1) {
                    skip_ws(header, &pos);
                    if (pos < header.len and header[pos] == '}') {
                        pos += 1;
                        break;
                    }
                    const fk_span = parse_raw_json_string(header, &pos);
                    const fk_name = header[fk_span.start .. fk_span.start + fk_span.length];
                    skip_ws(header, &pos);
                    tiger.assert(pos < header.len and header[pos] == ':');
                    pos += 1;
                    skip_ws(header, &pos);

                    if (std.mem.eql(u8, fk_name, "data_offsets")) {
                        tiger.assert(pos < header.len and header[pos] == '[');
                        pos += 1;
                        start_off = parse_u64(header, &pos);
                        skip_ws(header, &pos);
                        tiger.assert(pos < header.len and header[pos] == ',');
                        pos += 1;
                        end_off = parse_u64(header, &pos);
                        skip_ws(header, &pos);
                        tiger.assert(pos < header.len and header[pos] == ']');
                        pos += 1;
                        saw_offsets = true;
                    } else {
                        skip_json_value(header, &pos);
                    }
                    skip_ws(header, &pos);
                    if (pos < header.len and header[pos] == ',') pos += 1;
                }

                tiger.assert(saw_offsets);
                tiger.assert(end_off >= start_off);
                tiger.assert(self.data_base_offset + end_off <= self.file_bytes);

                self.insert_tensor(header, .{
                    .name_offset = @intCast(key_span.start),
                    .name_length = @intCast(key_span.length),
                    .data_offset = start_off,
                    .byte_length = end_off - start_off,
                });
            }

            skip_ws(header, &pos);
            if (pos < header.len and header[pos] == ',') pos += 1;
        }
    }
};

const Span = struct {
    start: usize,
    length: usize,
};

inline fn skip_ws(data: []const u8, pos: *usize) void {
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
            pos.* += 1;
        } else {
            break;
        }
    }
}

fn parse_raw_json_string(data: []const u8, pos: *usize) Span {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len and data[pos.*] == '"');
    pos.* += 1;
    const start = pos.*;
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == '\\') {
            pos.* += 2;
            continue;
        }
        if (c == '"') {
            const end = pos.*;
            pos.* += 1;
            return .{ .start = start, .length = end - start };
        }
        pos.* += 1;
    }
    @panic("Unterminated JSON string in safetensors header");
}

fn parse_u64(data: []const u8, pos: *usize) u64 {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len);
    var value: u64 = 0;
    var digits: u32 = 0;
    while (pos.* < data.len and digits < 20) : (digits += 1) {
        const c = data[pos.*];
        if (c < '0' or c > '9') break;
        value = value * 10 + @as(u64, c - '0');
        pos.* += 1;
    }
    tiger.assert(digits > 0);
    return value;
}

fn skip_json_value(data: []const u8, pos: *usize) void {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len);
    const first = data[pos.*];
    if (first == '"') {
        _ = parse_raw_json_string(data, pos);
        return;
    }
    if (first == '{' or first == '[') {
        var depth: i32 = 0;
        while (pos.* < data.len) {
            const c = data[pos.*];
            if (c == '"') {
                _ = parse_raw_json_string(data, pos);
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
        @panic("Unterminated JSON container in safetensors header");
    }
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ',' or c == '}' or c == ']' or c == ' ' or c == '\t' or c == '\n' or c == '\r') break;
        pos.* += 1;
    }
}
