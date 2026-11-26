// TigerStyle Windows core primitives, always-on release assertions, monotonic timing,
// Win32 mmap & Winsock2 ABI, fixed-capacity ByteWriter, and direct stdout/stderr.
// Zero dynamic allocations during request or token execution.

const std = @import("std");

/// TigerStyle always-on assertion: remains active even in ReleaseFast builds.
/// Crashes loudly on invariant violation rather than silently corrupting state.
pub inline fn assert(condition: bool) void {
    if (!condition) {
        @branchHint(.cold);
        @panic("TigerStyle assertion failed");
    }
}

/// Always-on assertion for Vulkan status codes (`0 == VK_SUCCESS`).
pub inline fn assert_vk(result: i32) void {
    if (result != 0) {
        @branchHint(.cold);
        print_stderr("[vulkan] Call failed with VkResult={d}\n", .{result});
        @panic("Vulkan assertion failed");
    }
}

/// Align `value` upward to a power-of-two `alignment` boundary.
pub inline fn align_up_u64(value: u64, alignment: u64) u64 {
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0);
    const mask: u64 = alignment - 1;
    const result: u64 = (value + mask) & ~mask;
    assert(result >= value);
    assert((result & mask) == 0);
    return result;
}

pub inline fn min_u64(a: u64, b: u64) u64 {
    return if (a < b) a else b;
}

pub inline fn max_u64(a: u64, b: u64) u64 {
    return if (a > b) a else b;
}

pub inline fn min_i32(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

pub inline fn max_i32(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

/// FNV-1a 64-bit hash for flat static open-addressed tables.
pub inline fn hash_bytes_u64(bytes: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (bytes) |byte| {
        hash ^= @as(u64, byte);
        hash *%= 1099511628211;
    }
    return if (hash == 0) 1 else hash;
}

/// SplitMix64 mixer for 64-bit integer keys (e.g., BPE token pairs).
pub inline fn hash_pair_u64(key: u64) u64 {
    var mixed: u64 = key +% 0x9e3779b97f4a7c15;
    mixed = (mixed ^ (mixed >> 30)) *% 0xbf58476d1ce4e5b9;
    mixed = (mixed ^ (mixed >> 27)) *% 0x94d049bb133111eb;
    mixed = mixed ^ (mixed >> 31);
    return if (mixed == 0) 1 else mixed;
}

// --- Win32 & C-Runtime ABI Declarations ---

pub const HANDLE = ?*anyopaque;
pub const HMODULE = ?*anyopaque;
pub const INVALID_HANDLE_VALUE: HANDLE = @ptrFromInt(std.math.maxInt(usize));

pub const GENERIC_READ: u32 = 0x80000000;
pub const GENERIC_WRITE: u32 = 0x40000000;
pub const FILE_SHARE_READ: u32 = 0x00000001;
pub const OPEN_EXISTING: u32 = 3;
pub const CREATE_ALWAYS: u32 = 2;
pub const FILE_ATTRIBUTE_NORMAL: u32 = 0x00000080;
pub const PAGE_READONLY: u32 = 0x02;
pub const FILE_MAP_READ: u32 = 0x0004;
pub const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
pub const STD_ERROR_HANDLE: u32 = @bitCast(@as(i32, -12));

pub const FILETIME = extern struct {
    dwLowDateTime: u32,
    dwHighDateTime: u32,
};

pub extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) HMODULE;
pub extern "kernel32" fn GetProcAddress(hModule: HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
pub extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;
pub extern "kernel32" fn GetSystemTimeAsFileTime(lpSystemTimeAsFileTime: *FILETIME) callconv(.winapi) void;
pub extern "kernel32" fn GetStdHandle(nStdHandle: u32) callconv(.winapi) HANDLE;
pub extern "kernel32" fn WriteFile(
    hFile: HANDLE,
    lpBuffer: [*]const u8,
    nNumberOfBytesToWrite: u32,
    lpNumberOfBytesWritten: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;
pub extern "kernel32" fn ReadFile(
    hFile: HANDLE,
    lpBuffer: [*]u8,
    nNumberOfBytesToRead: u32,
    lpNumberOfBytesRead: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;
pub extern "kernel32" fn CreateFileA(
    lpFileName: [*:0]const u8,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: ?*anyopaque,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: HANDLE,
) callconv(.winapi) HANDLE;
pub extern "kernel32" fn GetFileSizeEx(hFile: HANDLE, lpFileSize: *i64) callconv(.winapi) i32;
pub extern "kernel32" fn CreateFileMappingA(
    hFile: HANDLE,
    lpFileMappingAttributes: ?*anyopaque,
    flProtect: u32,
    dwMaximumSizeHigh: u32,
    dwMaximumSizeLow: u32,
    lpName: ?[*:0]const u8,
) callconv(.winapi) HANDLE;
pub extern "kernel32" fn MapViewOfFile(
    hFileMappingObject: HANDLE,
    dwDesiredAccess: u32,
    dwFileOffsetHigh: u32,
    dwFileOffsetLow: u32,
    dwNumberOfBytesToMap: usize,
) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: *const anyopaque) callconv(.winapi) i32;
pub extern "kernel32" fn CloseHandle(hObject: HANDLE) callconv(.winapi) i32;
pub extern "kernel32" fn GetEnvironmentVariableA(
    lpName: [*:0]const u8,
    lpBuffer: [*]u8,
    nSize: u32,
) callconv(.winapi) u32;
pub extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.winapi) void;

pub extern "c" fn malloc(size: usize) ?*anyopaque;
pub extern "c" fn calloc(count: usize, size: usize) ?*anyopaque;
pub extern "c" fn free(ptr: ?*anyopaque) void;
pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

// --- Winsock2 ABI Declarations ---

pub const SOCKET = usize;
pub const INVALID_SOCKET: SOCKET = std.math.maxInt(usize);
pub const SOCKET_ERROR: i32 = -1;

pub const AF_INET: i32 = 2;
pub const AF_INET6: i32 = 23;
pub const SOCK_STREAM: i32 = 1;
pub const SOL_SOCKET: i32 = 0xffff;
pub const SO_REUSEADDR: i32 = 0x0004;
pub const IPPROTO_TCP: i32 = 6;
pub const IPPROTO_IPV6: i32 = 41;
pub const TCP_NODELAY: i32 = 1;
pub const IPV6_V6ONLY: i32 = 27;
pub const POLLRDNORM: i16 = 0x0100;

pub const SockaddrIn = extern struct {
    sin_family: u16,
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
};

pub const SockaddrIn6 = extern struct {
    sin6_family: u16,
    sin6_port: u16,
    sin6_flowinfo: u32 = 0,
    sin6_addr: [16]u8 = @splat(0),
    sin6_scope_id: u32 = 0,
};

pub const WSAPOLLFD = extern struct {
    fd: SOCKET,
    events: i16,
    revents: i16 = 0,
};

pub const WSAData = extern struct {
    wVersion: u16,
    wHighVersion: u16,
    iMaxSockets: u16,
    iMaxUdpDg: u16,
    lpVendorInfo: ?[*]u8,
    szDescription: [257]u8,
    szSystemStatus: [129]u8,
};

comptime {
    if (@sizeOf(SockaddrIn) != 16) {
        @compileError("SockaddrIn must be exactly 16 bytes");
    }
    if (@sizeOf(SockaddrIn6) != 28) {
        @compileError("SockaddrIn6 must be exactly 28 bytes");
    }
}

pub extern "ws2_32" fn WSAStartup(wVersionRequested: u16, lpWSAData: *WSAData) callconv(.winapi) i32;
pub extern "ws2_32" fn WSACleanup() callconv(.winapi) i32;
pub extern "ws2_32" fn socket(af: i32, sock_type: i32, protocol: i32) callconv(.winapi) SOCKET;
pub extern "ws2_32" fn closesocket(s: SOCKET) callconv(.winapi) i32;
pub extern "ws2_32" fn setsockopt(s: SOCKET, level: i32, optname: i32, optval: *const anyopaque, optlen: i32) callconv(.winapi) i32;
pub extern "ws2_32" fn bind(s: SOCKET, name: *const anyopaque, namelen: i32) callconv(.winapi) i32;
pub extern "ws2_32" fn listen(s: SOCKET, backlog: i32) callconv(.winapi) i32;
pub extern "ws2_32" fn accept(s: SOCKET, addr: ?*anyopaque, addrlen: ?*i32) callconv(.winapi) SOCKET;
pub extern "ws2_32" fn recv(s: SOCKET, buf: [*]u8, len: i32, flags: i32) callconv(.winapi) i32;
pub extern "ws2_32" fn send(s: SOCKET, buf: [*]const u8, len: i32, flags: i32) callconv(.winapi) i32;
pub extern "ws2_32" fn inet_pton(family: i32, pszAddrString: [*:0]const u8, pAddrBuf: *anyopaque) callconv(.winapi) i32;
pub extern "ws2_32" fn WSAPoll(fdArray: [*]WSAPOLLFD, fds: u32, timeout: i32) callconv(.winapi) i32;

pub inline fn htons(value: u16) u16 {
    return @byteSwap(value);
}

var qpc_freq_cached: i64 = 0;

/// Monotonic wall clock in nanoseconds via QueryPerformanceCounter.
pub fn clock_monotonic_ns() u64 {
    if (qpc_freq_cached == 0) {
        var freq: i64 = 0;
        const ok = QueryPerformanceFrequency(&freq);
        assert(ok != 0 and freq > 0);
        qpc_freq_cached = freq;
    }
    var count: i64 = 0;
    const ok = QueryPerformanceCounter(&count);
    assert(ok != 0 and count >= 0);
    const num: u128 = @as(u128, @intCast(count)) * 1_000_000_000;
    return @intCast(num / @as(u128, @intCast(qpc_freq_cached)));
}

/// UNIX epoch timestamp in seconds (for OpenAI JSON `created` fields).
pub fn epoch_seconds() i64 {
    var ft: FILETIME = .{ .dwLowDateTime = 0, .dwHighDateTime = 0 };
    GetSystemTimeAsFileTime(&ft);
    const ticks: u64 = (@as(u64, ft.dwHighDateTime) << 32) | @as(u64, ft.dwLowDateTime);
    const epoch_offset: u64 = 116444736000000000; // 1601->1970 in 100ns ticks
    if (ticks < epoch_offset) return 0;
    return @intCast((ticks - epoch_offset) / 10_000_000);
}

/// Allocate zero-initialized host memory at startup (never called on hot paths).
pub fn alloc_host_zeroed(comptime T: type, count: usize) []T {
    assert(count > 0);
    const raw: ?*anyopaque = calloc(count, @sizeOf(T));
    assert(raw != null);
    const typed: [*]T = @ptrCast(@alignCast(raw.?));
    return typed[0..count];
}

/// Free startup-allocated host memory during teardown.
pub fn free_host_slice(comptime T: type, slice: []T) void {
    assert(slice.len > 0);
    free(@ptrCast(slice.ptr));
}

// --- Zero-Allocation Fixed-Buffer Writer & Direct Win32 Stdout/Stderr ---

pub const ByteWriter = struct {
    buffer: []u8,
    length: usize,

    pub fn init(buffer: []u8) ByteWriter {
        assert(buffer.len > 0);
        return .{ .buffer = buffer, .length = 0 };
    }

    pub fn reset(self: *ByteWriter) void {
        assert(self.length <= self.buffer.len);
        self.length = 0;
    }

    pub fn slice(self: *const ByteWriter) []const u8 {
        assert(self.length <= self.buffer.len);
        return self.buffer[0..self.length];
    }

    pub fn remaining(self: *const ByteWriter) usize {
        assert(self.length <= self.buffer.len);
        return self.buffer.len - self.length;
    }

    pub fn append_slice(self: *ByteWriter, text: []const u8) void {
        assert(self.length <= self.buffer.len);
        assert(text.len <= self.buffer.len - self.length);
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
    }

    pub fn try_append_slice(self: *ByteWriter, text: []const u8) bool {
        assert(self.length <= self.buffer.len);
        if (text.len > self.buffer.len - self.length) return false;
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
        return true;
    }

    pub fn append_byte(self: *ByteWriter, byte: u8) void {
        assert(self.length < self.buffer.len);
        self.buffer[self.length] = byte;
        self.length += 1;
    }

    pub fn print(self: *ByteWriter, comptime fmt: []const u8, args: anytype) void {
        const res = std.fmt.bufPrint(self.buffer[self.length..], fmt, args) catch {
            self.length = self.buffer.len;
            return;
        };
        self.length += res.len;
    }

    pub fn try_append_byte(self: *ByteWriter, byte: u8) bool {
        assert(self.length <= self.buffer.len);
        if (self.length >= self.buffer.len) return false;
        self.buffer[self.length] = byte;
        self.length += 1;
        return true;
    }

    pub fn append_fmt(self: *ByteWriter, comptime fmt: []const u8, args: anytype) void {
        assert(self.length <= self.buffer.len);
        const written = std.fmt.bufPrint(self.buffer[self.length..], fmt, args) catch {
            @panic("ByteWriter.append_fmt buffer overflow");
        };
        self.length += written.len;
        assert(self.length <= self.buffer.len);
    }

    pub fn append_json_escaped(self: *ByteWriter, text: []const u8) void {
        assert(self.length <= self.buffer.len);
        self.append_byte('"');
        const hex_digits = "0123456789abcdef";
        for (text) |byte| {
            switch (byte) {
                '"' => self.append_slice("\\\""),
                '\\' => self.append_slice("\\\\"),
                '\n' => self.append_slice("\\n"),
                '\r' => self.append_slice("\\r"),
                '\t' => self.append_slice("\\t"),
                0x08 => self.append_slice("\\b"),
                0x0c => self.append_slice("\\f"),
                else => {
                    if (byte < 0x20) {
                        var esc: [6]u8 = .{ '\\', 'u', '0', '0', hex_digits[(byte >> 4) & 0x0f], hex_digits[byte & 0x0f] };
                        self.append_slice(&esc);
                    } else {
                        self.append_byte(byte);
                    }
                },
            }
        }
        self.append_byte('"');
    }

    pub fn append_c_string_terminated(self: *ByteWriter, text: []const u8) [*:0]const u8 {
        const start = self.length;
        self.append_slice(text);
        self.append_byte(0);
        return @ptrCast(self.buffer[start..self.length].ptr);
    }
};

pub fn write_handle_all(handle: HANDLE, bytes: []const u8) void {
    if (handle == null or handle == INVALID_HANDLE_VALUE) return;
    var sent: usize = 0;
    const iterations_max: usize = bytes.len + 64;
    var iteration: usize = 0;
    while (sent < bytes.len and iteration < iterations_max) : (iteration += 1) {
        const chunk: u32 = @intCast(@min(bytes.len - sent, 1024 * 1024));
        var written: u32 = 0;
        const ok = WriteFile(handle, bytes[sent..].ptr, chunk, &written, null);
        if (ok == 0 or written == 0) return;
        sent += written;
    }
}

pub fn print_stdout(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [8192]u8 = undefined;
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch stack_buffer[0..];
    write_handle_all(GetStdHandle(STD_OUTPUT_HANDLE), text);
}

pub fn print_stderr(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [4096]u8 = undefined;
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch stack_buffer[0..];
    write_handle_all(GetStdHandle(STD_ERROR_HANDLE), text);
}

/// Build a null-terminated path `<directory>/<suffix>` inside caller stack buffer.
pub fn format_path_z(buffer: []u8, directory: []const u8, suffix: []const u8) [*:0]const u8 {
    assert(buffer.len > directory.len + suffix.len + 1);
    assert(directory.len > 0);
    @memcpy(buffer[0..directory.len], directory);
    @memcpy(buffer[directory.len .. directory.len + suffix.len], suffix);
    buffer[directory.len + suffix.len] = 0;
    return @ptrCast(buffer[0 .. directory.len + suffix.len + 1].ptr);
}
