// TigerStyle Multi-Slab Vulkan BDA VRAM Arena.
// Respects AMD RDNA 3's 2 GiB maxMemoryAllocationSize while minimizing slack via
// best-fit slab scanning and adaptive tail slab sizing (zero wasted VRAM!).

const std = @import("std");
const constants = @import("constants.zig");
const tiger = @import("tiger.zig");
const vk = @import("vulkan.zig");

pub const max_slabs: usize = 128;
pub const default_slab_bytes: u64 = 1024 * 1024 * 1024; // 1.0 GiB slabs (fits 2x w13 + 2x w2 cleanly!)

pub const Allocation = struct {
    addr: vk.VkDeviceAddress = 0,
    slab_idx: u32 = 0,
    offset: u64 = 0,
    size: u64 = 0,
    mapped: ?[*]u8 = null,
};

pub const VRAMArena = struct {
    ctx: ?*const vk.Context = null,
    slabs: [max_slabs]vk.GpuBuffer = @splat(.{}),
    slab_used: [max_slabs]u64 = @splat(0),
    slab_is_fallback: [max_slabs]bool = @splat(false),
    slab_count: u32 = 0,
    slab_size_max: u64 = default_slab_bytes,
    mem_type_idx: u32 = 0,
    fallback_mem_type_idx: u32 = std.math.maxInt(u32),
    map_host: bool = false,
    total_allocated_bytes: u64 = 0,
    total_reserved_bytes: u64 = 0,
    total_fallback_bytes: u64 = 0,
    total_donated_bytes: u64 = 0,
    donated: [64]struct { buffer: vk.VkBuffer, offset: u64, size: u64 } = undefined,
    donated_count: u32 = 0,
    allocation_count: u32 = 0,
    /// Optional arena whose VRAM slab tails are offered first (best-fit) before this arena
    /// reserves new slabs. The donor must outlive every allocation made from it.
    donor: ?*VRAMArena = null,

    pub fn init(ctx: *const vk.Context, mem_type_idx: u32, map_host: bool) VRAMArena {
        return .{
            .ctx = ctx,
            .mem_type_idx = mem_type_idx,
            .map_host = map_host,
        };
    }

    pub fn init_exact_slab(ctx: *const vk.Context, total_bytes: u64, mem_type_idx: u32, map_host: bool) VRAMArena {
        var arena = init(ctx, mem_type_idx, map_host);
        if (total_bytes > 0 and total_bytes <= 1792 * 1024 * 1024) {
            arena.add_slab(tiger.align_up_u64(total_bytes, constants.vram_alignment_bytes));
        }
        return arena;
    }

    fn add_slab(self: *VRAMArena, min_bytes: u64) void {
        if (!self.try_add_vram_slab(min_bytes)) self.add_fallback_slab(min_bytes);
    }

    fn try_add_vram_slab(self: *VRAMArena, min_bytes: u64) bool {
        tiger.assert(self.ctx != null);
        tiger.assert(self.slab_count < max_slabs);
        const ctx = self.ctx.?;
        const idx = self.slab_count;

        // Keep tail slabs tight near 13 GiB so we never waste 300+ MiB of VRAM slack!
        const pref: u64 = if (self.total_reserved_bytes >= 12 * 1024 * 1024 * 1024)
            @max(min_bytes, 256 * 1024 * 1024)
        else
            @max(self.slab_size_max, min_bytes);
        const min_vram_slab: u64 = if (self.fallback_mem_type_idx != std.math.maxInt(u32))
            @max(min_bytes, 16 * 1024 * 1024)
        else
            min_bytes;
        const candidates = [_]u64{
            pref,
            @max(512 * 1024 * 1024, min_vram_slab),
            @max(256 * 1024 * 1024, min_vram_slab),
            @max(128 * 1024 * 1024, min_vram_slab),
            @max(64 * 1024 * 1024, min_vram_slab),
            min_vram_slab,
        };
        for (candidates) |sz| {
            if (sz > 1792 * 1024 * 1024) continue;
            if (ctx.try_create_buffer(sz, self.mem_type_idx, self.map_host)) |buf| {
                self.slabs[idx] = buf;
                self.slab_used[idx] = 0;
                self.slab_is_fallback[idx] = false;
                self.slab_count += 1;
                self.total_reserved_bytes += sz;
                return true;
            }
        }
        return false;
    }

    fn add_fallback_slab(self: *VRAMArena, min_bytes: u64) void {
        tiger.assert(self.slab_count < max_slabs);
        const ctx = self.ctx.?;
        const idx = self.slab_count;
        // Optional graceful fallback to zero-copy pinned DDR5 (tries both host_wc and host_cached!)
        if (self.fallback_mem_type_idx != std.math.maxInt(u32)) {
            const alt_mt = if (self.fallback_mem_type_idx == ctx.mem_type_host_wc) ctx.mem_type_host_cached else ctx.mem_type_host_wc;
            const mts = [_]u32{ self.fallback_mem_type_idx, alt_mt };
            const fb_cands = [_]u64{
                @max(min_bytes, 256 * 1024 * 1024),
                @max(min_bytes, 64 * 1024 * 1024),
                min_bytes,
            };
            for (mts) |fmt| {
                if (fmt == std.math.maxInt(u32)) continue;
                for (fb_cands) |fsz| {
                    if (ctx.try_create_buffer(fsz, fmt, false)) |buf| {
                        self.slabs[idx] = buf;
                        self.slab_used[idx] = 0;
                        self.slab_is_fallback[idx] = true;
                        self.slab_count += 1;
                        self.total_fallback_bytes += fsz;
                        return;
                    }
                }
            }
        }
        tiger.print_stderr("[arena] OOM allocating slab (min={d:.1} MiB, already reserved={d:.2} GiB, used={d:.2} GiB)\n", .{
            @as(f64, @floatFromInt(min_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.total_reserved_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.total_allocated_bytes)) / (1024.0 * 1024.0 * 1024.0),
        });
        @panic("VRAMArena OOM");
    }

    /// Allocate `bytes` aligned to 256 bytes using best-fit across existing slabs,
    /// returning a contiguous 64-bit GPU BDA address.
    pub fn alloc(self: *VRAMArena, bytes: u64) Allocation {
        tiger.assert(self.ctx != null);
        tiger.assert(bytes > 0);
        const aligned = tiger.align_up_u64(bytes, constants.vram_alignment_bytes);
        tiger.assert(aligned <= 1792 * 1024 * 1024);

        if (self.donor) |d| {
            if (d.best_fit(aligned, false)) |di| {
                const off = d.slab_used[di];
                d.slab_used[di] += aligned;
                d.total_allocated_bytes += aligned;
                self.total_donated_bytes += aligned;
                if (self.donated_count < self.donated.len) {
                    self.donated[self.donated_count] = .{ .buffer = d.slabs[di].buffer, .offset = off, .size = aligned };
                    self.donated_count += 1;
                }
                self.allocation_count += 1;
                const slab = &d.slabs[di];
                return .{
                    .addr = slab.address + off,
                    .slab_idx = std.math.maxInt(u32), // not one of OUR slabs
                    .offset = off,
                    .size = aligned,
                    .mapped = if (slab.mapped) |p| p + @as(usize, @intCast(off)) else null,
                };
            }
        }
        // Best-fit across VRAM slabs first; then try a NEW VRAM slab; only then DDR5 fallback slabs.
        // (A fallback slab must never attract allocations that could still live in VRAM.)
        var chosen: ?u32 = self.best_fit(aligned, false);
        if (chosen == null) {
            if (self.try_add_vram_slab(aligned)) {
                chosen = self.slab_count - 1;
            } else {
                chosen = self.best_fit(aligned, true);
                if (chosen == null) {
                    self.add_fallback_slab(aligned);
                    chosen = self.slab_count - 1;
                }
            }
        }

        const si = chosen.?;
        const off = self.slab_used[si];
        self.slab_used[si] += aligned;
        self.total_allocated_bytes += aligned;
        self.allocation_count += 1;

        const slab = &self.slabs[si];
        return .{
            .addr = slab.address + off,
            .slab_idx = si,
            .offset = off,
            .size = aligned,
            .mapped = if (slab.mapped) |p| p + @as(usize, @intCast(off)) else null,
        };
    }

    fn best_fit(self: *const VRAMArena, aligned: u64, fallback: bool) ?u32 {
        var chosen: ?u32 = null;
        var best_rem: u64 = std.math.maxInt(u64);
        var i: u32 = 0;
        while (i < self.slab_count) : (i += 1) {
            if (self.slab_is_fallback[i] != fallback) continue;
            const avail = self.slabs[i].size - self.slab_used[i];
            if (avail >= aligned and avail < best_rem) {
                best_rem = avail;
                chosen = i;
            }
        }
        return chosen;
    }

    pub fn alloc_addr(self: *VRAMArena, bytes: u64) vk.VkDeviceAddress {
        return self.alloc(bytes).addr;
    }

    pub fn upload_bytes(self: *VRAMArena, src: []const u8, stage: *vk.GpuBuffer) Allocation {
        tiger.assert(src.len > 0);
        const a = self.alloc(src.len);
        if (a.mapped) |dst| {
            @memcpy(dst[0..src.len], src);
        } else {
            self.copy_via_stage(a, src, stage);
        }
        return a;
    }

    pub fn copy_via_stage(self: *const VRAMArena, dst: Allocation, src: []const u8, stage: *vk.GpuBuffer) void {
        const ctx = self.ctx.?;
        tiger.assert(stage.mapped != null and stage.size > 0);
        var sent: usize = 0;
        while (sent < src.len) {
            const chunk: usize = @intCast(@min(@as(u64, src.len - sent), stage.size));
            @memcpy(stage.mapped.?[0..chunk], src[sent .. sent + chunk]);

            const begin_info: vk.VkCommandBufferBeginInfo = .{
                .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            tiger.assert_vk(ctx.vkResetCommandBuffer(ctx.imm_cmd, 0));
            tiger.assert_vk(ctx.vkBeginCommandBuffer(ctx.imm_cmd, &begin_info));
            const region = [_]vk.VkBufferCopy{.{
                .srcOffset = 0,
                .dstOffset = dst.offset + sent,
                .size = chunk,
            }};
            ctx.vkCmdCopyBuffer(ctx.imm_cmd, stage.buffer, self.slabs[dst.slab_idx].buffer, 1, &region);
            tiger.assert_vk(ctx.vkEndCommandBuffer(ctx.imm_cmd));
            ctx.submit_and_wait(ctx.imm_cmd);
            sent += chunk;
        }
    }

    pub fn deinit(self: *VRAMArena) void {
        if (self.ctx) |ctx| {
            var i: u32 = 0;
            while (i < self.slab_count) : (i += 1) {
                ctx.destroy_buffer(&self.slabs[i]);
            }
        }
        self.* = .{};
    }
};
