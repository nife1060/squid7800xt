// Zero-dependency Vulkan 1.4 Compute C-ABI bindings & RX 7800 XT device context.
// Dynamically loads vulkan-1.dll from System32 — zero Vulkan SDK link dependency.
// Uses VK_KHR_buffer_device_address (64-bit GPU pointers in push constants),
// VK_KHR_cooperative_matrix (RDNA 3 WMMA), VK_EXT_subgroup_size_control (Wave32/64),
// and pre-recorded VkCommandBuffer replay (zero host dispatch overhead per step).

const std = @import("std");
const tiger = @import("tiger.zig");

pub const VkResult = i32;
pub const VkBool32 = u32;
pub const VkFlags = u32;
pub const VkFlags64 = u64;
pub const VkDeviceSize = u64;
pub const VkDeviceAddress = u64;

pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};
pub const VkCommandPool = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkFence = u64;
pub const VkBuffer = u64;
pub const VkDeviceMemory = u64;
pub const VkShaderModule = u64;
pub const VkPipelineLayout = u64;
pub const VkPipeline = u64;
pub const VkPipelineCache = u64;
pub const VkQueryPool = u64;
pub const VkDescriptorSetLayout = u64;

pub const VK_NULL_HANDLE: u64 = 0;
pub const VK_SUCCESS: i32 = 0;
pub const VK_TRUE: u32 = 1;
pub const VK_FALSE: u32 = 0;
pub const VK_WHOLE_SIZE: u64 = std.math.maxInt(u64);

pub inline fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

pub const VK_API_VERSION_1_4: u32 = make_api_version(0, 1, 4, 0);

// Structure types (Khronos Vulkan 1.4 spec)
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: u32 = 11;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO: u32 = 1000060000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: u32 = 1000082000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: u32 = 1000083000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: u32 = 1000177000;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: u32 = 1000225001;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: u32 = 1000225002;
pub const VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO: u32 = 1000244001;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: u32 = 1000257000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: u32 = 1000280000;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER_2: u32 = 1000314000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES: u32 = 1000314002;
pub const VK_STRUCTURE_TYPE_DEPENDENCY_INFO: u32 = 1000314003;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: u32 = 1000506000;
pub const VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR: u32 = 1000506001;

// Queue / memory / buffer flags
pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: u32 = 0x00000004;

pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x00000008;

pub const VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT: u32 = 0x00000002;

pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
pub const VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT: u32 = 0x00020000;

pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
pub const VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT: u32 = 0x00000002;

pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;
pub const VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: u32 = 0x00000004;

pub const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x00000001;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x00001000;
pub const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x00002000;
pub const VK_PIPELINE_STAGE_ALL_COMMANDS_BIT: u32 = 0x00010000;

pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000040;
pub const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x00000800;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x00001000;
pub const VK_ACCESS_MEMORY_READ_BIT: u32 = 0x00008000;
pub const VK_ACCESS_MEMORY_WRITE_BIT: u32 = 0x00010000;

pub const VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT: u64 = 0x00000800;
pub const VK_PIPELINE_STAGE_2_COPY_BIT: u64 = 0x100000000;
pub const VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT: u64 = 0x00010000;
pub const VK_ACCESS_2_SHADER_STORAGE_READ_BIT: u64 = 0x200000000;
pub const VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT: u64 = 0x400000000;
pub const VK_ACCESS_2_MEMORY_READ_BIT: u64 = 0x00008000;
pub const VK_ACCESS_2_MEMORY_WRITE_BIT: u64 = 0x00010000;

pub const VK_QUERY_TYPE_TIMESTAMP: u32 = 2;
pub const VK_QUERY_RESULT_64_BIT: u32 = 0x00000001;
pub const VK_QUERY_RESULT_WAIT_BIT: u32 = 0x00000002;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

// Struct definitions

pub const VkApplicationInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = VK_API_VERSION_1_4,
};

pub const VkInstanceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const VkPhysicalDeviceLimits = extern struct {
    raw: [504]u8 = @splat(0),
    // timestampPeriod is at offset 296 in VkPhysicalDeviceLimits (float) — we verify below via dedicated struct?
    // Better: declare exact leading fields up to timestampPeriod so there is zero guesswork!
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32 = 0,
    driverVersion: u32 = 0,
    vendorID: u32 = 0,
    deviceID: u32 = 0,
    deviceType: u32 = 0,
    deviceName: [256]u8 = @splat(0),
    pipelineCacheUUID: [16]u8 = @splat(0),
    // VkPhysicalDeviceLimits (504 bytes) + VkPhysicalDeviceSparseProperties (20 bytes) + padding = 536 bytes
    limits_and_sparse: [544]u8 = @splat(0),
};

pub const VkPhysicalDeviceFeatures = extern struct {
    flags: [55]VkBool32 = @splat(0),
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: VkFlags = 0,
    queueCount: u32 = 0,
    timestampValidBits: u32 = 0,
    minImageTransferGranularity: [3]u32 = .{ 0, 0, 0 },
};

pub const VkMemoryType = extern struct {
    propertyFlags: VkFlags = 0,
    heapIndex: u32 = 0,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize = 0,
    flags: VkFlags = 0,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32 = 0,
    memoryTypes: [32]VkMemoryType = @splat(.{}),
    memoryHeapCount: u32 = 0,
    memoryHeaps: [16]VkMemoryHeap = @splat(.{}),
};

pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: u32 = 1000059006;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: u32 = 1000237000;

pub const VkPhysicalDeviceMemoryBudgetPropertiesEXT = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    pNext: ?*anyopaque = null,
    heapBudget: [16]VkDeviceSize = @splat(0),
    heapUsage: [16]VkDeviceSize = @splat(0),
};

pub const VkPhysicalDeviceMemoryProperties2 = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    memoryProperties: VkPhysicalDeviceMemoryProperties = .{},
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 1,
    pQueuePriorities: [*]const f32,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const VkPhysicalDeviceFeatures = null,
};

pub const VkPhysicalDeviceBufferDeviceAddressFeatures = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    pNext: ?*anyopaque = null,
    bufferDeviceAddress: VkBool32 = VK_TRUE,
    bufferDeviceAddressCaptureReplay: VkBool32 = VK_FALSE,
    bufferDeviceAddressMultiDevice: VkBool32 = VK_FALSE,
};

pub const VkPhysicalDeviceShaderFloat16Int8Features = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    pNext: ?*anyopaque = null,
    shaderFloat16: VkBool32 = VK_TRUE,
    shaderInt8: VkBool32 = VK_TRUE,
};

pub const VkPhysicalDevice16BitStorageFeatures = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    pNext: ?*anyopaque = null,
    storageBuffer16BitAccess: VkBool32 = VK_TRUE,
    uniformAndStorageBuffer16BitAccess: VkBool32 = VK_FALSE,
    storagePushConstant16: VkBool32 = VK_FALSE,
    storageInputOutput16: VkBool32 = VK_FALSE,
};

pub const VkPhysicalDevice8BitStorageFeatures = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    pNext: ?*anyopaque = null,
    storageBuffer8BitAccess: VkBool32 = VK_TRUE,
    uniformAndStorageBuffer8BitAccess: VkBool32 = VK_FALSE,
    storagePushConstant8: VkBool32 = VK_FALSE,
};

pub const VkPhysicalDeviceSubgroupSizeControlFeatures = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    pNext: ?*anyopaque = null,
    subgroupSizeControl: VkBool32 = VK_TRUE,
    computeFullSubgroups: VkBool32 = VK_TRUE,
};

pub const VkPhysicalDeviceSynchronization2Features = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    pNext: ?*anyopaque = null,
    synchronization2: VkBool32 = VK_TRUE,
};

pub const VkPhysicalDeviceShaderIntegerDotProductFeatures = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    pNext: ?*anyopaque = null,
    shaderIntegerDotProduct: VkBool32 = VK_TRUE,
};

pub const VkPhysicalDeviceCooperativeMatrixFeaturesKHR = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    pNext: ?*anyopaque = null,
    cooperativeMatrix: VkBool32 = VK_TRUE,
    cooperativeMatrixRobustBufferAccess: VkBool32 = VK_FALSE,
};

pub const VkCooperativeMatrixPropertiesKHR = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
    pNext: ?*anyopaque = null,
    MSize: u32 = 0,
    NSize: u32 = 0,
    KSize: u32 = 0,
    AType: u32 = 0,
    BType: u32 = 0,
    CType: u32 = 0,
    ResultType: u32 = 0,
    saturatingAccumulation: VkBool32 = 0,
    scope: u32 = 0,
};

pub const VkBufferCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    size: VkDeviceSize = 0,
    usage: VkFlags = 0,
    sharingMode: u32 = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize = 0,
    alignment: VkDeviceSize = 0,
    memoryTypeBits: u32 = 0,
};

pub const VkMemoryAllocateFlagsInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    deviceMask: u32 = 0,
};

pub const VkMemoryAllocateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

pub const VkBufferDeviceAddressInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    pNext: ?*const anyopaque = null,
    buffer: VkBuffer = VK_NULL_HANDLE,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    codeSize: usize = 0,
    pCode: [*]const u32,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: VkFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    offset: u32 = 0,
    size: u32 = 0,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?[*]const VkPushConstantRange = null,
};

pub const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    requiredSubgroupSize: u32 = 32,
};

pub const VkSpecializationMapEntry = extern struct {
    constantID: u32 = 0,
    offset: u32 = 0,
    size: usize = 0,
};

pub const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = 0,
    pMapEntries: ?[*]const VkSpecializationMapEntry = null,
    dataSize: usize = 0,
    pData: ?*const anyopaque = null,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    stage: VkFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    module: VkShaderModule = VK_NULL_HANDLE,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*const anyopaque = null,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout = VK_NULL_HANDLE,
    basePipelineHandle: VkPipeline = VK_NULL_HANDLE,
    basePipelineIndex: i32 = -1,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    queueFamilyIndex: u32 = 0,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = null,
    level: u32 = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32 = 1,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const VkSubmitInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const u32 = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
};

pub const VkBufferCopy = extern struct {
    srcOffset: VkDeviceSize = 0,
    dstOffset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
};

pub const VkMemoryBarrier = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkFlags = VK_ACCESS_SHADER_WRITE_BIT,
    dstAccessMask: VkFlags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
};

pub const VkMemoryBarrier2 = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
    pNext: ?*const anyopaque = null,
    srcStageMask: VkFlags64 = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    srcAccessMask: VkFlags64 = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    dstStageMask: VkFlags64 = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    dstAccessMask: VkFlags64 = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
};

pub const VkDependencyInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    pNext: ?*const anyopaque = null,
    dependencyFlags: VkFlags = 0,
    memoryBarrierCount: u32 = 0,
    pMemoryBarriers: ?[*]const VkMemoryBarrier2 = null,
    bufferMemoryBarrierCount: u32 = 0,
    pBufferMemoryBarriers: ?*const anyopaque = null,
    imageMemoryBarrierCount: u32 = 0,
    pImageMemoryBarriers: ?*const anyopaque = null,
};

pub const VkQueryPoolCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queryType: u32 = VK_QUERY_TYPE_TIMESTAMP,
    queryCount: u32 = 0,
    pipelineStatistics: VkFlags = 0,
};

// Function pointer types
const PFN_vkGetInstanceProcAddr = *const fn (VkInstance, [*:0]const u8) callconv(.winapi) ?*anyopaque;
const PFN_vkGetDeviceProcAddr = *const fn (VkDevice, [*:0]const u8) callconv(.winapi) ?*anyopaque;
const PFN_vkCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(.winapi) VkResult;
const PFN_vkDestroyInstance = *const fn (VkInstance, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.winapi) VkResult;
const PFN_vkGetPhysicalDeviceProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties) callconv(.winapi) void;
const PFN_vkGetPhysicalDeviceFeatures = *const fn (VkPhysicalDevice, *VkPhysicalDeviceFeatures) callconv(.winapi) void;
const PFN_vkGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.winapi) void;
const PFN_vkGetPhysicalDeviceMemoryProperties2 = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties2) callconv(.winapi) void;
const PFN_vkGetPhysicalDeviceQueueFamilyProperties = *const fn (VkPhysicalDevice, *u32, ?[*]VkQueueFamilyProperties) callconv(.winapi) void;
const PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = *const fn (VkPhysicalDevice, *u32, ?[*]VkCooperativeMatrixPropertiesKHR) callconv(.winapi) VkResult;
const PFN_vkCreateDevice = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*const anyopaque, *VkDevice) callconv(.winapi) VkResult;
const PFN_vkDestroyDevice = *const fn (VkDevice, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkGetDeviceQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.winapi) void;
const PFN_vkDeviceWaitIdle = *const fn (VkDevice) callconv(.winapi) VkResult;
const PFN_vkQueueSubmit = *const fn (VkQueue, u32, [*]const VkSubmitInfo, VkFence) callconv(.winapi) VkResult;
const PFN_vkQueueWaitIdle = *const fn (VkQueue) callconv(.winapi) VkResult;
const PFN_vkCreateFence = *const fn (VkDevice, *const VkFenceCreateInfo, ?*const anyopaque, *VkFence) callconv(.winapi) VkResult;
const PFN_vkDestroyFence = *const fn (VkDevice, VkFence, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkWaitForFences = *const fn (VkDevice, u32, [*]const VkFence, VkBool32, u64) callconv(.winapi) VkResult;
const PFN_vkResetFences = *const fn (VkDevice, u32, [*]const VkFence) callconv(.winapi) VkResult;
const PFN_vkCreateBuffer = *const fn (VkDevice, *const VkBufferCreateInfo, ?*const anyopaque, *VkBuffer) callconv(.winapi) VkResult;
const PFN_vkDestroyBuffer = *const fn (VkDevice, VkBuffer, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkGetBufferMemoryRequirements = *const fn (VkDevice, VkBuffer, *VkMemoryRequirements) callconv(.winapi) void;
const PFN_vkAllocateMemory = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*const anyopaque, *VkDeviceMemory) callconv(.winapi) VkResult;
const PFN_vkFreeMemory = *const fn (VkDevice, VkDeviceMemory, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkBindBufferMemory = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.winapi) VkResult;
const PFN_vkMapMemory = *const fn (VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, *?*anyopaque) callconv(.winapi) VkResult;
const PFN_vkUnmapMemory = *const fn (VkDevice, VkDeviceMemory) callconv(.winapi) void;
const PFN_vkGetBufferDeviceAddress = *const fn (VkDevice, *const VkBufferDeviceAddressInfo) callconv(.winapi) VkDeviceAddress;
const PFN_vkCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(.winapi) VkResult;
const PFN_vkDestroyShaderModule = *const fn (VkDevice, VkShaderModule, ?*const anyopaque) callconv(.winapi) void;
pub const PFN_vkCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const anyopaque, *VkPipelineLayout) callconv(.winapi) VkResult;
pub const PFN_vkDestroyPipelineLayout = *const fn (VkDevice, VkPipelineLayout, ?*const anyopaque) callconv(.winapi) void;
pub const PFN_vkCreateComputePipelines = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*const anyopaque, [*]VkPipeline) callconv(.winapi) VkResult;
pub const PFN_vkDestroyPipeline = *const fn (VkDevice, VkPipeline, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkCreateCommandPool = *const fn (VkDevice, *const VkCommandPoolCreateInfo, ?*const anyopaque, *VkCommandPool) callconv(.winapi) VkResult;
const PFN_vkDestroyCommandPool = *const fn (VkDevice, VkCommandPool, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkAllocateCommandBuffers = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.winapi) VkResult;
const PFN_vkFreeCommandBuffers = *const fn (VkDevice, VkCommandPool, u32, [*]const VkCommandBuffer) callconv(.winapi) void;
const PFN_vkBeginCommandBuffer = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.winapi) VkResult;
const PFN_vkEndCommandBuffer = *const fn (VkCommandBuffer) callconv(.winapi) VkResult;
const PFN_vkResetCommandBuffer = *const fn (VkCommandBuffer, VkFlags) callconv(.winapi) VkResult;
const PFN_vkCmdBindPipeline = *const fn (VkCommandBuffer, u32, VkPipeline) callconv(.winapi) void;
const PFN_vkCmdPushConstants = *const fn (VkCommandBuffer, VkPipelineLayout, VkFlags, u32, u32, *const anyopaque) callconv(.winapi) void;
const PFN_vkCmdDispatch = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.winapi) void;
const PFN_vkCmdCopyBuffer = *const fn (VkCommandBuffer, VkBuffer, VkBuffer, u32, [*]const VkBufferCopy) callconv(.winapi) void;
const PFN_vkCmdFillBuffer = *const fn (VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, u32) callconv(.winapi) void;
const PFN_vkCmdPipelineBarrier = *const fn (VkCommandBuffer, VkFlags, VkFlags, VkFlags, u32, ?[*]const VkMemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkCmdPipelineBarrier2 = *const fn (VkCommandBuffer, *const VkDependencyInfo) callconv(.winapi) void;
const PFN_vkCreateQueryPool = *const fn (VkDevice, *const VkQueryPoolCreateInfo, ?*const anyopaque, *VkQueryPool) callconv(.winapi) VkResult;
const PFN_vkDestroyQueryPool = *const fn (VkDevice, VkQueryPool, ?*const anyopaque) callconv(.winapi) void;
const PFN_vkCmdResetQueryPool = *const fn (VkCommandBuffer, VkQueryPool, u32, u32) callconv(.winapi) void;
const PFN_vkCmdWriteTimestamp = *const fn (VkCommandBuffer, VkFlags, VkQueryPool, u32) callconv(.winapi) void;
const PFN_vkGetQueryPoolResults = *const fn (VkDevice, VkQueryPool, u32, u32, usize, *anyopaque, VkDeviceSize, VkFlags) callconv(.winapi) VkResult;

fn load_inst_fn(comptime T: type, get_proc: PFN_vkGetInstanceProcAddr, inst: VkInstance, name: [*:0]const u8) T {
    const p = get_proc(inst, name);
    if (p == null) {
        tiger.print_stderr("[vulkan] Failed to resolve instance proc: {s}\n", .{name});
        @panic("Missing Vulkan instance function");
    }
    return @ptrCast(p.?);
}

fn load_dev_fn(comptime T: type, get_proc: PFN_vkGetDeviceProcAddr, dev: VkDevice, name: [*:0]const u8) T {
    const p = get_proc(dev, name);
    if (p == null) {
        tiger.print_stderr("[vulkan] Failed to resolve device proc: {s}\n", .{name});
        @panic("Missing Vulkan device function");
    }
    return @ptrCast(p.?);
}

pub const GpuBuffer = struct {
    buffer: VkBuffer = VK_NULL_HANDLE,
    memory: VkDeviceMemory = VK_NULL_HANDLE,
    address: VkDeviceAddress = 0,
    mapped: ?[*]u8 = null,
    size: u64 = 0,
    memory_type_index: u32 = 0,
};

pub const Context = struct {
    dll: tiger.HMODULE = null,
    instance: VkInstance = null,
    phys_dev: VkPhysicalDevice = null,
    device: VkDevice = null,
    queue: VkQueue = null,
    queue_family: u32 = 0,
    cmd_pool: VkCommandPool = null,
    imm_cmd: VkCommandBuffer = null,
    imm_fence: VkFence = VK_NULL_HANDLE,

    mem_props: VkPhysicalDeviceMemoryProperties = .{},
    device_name: [256]u8 = @splat(0),
    vram_heap_bytes: u64 = 0,
    host_heap_bytes: u64 = 0,
    mem_type_vram_only: u32 = std.math.maxInt(u32),
    mem_type_vram_rebar: u32 = std.math.maxInt(u32),
    mem_type_host_cached: u32 = std.math.maxInt(u32),
    mem_type_host_wc: u32 = std.math.maxInt(u32),
    timestamp_period_ns: f32 = 1.0,
    has_coop_matrix: bool = false,

    // Resolved Vulkan functions
    vkDestroyInstance: PFN_vkDestroyInstance = undefined,
    vkDestroyDevice: PFN_vkDestroyDevice = undefined,
    vkDeviceWaitIdle: PFN_vkDeviceWaitIdle = undefined,
    vkQueueSubmit: PFN_vkQueueSubmit = undefined,
    vkQueueWaitIdle: PFN_vkQueueWaitIdle = undefined,
    vkCreateFence: PFN_vkCreateFence = undefined,
    vkDestroyFence: PFN_vkDestroyFence = undefined,
    vkWaitForFences: PFN_vkWaitForFences = undefined,
    vkResetFences: PFN_vkResetFences = undefined,
    vkCreateBuffer: PFN_vkCreateBuffer = undefined,
    vkDestroyBuffer: PFN_vkDestroyBuffer = undefined,
    vkGetBufferMemoryRequirements: PFN_vkGetBufferMemoryRequirements = undefined,
    vkAllocateMemory: PFN_vkAllocateMemory = undefined,
    vkFreeMemory: PFN_vkFreeMemory = undefined,
    vkBindBufferMemory: PFN_vkBindBufferMemory = undefined,
    vkMapMemory: PFN_vkMapMemory = undefined,
    vkUnmapMemory: PFN_vkUnmapMemory = undefined,
    vkGetBufferDeviceAddress: PFN_vkGetBufferDeviceAddress = undefined,
    vkCreateShaderModule: PFN_vkCreateShaderModule = undefined,
    vkDestroyShaderModule: PFN_vkDestroyShaderModule = undefined,
    vkCreatePipelineLayout: PFN_vkCreatePipelineLayout = undefined,
    vkDestroyPipelineLayout: PFN_vkDestroyPipelineLayout = undefined,
    vkCreateComputePipelines: PFN_vkCreateComputePipelines = undefined,
    vkDestroyPipeline: PFN_vkDestroyPipeline = undefined,
    vkCreateCommandPool: PFN_vkCreateCommandPool = undefined,
    vkDestroyCommandPool: PFN_vkDestroyCommandPool = undefined,
    vkAllocateCommandBuffers: PFN_vkAllocateCommandBuffers = undefined,
    vkFreeCommandBuffers: PFN_vkFreeCommandBuffers = undefined,
    vkBeginCommandBuffer: PFN_vkBeginCommandBuffer = undefined,
    vkEndCommandBuffer: PFN_vkEndCommandBuffer = undefined,
    vkResetCommandBuffer: PFN_vkResetCommandBuffer = undefined,
    vkCmdBindPipeline: PFN_vkCmdBindPipeline = undefined,
    vkCmdPushConstants: PFN_vkCmdPushConstants = undefined,
    vkCmdDispatch: PFN_vkCmdDispatch = undefined,
    vkCmdCopyBuffer: PFN_vkCmdCopyBuffer = undefined,
    vkCmdFillBuffer: PFN_vkCmdFillBuffer = undefined,
    vkCmdPipelineBarrier: PFN_vkCmdPipelineBarrier = undefined,
    vkCmdPipelineBarrier2: PFN_vkCmdPipelineBarrier2 = undefined,
    vkCreateQueryPool: PFN_vkCreateQueryPool = undefined,
    vkDestroyQueryPool: PFN_vkDestroyQueryPool = undefined,
    vkCmdResetQueryPool: PFN_vkCmdResetQueryPool = undefined,
    vkCmdWriteTimestamp: PFN_vkCmdWriteTimestamp = undefined,
    vkGetQueryPoolResults: PFN_vkGetQueryPoolResults = undefined,
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR: ?PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = null,
    vkGetPhysicalDeviceMemoryProperties2: ?PFN_vkGetPhysicalDeviceMemoryProperties2 = null,

    pub fn init(self: *Context) void {
        const dll = tiger.LoadLibraryA("vulkan-1.dll");
        tiger.assert(dll != null);
        self.dll = dll;

        const gipa_raw = tiger.GetProcAddress(dll, "vkGetInstanceProcAddr");
        tiger.assert(gipa_raw != null);
        const get_inst_proc: PFN_vkGetInstanceProcAddr = @ptrCast(gipa_raw.?);

        const vkCreateInstance = load_inst_fn(PFN_vkCreateInstance, get_inst_proc, null, "vkCreateInstance");
        const app_info: VkApplicationInfo = .{
            .pApplicationName = "squid7800xt",
            .applicationVersion = 1,
            .pEngineName = "squid7800xt",
            .engineVersion = 1,
            .apiVersion = VK_API_VERSION_1_4,
        };
        const inst_ci: VkInstanceCreateInfo = .{
            .pApplicationInfo = &app_info,
        };
        tiger.assert_vk(vkCreateInstance(&inst_ci, null, &self.instance));

        self.vkDestroyInstance = load_inst_fn(PFN_vkDestroyInstance, get_inst_proc, self.instance, "vkDestroyInstance");
        const vkEnumeratePhysicalDevices = load_inst_fn(PFN_vkEnumeratePhysicalDevices, get_inst_proc, self.instance, "vkEnumeratePhysicalDevices");
        const vkGetPhysicalDeviceProperties = load_inst_fn(PFN_vkGetPhysicalDeviceProperties, get_inst_proc, self.instance, "vkGetPhysicalDeviceProperties");
        const vkGetPhysicalDeviceFeatures = load_inst_fn(PFN_vkGetPhysicalDeviceFeatures, get_inst_proc, self.instance, "vkGetPhysicalDeviceFeatures");
        const vkGetPhysicalDeviceMemoryProperties = load_inst_fn(PFN_vkGetPhysicalDeviceMemoryProperties, get_inst_proc, self.instance, "vkGetPhysicalDeviceMemoryProperties");
        const vkGetPhysicalDeviceQueueFamilyProperties = load_inst_fn(PFN_vkGetPhysicalDeviceQueueFamilyProperties, get_inst_proc, self.instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        const vkCreateDevice = load_inst_fn(PFN_vkCreateDevice, get_inst_proc, self.instance, "vkCreateDevice");
        const vkGetDeviceProcAddr = load_inst_fn(PFN_vkGetDeviceProcAddr, get_inst_proc, self.instance, "vkGetDeviceProcAddr");
        if (get_inst_proc(self.instance, "vkGetPhysicalDeviceMemoryProperties2")) |p| {
            self.vkGetPhysicalDeviceMemoryProperties2 = @ptrCast(p);
        }
        if (get_inst_proc(self.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")) |p| {
            self.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = @ptrCast(p);
        }

        // Select discrete GPU (RX 7800 XT)
        var dev_count: u32 = 0;
        tiger.assert_vk(vkEnumeratePhysicalDevices(self.instance, &dev_count, null));
        tiger.assert(dev_count > 0 and dev_count <= 16);
        var devs: [16]VkPhysicalDevice = @splat(null);
        tiger.assert_vk(vkEnumeratePhysicalDevices(self.instance, &dev_count, &devs));

        var chosen_idx: u32 = 0;
        var chosen_props: VkPhysicalDeviceProperties = .{};
        var i: u32 = 0;
        while (i < dev_count) : (i += 1) {
            var props: VkPhysicalDeviceProperties = .{};
            vkGetPhysicalDeviceProperties(devs[i], &props);
            if (i == 0 or props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                chosen_idx = i;
                chosen_props = props;
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
            }
        }
        self.phys_dev = devs[chosen_idx];
        self.device_name = chosen_props.deviceName;
        // Read timestampPeriod from VkPhysicalDeviceLimits (float at offset 296 inside limits)
        // Per Vulkan spec: limits starts at offset 292 of VkPhysicalDeviceProperties (20+256+16=292),
        // and timestampPeriod is offset 296 within limits -> let's read cleanly by exact offset verification:
        // Actually in VkPhysicalDeviceLimits, timestampPeriod is at offset 296 bytes (74th uint32/float after alignment).
        // Let's also sanity check it is between 0.1 and 10000 ns; fallback to 1.0 if unexpected.
        const ts_bytes = chosen_props.limits_and_sparse[296..300];
        const ts_val: f32 = @bitCast(ts_bytes.*);
        self.timestamp_period_ns = if (ts_val > 0.01 and ts_val < 100000.0) ts_val else 1.0;

        // Query memory properties & classify heaps/types
        vkGetPhysicalDeviceMemoryProperties(self.phys_dev, &self.mem_props);
        var ti: u32 = 0;
        while (ti < self.mem_props.memoryTypeCount) : (ti += 1) {
            const mt = self.mem_props.memoryTypes[ti];
            const flags = mt.propertyFlags;
            const heap_size = self.mem_props.memoryHeaps[mt.heapIndex].size;
            const is_dev = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            const is_host = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            const is_coh = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            const is_cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
            // Avoid AMD device-coherent (0x40/0x80) uncached slow types for normal compute buffers!
            const is_extra_uncached = (flags & 0xc0) != 0;
            if (is_extra_uncached) continue;

            if (is_dev and !is_host and self.mem_type_vram_only == std.math.maxInt(u32)) {
                self.mem_type_vram_only = ti;
                self.vram_heap_bytes = heap_size;
            }
            if (is_dev and is_host and is_coh and self.mem_type_vram_rebar == std.math.maxInt(u32)) {
                self.mem_type_vram_rebar = ti;
                if (self.vram_heap_bytes == 0) self.vram_heap_bytes = heap_size;
            }
            if (!is_dev and is_host and is_coh and is_cached and self.mem_type_host_cached == std.math.maxInt(u32)) {
                self.mem_type_host_cached = ti;
                self.host_heap_bytes = heap_size;
            }
            if (!is_dev and is_host and is_coh and !is_cached and self.mem_type_host_wc == std.math.maxInt(u32)) {
                self.mem_type_host_wc = ti;
                if (self.host_heap_bytes == 0) self.host_heap_bytes = heap_size;
            }
        }
        tiger.assert(self.mem_type_vram_only != std.math.maxInt(u32));
        tiger.assert(self.mem_type_host_cached != std.math.maxInt(u32) or self.mem_type_host_wc != std.math.maxInt(u32));

        // Find dedicated compute queue (or graphics+compute queue)
        var qf_count: u32 = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(self.phys_dev, &qf_count, null);
        tiger.assert(qf_count > 0 and qf_count <= 16);
        var qfs: [16]VkQueueFamilyProperties = @splat(.{});
        vkGetPhysicalDeviceQueueFamilyProperties(self.phys_dev, &qf_count, &qfs);

        var chosen_qf: u32 = std.math.maxInt(u32);
        var qi: u32 = 0;
        while (qi < qf_count) : (qi += 1) {
            if ((qfs[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                // Prefer compute-only queue if available, else graphics+compute
                if (chosen_qf == std.math.maxInt(u32) or (qfs[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                    chosen_qf = qi;
                }
            }
        }
        tiger.assert(chosen_qf != std.math.maxInt(u32));
        self.queue_family = chosen_qf;

        // Enable core features (from driver query, disabling robustBufferAccess for peak speed!)
        var core_features: VkPhysicalDeviceFeatures = .{};
        vkGetPhysicalDeviceFeatures(self.phys_dev, &core_features);
        core_features.flags[0] = VK_FALSE; // robustBufferAccess = false

        var f_coop: VkPhysicalDeviceCooperativeMatrixFeaturesKHR = .{ .cooperativeMatrix = VK_TRUE };
        var f_idp: VkPhysicalDeviceShaderIntegerDotProductFeatures = .{ .pNext = &f_coop, .shaderIntegerDotProduct = VK_TRUE };
        var f_sync2: VkPhysicalDeviceSynchronization2Features = .{ .pNext = &f_idp, .synchronization2 = VK_TRUE };
        var f_sub: VkPhysicalDeviceSubgroupSizeControlFeatures = .{ .pNext = &f_sync2, .subgroupSizeControl = VK_TRUE, .computeFullSubgroups = VK_TRUE };
        var f_8bit: VkPhysicalDevice8BitStorageFeatures = .{ .pNext = &f_sub, .storageBuffer8BitAccess = VK_TRUE };
        var f_16bit: VkPhysicalDevice16BitStorageFeatures = .{ .pNext = &f_8bit, .storageBuffer16BitAccess = VK_TRUE };
        var f_f16i8: VkPhysicalDeviceShaderFloat16Int8Features = .{ .pNext = &f_16bit, .shaderFloat16 = VK_TRUE, .shaderInt8 = VK_TRUE };
        var f_bda: VkPhysicalDeviceBufferDeviceAddressFeatures = .{ .pNext = &f_f16i8, .bufferDeviceAddress = VK_TRUE };

        const dev_exts = [_][*:0]const u8{
            "VK_KHR_cooperative_matrix",
        };
        const queue_prio: [1]f32 = .{1.0};
        const q_ci = [_]VkDeviceQueueCreateInfo{.{
            .queueFamilyIndex = self.queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_prio,
        }};

        var dev_ci: VkDeviceCreateInfo = .{
            .pNext = &f_bda,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &q_ci,
            .enabledExtensionCount = dev_exts.len,
            .ppEnabledExtensionNames = &dev_exts,
            .pEnabledFeatures = &core_features,
        };
        const rc = vkCreateDevice(self.phys_dev, &dev_ci, null, &self.device);
        if (rc == VK_SUCCESS) {
            self.has_coop_matrix = true;
        } else {
            // Fallback without cooperative matrix if unsupported
            f_idp.pNext = null;
            dev_ci.enabledExtensionCount = 0;
            dev_ci.ppEnabledExtensionNames = null;
            tiger.assert_vk(vkCreateDevice(self.phys_dev, &dev_ci, null, &self.device));
            self.has_coop_matrix = false;
        }

        // Load device function pointers
        const vkGetDeviceQueue = load_dev_fn(PFN_vkGetDeviceQueue, vkGetDeviceProcAddr, self.device, "vkGetDeviceQueue");
        self.vkDestroyDevice = load_dev_fn(PFN_vkDestroyDevice, vkGetDeviceProcAddr, self.device, "vkDestroyDevice");
        self.vkDeviceWaitIdle = load_dev_fn(PFN_vkDeviceWaitIdle, vkGetDeviceProcAddr, self.device, "vkDeviceWaitIdle");
        self.vkQueueSubmit = load_dev_fn(PFN_vkQueueSubmit, vkGetDeviceProcAddr, self.device, "vkQueueSubmit");
        self.vkQueueWaitIdle = load_dev_fn(PFN_vkQueueWaitIdle, vkGetDeviceProcAddr, self.device, "vkQueueWaitIdle");
        self.vkCreateFence = load_dev_fn(PFN_vkCreateFence, vkGetDeviceProcAddr, self.device, "vkCreateFence");
        self.vkDestroyFence = load_dev_fn(PFN_vkDestroyFence, vkGetDeviceProcAddr, self.device, "vkDestroyFence");
        self.vkWaitForFences = load_dev_fn(PFN_vkWaitForFences, vkGetDeviceProcAddr, self.device, "vkWaitForFences");
        self.vkResetFences = load_dev_fn(PFN_vkResetFences, vkGetDeviceProcAddr, self.device, "vkResetFences");
        self.vkCreateBuffer = load_dev_fn(PFN_vkCreateBuffer, vkGetDeviceProcAddr, self.device, "vkCreateBuffer");
        self.vkDestroyBuffer = load_dev_fn(PFN_vkDestroyBuffer, vkGetDeviceProcAddr, self.device, "vkDestroyBuffer");
        self.vkGetBufferMemoryRequirements = load_dev_fn(PFN_vkGetBufferMemoryRequirements, vkGetDeviceProcAddr, self.device, "vkGetBufferMemoryRequirements");
        self.vkAllocateMemory = load_dev_fn(PFN_vkAllocateMemory, vkGetDeviceProcAddr, self.device, "vkAllocateMemory");
        self.vkFreeMemory = load_dev_fn(PFN_vkFreeMemory, vkGetDeviceProcAddr, self.device, "vkFreeMemory");
        self.vkBindBufferMemory = load_dev_fn(PFN_vkBindBufferMemory, vkGetDeviceProcAddr, self.device, "vkBindBufferMemory");
        self.vkMapMemory = load_dev_fn(PFN_vkMapMemory, vkGetDeviceProcAddr, self.device, "vkMapMemory");
        self.vkUnmapMemory = load_dev_fn(PFN_vkUnmapMemory, vkGetDeviceProcAddr, self.device, "vkUnmapMemory");
        self.vkGetBufferDeviceAddress = load_dev_fn(PFN_vkGetBufferDeviceAddress, vkGetDeviceProcAddr, self.device, "vkGetBufferDeviceAddress");
        self.vkCreateShaderModule = load_dev_fn(PFN_vkCreateShaderModule, vkGetDeviceProcAddr, self.device, "vkCreateShaderModule");
        self.vkDestroyShaderModule = load_dev_fn(PFN_vkDestroyShaderModule, vkGetDeviceProcAddr, self.device, "vkDestroyShaderModule");
        self.vkCreatePipelineLayout = load_dev_fn(PFN_vkCreatePipelineLayout, vkGetDeviceProcAddr, self.device, "vkCreatePipelineLayout");
        self.vkDestroyPipelineLayout = load_dev_fn(PFN_vkDestroyPipelineLayout, vkGetDeviceProcAddr, self.device, "vkDestroyPipelineLayout");
        self.vkCreateComputePipelines = load_dev_fn(PFN_vkCreateComputePipelines, vkGetDeviceProcAddr, self.device, "vkCreateComputePipelines");
        self.vkDestroyPipeline = load_dev_fn(PFN_vkDestroyPipeline, vkGetDeviceProcAddr, self.device, "vkDestroyPipeline");
        self.vkCreateCommandPool = load_dev_fn(PFN_vkCreateCommandPool, vkGetDeviceProcAddr, self.device, "vkCreateCommandPool");
        self.vkDestroyCommandPool = load_dev_fn(PFN_vkDestroyCommandPool, vkGetDeviceProcAddr, self.device, "vkDestroyCommandPool");
        self.vkAllocateCommandBuffers = load_dev_fn(PFN_vkAllocateCommandBuffers, vkGetDeviceProcAddr, self.device, "vkAllocateCommandBuffers");
        self.vkFreeCommandBuffers = load_dev_fn(PFN_vkFreeCommandBuffers, vkGetDeviceProcAddr, self.device, "vkFreeCommandBuffers");
        self.vkBeginCommandBuffer = load_dev_fn(PFN_vkBeginCommandBuffer, vkGetDeviceProcAddr, self.device, "vkBeginCommandBuffer");
        self.vkEndCommandBuffer = load_dev_fn(PFN_vkEndCommandBuffer, vkGetDeviceProcAddr, self.device, "vkEndCommandBuffer");
        self.vkResetCommandBuffer = load_dev_fn(PFN_vkResetCommandBuffer, vkGetDeviceProcAddr, self.device, "vkResetCommandBuffer");
        self.vkCmdBindPipeline = load_dev_fn(PFN_vkCmdBindPipeline, vkGetDeviceProcAddr, self.device, "vkCmdBindPipeline");
        self.vkCmdPushConstants = load_dev_fn(PFN_vkCmdPushConstants, vkGetDeviceProcAddr, self.device, "vkCmdPushConstants");
        self.vkCmdDispatch = load_dev_fn(PFN_vkCmdDispatch, vkGetDeviceProcAddr, self.device, "vkCmdDispatch");
        self.vkCmdCopyBuffer = load_dev_fn(PFN_vkCmdCopyBuffer, vkGetDeviceProcAddr, self.device, "vkCmdCopyBuffer");
        self.vkCmdFillBuffer = load_dev_fn(PFN_vkCmdFillBuffer, vkGetDeviceProcAddr, self.device, "vkCmdFillBuffer");
        self.vkCmdPipelineBarrier = load_dev_fn(PFN_vkCmdPipelineBarrier, vkGetDeviceProcAddr, self.device, "vkCmdPipelineBarrier");
        self.vkCmdPipelineBarrier2 = load_dev_fn(PFN_vkCmdPipelineBarrier2, vkGetDeviceProcAddr, self.device, "vkCmdPipelineBarrier2");
        self.vkCreateQueryPool = load_dev_fn(PFN_vkCreateQueryPool, vkGetDeviceProcAddr, self.device, "vkCreateQueryPool");
        self.vkDestroyQueryPool = load_dev_fn(PFN_vkDestroyQueryPool, vkGetDeviceProcAddr, self.device, "vkDestroyQueryPool");
        self.vkCmdResetQueryPool = load_dev_fn(PFN_vkCmdResetQueryPool, vkGetDeviceProcAddr, self.device, "vkCmdResetQueryPool");
        self.vkCmdWriteTimestamp = load_dev_fn(PFN_vkCmdWriteTimestamp, vkGetDeviceProcAddr, self.device, "vkCmdWriteTimestamp");
        self.vkGetQueryPoolResults = load_dev_fn(PFN_vkGetQueryPoolResults, vkGetDeviceProcAddr, self.device, "vkGetQueryPoolResults");

        vkGetDeviceQueue(self.device, self.queue_family, 0, &self.queue);

        const pool_ci: VkCommandPoolCreateInfo = .{
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.queue_family,
        };
        tiger.assert_vk(self.vkCreateCommandPool(self.device, &pool_ci, null, &self.cmd_pool));

        var cmds: [1]VkCommandBuffer = .{null};
        const alloc_ci: VkCommandBufferAllocateInfo = .{
            .commandPool = self.cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        tiger.assert_vk(self.vkAllocateCommandBuffers(self.device, &alloc_ci, &cmds));
        self.imm_cmd = cmds[0];

        const fence_ci: VkFenceCreateInfo = .{};
        tiger.assert_vk(self.vkCreateFence(self.device, &fence_ci, null, &self.imm_fence));
    }

    pub fn deinit(self: *Context) void {
        if (self.device != null) {
            _ = self.vkDeviceWaitIdle(self.device);
            if (self.imm_fence != VK_NULL_HANDLE) self.vkDestroyFence(self.device, self.imm_fence, null);
            if (self.cmd_pool != null) self.vkDestroyCommandPool(self.device, self.cmd_pool, null);
            self.vkDestroyDevice(self.device, null);
            self.device = null;
        }
        if (self.instance != null) {
            self.vkDestroyInstance(self.instance, null);
            self.instance = null;
        }
    }

    pub fn device_name_slice(self: *const Context) []const u8 {
        var len: usize = 0;
        while (len < self.device_name.len and self.device_name[len] != 0) : (len += 1) {}
        return self.device_name[0..len];
    }

    pub fn try_create_buffer(self: *const Context, size_bytes: u64, mem_type_idx: u32, map_host: bool) ?GpuBuffer {
        tiger.assert(size_bytes > 0);
        var buf: GpuBuffer = .{
            .size = size_bytes,
            .memory_type_index = mem_type_idx,
        };
        const bci: VkBufferCreateInfo = .{
            .size = size_bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };
        if (self.vkCreateBuffer(self.device, &bci, null, &buf.buffer) != VK_SUCCESS) return null;

        var req: VkMemoryRequirements = .{};
        self.vkGetBufferMemoryRequirements(self.device, buf.buffer, &req);
        tiger.assert((req.memoryTypeBits & (@as(u32, 1) << @intCast(mem_type_idx))) != 0);

        const alloc_flags: VkMemoryAllocateFlagsInfo = .{
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        };
        const mai: VkMemoryAllocateInfo = .{
            .pNext = &alloc_flags,
            .allocationSize = req.size,
            .memoryTypeIndex = mem_type_idx,
        };
        if (self.vkAllocateMemory(self.device, &mai, null, &buf.memory) != VK_SUCCESS) {
            self.vkDestroyBuffer(self.device, buf.buffer, null);
            return null;
        }
        tiger.assert_vk(self.vkBindBufferMemory(self.device, buf.buffer, buf.memory, 0));

        const bda_info: VkBufferDeviceAddressInfo = .{ .buffer = buf.buffer };
        buf.address = self.vkGetBufferDeviceAddress(self.device, &bda_info);
        tiger.assert(buf.address != 0);

        if (map_host) {
            var ptr: ?*anyopaque = null;
            tiger.assert_vk(self.vkMapMemory(self.device, buf.memory, 0, VK_WHOLE_SIZE, 0, &ptr));
            tiger.assert(ptr != null);
            buf.mapped = @ptrCast(ptr.?);
        }
        return buf;
    }

    pub fn create_buffer(self: *const Context, size_bytes: u64, mem_type_idx: u32, map_host: bool) GpuBuffer {
        if (self.try_create_buffer(size_bytes, mem_type_idx, map_host)) |b| return b;
        tiger.print_stderr("[vulkan] create_buffer failed (size={d:.2} MiB, memType={d})\n", .{
            @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0),
            mem_type_idx,
        });
        @panic("Vulkan OOM");
    }

    pub const VramBudget = struct { budget: u64, usage: u64 };

    /// VK_EXT_memory_budget view of the device-local heap: `budget` is what this process may use
    /// (WDDM subtracts other processes, e.g. the desktop compositor), `usage` what it already uses.
    pub fn vram_budget(self: *const Context) ?VramBudget {
        const f = self.vkGetPhysicalDeviceMemoryProperties2 orelse return null;
        var bud: VkPhysicalDeviceMemoryBudgetPropertiesEXT = .{};
        var props: VkPhysicalDeviceMemoryProperties2 = .{ .pNext = &bud };
        f(self.phys_dev, &props);
        const heap = self.mem_props.memoryTypes[self.mem_type_vram_only].heapIndex;
        if (bud.heapBudget[heap] == 0) return null;
        return .{ .budget = bud.heapBudget[heap], .usage = bud.heapUsage[heap] };
    }

    /// Bytes of device-local memory actually allocatable right now (allocate-until-failure probe,
    /// then free). On Windows the WDDM budget over-reports when the desktop holds VRAM.
    pub fn probe_vram_allocatable(self: *const Context) u64 {
        var bufs: [160]GpuBuffer = undefined;
        var n: usize = 0;
        var total: u64 = 0;
        const sizes = [_]u64{ 256 * 1024 * 1024, 32 * 1024 * 1024 };
        for (sizes) |sz| {
            while (n < bufs.len) {
                const b = self.try_create_buffer(sz, self.mem_type_vram_only, false) orelse break;
                bufs[n] = b;
                n += 1;
                total += sz;
            }
        }
        for (bufs[0..n]) |*b| self.destroy_buffer(b);
        return total;
    }

    pub fn try_create_host_buffer(self: *const Context, size_bytes: u64, prefer_cached: bool, map_host: bool) ?GpuBuffer {
        const mt_a = if (prefer_cached) self.mem_type_host_cached else self.mem_type_host_wc;
        const mt_b = if (prefer_cached) self.mem_type_host_wc else self.mem_type_host_cached;
        if (mt_a != std.math.maxInt(u32)) {
            if (self.try_create_buffer(size_bytes, mt_a, map_host)) |b| return b;
        }
        if (mt_b != std.math.maxInt(u32) and mt_b != mt_a) {
            if (self.try_create_buffer(size_bytes, mt_b, map_host)) |b| return b;
        }
        if (size_bytes <= 64 * 1024 * 1024 and self.mem_type_vram_rebar != std.math.maxInt(u32)) {
            if (self.try_create_buffer(size_bytes, self.mem_type_vram_rebar, map_host)) |b| return b;
        }
        return null;
    }

    pub fn create_host_buffer(self: *const Context, size_bytes: u64, prefer_cached: bool, map_host: bool) GpuBuffer {
        if (self.try_create_host_buffer(size_bytes, prefer_cached, map_host)) |b| return b;
        tiger.print_stderr("[vulkan] create_host_buffer failed across all host types (size={d:.2} MiB)\n", .{
            @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0),
        });
        @panic("Vulkan Host OOM");
    }

    pub fn destroy_buffer(self: *const Context, buf: *GpuBuffer) void {
        if (buf.mapped != null) {
            self.vkUnmapMemory(self.device, buf.memory);
            buf.mapped = null;
        }
        if (buf.buffer != VK_NULL_HANDLE) {
            self.vkDestroyBuffer(self.device, buf.buffer, null);
            buf.buffer = VK_NULL_HANDLE;
        }
        if (buf.memory != VK_NULL_HANDLE) {
            self.vkFreeMemory(self.device, buf.memory, null);
            buf.memory = VK_NULL_HANDLE;
        }
        buf.address = 0;
        buf.size = 0;
    }

    pub fn create_pipeline_layout(self: *const Context, push_constant_bytes: u32) VkPipelineLayout {
        const range = [_]VkPushConstantRange{.{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_constant_bytes,
        }};
        const ci: VkPipelineLayoutCreateInfo = .{
            .pushConstantRangeCount = if (push_constant_bytes > 0) 1 else 0,
            .pPushConstantRanges = if (push_constant_bytes > 0) &range else null,
        };
        var layout: VkPipelineLayout = VK_NULL_HANDLE;
        tiger.assert_vk(self.vkCreatePipelineLayout(self.device, &ci, null, &layout));
        return layout;
    }

    pub fn create_compute_pipeline(
        self: *const Context,
        layout: VkPipelineLayout,
        spv_bytes: []const u8,
        required_subgroup_size: u32,
    ) VkPipeline {
        return self.create_compute_pipeline_spec(layout, spv_bytes, required_subgroup_size, null);
    }

    pub fn create_compute_pipeline_spec(
        self: *const Context,
        layout: VkPipelineLayout,
        spv_bytes: []const u8,
        required_subgroup_size: u32,
        spec_u32: ?u32,
    ) VkPipeline {
        tiger.assert(spv_bytes.len > 0 and spv_bytes.len % 4 == 0);
        const mod_ci: VkShaderModuleCreateInfo = .{
            .codeSize = spv_bytes.len,
            .pCode = @ptrCast(@alignCast(spv_bytes.ptr)),
        };
        var module: VkShaderModule = VK_NULL_HANDLE;
        tiger.assert_vk(self.vkCreateShaderModule(self.device, &mod_ci, null, &module));
        defer self.vkDestroyShaderModule(self.device, module, null);

        var sub_ci: VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = .{
            .requiredSubgroupSize = required_subgroup_size,
        };
        var spec_val: u32 = spec_u32 orelse 0;
        const spec_entry = [_]VkSpecializationMapEntry{.{
            .constantID = 0,
            .offset = 0,
            .size = @sizeOf(u32),
        }};
        const spec_info: VkSpecializationInfo = .{
            .mapEntryCount = 1,
            .pMapEntries = &spec_entry,
            .dataSize = @sizeOf(u32),
            .pData = &spec_val,
        };
        const stage: VkPipelineShaderStageCreateInfo = .{
            .pNext = if (required_subgroup_size > 0) &sub_ci else null,
            .flags = if (required_subgroup_size > 0) VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT else 0,
            .module = module,
            .pName = "main",
            .pSpecializationInfo = if (spec_u32 != null) &spec_info else null,
        };
        const pipe_ci = [_]VkComputePipelineCreateInfo{.{
            .stage = stage,
            .layout = layout,
        }};
        var pipes: [1]VkPipeline = .{VK_NULL_HANDLE};
        tiger.assert_vk(self.vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, &pipe_ci, null, &pipes));
        return pipes[0];
    }

    pub fn alloc_command_buffer(self: *const Context) VkCommandBuffer {
        var cmds: [1]VkCommandBuffer = .{null};
        const ai: VkCommandBufferAllocateInfo = .{
            .commandPool = self.cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        tiger.assert_vk(self.vkAllocateCommandBuffers(self.device, &ai, &cmds));
        return cmds[0];
    }

    pub fn submit_and_wait(self: *const Context, cmd: VkCommandBuffer) void {
        const fences = [_]VkFence{self.imm_fence};
        tiger.assert_vk(self.vkResetFences(self.device, 1, &fences));
        const cmds = [_]VkCommandBuffer{cmd};
        const si = [_]VkSubmitInfo{.{
            .commandBufferCount = 1,
            .pCommandBuffers = &cmds,
        }};
        tiger.assert_vk(self.vkQueueSubmit(self.queue, 1, &si, self.imm_fence));
        tiger.assert_vk(self.vkWaitForFences(self.device, 1, &fences, VK_TRUE, std.math.maxInt(u64)));
    }

    pub fn cmd_compute_barrier(self: *const Context, cmd: VkCommandBuffer) void {
        const mb = [_]VkMemoryBarrier2{.{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        }};
        const dep: VkDependencyInfo = .{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
        };
        self.vkCmdPipelineBarrier2(cmd, &dep);
    }

    pub fn cmd_full_barrier(self: *const Context, cmd: VkCommandBuffer) void {
        const mb = [_]VkMemoryBarrier2{.{
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        }};
        const dep: VkDependencyInfo = .{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
        };
        self.vkCmdPipelineBarrier2(cmd, &dep);
    }
};
