#pragma once
// vk_util.hpp - tiny header-only Vulkan compute utility layer.
//
// vector_add.cpp shows every step raw. Every subsequent kernel would drown
// in the same 200 lines of boilerplate, so we lift instance/device/queue/
// buffer/shader creation into this header. Kernel .cpp files then focus on
// the kernel-specific bits: descriptor layout, push constants, dispatch.
//
// Deliberately minimal - no exceptions, no RAII wrappers, no CRTP. Read it
// top-to-bottom and every line maps to a Vulkan call you already saw in
// vector_add.cpp.

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>

namespace vku {

#define VKU_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ============================================================
// Context - everything you need once, kept around for the whole run.
// ============================================================
struct Context {
    VkInstance                 instance         = VK_NULL_HANDLE;
    VkPhysicalDevice           physical_device  = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties       = {};
    VkDevice                   device           = VK_NULL_HANDLE;
    VkQueue                    queue            = VK_NULL_HANDLE;
    uint32_t                   queue_family     = UINT32_MAX;
    VkCommandPool              command_pool     = VK_NULL_HANDLE;
    // Async upload path (task #29): a SECOND VkQueue from a DIFFERENT
    // queue family (probed on Navi10: family 1 = COMPUTE|TRANSFER x4,
    // distinct from the graphics family 0). Used for vkCmdCopyBuffer
    // H2D uploads with semaphore handoff so the DMA overlaps the host
    // record window instead of serializing on the main queue. All
    // buffers created while has_queue2 is set become CONCURRENT-shared
    // between the two families (no ownership-transfer ceremony); the
    // semaphore orders execution + a consumer-side barrier handles
    // memory visibility.
    VkQueue                    queue2           = VK_NULL_HANDLE;
    uint32_t                   queue2_family    = UINT32_MAX;
    VkCommandPool              command_pool2    = VK_NULL_HANDLE;
    bool                       has_queue2       = false;
    // VK_EXT_host_query_reset (probed at device creation): lets the batch
    // timeline instrumentation reset its timestamp query pool from the host
    // between batches instead of recording vkCmdResetQueryPool in every
    // command buffer.  Absent -> the in-command reset fallback.
    bool                       has_host_query_reset = false;
};

inline Context create_context() {
    Context ctx{};

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "freetoken-vulkan";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VKU_CHECK(vkCreateInstance(&ii, nullptr, &ctx.instance));

    // Pick a physical device - prefer discrete.
    uint32_t n = 0;
    VKU_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &n, nullptr));
    if (n == 0) { std::fprintf(stderr, "no Vulkan GPU\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> pds(n);
    VKU_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &n, pds.data()));
    ctx.physical_device = pds[0];
    for (auto& cand : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(cand, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physical_device = cand; break;
        }
    }
    vkGetPhysicalDeviceProperties(ctx.physical_device, &ctx.properties);

    // Find compute queue family.
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &qn, qfs.data());
    for (uint32_t i = 0; i < qn; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queue_family = i; break; }
    }
    if (ctx.queue_family == UINT32_MAX) { std::fprintf(stderr, "no compute queue\n"); std::exit(1); }

    // Second queue from a DIFFERENT family with COMPUTE|TRANSFER
    // (async upload path, task #29). Skipped silently when absent so the
    // context stays single-queue (legacy behavior, clean no-overlap
    // fallback) - a second queue is never created unless a distinct
    // family is actually available.
    for (uint32_t i = 0; i < qn; ++i) {
        if (i == ctx.queue_family) continue;
        const uint32_t want = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        if ((qfs[i].queueFlags & want) == want && qfs[i].queueCount >= 1) {
            ctx.queue2_family = i;
            ctx.has_queue2 = true;
            break;
        }
    }

    // Enable FP16 + INT8 features. Chain: Vulkan11Features (16-bit storage) ->
    // Vulkan12Features (shaderFloat16 + shaderInt8 + 8-bit storage) -> attached
    // to DeviceCreateInfo.pNext. 6800 XT / RDNA2 supports all four.
    // Int8 + 8-bit storage are needed for MXFP4/NVFP4 packed weights (uint8
    // blocks + uint8 scales).
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderFloat16              = VK_TRUE;
    v12.shaderInt8                 = VK_TRUE;
    v12.storageBuffer8BitAccess    = VK_TRUE;
    v12.uniformAndStorageBuffer8BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess = VK_TRUE;
    v11.pNext = &v12;

    // Optional device extensions.  Only VK_EXT_host_query_reset today
    // (timestamp-pool resets between batches without touching the command
    // buffer); absence is fine - the in-command vkCmdResetQueryPool
    // fallback covers it.  Never enable anything not actually exposed.
    std::vector<const char*> dev_exts;
    {
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(ctx.physical_device, nullptr,
                                             &en, nullptr);
        std::vector<VkExtensionProperties> eps(en);
        vkEnumerateDeviceExtensionProperties(ctx.physical_device, nullptr,
                                             &en, eps.data());
        for (const auto& e : eps) {
            if (!std::strcmp(e.extensionName,
                             VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME)) {
                dev_exts.push_back(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
                ctx.has_host_query_reset = true;
            }
        }
    }

    // Logical device + queues (main + optional second family).
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qcis[2]{};
    qcis[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qcis[0].queueFamilyIndex = ctx.queue_family;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &qprio;
    uint32_t nqci = 1;
    if (ctx.has_queue2) {
        qcis[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qcis[1].queueFamilyIndex = ctx.queue2_family;
        qcis[1].queueCount = 1;
        qcis[1].pQueuePriorities = &qprio;
        nqci = 2;
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = nqci;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.data();
    dci.pNext = &v11;   // enable the fp16 + 16bit-storage features
    VKU_CHECK(vkCreateDevice(ctx.physical_device, &dci, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);
    if (ctx.has_queue2)
        vkGetDeviceQueue(ctx.device, ctx.queue2_family, 0, &ctx.queue2);

    // Command pools.
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = ctx.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VKU_CHECK(vkCreateCommandPool(ctx.device, &cpci, nullptr, &ctx.command_pool));
    if (ctx.has_queue2) {
        VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci2.queueFamilyIndex = ctx.queue2_family;
        cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VKU_CHECK(vkCreateCommandPool(ctx.device, &cpci2, nullptr, &ctx.command_pool2));
    }

    return ctx;
}

inline void destroy_context(Context& ctx) {
    if (ctx.command_pool2) vkDestroyCommandPool(ctx.device, ctx.command_pool2, nullptr);
    if (ctx.command_pool)  vkDestroyCommandPool(ctx.device, ctx.command_pool, nullptr);
    if (ctx.device)        vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance)      vkDestroyInstance(ctx.instance, nullptr);
    ctx = {};
}

// ============================================================
// Buffer helpers.
// ============================================================
struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    // HOST_VISIBLE buffers are mapped once in make_buffer and kept mapped
    // for the buffer's lifetime. Per-upload vkMapMemory/vkUnmapMemory costs
    // ~130 us/MB of driver page-table work on RADV/Navi10 and caps the
    // staging record path at ~4-5 GB/s (the upload fix).
    void*          mapped = nullptr;
    // Coherence of the ACTUAL memory type, not of what was requested: only a
    // non-coherent allocation needs vkFlushMappedMemoryRanges / vkInvalidate-
    // MappedMemoryRanges over exactly the touched range before the GPU reads
    // (resp. before the host reads). Coherent host memory needs no call -
    // and must not get one, because RADV walks the whole mapped range of a
    // coherent allocation on flush.
    bool           host_coherent = true;
};

// Staging map style, FTV_STAGING_MAP:
//   0  temporary map/unmap per host access (pre-fix behaviour)
//   1  persistent map on every host-visible buffer (default)
//   2  persistent map except for upload_via_staging's one-shot sync staging
// Non-coherent allocations always flush the exact written range either way;
// the knob only picks who owns the mapping.
inline int staging_map_mode() {
    static const int mode = [] {
        const char* e = std::getenv("FTV_STAGING_MAP");
        if (!e || !*e) return 1;
        if (e[0] == '0' && !e[1]) return 0;
        if (e[0] == '2' && !e[1]) return 2;
        return 1;
    }();
    return mode;
}

inline uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t type_bits,
                                 VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    std::fprintf(stderr, "no matching memory type\n"); std::exit(1);
}

// Same lookup but reports whether the type exists plus its real flags, so
// callers can ask for non-coherent host memory first and fall back. `forbid`
// is a flag mask the type must not carry (0 = no restriction).
inline bool find_memory_type_opt(VkPhysicalDevice pd, uint32_t type_bits,
                                 VkMemoryPropertyFlags want,
                                 VkMemoryPropertyFlags forbid,
                                 uint32_t* out_index,
                                 VkMemoryPropertyFlags* out_flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (!(type_bits & (1u << i)) || (f & want) != want || (f & forbid)) continue;
        *out_index = i;
        *out_flags = f;
        return true;
    }
    return false;
}

inline Buffer make_buffer(const Context& ctx, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          bool map_persistently = true) {
    Buffer b{};
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    // CONCURRENT sharing between the main + upload families (task #29):
    // vkCmdCopyBuffer uploads on the second queue write buffers that the
    // main queue's compute dispatches read; concurrent sharing removes
    // the queue-family-ownership-transfer ceremony (semaphore + consumer
    // barrier still order execution/visibility). When the context has no
    // second family this is EXCLUSIVE as before.
    uint32_t fams[2] = {ctx.queue_family, ctx.queue2_family};
    if (ctx.has_queue2) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2;
        bi.pQueueFamilyIndices = fams;
    }
    VKU_CHECK(vkCreateBuffer(ctx.device, &bi, nullptr, &b.buf));
    b.size = size;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx.device, b.buf, &mr);

    uint32_t mt = 0;
    VkMemoryPropertyFlags mt_flags = 0;
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((props & hv) && !(props & hc) &&
        !find_memory_type_opt(ctx.physical_device, mr.memoryTypeBits, hv,
                              hc, &mt, &mt_flags)) {
        // No non-coherent host type: fall back to the coherent one below.
        props |= hc;
    }
    if ((mt_flags == 0) &&
        !find_memory_type_opt(ctx.physical_device, mr.memoryTypeBits, props,
                              0, &mt, &mt_flags)) {
        // Preferred (staging asks for HOST_CACHED) not offered: degrade to
        // any type matching the rest instead of failing the device.
        props &= ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (!find_memory_type_opt(ctx.physical_device, mr.memoryTypeBits,
                                  props, 0, &mt, &mt_flags)) {
            std::fprintf(stderr, "no matching memory type\n"); std::exit(1);
        }
    }
    if (std::getenv("FREETOKEN_VULKAN_LOG_MEMTYPE")) {
        std::fprintf(stderr, "[vk] alloc %llu B props=0x%x -> type %u flags=0x%x\n",
                     (unsigned long long)size, props, mt, mt_flags);
    }

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mt;
    VKU_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &b.mem));
    VKU_CHECK(vkBindBufferMemory(ctx.device, b.buf, b.mem, 0));
    b.host_coherent = (mt_flags & hc) != 0;
    if ((mt_flags & hv) && map_persistently)
        VKU_CHECK(vkMapMemory(ctx.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
}

inline void destroy_buffer(const Context& ctx, Buffer& b) {
    if (b.mapped) vkUnmapMemory(ctx.device, b.mem);
    if (b.buf) vkDestroyBuffer(ctx.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(ctx.device, b.mem, nullptr);
    b = {};
}

// End of a legal flush/invalidate range: round up to the non-coherent atom,
// never past the buffer (offset+size at or under the memory size is the
// other legal form of the range). Also the size to map when the caller maps
// only what it touches, so the flushed range stays inside the mapping.
inline VkDeviceSize range_end(const Context& ctx, const Buffer& b,
                              VkDeviceSize offset, VkDeviceSize bytes) {
    const VkDeviceSize atom = ctx.properties.limits.nonCoherentAtomSize;
    const VkDeviceSize end = offset + bytes;
    VkDeviceSize aligned = (end + atom - 1) / atom * atom;
    if (aligned > b.size) aligned = b.size;
    return aligned;
}

// Make a host WRITE visible to the GPU before the copy/dispatch that reads
// it is submitted. No-op on coherent memory (spec) and when nothing moved.
inline void flush_host_write(const Context& ctx, const Buffer& b,
                             VkDeviceSize offset, VkDeviceSize bytes) {
    if (b.host_coherent || bytes == 0) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.mem;
    r.offset = offset;
    r.size = range_end(ctx, b, offset, bytes);
    vkFlushMappedMemoryRanges(ctx.device, 1, &r);
}

// Make a GPU WRITE visible to the host before it reads it back.
inline void invalidate_host_read(const Context& ctx, const Buffer& b,
                                 VkDeviceSize offset, VkDeviceSize bytes) {
    if (b.host_coherent || bytes == 0) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.mem;
    r.offset = offset;
    r.size = range_end(ctx, b, offset, bytes);
    vkInvalidateMappedMemoryRanges(ctx.device, 1, &r);
}

// Begin a host write: returns the destination pointer (persistent mapping,
// or a temporary map the caller pairs with host_write_end).
inline void* host_write_begin(const Context& ctx, Buffer& b,
                              VkDeviceSize bytes) {
    if (b.mapped) return b.mapped;
    void* p = nullptr;
    VKU_CHECK(vkMapMemory(ctx.device, b.mem, 0, range_end(ctx, b, 0, bytes), 0, &p));
    return p;
}

// End a host write: flush the exact range if the memory is non-coherent,
// then drop a temporary mapping (a persistent one stays).
inline void host_write_end(const Context& ctx, Buffer& b,
                           VkDeviceSize bytes, void* p) {
    flush_host_write(ctx, b, 0, bytes);
    if (p != b.mapped) vkUnmapMemory(ctx.device, b.mem);
}

// Host-visible staging/scratch buffer. Prefers HOST_VISIBLE|HOST_CACHED:
// on this device that is coherent write-back system RAM, and the CPU read
// of a download staging matters - the non-cached type the old preference
// resolved to reads back at ~340 MB/s (aperture), which cost the 63 MB
// GDN snapshot pull-down 185 ms. Falls back through the same chain as
// before (non-coherent host type, then any coherent host type).
inline Buffer make_host_ssbo(const Context& ctx, VkDeviceSize size,
                             bool map_persistently = true) {
    return make_buffer(ctx, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, map_persistently);
}

// Device-local buffer, resides in VRAM. Faster for GPU compute but not
// directly writable from CPU; must be filled via a staging buffer +
// vkCmdCopyBuffer. Use for weights that live across many kernel calls.
inline Buffer make_device_ssbo(const Context& ctx, VkDeviceSize size) {
    return make_buffer(ctx, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// ============================================================
// Buffer pool: recycles HOST_VISIBLE VkBuffers across kernel calls.
//
// Problem: allocating and freeing hundreds of small VkBuffers per forward
// pass exhausts the driver's memory-allocation tracking on AMD/Windows,
// hitting VK_ERROR_OUT_OF_DEVICE_MEMORY around forward pass #3 when a
// large baseline (like resident MoE weights, ~10 GB) is already pinned.
//
// Solution: bucket buffers by log2 size. Acquire returns an existing free
// buffer of >= the requested size, or allocates a new one at the bucket
// size. Release puts it back on the free list for the same bucket instead
// of freeing.
//
// - Growth-only (never returns memory to the OS while the pool is alive)
// - Bucket boundaries are powers of 2 from 4 KiB up to 512 MiB
// - Oversized requests receive an exact-size unpooled allocation
// - Caps the free list per bucket to avoid unbounded growth of small buffers
// ============================================================
struct BufferPool {
    Context* ctx = nullptr;
    static constexpr uint32_t kNumBuckets = 18; // 4 KiB .. 512 MiB
    static constexpr size_t   kMaxPerBucket = 64;
    // Device-local free lists for the sync one-shot upload path, keyed by
    // EXACT byte size - NOT power-of-two buckets. The engine's VRAM budget
    // (FTV_EXP_CACHE_MB) is counted in requested bytes, so bucket rounding
    // silently inflates driver VRAM by ~13% (measured: +1.1 GB over the
    // 2048 MiB budget, at the GTT-spill edge of this 8 GB card). Alloc
    // sizes repeat per layer shape, so exact-size reuse still hits on the
    // hot paths. Two bounds so evictions still return VRAM: at most 16
    // free buffers per size and 128 MiB TOTAL across all sizes.
    static constexpr uint32_t kMaxPerSizeDev = 16;
    static constexpr size_t   kMaxTotalDevBytes = 128ull << 20;

    std::vector<Buffer> host_pool[kNumBuckets];
    std::unordered_map<size_t, std::vector<Buffer>> dev_pool;
    size_t dev_held_bytes = 0;

    static uint32_t bucket_of(size_t bytes) {
        // Smallest bucket = 4 KiB. Bucket i holds sizes (2^(11+i), 2^(12+i)].
        // Bucket 0: (4KB, 8KB]; 1: (8, 16]; ...; 17: (512MB, 1GB].
        // For bytes <= 4KB use bucket 0 (waste at most 4KB per allocation).
        uint32_t idx = 0;
        size_t cap = 4 * 1024;
        while (idx + 1 < kNumBuckets && cap < bytes) {
            cap *= 2;
            idx++;
        }
        return idx;
    }

    static size_t bucket_size(uint32_t idx) {
        size_t cap = 4 * 1024;
        for (uint32_t i = 0; i < idx; ++i) cap *= 2;
        return cap;
    }

    static size_t max_bucket_size() {
        return bucket_size(kNumBuckets - 1);
    }

    Buffer acquire_host(size_t bytes) {
        const bool pm = staging_map_mode() != 0;
        if (bytes > max_bucket_size()) {
            return make_host_ssbo(*ctx, bytes, pm);
        }
        uint32_t idx = bucket_of(bytes);
        auto& bucket = host_pool[idx];
        if (!bucket.empty()) {
            Buffer b = bucket.back();
            bucket.pop_back();
            return b;
        }
        return make_host_ssbo(*ctx, bucket_size(idx), pm);
    }

    void release_host(Buffer& b) {
        if (!b.buf) return;
        if ((size_t)b.size > max_bucket_size()) {
            destroy_buffer(*ctx, b);
            return;
        }
        uint32_t idx = bucket_of((size_t)b.size);
        auto& bucket = host_pool[idx];
        if (bucket.size() < kMaxPerBucket) {
            bucket.push_back(b);
            b = {};
        } else {
            destroy_buffer(*ctx, b);
        }
    }

    // Device-local destination for the sync one-shot upload path. Reusing
    // a pooled buffer skips vkCreateBuffer + vkAllocateMemory +
    // vkBindBufferMemory per upload, the churn that kept the sync path at
    // ~4-5 GB/s while the pooled batched path ran at ~12. Exact-size key,
    // so a hit returns a buffer of EXACTLY the requested bytes (VRAM
    // accounting identical to a fresh allocation).
    Buffer acquire_device(size_t bytes) {
        auto it = dev_pool.find(bytes);
        if (it != dev_pool.end() && !it->second.empty()) {
            Buffer b = it->second.back();
            it->second.pop_back();
            dev_held_bytes -= (size_t)b.size;
            if (it->second.empty()) dev_pool.erase(it);
            return b;
        }
        return make_device_ssbo(*ctx, bytes);
    }

    // Returns true when the buffer was DESTROYED (per-size free list full,
    // or the 128 MiB global budget would be exceeded) - the caller must
    // then purge anything that may still reference the dead VkBuffer
    // (descriptor-set cache). A pooled buffer stays alive, so its cached
    // descriptor sets remain valid.
    bool release_device(Buffer& b) {
        if (!b.buf) return false;
        const size_t sz = (size_t)b.size;
        auto it = dev_pool.find(sz);
        const bool full = it != dev_pool.end()
                          && it->second.size() >= kMaxPerSizeDev;
        if (!full && dev_held_bytes + sz <= kMaxTotalDevBytes) {
            dev_pool[sz].push_back(b);
            dev_held_bytes += sz;
            b = {};
            return false;
        }
        destroy_buffer(*ctx, b);
        return true;
    }

    void destroy_all() {
        for (auto& bucket : host_pool) {
            for (auto& b : bucket) destroy_buffer(*ctx, b);
            bucket.clear();
        }
        for (auto& [sz, bufs] : dev_pool) {
            (void)sz;
            for (auto& b : bufs) destroy_buffer(*ctx, b);
            bufs.clear();
        }
        dev_pool.clear();
        dev_held_bytes = 0;
    }

    size_t total_bytes_held() const {
        size_t total = dev_held_bytes;
        for (const auto& bucket : host_pool) {
            for (const auto& b : bucket) total += (size_t)b.size;
        }
        return total;
    }
};

// ------------------------------------------------------------
// Large host-read helper for download staging.
//
// Second-order on top of the HOST_CACHED staging choice (that was the
// 21x: uncached type read at 340 MB/s, cached at ~10 GB/s - see
// make_host_ssbo): unrolled movntdqa loads win another ~15% on big
// readbacks (63 MB pull-down 5.80 -> 4.95 ms), byte-identical to memcpy.
// movntdqa is SSE4.1; without it, plain memcpy. Tail included.
// ------------------------------------------------------------
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define VKU_HAS_X86_INTRIN 1
#include <immintrin.h>

// Per-function target attribute: only this helper gets SSE4.1 codegen;
// the rest of the extension keeps the baseline ISA.
inline void memcpy_bar_nt(char* d, const char* s, size_t n)
    __attribute__((target("sse4.1")));
inline void memcpy_bar_nt(char* d, const char* s, size_t n) {
    // movntdqa needs a 16-byte-aligned source; byte-prologue to reach it.
    size_t i = 0;
    while (i < n && (reinterpret_cast<uintptr_t>(s + i) & 15u) != 0) {
        d[i] = s[i];
        ++i;
    }
    while (i + 64 <= n) {
        char* sp = const_cast<char*>(s + i);
        __m128i a = _mm_stream_load_si128(reinterpret_cast<__m128i*>(sp));
        __m128i b = _mm_stream_load_si128(
            reinterpret_cast<__m128i*>(sp + 16));
        __m128i c = _mm_stream_load_si128(
            reinterpret_cast<__m128i*>(sp + 32));
        __m128i e = _mm_stream_load_si128(
            reinterpret_cast<__m128i*>(sp + 48));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), a);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i + 16), b);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i + 32), c);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i + 48), e);
        i += 64;
    }
    std::memcpy(d + i, s + i, n - i);
}
#endif

inline void memcpy_bar(void* dst0, const void* src0, size_t n) {
#ifdef VKU_HAS_X86_INTRIN
    static const bool has_sse41 = __builtin_cpu_supports("sse4.1");
    if (has_sse41 && n >= 1024) {
        memcpy_bar_nt(static_cast<char*>(dst0),
                      static_cast<const char*>(src0), n);
        return;
    }
#endif
    std::memcpy(dst0, src0, n);
}

inline void upload(const Context& ctx, Buffer& b, const void* src, VkDeviceSize bytes) {
    if (bytes > b.size) {
        throw std::runtime_error("upload exceeds Vulkan buffer capacity");
    }
    void* p = host_write_begin(ctx, b, bytes);
    std::memcpy(p, src, (size_t)bytes);
    host_write_end(ctx, b, bytes, p);
}

inline void download(const Context& ctx, Buffer& b, void* dst, VkDeviceSize bytes) {
    if (bytes > b.size) {
        throw std::runtime_error("download exceeds Vulkan buffer capacity");
    }
    if (b.mapped) {
        invalidate_host_read(ctx, b, 0, bytes);
        memcpy_bar(dst, b.mapped, (size_t)bytes);
        return;
    }
    void* p;
    VKU_CHECK(vkMapMemory(ctx.device, b.mem, 0, range_end(ctx, b, 0, bytes), 0, &p));
    invalidate_host_read(ctx, b, 0, bytes);
    memcpy_bar(dst, p, (size_t)bytes);
    vkUnmapMemory(ctx.device, b.mem);
}

// ============================================================
// One-shot command buffer helper - begin, record via callback, submit, wait.
// Returns wall-clock ms of the queue submit+wait.
// ============================================================
// Reusable resources for submit_and_wait - allocated once, reused across
// every kernel call. Skipping vkAllocateCommandBuffers / vkFreeCommandBuffers
// each call is worth ~50-200 us; using a fence instead of vkQueueWaitIdle
// avoids draining the whole queue.
namespace detail {
    inline VkCommandBuffer& shared_cbuf() {
        static VkCommandBuffer cbuf = VK_NULL_HANDLE;
        return cbuf;
    }
    inline VkFence& shared_fence() {
        static VkFence f = VK_NULL_HANDLE;
        return f;
    }
}

template <typename Fn>
inline double submit_and_wait(const Context& ctx, Fn record) {
    VkCommandBuffer& cbuf = detail::shared_cbuf();
    VkFence& fence = detail::shared_fence();

    if (cbuf == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKU_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &cbuf));
    } else {
        VKU_CHECK(vkResetCommandBuffer(cbuf, 0));
    }
    if (fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKU_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &fence));
    }

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKU_CHECK(vkBeginCommandBuffer(cbuf, &cbbi));
    record(cbuf);
    VKU_CHECK(vkEndCommandBuffer(cbuf));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbuf;

    auto t0 = std::chrono::steady_clock::now();
    VKU_CHECK(vkQueueSubmit(ctx.queue, 1, &si, fence));
    // Busy-poll the fence with vkGetFenceStatus instead of vkWaitForFences.
    // On Windows AMD, vkWaitForFences appears to sleep+wake on the OS
    // scheduler tick (~15.6 ms), turning even microsecond-fast GPU
    // dispatches into ~18 ms round-trips. Busy-polling avoids that at
    // the cost of a hot CPU thread - fine for our per-call overhead.
    while (true) {
        VkResult r = vkGetFenceStatus(ctx.device, fence);
        if (r == VK_SUCCESS) break;
        if (r != VK_NOT_READY) { VKU_CHECK(r); }
        // Optional: _mm_pause() to reduce CPU heat, but keep tight.
    }
    VKU_CHECK(vkResetFences(ctx.device, 1, &fence));
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Upload from CPU into a device-local Buffer via staging. When `pool` is
// given the staging buffer comes from (and returns to) its HOST_VISIBLE
// free list, and the copy reuses submit_and_wait's shared command buffer
// + fence - no per-call staging/cmd alloc, no vkQueueWaitIdle. Without a
// pool this is the legacy temporary-everything behaviour.
inline void upload_via_staging(const Context& ctx, Buffer& device_buf,
                                const void* src, VkDeviceSize bytes,
                                BufferPool* pool = nullptr) {
    // 1. Staging buffer + memcpy CPU -> staging. Mode 2 keeps this one-shot
    // sync path on a temporary map while the pooled/async paths stay
    // persistently mapped.
    Buffer staging;
    if (pool) staging = pool->acquire_host((size_t)bytes);
    else      staging = make_host_ssbo(ctx, bytes, staging_map_mode() == 1);
    upload(ctx, staging, src, bytes);

    // 2. Record staging -> device on the shared one-shot command buffer,
    // submit, and wait the fence (submit_and_wait reuses both).
    submit_and_wait(ctx, [&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.size = bytes;
        vkCmdCopyBuffer(cmd, staging.buf, device_buf.buf, 1, &copy);
    });

    // 3. Recycle or free the staging buffer.
    if (pool) pool->release_host(staging);
    else      destroy_buffer(ctx, staging);
}

// ============================================================
// Shader helpers.
// ============================================================
inline std::vector<uint32_t> load_spirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    size_t bytes = (size_t)f.tellg();
    std::vector<uint32_t> code(bytes / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), bytes);
    return code;
}

inline VkShaderModule make_shader_module(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spv.size() * 4;
    smi.pCode = spv.data();
    VkShaderModule sm;
    VKU_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    return sm;
}

// ============================================================
// GPU timestamp helper - measures actual on-GPU compute time via
// VkQueryPool, separate from queue submission + wait overhead.
//
// Usage:
//     TimestampQuery ts = make_timestamp_query(ctx);
//     double wall_ms = submit_with_timestamps(ctx, ts, [&](VkCommandBuffer cbuf){
//         cmd_reset_and_write_start(cbuf, ts);
//         // ... vkCmdDispatch calls ...
//         cmd_write_end(cbuf, ts);
//     });
//     double gpu_ms = read_gpu_ms(ctx, ts);
//     destroy_timestamp_query(ctx, ts);
//
// GPU-only ms is derived from two hardware timestamps captured
// pre/post compute. Requires the device to support timestamp queries
// on the compute queue (all discrete GPUs do; TimestampComputeAndGraphics
// or timestampPeriod must be > 0).
// ============================================================
struct TimestampQuery {
    VkQueryPool pool = VK_NULL_HANDLE;
    float       timestamp_period_ns = 1.0f;   // ns per tick, from device props
};

inline TimestampQuery make_timestamp_query(const Context& ctx) {
    TimestampQuery ts{};
    ts.timestamp_period_ns = ctx.properties.limits.timestampPeriod;
    if (ts.timestamp_period_ns == 0.0f) {
        std::fprintf(stderr,
            "warning: this queue does not support timestamps; results will be 0\n");
    }
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;   // start + end
    VKU_CHECK(vkCreateQueryPool(ctx.device, &qpci, nullptr, &ts.pool));
    return ts;
}

inline void destroy_timestamp_query(const Context& ctx, TimestampQuery& ts) {
    if (ts.pool) vkDestroyQueryPool(ctx.device, ts.pool, nullptr);
    ts = {};
}

// Reset queries + write timestamp at the top of the compute pipeline.
// Call at the START of your command buffer, before any dispatch.
inline void cmd_reset_and_write_start(VkCommandBuffer cbuf, const TimestampQuery& ts) {
    vkCmdResetQueryPool(cbuf, ts.pool, 0, 2);
    // TOP_OF_PIPE: measured when the command enters the pipeline. Fine for
    // compute-only workloads (no earlier graphics stage to matter).
    vkCmdWriteTimestamp(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts.pool, 0);
}

// Write end-timestamp after your dispatch calls. Uses BOTTOM_OF_PIPE so it
// fires only after the compute shader completes.
inline void cmd_write_end(VkCommandBuffer cbuf, const TimestampQuery& ts) {
    vkCmdWriteTimestamp(cbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts.pool, 1);
}

// Same shape as submit_and_wait but the callback is expected to sandwich
// its dispatch calls between cmd_reset_and_write_start / cmd_write_end.
// Returns wall-clock ms (same as submit_and_wait). Use read_gpu_ms for
// the GPU-only figure.
template <typename Fn>
inline double submit_with_timestamps(const Context& ctx, const TimestampQuery& ts,
                                     Fn record) {
    (void)ts;   // unused here but part of the signature contract
    return submit_and_wait(ctx, record);
}

inline double read_gpu_ms(const Context& ctx, const TimestampQuery& ts) {
    uint64_t stamps[2] = {0, 0};
    VKU_CHECK(vkGetQueryPoolResults(
        ctx.device, ts.pool, 0, 2, sizeof(stamps), stamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const uint64_t delta_ticks = stamps[1] - stamps[0];
    const double   delta_ns    = (double)delta_ticks * ts.timestamp_period_ns;
    return delta_ns / 1.0e6;   // ns -> ms
}

} // namespace vku
