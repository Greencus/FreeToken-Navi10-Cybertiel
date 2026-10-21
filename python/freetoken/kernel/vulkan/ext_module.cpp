// ext_module.cpp - Vulkan compute backend exposed as PyTorch custom ops.
//
// Extends the RMSNorm POC to the full set of ops used by our transformer
// demos: matmul (reg_tiled), softmax, flash_attention, swiglu, moe_router,
// moe_mlp. Everything shares a persistent VulkanContext + pipeline cache
// so a full 11-kernel transformer block runs in-process without disk I/O.
//
// Design:
//   - One `KernelPipeline` per unique shader.
//   - Pipelines lazy-init on first call.
//   - Per-call: allocate host-visible buffers, upload inputs, dispatch,
//     download outputs, free buffers. Buffer pooling is a future win
//     (SKIPPED.md) but not on the critical path.

#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <string>
#include <cstdlib>
#include <memory>
#include <future>
#include <unordered_map>

#include "vk_util.hpp"

// ============================================================
// Persistent Vulkan context (singleton).
// ============================================================
static vku::Context& get_ctx() {
    static vku::Context ctx = vku::create_context();
    return ctx;
}

// Shader dir - env-var configurable so tests can point at a build/ folder.
static std::string shader_path(const char* shader_name) {
    const char* env = std::getenv("FREETOKEN_VULKAN_SHADERS");
    if (env) return std::string(env) + "/" + shader_name;
    return std::string(R"(C:\Users\rohanborkar\Downloads\FreeToken-Vulkan\build\shaders\)")
           + shader_name;
}

// ============================================================
// Kernel pipeline: cached per shader path.
// ============================================================
struct KernelPipeline {
    VkShaderModule        sm              = VK_NULL_HANDLE;
    VkPipeline            pipeline        = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl             = VK_NULL_HANDLE;
    uint32_t              num_bindings    = 0;
    uint32_t              push_const_size = 0;
};

// Cache by shader filename.
static std::unordered_map<std::string, std::unique_ptr<KernelPipeline>> g_pipe_cache;

static KernelPipeline& get_pipeline(const std::string& shader_name,
                                    uint32_t num_bindings,
                                    uint32_t push_const_size);
static double run_kernel(const KernelPipeline& pipe,
                         const std::vector<VkBuffer>& buffers,
                         const void* push_const_data,
                         uint32_t gx, uint32_t gy, uint32_t gz);

// ============================================================
// Global buffer pool. Recycles HOST_VISIBLE buffers across kernel calls
// so we don't hammer vkAllocateMemory / vkFreeMemory. See vku::BufferPool.
// ============================================================
static vku::BufferPool* g_pool = nullptr;

static vku::BufferPool& get_pool() {
    if (!g_pool) {
        g_pool = new vku::BufferPool();
        g_pool->ctx = &get_ctx();
    }
    return *g_pool;
}

// ============================================================
// Resident VRAM buffers. Weights uploaded once, reused across many
// kernel calls. Referenced by int64_t handle from Python.
// ============================================================
static std::unordered_map<int64_t, vku::Buffer> g_resident;
static int64_t g_next_handle = 1;

static VkBuffer resident_buf(int64_t handle) {
    auto it = g_resident.find(handle);
    TORCH_CHECK(it != g_resident.end(),
                "resident buffer handle not found: ", handle);
    return it->second.buf;
}

int64_t upload_resident_vulkan(torch::Tensor t) {
    TORCH_CHECK(t.device().is_cpu(), "resident upload: tensor must be on CPU");
    auto tc = t.contiguous();
    const size_t bytes = (size_t)tc.numel() * tc.element_size();
    auto& ctx = get_ctx();
    vku::Buffer buf = vku::make_device_ssbo(ctx, bytes);
    vku::upload_via_staging(ctx, buf, tc.data_ptr(), bytes);
    const int64_t h = g_next_handle++;
    g_resident.emplace(h, buf);
    return h;
}

int64_t allocate_resident_vulkan(int64_t bytes) {
    TORCH_CHECK(bytes > 0, "resident allocation size must be positive");
    auto& ctx = get_ctx();
    vku::Buffer buf = vku::make_device_ssbo(ctx, (VkDeviceSize)bytes);
    const int64_t h = g_next_handle++;
    g_resident.emplace(h, buf);
    return h;
}

torch::Tensor download_resident_vulkan(int64_t handle,
                                       const std::vector<int64_t>& shape) {
    auto it = g_resident.find(handle);
    TORCH_CHECK(it != g_resident.end(), "resident buffer handle not found");
    int64_t elements = 1;
    for (int64_t dim : shape) {
        TORCH_CHECK(dim >= 0, "resident shape dimensions must be non-negative");
        elements *= dim;
    }
    const size_t bytes = (size_t)elements * sizeof(float);
    TORCH_CHECK(bytes <= (size_t)it->second.size,
                "resident download exceeds buffer capacity");
    auto output = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32));
    auto& ctx = get_ctx();
    vku::Buffer staging = get_pool().acquire_host(bytes);
    vku::submit_and_wait(ctx, [&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.size = (VkDeviceSize)bytes;
        vkCmdCopyBuffer(cmd, it->second.buf, staging.buf, 1, &copy);
    });
    vku::download(ctx, staging, output.data_ptr<float>(), bytes);
    get_pool().release_host(staging);
    return output;
}

torch::Tensor download_resident_i32_vulkan(
    int64_t handle, const std::vector<int64_t>& shape) {
    auto it = g_resident.find(handle);
    TORCH_CHECK(it != g_resident.end(), "resident buffer handle not found");
    int64_t elements = 1;
    for (int64_t dim : shape) elements *= dim;
    const size_t bytes = (size_t)elements * sizeof(int32_t);
    TORCH_CHECK(bytes <= (size_t)it->second.size,
                "resident int32 download exceeds buffer capacity");
    auto output = torch::empty(shape,
        torch::TensorOptions().dtype(torch::kInt32));
    auto& ctx = get_ctx();
    vku::Buffer staging = get_pool().acquire_host(bytes);
    vku::submit_and_wait(ctx, [&](VkCommandBuffer cmd) {
        VkBufferCopy copy{}; copy.size = (VkDeviceSize)bytes;
        vkCmdCopyBuffer(cmd, it->second.buf, staging.buf, 1, &copy);
    });
    vku::download(ctx, staging, output.data_ptr<int32_t>(), bytes);
    get_pool().release_host(staging);
    return output;
}

void topk_resident_vulkan(int64_t logits_handle, int64_t indices_handle,
                          int64_t weights_handle, int64_t rows) {
    TORCH_CHECK(rows > 0, "top-K rows must be positive");
    auto& pipe = get_pipeline("topk_softmax_128_f32.comp.spv", 3, 4);
    struct PC { uint32_t rows; } pc{(uint32_t)rows};
    run_kernel(pipe, {resident_buf(logits_handle), resident_buf(indices_handle),
                      resident_buf(weights_handle)},
               &pc, (uint32_t)rows, 1, 1);
}

void rmsnorm_resident_vulkan(int64_t x_handle, int64_t weight_handle,
                             int64_t y_handle, int64_t rows,
                             int64_t hidden, double eps) {
    TORCH_CHECK(rows > 0 && hidden > 0, "invalid RMSNorm shape");
    const size_t xy_bytes = (size_t)rows * hidden * sizeof(float);
    const size_t w_bytes = (size_t)hidden * sizeof(float);
    TORCH_CHECK(g_resident.at(x_handle).size >= xy_bytes, "resident x too small");
    TORCH_CHECK(g_resident.at(y_handle).size >= xy_bytes, "resident y too small");
    TORCH_CHECK(g_resident.at(weight_handle).size >= w_bytes, "resident weight too small");
    auto& pipe = get_pipeline("rmsnorm_f32.comp.spv", 3, 8);
    struct PC { uint32_t H; float eps; } pc = {
        (uint32_t)hidden, (float)eps
    };
    run_kernel(pipe, {resident_buf(x_handle), resident_buf(weight_handle),
                      resident_buf(y_handle)}, &pc, (uint32_t)rows, 1, 1);
}

void transpose_resident_vulkan(int64_t src_handle, int64_t dst_handle,
                               int64_t S, int64_t H, int64_t D,
                               bool inverse) {
    TORCH_CHECK(S > 0 && H > 0 && D > 0, "invalid transpose shape");
    auto& pipe = get_pipeline("transpose_shd_hsd_f32.comp.spv", 2, 16);
    struct PC { uint32_t S, H, D, inverse; } pc{
        (uint32_t)S, (uint32_t)H, (uint32_t)D,
        inverse ? 1u : 0u};
    uint32_t groups = (uint32_t)((S * H * D + 255) / 256);
    run_kernel(pipe, {resident_buf(src_handle), resident_buf(dst_handle)},
               &pc, groups, 1, 1);
}

void swiglu_resident_vulkan(int64_t gate_handle, int64_t up_handle,
                            int64_t out_handle, int64_t elements) {
    TORCH_CHECK(elements > 0, "resident SwiGLU size must be positive");
    auto& pipe = get_pipeline("swiglu_f32.comp.spv", 3, 4);
    struct PC { uint32_t n; } pc{(uint32_t)elements};
    run_kernel(
        pipe,
        {resident_buf(gate_handle), resident_buf(up_handle),
         resident_buf(out_handle)},
        &pc, (uint32_t)((elements + 255) / 256), 1, 1);
}

void add_resident_vulkan(int64_t a_handle, int64_t b_handle,
                         int64_t out_handle, int64_t elements) {
    TORCH_CHECK(elements > 0, "resident add element count must be positive");
    const size_t bytes = (size_t)elements * sizeof(float);
    TORCH_CHECK(g_resident.at(a_handle).size >= bytes, "resident a too small");
    TORCH_CHECK(g_resident.at(b_handle).size >= bytes, "resident b too small");
    TORCH_CHECK(g_resident.at(out_handle).size >= bytes, "resident output too small");
    auto& pipe = get_pipeline("vector_add.comp.spv", 3, 4);
    struct PC { uint32_t n; } pc = {(uint32_t)elements};
    uint32_t groups = ((uint32_t)elements + 255u) / 256u;
    run_kernel(pipe, {resident_buf(a_handle), resident_buf(b_handle),
                      resident_buf(out_handle)}, &pc, groups, 1, 1);
}

void upload_resident_batch_vulkan(
    const std::vector<int64_t>& handles,
    const std::vector<torch::Tensor>& tensors,
    const std::vector<int64_t>& offsets) {
    TORCH_CHECK(handles.size() == tensors.size(), "handles/tensors size mismatch");
    TORCH_CHECK(handles.size() == offsets.size(), "handles/offsets size mismatch");
    if (handles.empty()) return;

    auto& ctx = get_ctx();
    std::vector<torch::Tensor> contiguous;
    std::vector<vku::Buffer> staging;
    std::vector<void*> mapped;
    std::vector<size_t> bytes;
    contiguous.reserve(tensors.size());
    staging.reserve(tensors.size());
    mapped.reserve(tensors.size());
    bytes.reserve(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].device().is_cpu(), "resident upload tensor must be on CPU");
        TORCH_CHECK(offsets[i] >= 0, "resident upload offset must be non-negative");
        auto it = g_resident.find(handles[i]);
        TORCH_CHECK(it != g_resident.end(), "resident buffer handle not found: ", handles[i]);
        contiguous.push_back(tensors[i].contiguous());
        size_t n = (size_t)contiguous.back().numel() * contiguous.back().element_size();
        TORCH_CHECK((size_t)offsets[i] + n <= (size_t)it->second.size,
                    "resident batch upload exceeds destination capacity");
        bytes.push_back(n);
        staging.push_back(get_pool().acquire_host(n));
        mapped.push_back(vku::host_write_begin(ctx, staging.back(), n));
    }

    std::vector<std::future<void>> copies;
    copies.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        copies.push_back(std::async(std::launch::async, [&, i] {
            std::memcpy(mapped[i], contiguous[i].data_ptr(), bytes[i]);
        }));
    }
    for (auto& copy : copies) copy.get();
    // Flush the exact written range (non-coherent host memory only) and drop
    // temporary mappings before the copies are submitted.
    for (size_t i = 0; i < staging.size(); ++i)
        vku::host_write_end(ctx, staging[i], bytes[i], mapped[i]);

    vku::submit_and_wait(ctx, [&](VkCommandBuffer cmd) {
        for (size_t i = 0; i < tensors.size(); ++i) {
            VkBufferCopy copy{};
            copy.srcOffset = 0;
            copy.dstOffset = (VkDeviceSize)offsets[i];
            copy.size = (VkDeviceSize)bytes[i];
            vkCmdCopyBuffer(cmd, staging[i].buf,
                            g_resident.at(handles[i]).buf, 1, &copy);
        }
    });
    for (auto& buffer : staging) get_pool().release_host(buffer);
}

void update_resident_vulkan(int64_t handle, torch::Tensor tensor,
                            int64_t offset) {
    upload_resident_batch_vulkan({handle}, {tensor}, {offset});
}

// Forward: descriptor pool lives with the pool infra below.
static VkDescriptorPool g_desc_pool;
static uint32_t g_desc_pool_used;
static VkDescriptorPool g_desc_cache_pool;
static uint32_t g_desc_cache_pool_used;

// Descriptor-set cache (task #12 opt): decode re-binds the SAME
// (pipeline, buffers) combos every token (cache-hit experts, pinned
// dense weights, fixed scratch). Keyed lookup skips vkAllocate +
// vkUpdate entirely on hits (~800+ driver calls/token). Entries are
// valid as long as the VkBuffers live: pooled expert handles are reused
// in place (same VkBuffer, still valid), but free_resident DESTROYS the
// VkBuffer, so it must purge the cache (see free_resident_vulkan).
struct DescKey {
    const KernelPipeline* pipe = nullptr;
    uint8_t n = 0;
    VkBuffer b[9] = {};  // widened >4 so gdn_recur (9 buf) hits the cache
    bool operator==(const DescKey& o) const {
        if (pipe != o.pipe || n != o.n) return false;
        for (uint8_t i = 0; i < n; ++i)
            if (b[i] != o.b[i]) return false;
        return true;
    }
};
struct DescKeyHash {
    size_t operator()(const DescKey& k) const noexcept {
        size_t h = std::hash<const void*>{}(k.pipe) ^ (size_t)k.n;
        for (uint8_t i = 0; i < k.n; ++i) {
            size_t hb = std::hash<const void*>{}((const void*)k.b[i]);
            h ^= hb + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
static std::unordered_map<DescKey, VkDescriptorSet, DescKeyHash> g_desc_cache;
static uint64_t g_desc_hits = 0;
static uint64_t g_desc_miss = 0;
static uint64_t g_desc_clears = 0;

static void desc_cache_clear() {
    auto& ctx = get_ctx();
    if (g_desc_cache_pool != VK_NULL_HANDLE)
        vkResetDescriptorPool(ctx.device, g_desc_cache_pool, 0);
    g_desc_cache_pool_used = 0;
    g_desc_cache.clear();
    g_desc_clears++;
}

void free_resident_vulkan(int64_t handle) {
    auto it = g_resident.find(handle);
    if (it == g_resident.end()) return;
    auto& ctx = get_ctx();
    vku::destroy_buffer(ctx, it->second);
    g_resident.erase(it);
    // The destroyed VkBuffer may be referenced by cached descriptor sets;
    // purge the cache so no dangling set is ever bound (task #12 opt).
    // Pooled expert handles are NOT freed (reused in place), so steady-
    // state decode never pays this purge.
    if (!g_desc_cache.empty()) desc_cache_clear();
}

int64_t resident_bytes_total_vulkan() {
    size_t total = 0;
    for (auto& kv : g_resident) total += (size_t)kv.second.size;
    return (int64_t)total;
}

static KernelPipeline& get_pipeline(const std::string& shader_name,
                                    uint32_t num_bindings,
                                    uint32_t push_const_size) {
    auto it = g_pipe_cache.find(shader_name);
    if (it != g_pipe_cache.end()) return *it->second;

    auto& ctx = get_ctx();
    auto p = std::make_unique<KernelPipeline>();
    p->num_bindings = num_bindings;
    p->push_const_size = push_const_size;

    // Descriptor set layout: all bindings are STORAGE_BUFFER, compute-stage.
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = num_bindings;
    dsli.pBindings = bindings.data();
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &p->dsl));

    // Pipeline layout with optional push constants.
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = push_const_size;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p->dsl;
    if (push_const_size > 0) {
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
    }
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &p->pipeline_layout));

    // Shader + pipeline.
    auto spv = vku::load_spirv(shader_path(shader_name.c_str()).c_str());
    p->sm = vku::make_shader_module(ctx.device, spv);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = p->sm;
    cpi.stage.pName = "main";
    cpi.layout = p->pipeline_layout;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p->pipeline));

    auto& ref = *p;
    g_pipe_cache[shader_name] = std::move(p);
    return ref;
}

// ============================================================
// Kernel-call helper: allocate descriptor set from global pool, bind
// buffers, push constants, dispatch.  Returns wall-clock ms.
//
// Perf: caches a global VkDescriptorPool (up to kDescPoolMaxSets sets)
// and resets it when full. This saves the vkCreateDescriptorPool /
// vkDestroyDescriptorPool round-trip per kernel call - ~100-500 us each,
// which was measured to dominate MoE decode-step time.
// ============================================================
static constexpr uint32_t kDescPoolMaxSets = 4096;
static constexpr uint32_t kDescPoolMaxDesc = 65536;
// Dedicated pool for CACHED sets: never reset except by desc_cache_clear
// (free_resident). Transient sets keep using g_desc_pool with the old
// reset-when-full behavior. Split pools fix the reset-per-dispatch
// pathology where churn keys (fresh VkBuffers per cache miss) filled the
// shared pool and every new dispatch wiped the whole cache.
// (Storage is the forward declaration above; static zero-init suffices
// since ensure_desc_cache_pool assigns on first use.)
// Stop caching new keys past this many cached sets; excess keys fall back
// to the transient path. Stable decode keys (~1.5-2k) fit comfortably.
static constexpr uint32_t kDescCacheMaxSets = 3072;

static void ensure_desc_pool() {
    if (g_desc_pool != VK_NULL_HANDLE) return;
    auto& ctx = get_ctx();
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kDescPoolMaxDesc};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kDescPoolMaxSets;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &g_desc_pool));
    g_desc_pool_used = 0;
}

static void ensure_desc_cache_pool() {
    if (g_desc_cache_pool != VK_NULL_HANDLE) return;
    auto& ctx = get_ctx();
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kDescPoolMaxDesc};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kDescPoolMaxSets;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &g_desc_cache_pool));
    g_desc_cache_pool_used = 0;
}

// Descriptor-set cache lookup. Stable keys hit the dedicated cache pool
// (never reset in steady state); once the cache is full, new churn keys
// fall back to the transient path instead of wiping the cache.
// Returns VK_NULL_HANDLE when the caller should use make_descriptor_set.
static VkDescriptorSet make_descriptor_set_cached(
    const KernelPipeline& pipe, const VkBuffer* bufs, uint8_t n,
    bool* from_cache) {
    auto& ctx = get_ctx();
    DescKey key;
    key.pipe = &pipe;
    key.n = n;
    for (uint8_t i = 0; i < n; ++i) key.b[i] = bufs[i];
    auto it = g_desc_cache.find(key);
    if (it != g_desc_cache.end()) {
        *from_cache = true;
        g_desc_hits++;
        return it->second;
    }
    *from_cache = false;
    g_desc_miss++;
    if (g_desc_cache.size() >= kDescCacheMaxSets)
        return VK_NULL_HANDLE;  // cache full: transient fallback
    ensure_desc_cache_pool();
    if (g_desc_cache_pool_used + 1 > kDescPoolMaxSets)
        return VK_NULL_HANDLE;  // pool full: transient fallback
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_desc_cache_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &pipe.dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));
    g_desc_cache_pool_used++;
    // Stack arrays: no heap allocs on the hot path (n <= 9 always).
    VkDescriptorBufferInfo dbi[9];
    VkWriteDescriptorSet writes[9];
    for (uint8_t i = 0; i < n; ++i) {
        dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, (uint32_t)n, writes, 0, nullptr);
    g_desc_cache.emplace(key, dset);
    return dset;
}

static VkDescriptorSet make_descriptor_set(
    const KernelPipeline& pipe,
    const std::vector<VkBuffer>& buffers) {    auto& ctx = get_ctx();
    ensure_desc_pool();
    if (g_desc_pool_used + 1 > kDescPoolMaxSets) {
        vkResetDescriptorPool(ctx.device, g_desc_pool, 0);
        g_desc_pool_used = 0;
    }
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_desc_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &pipe.dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));
    g_desc_pool_used++;
    std::vector<VkDescriptorBufferInfo> dbi(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        dbi[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, (uint32_t)writes.size(),
                           writes.data(), 0, nullptr);
    return dset;
}

static double run_kernel(const KernelPipeline& pipe,
                         const std::vector<VkBuffer>& buffers,
                         const void* push_const_data,
                         uint32_t gx, uint32_t gy, uint32_t gz) {
    auto& ctx = get_ctx();
    VkDescriptorSet dset;
    if (buffers.size() <= 9) {
        VkBuffer small[9];
        for (size_t i = 0; i < buffers.size(); ++i) small[i] = buffers[i];
        bool hit = false;
        dset = make_descriptor_set_cached(
            pipe, small, (uint8_t)buffers.size(), &hit);
        if (dset == VK_NULL_HANDLE)
            dset = make_descriptor_set(pipe, buffers);
    } else {
        dset = make_descriptor_set(pipe, buffers);
    }
    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.pipeline_layout, 0, 1, &dset, 0, nullptr);
        if (pipe.push_const_size > 0 && push_const_data != nullptr) {
            vkCmdPushConstants(cbuf, pipe.pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, pipe.push_const_size, push_const_data);
        }
        vkCmdDispatch(cbuf, gx, gy, 1);
        (void)gz;
    });
    return ms;
}

// Small helper for per-op input/output buffers.
struct CallBuffers {
    std::vector<vku::Buffer> bufs;
    ~CallBuffers() {
        auto& pool = get_pool();
        for (auto& b : bufs) pool.release_host(b);
    }
};

// ============================================================
// Op 1: RMSNorm.
// ============================================================
torch::Tensor rmsnorm_vulkan(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.dtype() == torch::kFloat32,      "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(),
                "tensors must be on CPU");

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto orig_shape = x_c.sizes().vec();
    const int64_t H = orig_shape.back();
    const int64_t N = x_c.numel() / H;
    TORCH_CHECK(w_c.numel() == H, "weight size must match last dim of x");

    auto x2 = x_c.reshape({N, H});
    auto y  = torch::empty_like(x2);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("rmsnorm_f32.comp.spv", 3, 8);

    const size_t xy_bytes = (size_t)N * H * sizeof(float);
    const size_t w_bytes  = (size_t)H * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( xy_bytes));
    cb.bufs.push_back(get_pool().acquire_host( w_bytes));
    cb.bufs.push_back(get_pool().acquire_host( xy_bytes));

    vku::upload(ctx, cb.bufs[0], x2.data_ptr<float>(),  xy_bytes);
    vku::upload(ctx, cb.bufs[1], w_c.data_ptr<float>(), w_bytes);

    struct PC { uint32_t H; float eps; } pc = { (uint32_t)H, (float)eps };
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf},
               &pc, (uint32_t)N, 1, 1);

    vku::download(ctx, cb.bufs[2], y.data_ptr<float>(), xy_bytes);
    return y.reshape(orig_shape);
}

// ============================================================
// Op 2: GEMM (reg_tiled_f32). Simplest 2D dispatch case.
// ============================================================
torch::Tensor matmul_vulkan(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "matmul: both A and B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul: 2D only");
    TORCH_CHECK(A.size(1) == B.size(0),       "matmul: shape mismatch");

    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C = torch::empty({M, N}, A.options());

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("gemm_reg_tiled_f32.comp.spv", 3, 12);

    const size_t a_bytes = (size_t)M * K * sizeof(float);
    const size_t b_bytes = (size_t)K * N * sizeof(float);
    const size_t c_bytes = (size_t)M * N * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( a_bytes));
    cb.bufs.push_back(get_pool().acquire_host( b_bytes));
    cb.bufs.push_back(get_pool().acquire_host( c_bytes));

    vku::upload(ctx, cb.bufs[0], A_c.data_ptr<float>(), a_bytes);
    vku::upload(ctx, cb.bufs[1], B_c.data_ptr<float>(), b_bytes);

    struct PC { uint32_t M; uint32_t N; uint32_t K; } pc = { (uint32_t)M, (uint32_t)N, (uint32_t)K };
    // reg_tiled: each workgroup covers 64x64 of C; convention gx tiles N, gy tiles M.
    const uint32_t TM = 64, TN = 64;
    uint32_t gx = ((uint32_t)N + TN - 1) / TN;
    uint32_t gy = ((uint32_t)M + TM - 1) / TM;
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf}, &pc, gx, gy, 1);

    vku::download(ctx, cb.bufs[2], C.data_ptr<float>(), c_bytes);
    return C;
}

// ============================================================
// Op 3: Softmax over last dim.
// ============================================================
torch::Tensor softmax_vulkan(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax: x must be float32");
    auto x_c = x.contiguous();
    auto orig_shape = x_c.sizes().vec();
    const int64_t D = orig_shape.back();
    const int64_t N = x_c.numel() / D;

    auto x2 = x_c.reshape({N, D});
    auto y = torch::empty_like(x2);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("softmax_f32.comp.spv", 2, 4);

    const size_t bytes = (size_t)N * D * sizeof(float);
    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( bytes));
    cb.bufs.push_back(get_pool().acquire_host( bytes));
    vku::upload(ctx, cb.bufs[0], x2.data_ptr<float>(), bytes);

    uint32_t D_pc = (uint32_t)D;
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf}, &D_pc, (uint32_t)N, 1, 1);
    vku::download(ctx, cb.bufs[1], y.data_ptr<float>(), bytes);
    return y.reshape(orig_shape);
}

// ============================================================
// Op 4: FlashAttention single-head.
// ============================================================
torch::Tensor flash_attention_vulkan(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V, double scale) {
    TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32
                && V.dtype() == torch::kFloat32, "flash_attn: fp32 only");
    TORCH_CHECK(Q.dim() == 2 && Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "flash_attn: Q, K, V must be 2D and same shape");
    const int64_t S = Q.size(0), D = Q.size(1);

    auto Q_c = Q.contiguous(), K_c = K.contiguous(), V_c = V.contiguous();
    auto O = torch::empty_like(Q_c);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("flash_attention_f32.comp.spv", 4, 12);

    const size_t bytes = (size_t)S * D * sizeof(float);
    CallBuffers cb;
    for (int i = 0; i < 4; ++i)
        cb.bufs.push_back(get_pool().acquire_host( bytes));
    vku::upload(ctx, cb.bufs[0], Q_c.data_ptr<float>(), bytes);
    vku::upload(ctx, cb.bufs[1], K_c.data_ptr<float>(), bytes);
    vku::upload(ctx, cb.bufs[2], V_c.data_ptr<float>(), bytes);

    struct PC { uint32_t S; uint32_t D; float scale; } pc =
        { (uint32_t)S, (uint32_t)D, (float)scale };
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf},
               &pc, (uint32_t)S, 1, 1);
    vku::download(ctx, cb.bufs[3], O.data_ptr<float>(), bytes);
    return O;
}

// ============================================================
// Op 4b: Multi-head FlashAttention with optional causal mask.
// Q, K, V, O: [B, H, S, D] float32. causal = bool.
// ============================================================
torch::Tensor flash_attention_mh_vulkan(torch::Tensor Q, torch::Tensor K,
                                         torch::Tensor V, double scale,
                                         bool causal) {
    TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32
                && V.dtype() == torch::kFloat32, "flash_attn_mh: fp32 only");
    TORCH_CHECK(Q.dim() == 4 && Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "flash_attn_mh: Q, K, V must be 4D [B, H, S, D] and same shape");
    const int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    const int64_t BH = B * H;

    auto Q_c = Q.contiguous(), K_c = K.contiguous(), V_c = V.contiguous();
    auto O = torch::empty_like(Q_c);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("flash_attention_mh_f32.comp.spv", 4, 20);

    const size_t bytes = (size_t)BH * S * D * sizeof(float);
    CallBuffers cb;
    for (int i = 0; i < 4; ++i)
        cb.bufs.push_back(get_pool().acquire_host( bytes));
    vku::upload(ctx, cb.bufs[0], Q_c.data_ptr<float>(), bytes);
    vku::upload(ctx, cb.bufs[1], K_c.data_ptr<float>(), bytes);
    vku::upload(ctx, cb.bufs[2], V_c.data_ptr<float>(), bytes);

    struct PC {
        uint32_t S; uint32_t D; uint32_t BH; uint32_t causal; float scale;
    } pc = { (uint32_t)S, (uint32_t)D, (uint32_t)BH,
             causal ? 1u : 0u, (float)scale };

    // Dispatch: S workgroups along X (queries), BH along Y (batch*heads).
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf},
               &pc, (uint32_t)S, (uint32_t)BH, 1);
    vku::download(ctx, cb.bufs[3], O.data_ptr<float>(), bytes);
    return O;
}

// ============================================================
// Op 4c: gpt-oss FlashAttention - MH + GQA + causal + sliding window + sinks.
// Q [B, H_q, S, D], K, V [B, H_kv, S, D], sinks [H_q].
// ============================================================
torch::Tensor flash_attention_gpt_oss_vulkan(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor sinks, double scale,
    int64_t sliding_window, bool use_sinks
) {
    TORCH_CHECK(Q.dtype()==torch::kFloat32 && K.dtype()==torch::kFloat32
                && V.dtype()==torch::kFloat32, "fp32 only");
    TORCH_CHECK(sinks.dtype()==torch::kFloat32, "sinks fp32");
    TORCH_CHECK(Q.dim()==4 && K.dim()==4 && V.dim()==4,
                "Q, K, V must be 4D [B, H, S, D]");
    const int64_t B    = Q.size(0);
    const int64_t H_q  = Q.size(1);
    const int64_t S    = Q.size(2);
    const int64_t D    = Q.size(3);
    const int64_t H_kv = K.size(1);
    TORCH_CHECK(K.size(0)==B && K.size(2)==S && K.size(3)==D, "K shape");
    TORCH_CHECK(V.sizes() == K.sizes(), "V must match K");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
    TORCH_CHECK(sinks.numel() == H_q, "sinks size must equal H_q");
    const int64_t H_q_per_kv = H_q / H_kv;

    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    auto Sc = sinks.contiguous();
    auto O = torch::empty_like(Qc);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("flash_attention_gpt_oss_f32.comp.spv", 5, 32);

    const size_t qbytes  = (size_t)B * H_q  * S * D * sizeof(float);
    const size_t kvbytes = (size_t)B * H_kv * S * D * sizeof(float);
    const size_t sbytes  = (size_t)H_q * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( qbytes));
    cb.bufs.push_back(get_pool().acquire_host( kvbytes));
    cb.bufs.push_back(get_pool().acquire_host( kvbytes));
    cb.bufs.push_back(get_pool().acquire_host( sbytes));
    cb.bufs.push_back(get_pool().acquire_host( qbytes));
    vku::upload(ctx, cb.bufs[0], Qc.data_ptr<float>(), qbytes);
    vku::upload(ctx, cb.bufs[1], Kc.data_ptr<float>(), kvbytes);
    vku::upload(ctx, cb.bufs[2], Vc.data_ptr<float>(), kvbytes);
    vku::upload(ctx, cb.bufs[3], Sc.data_ptr<float>(), sbytes);

    struct PC {
        uint32_t S, D, H_q, H_kv, H_q_per_kv, sliding_window, use_sinks;
        float scale;
    } pc = {
        (uint32_t)S, (uint32_t)D, (uint32_t)H_q, (uint32_t)H_kv,
        (uint32_t)H_q_per_kv,
        sliding_window < 0 ? 0u : (uint32_t)sliding_window,
        use_sinks ? 1u : 0u,
        (float)scale
    };
    // Dispatch: (S, B*H_q, 1)
    run_kernel(pipe,
               {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf,
                cb.bufs[3].buf, cb.bufs[4].buf},
               &pc, (uint32_t)S, (uint32_t)(B * H_q), 1);
    vku::download(ctx, cb.bufs[4], O.data_ptr<float>(), qbytes);
    return O;
}

// ============================================================
// Op 4d: gpt-oss FlashAttention with KV cache - S_q may differ from S_kv,
// past_len is the absolute position of the first Q token.
// ============================================================
torch::Tensor flash_attention_gpt_oss_kv_vulkan(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor sinks, double scale,
    int64_t past_len, int64_t sliding_window, bool use_sinks
) {
    TORCH_CHECK(Q.dtype()==torch::kFloat32 && K.dtype()==torch::kFloat32
                && V.dtype()==torch::kFloat32, "fp32 only");
    TORCH_CHECK(sinks.dtype()==torch::kFloat32, "sinks fp32");
    TORCH_CHECK(Q.dim()==4 && K.dim()==4 && V.dim()==4,
                "Q, K, V must be 4D [B, H, S, D]");
    const int64_t B    = Q.size(0);
    const int64_t H_q  = Q.size(1);
    const int64_t S_q  = Q.size(2);
    const int64_t D    = Q.size(3);
    const int64_t H_kv = K.size(1);
    const int64_t S_kv = K.size(2);
    TORCH_CHECK(K.size(0)==B && K.size(3)==D, "K shape");
    TORCH_CHECK(V.sizes() == K.sizes(), "V must match K");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
    TORCH_CHECK(sinks.numel() == H_q, "sinks size must equal H_q");
    TORCH_CHECK(past_len + S_q <= S_kv,
                "past_len + S_q must equal S_kv (or less)");
    const int64_t H_q_per_kv = H_q / H_kv;

    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    auto Sc = sinks.contiguous();
    auto O = torch::empty_like(Qc);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("flash_attention_gpt_oss_kv_f32.comp.spv", 5, 40);

    const size_t qbytes  = (size_t)B * H_q  * S_q  * D * sizeof(float);
    const size_t kvbytes = (size_t)B * H_kv * S_kv * D * sizeof(float);
    const size_t sbytes  = (size_t)H_q * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( qbytes));
    cb.bufs.push_back(get_pool().acquire_host( kvbytes));
    cb.bufs.push_back(get_pool().acquire_host( kvbytes));
    cb.bufs.push_back(get_pool().acquire_host( sbytes));
    cb.bufs.push_back(get_pool().acquire_host( qbytes));
    vku::upload(ctx, cb.bufs[0], Qc.data_ptr<float>(), qbytes);
    vku::upload(ctx, cb.bufs[1], Kc.data_ptr<float>(), kvbytes);
    vku::upload(ctx, cb.bufs[2], Vc.data_ptr<float>(), kvbytes);
    vku::upload(ctx, cb.bufs[3], Sc.data_ptr<float>(), sbytes);

    struct PC {
        uint32_t S_q, S_kv, D, H_q, H_kv, H_q_per_kv,
                 past_len, sliding_window, use_sinks;
        float scale;
    } pc = {
        (uint32_t)S_q, (uint32_t)S_kv, (uint32_t)D,
        (uint32_t)H_q, (uint32_t)H_kv, (uint32_t)H_q_per_kv,
        (uint32_t)past_len,
        sliding_window < 0 ? 0u : (uint32_t)sliding_window,
        use_sinks ? 1u : 0u,
        (float)scale
    };
    run_kernel(pipe,
               {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf,
                cb.bufs[3].buf, cb.bufs[4].buf},
               &pc, (uint32_t)S_q, (uint32_t)(B * H_q), 1);
    vku::download(ctx, cb.bufs[4], O.data_ptr<float>(), qbytes);
    return O;
}

// ============================================================
// Op 4e: RoPE (partial rotary position embedding, in-place).
// Applies HuggingFace-style rotate_half rotation to the first rotary_dim
// dims of each head. cos/sin are precomputed on host with YARN / NTK scaling.
// ============================================================
torch::Tensor rope_partial_vulkan(torch::Tensor x, torch::Tensor cos,
                                    torch::Tensor sin, int64_t rotary_dim) {
    TORCH_CHECK(x.dtype()==torch::kFloat32 && cos.dtype()==torch::kFloat32
                && sin.dtype()==torch::kFloat32, "fp32 only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, S, D]");
    const int64_t B  = x.size(0);
    const int64_t H  = x.size(1);
    const int64_t S  = x.size(2);
    const int64_t D  = x.size(3);
    TORCH_CHECK(rotary_dim > 0 && rotary_dim <= D && rotary_dim % 2 == 0,
                "rotary_dim must be even and <= D");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == S && cos.size(1) == rotary_dim,
                "cos must be [S, rotary_dim]");
    TORCH_CHECK(sin.sizes() == cos.sizes(), "sin must match cos shape");

    auto xc = x.contiguous();
    auto cc = cos.contiguous();
    auto sc = sin.contiguous();
    auto y = torch::empty_like(xc);

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("rope_partial_f32.comp.spv", 4, 16);

    const size_t xbytes  = (size_t)B * H * S * D * sizeof(float);
    const size_t csbytes = (size_t)S * rotary_dim * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( xbytes));
    cb.bufs.push_back(get_pool().acquire_host( csbytes));
    cb.bufs.push_back(get_pool().acquire_host( csbytes));
    cb.bufs.push_back(get_pool().acquire_host( xbytes));
    vku::upload(ctx, cb.bufs[0], xc.data_ptr<float>(), xbytes);
    vku::upload(ctx, cb.bufs[1], cc.data_ptr<float>(), csbytes);
    vku::upload(ctx, cb.bufs[2], sc.data_ptr<float>(), csbytes);

    struct PC { uint32_t B_H; uint32_t S; uint32_t D; uint32_t rotary_dim; } pc =
        { (uint32_t)(B*H), (uint32_t)S, (uint32_t)D, (uint32_t)rotary_dim };
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf},
               &pc, (uint32_t)S, (uint32_t)(B*H), 1);
    vku::download(ctx, cb.bufs[3], y.data_ptr<float>(), xbytes);
    return y;
}

// ============================================================
// Op: FP32 linear layer (matmul + optional bias) with RESIDENT weights.
// y[t, n] = sum_k x[t, k] * W[n, k] + b[n]
// W and B are referenced by resident-VRAM handles; x is transient.
// Suitable for LM head, Q/K/V/O projections, router, etc.
// ============================================================
void rmsnorm_qkv_resident_vulkan(
    int64_t x_handle, int64_t norm_weight_handle,
    int64_t normalized_handle,
    int64_t q_weight_handle, int64_t q_bias_handle, int64_t q_handle,
    int64_t k_weight_handle, int64_t k_bias_handle, int64_t k_handle,
    int64_t v_weight_handle, int64_t v_bias_handle, int64_t v_handle,
    int64_t rows, int64_t hidden, int64_t q_size, int64_t kv_size,
    double eps) {
    auto& ctx = get_ctx();
    auto& norm = get_pipeline("rmsnorm_f32.comp.spv", 3, 8);
    auto& linear = get_pipeline("linear_fp32_resident_f32.comp.spv", 4, 16);
    struct NormPC { uint32_t H; float eps; } norm_pc = {
        (uint32_t)hidden, (float)eps};
    struct LinearPC { uint32_t T, N, K, use_bias; };
    LinearPC q_pc{(uint32_t)rows,(uint32_t)q_size,(uint32_t)hidden,1u};
    LinearPC kv_pc{(uint32_t)rows,(uint32_t)kv_size,(uint32_t)hidden,1u};
    VkDescriptorSet norm_set = make_descriptor_set(norm, {
        resident_buf(x_handle), resident_buf(norm_weight_handle),
        resident_buf(normalized_handle)});
    auto make_linear = [&](int64_t w, int64_t b, int64_t out) {
        return make_descriptor_set(linear, {
            resident_buf(w), resident_buf(b), resident_buf(normalized_handle),
            resident_buf(out)});
    };
    VkDescriptorSet q_set=make_linear(q_weight_handle,q_bias_handle,q_handle);
    VkDescriptorSet k_set=make_linear(k_weight_handle,k_bias_handle,k_handle);
    VkDescriptorSet v_set=make_linear(v_weight_handle,v_bias_handle,v_handle);
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,norm.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,norm.pipeline_layout,0,1,&norm_set,0,nullptr);
        vkCmdPushConstants(cmd,norm.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&norm_pc);
        vkCmdDispatch(cmd,(uint32_t)rows,1,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        auto dispatch=[&](VkDescriptorSet set,const LinearPC& pc){
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline_layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cmd,linear.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&pc);
            vkCmdDispatch(cmd,pc.N,pc.T,1);
        };
        dispatch(q_set,q_pc);dispatch(k_set,kv_pc);dispatch(v_set,kv_pc);
    });
}

void rope_kv_attention_resident_vulkan(
    int64_t q_handle, int64_t k_handle, int64_t v_handle,
    int64_t cos_handle, int64_t sin_handle,
    int64_t q_rot_handle, int64_t k_rot_handle,
    int64_t k_cache_handle, int64_t v_cache_handle,
    int64_t sinks_handle, int64_t output_handle,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Snew,
    int64_t D, int64_t start, int64_t capacity,
    int64_t sliding_window, bool use_sinks, double scale) {
    auto& ctx=get_ctx();
    auto& rope=get_pipeline("rope_partial_f32.comp.spv",4,16);
    auto& append=get_pipeline("kv_cache_append_f32.comp.spv",4,24);
    auto& attention=get_pipeline(
        "flash_attention_gpt_oss_kv_cache_f32.comp.spv",5,44);
    struct RopePC{uint32_t BH,S,D,rotary_dim;};
    RopePC qpc{(uint32_t)(B*Hq),(uint32_t)Snew,(uint32_t)D,(uint32_t)D};
    RopePC kpc{(uint32_t)(B*Hkv),(uint32_t)Snew,(uint32_t)D,(uint32_t)D};
    struct AppendPC{uint32_t B,Hkv,Snew,D,start,capacity;} apc{
        (uint32_t)B,(uint32_t)Hkv,(uint32_t)Snew,(uint32_t)D,
        (uint32_t)start,(uint32_t)capacity};
    struct AttnPC{
        uint32_t S_q,S_kv,D,H_q,H_kv,H_q_per_kv,past_len,
                 sliding_window,use_sinks,capacity;
        float scale;
    } attn{(uint32_t)Snew,(uint32_t)(start+Snew),(uint32_t)D,
        (uint32_t)Hq,(uint32_t)Hkv,(uint32_t)(Hq/Hkv),(uint32_t)start,
        (uint32_t)sliding_window,use_sinks?1u:0u,(uint32_t)capacity,
        (float)scale};
    VkDescriptorSet qset=make_descriptor_set(rope,{resident_buf(q_handle),resident_buf(cos_handle),resident_buf(sin_handle),resident_buf(q_rot_handle)});
    VkDescriptorSet kset=make_descriptor_set(rope,{resident_buf(k_handle),resident_buf(cos_handle),resident_buf(sin_handle),resident_buf(k_rot_handle)});
    VkDescriptorSet aset=make_descriptor_set(append,{resident_buf(k_rot_handle),resident_buf(v_handle),resident_buf(k_cache_handle),resident_buf(v_cache_handle)});
    VkDescriptorSet attnset=make_descriptor_set(attention,{resident_buf(q_rot_handle),resident_buf(k_cache_handle),resident_buf(v_cache_handle),resident_buf(sinks_handle),resident_buf(output_handle)});
    auto barrier=[&](VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){
        auto rope_dispatch=[&](VkDescriptorSet set,const RopePC& pc,uint32_t heads){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,rope.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,rope.pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(cmd,rope.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&pc);vkCmdDispatch(cmd,(uint32_t)Snew,heads,1);};
        rope_dispatch(qset,qpc,(uint32_t)(B*Hq));rope_dispatch(kset,kpc,(uint32_t)(B*Hkv));barrier(cmd);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,append.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,append.pipeline_layout,0,1,&aset,0,nullptr);vkCmdPushConstants(cmd,append.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,24,&apc);vkCmdDispatch(cmd,(uint32_t)((B*Hkv*Snew*D+255)/256),1,1);barrier(cmd);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,attention.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,attention.pipeline_layout,0,1,&attnset,0,nullptr);vkCmdPushConstants(cmd,attention.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,44,&attn);vkCmdDispatch(cmd,(uint32_t)Snew,(uint32_t)(B*Hq),1);
    });
}

void qwen3_decode_layer_awq_twostage_resident_vulkan(
 int64_t x,int64_t input_norm,int64_t normalized,int64_t qw,int64_t qz,int64_t qs,int64_t q,int64_t kw,int64_t kz,int64_t ks,int64_t k,int64_t vw,int64_t vz,int64_t vs,int64_t v,int64_t qnorm,int64_t knorm,int64_t cos,int64_t sin,int64_t qrot,int64_t krot,int64_t kcache,int64_t vcache,int64_t sinks,int64_t attn,int64_t ow,int64_t oz,int64_t os,int64_t projected,int64_t residual,int64_t postnorm,int64_t post,int64_t gatew,int64_t gatez,int64_t gates,int64_t gate,int64_t upw,int64_t upz,int64_t ups,int64_t up,int64_t activated,int64_t downw,int64_t downz,int64_t downs,int64_t mlp,int64_t out,int64_t partials,int64_t D,int64_t Dff,int64_t Hq,int64_t Hkv,int64_t Hd,int64_t position,int64_t capacity,double eps,double scale){
 auto&ctx=get_ctx();auto&norm=get_pipeline("rmsnorm_f32.comp.spv",3,8);auto&s1=get_pipeline("linear_awq4_stage1_f32.comp.spv",5,16);auto&s2=get_pipeline("linear_awq4_stage2_f32.comp.spv",2,8);auto&rope=get_pipeline("rope_partial_f32.comp.spv",4,16);auto&append=get_pipeline("kv_cache_append_f32.comp.spv",4,24);auto&attention=get_pipeline("flash_attention_gpt_oss_kv_cache_f32.comp.spv",5,44);auto&add=get_pipeline("vector_add.comp.spv",3,4);auto&glu=get_pipeline("swiglu_f32.comp.spv",3,4);
 struct NPC{uint32_t H;float eps;};NPC nd{(uint32_t)D,(float)eps},nh{(uint32_t)Hd,(float)eps};struct RPC{uint32_t BH,S,D,rotary_dim;}qrp{(uint32_t)Hq,1u,(uint32_t)Hd,(uint32_t)Hd},krp{(uint32_t)Hkv,1u,(uint32_t)Hd,(uint32_t)Hd};struct KPC{uint32_t B,Hkv,Snew,D,start,capacity;}cpc{1u,(uint32_t)Hkv,1u,(uint32_t)Hd,(uint32_t)position,(uint32_t)capacity};struct APC{uint32_t S_q,S_kv,D,H_q,H_kv,H_q_per_kv,past_len,sliding_window,use_sinks,capacity;float scale;}apc{1u,(uint32_t)(position+1),(uint32_t)Hd,(uint32_t)Hq,(uint32_t)Hkv,(uint32_t)(Hq/Hkv),(uint32_t)position,0u,0u,(uint32_t)capacity,(float)scale};struct VPC{uint32_t n;}vd{(uint32_t)D},vf{(uint32_t)Dff};
 struct Lin{VkDescriptorSet a,b;uint32_t N,K;};auto L=[&](int64_t w,int64_t z,int64_t sc,int64_t in,int64_t o,uint32_t N,uint32_t K){return Lin{make_descriptor_set(s1,{resident_buf(w),resident_buf(z),resident_buf(sc),resident_buf(in),resident_buf(partials)}),make_descriptor_set(s2,{resident_buf(partials),resident_buf(o)}),N,K};};auto Nrm=[&](int64_t a,int64_t w,int64_t o){return make_descriptor_set(norm,{resident_buf(a),resident_buf(w),resident_buf(o)});};auto Add=[&](int64_t a,int64_t b,int64_t o){return make_descriptor_set(add,{resident_buf(a),resident_buf(b),resident_buf(o)});};VkDescriptorSet n0=Nrm(x,input_norm,normalized),nq=Nrm(q,qnorm,q),nk=Nrm(k,knorm,k),rq=make_descriptor_set(rope,{resident_buf(q),resident_buf(cos),resident_buf(sin),resident_buf(qrot)}),rk=make_descriptor_set(rope,{resident_buf(k),resident_buf(cos),resident_buf(sin),resident_buf(krot)}),ca=make_descriptor_set(append,{resident_buf(krot),resident_buf(v),resident_buf(kcache),resident_buf(vcache)}),at=make_descriptor_set(attention,{resident_buf(qrot),resident_buf(kcache),resident_buf(vcache),resident_buf(sinks),resident_buf(attn)}),a0=Add(projected,x,residual),n1=Nrm(residual,postnorm,post),sg=make_descriptor_set(glu,{resident_buf(gate),resident_buf(up),resident_buf(activated)}),a1=Add(residual,mlp,out);Lin lq=L(qw,qz,qs,normalized,q,Hq*Hd,D),lk=L(kw,kz,ks,normalized,k,Hkv*Hd,D),lv=L(vw,vz,vs,normalized,v,Hkv*Hd,D),lo=L(ow,oz,os,attn,projected,D,Hq*Hd),lg=L(gatew,gatez,gates,post,gate,Dff,D),lu=L(upw,upz,ups,post,up,Dff,D),ld=L(downw,downz,downs,activated,mlp,D,Dff);
 vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){auto bar=[&](){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};auto run=[&](KernelPipeline&p,VkDescriptorSet ds,const void*pc,uint32_t pcs,uint32_t gx,uint32_t gy=1){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline_layout,0,1,&ds,0,nullptr);vkCmdPushConstants(cmd,p.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,pcs,pc);vkCmdDispatch(cmd,gx,gy,1);};auto linear=[&](Lin l){uint32_t chunks=(l.K+1023)/1024;struct P1{uint32_t N,K,group,chunk;}p1{l.N,l.K,128u,1024u};struct P2{uint32_t N,chunks;}p2{l.N,chunks};run(s1,l.a,&p1,16,(l.N+127)/128,chunks);bar();run(s2,l.b,&p2,8,(l.N+127)/128);bar();};run(norm,n0,&nd,8,1);bar();linear(lq);linear(lk);linear(lv);run(norm,nq,&nh,8,Hq);run(norm,nk,&nh,8,Hkv);bar();run(rope,rq,&qrp,16,1,Hq);run(rope,rk,&krp,16,1,Hkv);bar();run(append,ca,&cpc,24,(Hkv*Hd+255)/256);bar();run(attention,at,&apc,44,1,Hq);bar();linear(lo);run(add,a0,&vd,4,(D+255)/256);bar();run(norm,n1,&nd,8,1);bar();linear(lg);linear(lu);run(glu,sg,&vf,4,(Dff+255)/256);bar();linear(ld);run(add,a1,&vd,4,(D+255)/256);});
}

void qwen3_decode_layer_resident_vulkan(
 int64_t x,int64_t input_norm,int64_t normalized,int64_t qw,int64_t q,int64_t kw,int64_t k,int64_t vw,int64_t v,int64_t qnorm,int64_t knorm,int64_t cos,int64_t sin,int64_t qrot,int64_t krot,int64_t kcache,int64_t vcache,int64_t sinks,int64_t attn,int64_t ow,int64_t projected,int64_t residual,int64_t postnorm,int64_t post,int64_t gatew,int64_t gate,int64_t upw,int64_t up,int64_t activated,int64_t downw,int64_t mlp,int64_t out,int64_t qs,int64_t ks,int64_t vs,int64_t os,int64_t gates,int64_t ups,int64_t downs,bool fp8,int64_t D,int64_t Dff,int64_t Hq,int64_t Hkv,int64_t Hd,int64_t position,int64_t capacity,double eps,double scale){
 auto&ctx=get_ctx();auto&norm=get_pipeline("rmsnorm_f32.comp.spv",3,8);auto&lin=get_pipeline(fp8?"linear_fp8e4m3_resident_f32.comp.spv":"linear_bf16_resident_f32.comp.spv",4,16);auto&rope=get_pipeline("rope_partial_f32.comp.spv",4,16);auto&append=get_pipeline("kv_cache_append_f32.comp.spv",4,24);auto&attention=get_pipeline("flash_attention_gpt_oss_kv_cache_f32.comp.spv",5,44);auto&add=get_pipeline("vector_add.comp.spv",3,4);auto&glu=get_pipeline("swiglu_f32.comp.spv",3,4);
 struct NPC{uint32_t H;float eps;};NPC nd{(uint32_t)D,(float)eps},nh{(uint32_t)Hd,(float)eps};struct LPC{uint32_t T,N,K,aux;};LPC qpc{1u,(uint32_t)(Hq*Hd),(uint32_t)D,fp8?(uint32_t)((D+127)/128):0u},kpc{1u,(uint32_t)(Hkv*Hd),(uint32_t)D,fp8?(uint32_t)((D+127)/128):0u},dpc{1u,(uint32_t)Dff,(uint32_t)D,fp8?(uint32_t)((D+127)/128):0u},opc{1u,(uint32_t)D,(uint32_t)(Hq*Hd),fp8?(uint32_t)((Hq*Hd+127)/128):0u},downpc{1u,(uint32_t)D,(uint32_t)Dff,fp8?(uint32_t)((Dff+127)/128):0u};struct RPC{uint32_t BH,S,D,rotary_dim;}qrp{(uint32_t)Hq,1u,(uint32_t)Hd,(uint32_t)Hd},krp{(uint32_t)Hkv,1u,(uint32_t)Hd,(uint32_t)Hd};struct KPC{uint32_t B,Hkv,Snew,D,start,capacity;}cpc{1u,(uint32_t)Hkv,1u,(uint32_t)Hd,(uint32_t)position,(uint32_t)capacity};struct APC{uint32_t S_q,S_kv,D,H_q,H_kv,H_q_per_kv,past_len,sliding_window,use_sinks,capacity;float scale;}apc{1u,(uint32_t)(position+1),(uint32_t)Hd,(uint32_t)Hq,(uint32_t)Hkv,(uint32_t)(Hq/Hkv),(uint32_t)position,0u,0u,(uint32_t)capacity,(float)scale};struct VPC{uint32_t n;}vd{(uint32_t)D},vf{(uint32_t)Dff};
 auto L=[&](int64_t w,int64_t s,int64_t in,int64_t o){return make_descriptor_set(lin,{resident_buf(w),resident_buf(fp8?s:w),resident_buf(in),resident_buf(o)});};auto N=[&](int64_t a,int64_t w,int64_t o){return make_descriptor_set(norm,{resident_buf(a),resident_buf(w),resident_buf(o)});};auto A=[&](int64_t a,int64_t b,int64_t o){return make_descriptor_set(add,{resident_buf(a),resident_buf(b),resident_buf(o)});};VkDescriptorSet n0=N(x,input_norm,normalized),lq=L(qw,qs,normalized,q),lk=L(kw,ks,normalized,k),lv=L(vw,vs,normalized,v),nq=N(q,qnorm,q),nk=N(k,knorm,k),rq=make_descriptor_set(rope,{resident_buf(q),resident_buf(cos),resident_buf(sin),resident_buf(qrot)}),rk=make_descriptor_set(rope,{resident_buf(k),resident_buf(cos),resident_buf(sin),resident_buf(krot)}),ca=make_descriptor_set(append,{resident_buf(krot),resident_buf(v),resident_buf(kcache),resident_buf(vcache)}),at=make_descriptor_set(attention,{resident_buf(qrot),resident_buf(kcache),resident_buf(vcache),resident_buf(sinks),resident_buf(attn)}),lo=L(ow,os,attn,projected),a0=A(projected,x,residual),n1=N(residual,postnorm,post),lg=L(gatew,gates,post,gate),lu=L(upw,ups,post,up),sg=make_descriptor_set(glu,{resident_buf(gate),resident_buf(up),resident_buf(activated)}),ld=L(downw,downs,activated,mlp),a1=A(residual,mlp,out);
 vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){auto bar=[&](){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};auto run=[&](KernelPipeline&p,VkDescriptorSet ds,const void*pc,uint32_t pcs,uint32_t gx,uint32_t gy=1){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline_layout,0,1,&ds,0,nullptr);vkCmdPushConstants(cmd,p.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,pcs,pc);vkCmdDispatch(cmd,gx,gy,1);};run(norm,n0,&nd,8,1);bar();run(lin,lq,&qpc,16,qpc.N);run(lin,lk,&kpc,16,kpc.N);run(lin,lv,&kpc,16,kpc.N);bar();run(norm,nq,&nh,8,Hq);run(norm,nk,&nh,8,Hkv);bar();run(rope,rq,&qrp,16,1,Hq);run(rope,rk,&krp,16,1,Hkv);bar();run(append,ca,&cpc,24,(Hkv*Hd+255)/256);bar();run(attention,at,&apc,44,1,Hq);bar();run(lin,lo,&opc,16,opc.N);bar();run(add,a0,&vd,4,(D+255)/256);bar();run(norm,n1,&nd,8,1);bar();run(lin,lg,&dpc,16,dpc.N);run(lin,lu,&dpc,16,dpc.N);bar();run(glu,sg,&vf,4,(Dff+255)/256);bar();run(lin,ld,&downpc,16,downpc.N);bar();run(add,a1,&vd,4,(D+255)/256);});
}

void decode_pre_moe_all_resident_vulkan(
    int64_t x,int64_t input_norm,int64_t normalized,
    int64_t qw,int64_t qb,int64_t q,int64_t kw,int64_t kb,int64_t k,int64_t vw,int64_t vb,int64_t v,
    int64_t cos,int64_t sin,int64_t qrot,int64_t krot,int64_t kcache,int64_t vcache,int64_t sinks,int64_t attention_out,
    int64_t ow,int64_t ob,int64_t projected,int64_t residual,int64_t post_norm_weight,int64_t post_normalized,
    int64_t router_w,int64_t router_b,int64_t logits,
    int64_t hidden,int64_t Hq,int64_t Hkv,int64_t head_dim,int64_t experts,int64_t position,int64_t capacity,int64_t sliding_window,double eps,double scale) {
    auto& ctx=get_ctx();auto& norm=get_pipeline("rmsnorm_f32.comp.spv",3,8);auto& linear=get_pipeline("linear_fp32_resident_f32.comp.spv",4,16);auto& rope=get_pipeline("rope_partial_f32.comp.spv",4,16);auto& append=get_pipeline("kv_cache_append_f32.comp.spv",4,24);auto& attn=get_pipeline("flash_attention_gpt_oss_kv_cache_f32.comp.spv",5,44);auto& add=get_pipeline("vector_add.comp.spv",3,4);
    struct NPC{uint32_t H;float eps;}npc{(uint32_t)hidden,(float)eps};struct LPC{uint32_t T,N,K,use_bias;};LPC qpc{1u,(uint32_t)(Hq*head_dim),(uint32_t)hidden,1u},kvpc{1u,(uint32_t)(Hkv*head_dim),(uint32_t)hidden,1u},opc{1u,(uint32_t)hidden,(uint32_t)(Hq*head_dim),1u},rpc{1u,(uint32_t)experts,(uint32_t)hidden,1u};struct RPC{uint32_t BH,S,D,rotary_dim;}qrp{(uint32_t)Hq,1u,(uint32_t)head_dim,(uint32_t)head_dim},krp{(uint32_t)Hkv,1u,(uint32_t)head_dim,(uint32_t)head_dim};struct APC{uint32_t B,Hkv,Snew,D,start,capacity;}apc{1u,(uint32_t)Hkv,1u,(uint32_t)head_dim,(uint32_t)position,(uint32_t)capacity};struct ATPC{uint32_t S_q,S_kv,D,H_q,H_kv,H_q_per_kv,past_len,sliding_window,use_sinks,capacity;float scale;}atpc{1u,(uint32_t)(position+1),(uint32_t)head_dim,(uint32_t)Hq,(uint32_t)Hkv,(uint32_t)(Hq/Hkv),(uint32_t)position,(uint32_t)sliding_window,1u,(uint32_t)capacity,(float)scale};struct ADPC{uint32_t n;}adpc{(uint32_t)hidden};
    auto lin=[&](int64_t w,int64_t b,int64_t in,int64_t out){return make_descriptor_set(linear,{resident_buf(w),resident_buf(b),resident_buf(in),resident_buf(out)});};VkDescriptorSet ns=make_descriptor_set(norm,{resident_buf(x),resident_buf(input_norm),resident_buf(normalized)}),qs=lin(qw,qb,normalized,q),ks=lin(kw,kb,normalized,k),vs=lin(vw,vb,normalized,v),qrs=make_descriptor_set(rope,{resident_buf(q),resident_buf(cos),resident_buf(sin),resident_buf(qrot)}),krs=make_descriptor_set(rope,{resident_buf(k),resident_buf(cos),resident_buf(sin),resident_buf(krot)}),aps=make_descriptor_set(append,{resident_buf(krot),resident_buf(v),resident_buf(kcache),resident_buf(vcache)}),ats=make_descriptor_set(attn,{resident_buf(qrot),resident_buf(kcache),resident_buf(vcache),resident_buf(sinks),resident_buf(attention_out)}),os=lin(ow,ob,attention_out,projected),ads=make_descriptor_set(add,{resident_buf(projected),resident_buf(x),resident_buf(residual)}),pns=make_descriptor_set(norm,{resident_buf(residual),resident_buf(post_norm_weight),resident_buf(post_normalized)}),rs=lin(router_w,router_b,post_normalized,logits);
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){auto bar=[&](){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};auto bind=[&](KernelPipeline& p,VkDescriptorSet set,const void* pc,uint32_t pcs,uint32_t gx,uint32_t gy){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(cmd,p.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,pcs,pc);vkCmdDispatch(cmd,gx,gy,1);};bind(norm,ns,&npc,8,1,1);bar();bind(linear,qs,&qpc,16,qpc.N,1);bind(linear,ks,&kvpc,16,kvpc.N,1);bind(linear,vs,&kvpc,16,kvpc.N,1);bar();bind(rope,qrs,&qrp,16,1,(uint32_t)Hq);bind(rope,krs,&krp,16,1,(uint32_t)Hkv);bar();bind(append,aps,&apc,24,(uint32_t)((Hkv*head_dim+255)/256),1);bar();bind(attn,ats,&atpc,44,1,(uint32_t)Hq);bar();bind(linear,os,&opc,16,opc.N,1);bar();bind(add,ads,&adpc,4,(uint32_t)((hidden+255)/256),1);bar();bind(norm,pns,&npc,8,1,1);bar();bind(linear,rs,&rpc,16,rpc.N,1);});
}

void oproj_router_resident_vulkan(
    int64_t attention_handle, int64_t o_weight_handle,
    int64_t o_bias_handle, int64_t projected_handle,
    int64_t residual_handle, int64_t residual_out_handle,
    int64_t norm_weight_handle, int64_t normalized_handle,
    int64_t router_weight_handle, int64_t router_bias_handle,
    int64_t logits_handle, int64_t rows, int64_t hidden,
    int64_t attention_size, int64_t experts, double eps) {
    auto& ctx = get_ctx();
    auto& linear = get_pipeline("linear_fp32_resident_f32.comp.spv",4,16);
    auto& add = get_pipeline("vector_add.comp.spv",3,4);
    auto& norm = get_pipeline("rmsnorm_f32.comp.spv",3,8);
    struct LinearPC{uint32_t T,N,K,use_bias;};
    LinearPC o_pc{(uint32_t)rows,(uint32_t)hidden,
                  (uint32_t)attention_size,1u};
    LinearPC router_pc{(uint32_t)rows,(uint32_t)experts,(uint32_t)hidden,1u};
    struct AddPC{uint32_t n;} add_pc{(uint32_t)(rows*hidden)};
    struct NormPC{uint32_t H;float eps;} norm_pc{(uint32_t)hidden,(float)eps};
    auto linear_set=[&](int64_t w,int64_t b,int64_t x,int64_t y){return make_descriptor_set(linear,{resident_buf(w),resident_buf(b),resident_buf(x),resident_buf(y)});};
    VkDescriptorSet o_set=linear_set(o_weight_handle,o_bias_handle,attention_handle,projected_handle);
    VkDescriptorSet add_set=make_descriptor_set(add,{resident_buf(projected_handle),resident_buf(residual_handle),resident_buf(residual_out_handle)});
    VkDescriptorSet norm_set=make_descriptor_set(norm,{resident_buf(residual_out_handle),resident_buf(norm_weight_handle),resident_buf(normalized_handle)});
    VkDescriptorSet router_set=linear_set(router_weight_handle,router_bias_handle,normalized_handle,logits_handle);
    auto barrier=[&](VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline_layout,0,1,&o_set,0,nullptr);vkCmdPushConstants(cmd,linear.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&o_pc);vkCmdDispatch(cmd,(uint32_t)hidden,(uint32_t)rows,1);barrier(cmd);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,add.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,add.pipeline_layout,0,1,&add_set,0,nullptr);vkCmdPushConstants(cmd,add.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&add_pc);vkCmdDispatch(cmd,(uint32_t)((rows*hidden+255)/256),1,1);barrier(cmd);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,norm.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,norm.pipeline_layout,0,1,&norm_set,0,nullptr);vkCmdPushConstants(cmd,norm.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&norm_pc);vkCmdDispatch(cmd,(uint32_t)rows,1,1);barrier(cmd);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,linear.pipeline_layout,0,1,&router_set,0,nullptr);vkCmdPushConstants(cmd,linear.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&router_pc);vkCmdDispatch(cmd,(uint32_t)experts,(uint32_t)rows,1);
    });
}

void linear_awq4_twostage_resident_io_vulkan(
    int64_t x_handle, int64_t h_qweight, int64_t h_qzeros,
    int64_t h_scales, int64_t partials_handle, int64_t y_handle,
    int64_t N, int64_t K, int64_t group_size) {
    const uint32_t chunk_k=1024;
    const uint32_t chunks=(uint32_t)((K+chunk_k-1)/chunk_k);
    auto&ctx=get_ctx();auto&s1=get_pipeline("linear_awq4_stage1_f32.comp.spv",5,16);auto&s2=get_pipeline("linear_awq4_stage2_f32.comp.spv",2,8);
    struct P1{uint32_t N,K,group_size,chunk_k;}p1{(uint32_t)N,(uint32_t)K,(uint32_t)group_size,chunk_k};struct P2{uint32_t N,chunks;}p2{(uint32_t)N,chunks};
    VkDescriptorSet d1=make_descriptor_set(s1,{resident_buf(h_qweight),resident_buf(h_qzeros),resident_buf(h_scales),resident_buf(x_handle),resident_buf(partials_handle)});VkDescriptorSet d2=make_descriptor_set(s2,{resident_buf(partials_handle),resident_buf(y_handle)});
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s1.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s1.pipeline_layout,0,1,&d1,0,nullptr);vkCmdPushConstants(cmd,s1.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&p1);vkCmdDispatch(cmd,(uint32_t)((N+127)/128),chunks,1);VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s2.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s2.pipeline_layout,0,1,&d2,0,nullptr);vkCmdPushConstants(cmd,s2.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&p2);vkCmdDispatch(cmd,(uint32_t)((N+127)/128),1,1);});
}

void linear_awq4_resident_io_vulkan(
    int64_t x_handle, int64_t h_qweight, int64_t h_qzeros,
    int64_t h_scales, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, int64_t group_size) {
    auto& pipe=get_pipeline("linear_awq4_resident_f32.comp.spv",5,16);
    struct PC{uint32_t T,N,K,group_size;}pc{(uint32_t)T,(uint32_t)N,(uint32_t)K,(uint32_t)group_size};
    run_kernel(pipe,{resident_buf(h_qweight),resident_buf(h_qzeros),resident_buf(h_scales),resident_buf(x_handle),resident_buf(y_handle)},&pc,(uint32_t)((N+127)/128),(uint32_t)T,1);
}

void linear_ao_affine_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t h_scales,
    int64_t h_biases, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, int64_t bits, int64_t group_size) {
    auto& pipe = get_pipeline("linear_ao_affine_resident_f32.comp.spv", 5, 28);
    struct PC {
        uint32_t T, N, K, bits, group_size, words, ngroups;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, (uint32_t)bits,
          (uint32_t)group_size, (uint32_t)((K * bits + 31) / 32),
          (uint32_t)(K / group_size)};
    run_kernel(pipe, {resident_buf(h_packed), resident_buf(h_scales),
                      resident_buf(h_biases), resident_buf(x_handle),
                      resident_buf(y_handle)},
               &pc, (uint32_t)((N + 127) / 128), (uint32_t)T, 1);
}

void linear_nvfp4_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t h_scales,
    int64_t y_handle, int64_t T, int64_t N, int64_t K,
    double gscale, int64_t slot0) {
    auto& pipe = get_pipeline("linear_nvfp4_resident_f32.comp.spv", 4, 28);
    struct PC {
        uint32_t T, N, K, wpr, ng32;
        float gscale;
        uint32_t slot0;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
          (uint32_t)(K / 64), (float)gscale, (uint32_t)slot0};
    // K-tiled layout (task #10): one workgroup per output row, 128 lanes
    // tile the K loop -> grid.x = N (was N/128 with 128 rows per WG).
    run_kernel(pipe, {resident_buf(h_packed), resident_buf(h_scales),
                      resident_buf(x_handle), resident_buf(y_handle)},
               &pc, (uint32_t)N, (uint32_t)T, 1);
}

// Fused gate+up with SiLU fold (task #11). Same signature + push
// constants as linear_nvfp4_resident_io but the shader reads rows n
// (gate) and n+N (up) of the stacked bundle and writes Y = silu(gate)*up;
// the caller feeds that resident Y straight into the down matvec.
void linear_nvfp4_gateup_silu_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t h_scales,
    int64_t y_handle, int64_t T, int64_t N, int64_t K,
    double gscale, int64_t slot0) {
    auto& pipe = get_pipeline("linear_nvfp4_gateup_silu_f32.comp.spv", 4, 28);
    struct PC {
        uint32_t T, N, K, wpr, ng32;
        float gscale;
        uint32_t slot0;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
          (uint32_t)(K / 64), (float)gscale, (uint32_t)slot0};
    // Stacked grid: N gate rows (each WG also computes its up twin row).
    run_kernel(pipe, {resident_buf(h_packed), resident_buf(h_scales),
                      resident_buf(x_handle), resident_buf(y_handle)},
               &pc, (uint32_t)N, (uint32_t)T, 1);
}

// ============================================================
// ggml K-quant (Q4_K / Q5_K / Q8_0) resident matvecs.
//
// The packed GGUF bytes are uploaded VERBATIM and unpacked inside the
// shader (linear_q*k_*_f32.comp), so the host never dequantizes and the
// per-expert H2D transfer is the packed bytes only (1.125 / 1.375 /
// 1.0625 B per weight instead of 2 for BF16). This is the fix for the
// CyberTiel host-dequant bottleneck (3.6-5.5 s/token of numpy dequant +
// 1.9 GB/token of BF16 uploads before it).
//
// One 32-byte push block shared by all three types:
//   wpr       loop trips per row  (K/8 for every type: K/256*32 K-quant
//                                 qs words, or K/32*4 Q8_0 quarter-blocks)
//   ng32      items per block     (36 / 44 words for Q4_K / Q5_K incl. the
//                                 d/dmin/scales[/(qh)] header; 4 quarter
//                                 chunks for Q8_0)
//   rowstride words per packed row (block count * ng32; K/64*17 for Q8_0)
//   slot0     first row of this matrix inside a packed resident arena
// Buffers per dispatch: {packed W, X, Y} - K-quants carry their scales
// inside the block, unlike the NVFP4 shaders' separate scale buffer.
// ============================================================
struct KQuantPC {
    uint32_t T, N, K, wpr, ng32;
    float gscale;
    uint32_t slot0, rowstride;
};

static inline const char* kquant_shader(const std::string& qtype,
                                        const std::string& kind) {
    const bool gu = kind == "gateup";
    const bool ac = kind == "acc";
    if (qtype == "q4_k") {
        return gu ? "linear_q4k_gateup_silu_f32.comp.spv"
                  : ac ? "linear_q4k_acc_resident_f32.comp.spv"
                       : "linear_q4k_resident_f32.comp.spv";
    }
    if (qtype == "q5_k") {
        return gu ? "linear_q5k_gateup_silu_f32.comp.spv"
                  : ac ? "linear_q5k_acc_resident_f32.comp.spv"
                       : "linear_q5k_resident_f32.comp.spv";
    }
    TORCH_CHECK(qtype == "q8_0", "unknown K-quant type: ", qtype);
    TORCH_CHECK(!gu && !ac, "q8_0 has no gateup/acc shader variant");
    return "linear_q8_0_resident_f32.comp.spv";
}

static inline KQuantPC kquant_pc(const std::string& qtype, int64_t T,
                                 int64_t N, int64_t K, double gscale,
                                 int64_t slot0) {
    TORCH_CHECK(T > 0 && N > 0 && K > 0 && K % 256 == 0,
                "K-quant matvec needs T,N,K > 0 and K a multiple of 256");
    KQuantPC pc;
    pc.T = (uint32_t)T;
    pc.N = (uint32_t)N;
    pc.K = (uint32_t)K;
    pc.gscale = (float)gscale;
    pc.slot0 = (uint32_t)slot0;
    if (qtype == "q4_k") {
        pc.wpr = (uint32_t)(K / 256 * 32);
        pc.ng32 = 36u;
        pc.rowstride = (uint32_t)(K / 256 * 36);
    } else if (qtype == "q5_k") {
        pc.wpr = (uint32_t)(K / 256 * 32);
        pc.ng32 = 44u;
        pc.rowstride = (uint32_t)(K / 256 * 44);
    } else if (qtype == "q8_0") {
        TORCH_CHECK(K % 64 == 0, "q8_0 needs K a multiple of 64");
        pc.wpr = (uint32_t)(K / 32 * 4);
        pc.ng32 = 4u;
        pc.rowstride = (uint32_t)(K / 64 * 17);   // 34 B per block / 4
    } else {
        TORCH_CHECK(false, "unknown K-quant type: ", qtype);
    }
    return pc;
}

// kind: "plain" | "gateup" | "acc" (gateup/acc are q4_k/q5_k only).
void linear_kquant_resident_io_vulkan(
    const std::string& qtype, const std::string& kind, int64_t x_handle,
    int64_t h_packed, int64_t y_handle, int64_t T, int64_t N, int64_t K,
    double gscale, int64_t slot0) {
    const char* shader = kquant_shader(qtype, kind);
    KQuantPC pc = kquant_pc(qtype, T, N, K, gscale, slot0);
    auto& pipe = get_pipeline(shader, 3, sizeof(pc));
    run_kernel(pipe, {resident_buf(h_packed), resident_buf(x_handle),
                      resident_buf(y_handle)},
               &pc, (uint32_t)N, (uint32_t)T, 1);
}

void linear_q4k_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q4_k", "plain", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q5k_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q5_k", "plain", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q8_0_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q8_0", "plain", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q4k_gateup_silu_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q4_k", "gateup", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q5k_gateup_silu_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q5_k", "gateup", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q4k_acc_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q4_k", "acc", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_q5k_acc_resident_io_vulkan(
    int64_t x_handle, int64_t h_packed, int64_t y_handle, int64_t T,
    int64_t N, int64_t K, double gscale, int64_t slot0) {
    linear_kquant_resident_io_vulkan("q5_k", "acc", x_handle, h_packed,
                                     y_handle, T, N, K, gscale, slot0);
}

void linear_fp8e4m3_resident_io_vulkan(
    int64_t x_handle, int64_t h_w, int64_t h_scales, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, int64_t scale_cols) {
    auto& pipe=get_pipeline("linear_fp8e4m3_resident_f32.comp.spv",4,16);
    struct PC{uint32_t T,N,K,scale_cols;}pc{(uint32_t)T,(uint32_t)N,(uint32_t)K,(uint32_t)scale_cols};
    run_kernel(pipe,{resident_buf(h_w),resident_buf(h_scales),resident_buf(x_handle),resident_buf(y_handle)},&pc,(uint32_t)N,(uint32_t)T,1);
}

// ============================================================
// Batching: a decode step's dispatches and readbacks are collected into
// ONE command buffer, submitted once, waited once, then readbacks are
// flushed.  A single submission means ordering between recorded ops must
// come from barriers; task #17 makes those barriers hazard-aware (a full
// barrier is only recorded when two in-batch ops share a buffer with a
// write between them), instead of the original barrier-before-every-op.
// ============================================================
static uint64_t g_barrier_total = 0;  // cumulative record barriers
static uint64_t g_barrier_batch = 0;  // barriers in the open batch
static uint64_t g_submit_total = 0;   // cumulative group submissions
static uint64_t g_submit_batch = 0;   // submissions in the open batch
static struct BatchState {
    bool active = false;
    VkCommandBuffer cbuf  = VK_NULL_HANDLE;
    VkFence         fence = VK_NULL_HANDLE;
    std::vector<vku::Buffer> uploads;  // held until submit so the pool
                                       // cannot recycle them mid-batch
    struct Readback {
        vku::Buffer staging;
        torch::Tensor out;
        size_t bytes;
    };
    std::vector<Readback> readbacks;
    // Hazard-aware barrier tracking (task #17): the queue is in-order, so
    // a full memory barrier is only REQUIRED when the recorded op shares a
    // buffer with a prior op that wrote it (RAW), or writes a buffer a
    // prior op read or wrote (WAR/WAW).  Per-buffer access flags, cleared
    // by every emitted barrier (a full barrier orders ALL prior accesses)
    // and by batch_begin (a new submission follows the fence wait).
    struct Access {
        bool read = false;
        bool write = false;
    };
    std::unordered_map<VkBuffer, Access> acc;
} g_batch;

// ============================================================
// Async upload queue (task #29): vkCmdCopyBuffer H2D uploads run on a
// SECOND VkQueue from a DIFFERENT queue family (probed on Navi10:
// family 1 = COMPUTE|TRANSFER x4, distinct from the family-0 graphics
// queue).  The copy DMA overlaps the host record window instead of
// serializing on the main queue's critical path; the consumer batch
// submits with a binary-semaphore wait (ordering) and records a
// cross-queue memory barrier (visibility).  All buffers are CONCURRENT-
// shared between the two families (see vk_util.hpp make_buffer), so no
// queue-family-ownership-transfer ceremony is needed.
//
// One upload batch per consumer (per MoE layer): Python packs the arena
// content, records copies via batch_upload_async, then flushes with
// batch_upload_async_flush (returns an int64 cookie for the semaphore);
// the MoE batch's batch_submit(wait_semaphores=[cookie]) consumes it,
// and the staging buffers + semaphore are only recycled after the
// consumer's fence completes.
struct SubmittedUpload {
    VkSemaphore sem = VK_NULL_HANDLE;
    VkFence     fence = VK_NULL_HANDLE;
    uint32_t    slot = 0;
    std::vector<vku::Buffer> staging;
};

static struct UploadState {
    VkCommandBuffer cbufs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence         fences[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t        cur = 0;
    bool            recording = false;   // cbufs[cur] is open
    std::vector<vku::Buffer> staging;    // held while this batch is in flight
} g_upload;

static std::vector<SubmittedUpload> g_uploads_inflight;  // FIFO, host order
static std::vector<VkSemaphore> g_sem_pool;
static int64_t g_sem_cookie = 0;
static std::unordered_map<int64_t, VkSemaphore> g_sem_by_cookie;
static uint64_t g_cross_total = 0;  // cross-queue barriers (separate counter)
// Queue-2 (upload queue) vkQueueSubmit calls: one per flush, plus the
// safety-net drain's submit of a half-recorded batch.  Separate from
// g_submit_total (MAIN-queue submissions) so counter columns can show
// that a token's dismissals moved to queue 2 without rising on the
// main queue (oracle-11 fix 2).
static uint64_t g_upload_submit_total = 0;

static VkSemaphore sem_acquire() {
    auto& ctx = get_ctx();
    if (!g_sem_pool.empty()) {
        VkSemaphore s = g_sem_pool.back();
        g_sem_pool.pop_back();
        return s;
    }
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore s = VK_NULL_HANDLE;
    VKU_CHECK(vkCreateSemaphore(ctx.device, &sci, nullptr, &s));
    return s;
}

static void sem_release(VkSemaphore s) {
    auto& ctx = get_ctx();
    if (s == VK_NULL_HANDLE) return;
    if (g_sem_pool.size() < 128) {
        g_sem_pool.push_back(s);
    } else {
        vkDestroySemaphore(ctx.device, s, nullptr);
    }
}

// Begin recording on the current upload slot.  Reusing a slot whose
// previous flush is still executing on queue 2 would be UB; wait that
// slot's fence first (already signaled in the normal flow because the
// consumer batch waits the semaphore before completing).
static void upload_begin_locked() {
    auto& ctx = get_ctx();
    if (g_upload.cbufs[g_upload.cur] == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.command_pool2;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKU_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai,
                                           &g_upload.cbufs[g_upload.cur]));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKU_CHECK(vkCreateFence(ctx.device, &fci, nullptr,
                                &g_upload.fences[g_upload.cur]));
    } else {
        VKU_CHECK(vkWaitForFences(ctx.device, 1, &g_upload.fences[g_upload.cur],
                                  VK_TRUE, UINT64_C(30'000'000'000)));
        VKU_CHECK(vkResetFences(ctx.device, 1, &g_upload.fences[g_upload.cur]));
        VKU_CHECK(vkResetCommandBuffer(g_upload.cbufs[g_upload.cur], 0));
    }
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKU_CHECK(vkBeginCommandBuffer(g_upload.cbufs[g_upload.cur], &cbbi));
    g_upload.recording = true;
}

// Record one host->device copy on the async upload queue.  The tensors
// are memcpy'd into pooled staging buffers (same pool the main queue
// uses); a coherent staging type needs no flush, a non-coherent one is
// flushed for exactly the written range in host_write_end.  Distinct dst
// ranges per call, one submit per batch -> no intra-batch barriers needed.
static void batch_upload_async_vulkan(int64_t handle, torch::Tensor tensor,
                                      int64_t offset) {
    auto& ctx = get_ctx();
    TORCH_CHECK(ctx.has_queue2,
                "batch_upload_async needs a distinct second queue family");
    TORCH_CHECK(!g_batch.active,
                "batch_upload_async must run outside the open main batch");
    auto tc = tensor.contiguous();
    const size_t bytes = (size_t)tc.numel() * tc.element_size();
    auto& dst = g_resident.at(handle);
    TORCH_CHECK((size_t)offset + bytes <= dst.size,
                "async upload exceeds resident capacity");
    if (!g_upload.recording) upload_begin_locked();
    vku::Buffer staging = get_pool().acquire_host(bytes);
    void* p = vku::host_write_begin(ctx, staging, bytes);
    std::memcpy(p, tc.data_ptr(), bytes);
    vku::host_write_end(ctx, staging, bytes, p);
    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = (VkDeviceSize)offset;
    c.size = (VkDeviceSize)bytes;
    vkCmdCopyBuffer(g_upload.cbufs[g_upload.cur], staging.buf, dst.buf, 1, &c);
    g_upload.staging.push_back(staging);
}

// End + submit the open upload batch on queue 2; signal a fresh binary
// semaphore and return its cookie.  The consumer (batch_submit with
// wait_semaphores) waits it; staging + semaphore recycle there.
static int64_t batch_upload_async_flush_vulkan() {
    auto& ctx = get_ctx();
    TORCH_CHECK(ctx.has_queue2,
                "batch_upload_async_flush needs a second queue family");
    TORCH_CHECK(g_upload.recording,
                "batch_upload_async_flush with nothing recorded");
    VKU_CHECK(vkEndCommandBuffer(g_upload.cbufs[g_upload.cur]));
    g_upload.recording = false;
    VkSemaphore sem = sem_acquire();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_upload.cbufs[g_upload.cur];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sem;
    VKU_CHECK(vkQueueSubmit(ctx.queue2, 1, &si, g_upload.fences[g_upload.cur]));
    ++g_upload_submit_total;
    g_uploads_inflight.push_back(
        {sem, g_upload.fences[g_upload.cur], g_upload.cur,
         std::move(g_upload.staging)});
    g_upload.staging.clear();
    const int64_t cookie = ++g_sem_cookie;
    g_sem_by_cookie.emplace(cookie, sem);
    g_upload.cur ^= 1;  // next batch lazily begins the other slot
    return cookie;
}

// Cross-queue memory barrier on the OPEN main batch: makes queue-2 copy
// writes visible to this batch's compute dispatches (semaphore handles
// ordering; this handles device-local L2 visibility).  Recorded at the
// consumer batch's start; counted separately from batch_barrier so the
// per-token hazard-barrier counter stays comparable to the pre-task
// baseline.
static void batch_cross_barrier_vulkan() {
    TORCH_CHECK(g_batch.active, "batch_cross_barrier outside an open batch");
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(g_batch.cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                             | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    ++g_cross_total;
}

// Safety net: wait every in-flight queue-2 batch and recycle its staging
// + semaphores.  No-op in the normal decode flow (every flush was
// consumed by its MoE batch's submit).
//
// FENCE INVARIANT (oracle-11 fix 1): the upload-slot fences are left
// SIGNALED by the queue-2 completion signal and are reset ONLY by
// upload_begin_locked (wait -> reset -> begin) when the slot is next
// opened — a wait here must therefore NOT reset, or the next open of
// that slot hangs on a fence nothing will ever signal (30 s VK_TIMEOUT
// at upload_begin_locked).  The wait alone is the drain: it guarantees
// the copies completed before their staging is recycled, exactly like
// the consumer batch_submit does for consumed cookies.
static void batch_upload_async_drain_vulkan() {
    auto& ctx = get_ctx();
    if (!ctx.has_queue2) return;
    if (g_upload.recording) {
        VKU_CHECK(vkEndCommandBuffer(g_upload.cbufs[g_upload.cur]));
        g_upload.recording = false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_upload.cbufs[g_upload.cur];
        VKU_CHECK(vkQueueSubmit(ctx.queue2, 1, &si,
                                g_upload.fences[g_upload.cur]));
        ++g_upload_submit_total;
        VKU_CHECK(vkWaitForFences(ctx.device, 1, &g_upload.fences[g_upload.cur],
                                  VK_TRUE, UINT64_C(30'000'000'000)));
        for (auto& b : g_upload.staging) get_pool().release_host(b);
        g_upload.staging.clear();
    }
    for (auto& u : g_uploads_inflight) {
        VKU_CHECK(vkWaitForFences(ctx.device, 1, &u.fence, VK_TRUE,
                                  UINT64_C(30'000'000'000)));
        for (auto& b : u.staging) get_pool().release_host(b);
        sem_release(u.sem);
    }
    g_uploads_inflight.clear();
}

static void batch_barrier(VkCommandBuffer cb, VkPipelineStageFlags src,
                          VkPipelineStageFlags dst, bool write) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
                       | (write ? VK_ACCESS_MEMORY_WRITE_BIT : 0u);
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    ++g_barrier_total;
    ++g_barrier_batch;
}

static void batch_begin_vulkan() {
    auto& ctx = get_ctx();
    g_barrier_batch = 0;
    g_submit_batch = 0;
    if (g_batch.cbuf == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKU_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &g_batch.cbuf));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKU_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &g_batch.fence));
    } else {
        VKU_CHECK(vkResetCommandBuffer(g_batch.cbuf, 0));
    }
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKU_CHECK(vkBeginCommandBuffer(g_batch.cbuf, &cbbi));
    g_batch.active = true;
    g_batch.acc.clear();  // fresh submission: fence already ordered all prior
                          // accesses, so hazard state starts empty
}

// Order a recorded op against prior accesses in this batch.  `touches` is
// (buffer, is_write) per buffer the op accesses; write covers read-modify-
// write outputs.  Emits a full memory barrier only when a hazard exists,
// then records this op's accesses.
static void batch_ensure_ordered(
    const std::vector<std::pair<VkBuffer, bool>>& touches) {
    if (g_batch.acc.empty()) {
        for (auto& [b, w] : touches) {
            auto& a = g_batch.acc[b];
            if (w) a.write = true; else a.read = true;
        }
        return;
    }
    bool hazard = false;
    for (auto& [b, w] : touches) {
        auto it = g_batch.acc.find(b);
        if (it == g_batch.acc.end()) continue;
        if (!w && it->second.write) { hazard = true; break; }   // RAW
        if (w && (it->second.read || it->second.write)) {
            hazard = true;                                      // WAR/WAW
            break;
        }
    }
    if (hazard) {
        batch_barrier(g_batch.cbuf,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                          | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                          | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      true);
        g_batch.acc.clear();
    }
    for (auto& [b, w] : touches) {
        auto& a = g_batch.acc[b];
        if (w) a.write = true; else a.read = true;
    }
}

static void batch_upload_vulkan(int64_t handle, torch::Tensor tensor,
                                int64_t offset) {
    if (!g_batch.active) {
        // No open batch: fall back to the synchronous upload path. A one-shot
        // upload outside the decode loop (an expert pre-pin at init) is
        // correct but not free; the decode path always uploads inside a batch.
        upload_resident_batch_vulkan({handle}, {std::move(tensor)}, {offset});
        return;
    }
    auto tc = tensor.contiguous();
    const size_t bytes = (size_t)tc.numel() * tc.element_size();
    auto& dst = g_resident.at(handle);
    TORCH_CHECK((size_t)offset + bytes <= dst.size,
                "batch upload exceeds resident capacity");
    auto& ctx = get_ctx();
    vku::Buffer staging = get_pool().acquire_host(bytes);
    void* p = vku::host_write_begin(ctx, staging, bytes);
    std::memcpy(p, tc.data_ptr(), bytes);
    vku::host_write_end(ctx, staging, bytes, p);
    // The first op reading this buffer will see the copy only via the
    // barrier emitted by its own RAW hazard check.
    batch_ensure_ordered({{dst.buf, true}});
    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = (VkDeviceSize)offset;
    c.size = (VkDeviceSize)bytes;
    vkCmdCopyBuffer(g_batch.cbuf, staging.buf, dst.buf, 1, &c);
    g_batch.uploads.push_back(staging);
}

static void batch_fill_vulkan(int64_t handle, float value, int64_t count) {
    TORCH_CHECK(g_batch.active, "batch_fill outside an open batch");
    auto& dst = g_resident.at(handle);
    const size_t bytes = (size_t)count * sizeof(float);
    TORCH_CHECK(bytes <= dst.size, "batch fill exceeds resident capacity");
    batch_ensure_ordered({{dst.buf, true}});
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    vkCmdFillBuffer(g_batch.cbuf, dst.buf, 0, bytes, bits);
}

static void batch_kernel_record(const std::string& shader,
                                const std::vector<VkBuffer>& buffers,
                                const std::vector<uint8_t>& access,
                                const void* push, uint32_t push_size,
                                uint32_t gx, uint32_t gy) {
    TORCH_CHECK(g_batch.active, "batch kernel outside an open batch");
    TORCH_CHECK(buffers.size() == access.size(),
                "batch_kernel_record access list size mismatch");
    auto& ctx = get_ctx();
    // Order only against real hazards (shared buffer + prior write); the
    // in-order queue plus RAW/WAR/WAW barriers is a valid execution order.
    std::vector<std::pair<VkBuffer, bool>> touches;
    touches.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
        touches.emplace_back(buffers[i], access[i] != 0);
    batch_ensure_ordered(touches);
    auto& pipe = get_pipeline(shader, (uint32_t)buffers.size(), push_size);
    VkBuffer small[9];
    const VkBuffer* bp = nullptr;
    uint8_t bn = 0;
    if (buffers.size() <= 9) {
        for (size_t i = 0; i < buffers.size(); ++i) small[i] = buffers[i];
        bp = small;
        bn = (uint8_t)buffers.size();
    }
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (bp != nullptr) {
        bool hit = false;
        dset = make_descriptor_set_cached(pipe, bp, bn, &hit);
    }
    if (dset == VK_NULL_HANDLE)
        dset = make_descriptor_set(pipe, buffers);
    vkCmdBindPipeline(g_batch.cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipe.pipeline);
    vkCmdBindDescriptorSets(g_batch.cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipeline_layout, 0, 1, &dset, 0, nullptr);
    if (push_size > 0 && push != nullptr) {
        vkCmdPushConstants(g_batch.cbuf, pipe.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    }
    vkCmdDispatch(g_batch.cbuf, gx, gy, 1);
}

// NVFP4 dispatch inside a batch. kind: "plain" | "gateup" | "acc".
// M=1 microkernel (task #20): when the env var FTV_NVFP4_M1 is nonzero,
// all three kinds route to the Wave64 single-wavefront shader
// linear_nvfp4_m1 (lane-private acc + shuffle butterfly, no barriers).
// Default OFF: unset -> today's exact resident shaders. Gate is read once
// at first call (cheap + stable for the whole process). The pipelined
// m1 kernel adds a `kind` selector u32 to the 24-byte push block (28 B).
static void batch_nvfp4_io_vulkan(
    const std::string& kind, int64_t h_packed, int64_t h_scales,
    int64_t x_handle, int64_t y_handle, int64_t T, int64_t N, int64_t K,
    double gscale, int64_t slot0) {
    // Task #20 drop-in: env-gated NVFP4 single-wavefront microkernel.
    // Enable with FTV_NVFP4_M1=<nonzero int> (atoi("true")=0, OFF by default).
    // Measured ~1.04× full-model / 1.148× isolated MoE; sync-path shaders
    // (prefill) do NOT use this gate — enabling during mixed sessions splits numerics.
    static const bool use_m1 = []() {
        const char* e = std::getenv("FTV_NVFP4_M1");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (use_m1) {
        struct M1PC {  // 32 B: resident fields + kind + slot0
            uint32_t T, N, K, wpr, ng32;
            float gscale;
            uint32_t kind;
            uint32_t slot0;
        } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
              (uint32_t)(K / 64), (float)gscale,
              kind == "acc" ? 2u : kind == "gateup" ? 1u : 0u,
              (uint32_t)slot0};
        batch_kernel_record("linear_nvfp4_m1_f32.comp.spv",
                            {resident_buf(h_packed), resident_buf(h_scales),
                             resident_buf(x_handle), resident_buf(y_handle)},
                            {0, 0, 0, 1},   // W, S, X read; Y written (acc: RMW)
                            &pc, sizeof(pc), (uint32_t)N, (uint32_t)T);
        return;
    }
    const char* shader =
        kind == "acc"    ? "linear_nvfp4_acc_resident_f32.comp.spv"
        : kind == "gateup"
            ? "linear_nvfp4_gateup_silu_f32.comp.spv"
            : "linear_nvfp4_resident_f32.comp.spv";
    struct PC {  // 28 B: resident fields + slot0 (row offset in arena)
        uint32_t T, N, K, wpr, ng32;
        float gscale;
        uint32_t slot0;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
          (uint32_t)(K / 64), (float)gscale, (uint32_t)slot0};
    batch_kernel_record(shader,
                        {resident_buf(h_packed), resident_buf(h_scales),
                         resident_buf(x_handle), resident_buf(y_handle)},
                        {0, 0, 0, 1},   // W, S, X read; Y written (acc: RMW)
                        &pc, sizeof(pc), (uint32_t)N, (uint32_t)T);
}

// K-quant dispatch inside a batch: same contract as batch_nvfp4_io but
// with NO separate scale buffer (the K-quant scales live in the packed
// block) and a qtype selector. kind: "plain" | "gateup" | "acc".
static void batch_kquant_io_vulkan(
    const std::string& qtype, const std::string& kind, int64_t h_packed,
    int64_t x_handle, int64_t y_handle, int64_t T, int64_t N, int64_t K,
    double gscale, int64_t slot0) {
    const char* shader = kquant_shader(qtype, kind);
    KQuantPC pc = kquant_pc(qtype, T, N, K, gscale, slot0);
    batch_kernel_record(shader,
                        {resident_buf(h_packed), resident_buf(x_handle),
                         resident_buf(y_handle)},
                        {0, 0, 1},   // packed + X read; Y written (all
                                     // kinds: plain/gateup write Y, acc
                                     // is RMW -- matches batch_nvfp4_io)
                        &pc, sizeof(pc), (uint32_t)N, (uint32_t)T);
}

// Fused MoE-layer record for K-quant experts: the mirror of
// batch_moe_record_vulkan, minus the NVFP4 scale buffers and minus the
// shared-expert / download tail (those stay BF16 ops the caller records
// around it). 8 packed gateup-silu + 8 packed down-accumulate in ONE
// pybind crossing; gate rows of expert e sit at packed rows
// e*2*gu_rows.. (up rows follow at +gu_rows), down rows at dn_slot0[e]..
// gu_rows is the GATE row count N (the bundle holds 2N rows), matching
// batch_moe_record's gu_rows convention.
static void batch_moe_kquant_record_vulkan(
    int64_t h_x_fused, int64_t h_y_acc,
    const std::string& qtype,
    const std::vector<int64_t>& silu_slots,
    const std::vector<int64_t>& gu_w, const std::vector<double>& gu_scale,
    int64_t gu_rows, int64_t gu_cols,
    const std::vector<int64_t>& dn_w, const std::vector<double>& dn_scale,
    int64_t dn_rows, int64_t dn_cols,
    const std::vector<int64_t>& gu_slot0, const std::vector<int64_t>& dn_slot0) {
    TORCH_CHECK(g_batch.active, "batch_moe_kquant_record outside an open batch");
    const size_t ne = silu_slots.size();
    TORCH_CHECK(gu_w.size() == ne && gu_scale.size() == ne,
                "batch_moe_kquant_record gateup list size mismatch");
    TORCH_CHECK(dn_w.size() == ne && dn_scale.size() == ne,
                "batch_moe_kquant_record down list size mismatch");
    for (size_t e = 0; e < ne; ++e) {
        int64_t gs = (gu_slot0.size() > e) ? gu_slot0[e] : 0;
        int64_t ds = (dn_slot0.size() > e) ? dn_slot0[e] : 0;
        batch_kquant_io_vulkan(qtype, "gateup", gu_w[e], h_x_fused,
                               silu_slots[e], 1, gu_rows, gu_cols,
                               gu_scale[e], gs);
        batch_kquant_io_vulkan(qtype, "acc", dn_w[e], silu_slots[e],
                               h_y_acc, 1, dn_rows, dn_cols, dn_scale[e], ds);
    }
}

// Forward: defined below; the fused MoE recorder calls them.
static void batch_bf16_io_vulkan(int64_t x_handle, int64_t h_w, int64_t h_b,
                                 int64_t y_handle, int64_t T, int64_t N,
                                 int64_t K, bool use_bias);
static void batch_swiglu_io_vulkan(int64_t h_gate, int64_t h_up,
                                   int64_t h_out, int64_t N);
static void batch_download_vulkan(int64_t handle, torch::Tensor out,
                                  int64_t offset);
// Fused MoE-layer record (task #12 opt): the whole batch-3 MoE sequence
// (8 gateup + 8 acc + shared gate/up/gsc + swiglu fold + shared down +
// 3 downloads) in ONE pybind crossing instead of ~24. Calls the exact
// same static record helpers in the exact same order, so results are
// bit-identical to the split calls; only crossing + arg-parse overhead
// disappears (~10-25 ms/token on host-loaded boxes).
static void batch_moe_record_vulkan(
    int64_t h_x_fused, int64_t h_y_acc,
    int64_t h_sg, int64_t h_su, int64_t h_gsc,
    int64_t h_x3, int64_t h_sdown,
    const std::vector<int64_t>& silu_slots,
    const std::vector<int64_t>& gu_w, const std::vector<int64_t>& gu_s,
    const std::vector<double>& gu_scale, int64_t gu_rows, int64_t gu_cols,
    const std::vector<int64_t>& dn_w, const std::vector<int64_t>& dn_s,
    const std::vector<double>& dn_scale, int64_t dn_rows, int64_t dn_cols,
    int64_t sg_w, int64_t sg_rows, int64_t sg_cols,
    int64_t su_w, int64_t su_rows, int64_t su_cols,
    int64_t gsc_w, int64_t gsc_rows, int64_t gsc_cols,
    int64_t sdp_w, int64_t sdp_rows, int64_t sdp_cols,
    int64_t gate_rows,
    const std::vector<int64_t>& gu_slot0, const std::vector<int64_t>& dn_slot0,
    torch::Tensor acc_out, torch::Tensor sd_out, torch::Tensor gsc_out) {
    TORCH_CHECK(g_batch.active, "batch_moe_record outside an open batch");
    const size_t ne = silu_slots.size();
    TORCH_CHECK(gu_w.size() == ne && gu_s.size() == ne && gu_scale.size() == ne,
                "batch_moe_record gateup list size mismatch");
    TORCH_CHECK(dn_w.size() == ne && dn_s.size() == ne && dn_scale.size() == ne,
                "batch_moe_record down list size mismatch");
    for (size_t e = 0; e < ne; ++e) {
        int64_t gs = (gu_slot0.size() > e) ? gu_slot0[e] : 0;
        int64_t ds = (dn_slot0.size() > e) ? dn_slot0[e] : 0;
        batch_nvfp4_io_vulkan("gateup", gu_w[e], gu_s[e], h_x_fused,
                              silu_slots[e], 1, gu_rows, gu_cols, gu_scale[e], gs);
        batch_nvfp4_io_vulkan("acc", dn_w[e], dn_s[e], silu_slots[e],
                              h_y_acc, 1, dn_rows, dn_cols, dn_scale[e], ds);
    }
    batch_bf16_io_vulkan(h_x_fused, sg_w, sg_w, h_sg, 1, sg_rows, sg_cols, false);
    batch_bf16_io_vulkan(h_x_fused, su_w, su_w, h_su, 1, su_rows, su_cols, false);
    batch_bf16_io_vulkan(h_x_fused, gsc_w, gsc_w, h_gsc, 1, gsc_rows, gsc_cols, false);
    batch_swiglu_io_vulkan(h_sg, h_su, h_x3, gate_rows);
    batch_bf16_io_vulkan(h_x3, sdp_w, sdp_w, h_sdown, 1, sdp_rows, sdp_cols, false);
    batch_download_vulkan(h_y_acc, acc_out, 0);
    batch_download_vulkan(h_sdown, sd_out, 0);
    batch_download_vulkan(h_gsc, gsc_out, 0);
}

// ============================================================
// Fused MUL_MAT_ID-style MoE dispatches (arena): the 8 routed
// experts of a layer collapse from 16 dispatches (8 gateup+silu +
// 8 down-accumulate) to 2.  Both shaders index the per-layer arena
// with a resident u32[8] ids buffer (`_arena_ids`, slot index per
// canonical expert) and a resident float[16] gs buffer (`_arena_gs`,
// gateup scales at [0..8), accumulated scales at [8..16)).
//
// Numeric contract (verified by test_batched_fork2):
//  - gateup8: every (j, n) row dot is bit-identical to the per-expert
//    gateup_silu kernel (same slot row offset, lane/word split, scale
//    indexing, FMA chain, two-tree gate+up reduction); output row is
//    j*N + n of h_silu_all instead of row n of slot buffer e.
//  - acc8: every expert's dot d_j is bit-identical to the per-expert
//    acc kernel's partial[0]; one WG per output row sums the 8 d_j in
//    canonical order j=0..7 into a register and applies a single add
//    Y[n] += r.  With h_y_acc pre-filled to exact +0.0 this equals
//    ((0+d0)+d1)..+d7, the old 8-dispatch add chain — bit-exact, and
//    deterministic (no cross-WG RMW race on a shared row).
// ============================================================
static void batch_nvfp4_gateup8_io_vulkan(
    int64_t h_x_fused, int64_t h_arena_packed, int64_t h_arena_scales,
    int64_t h_silu_all, int64_t h_ids, int64_t h_gs,
    int64_t N, int64_t K, int64_t rows_per_slot) {
    struct PC {
        uint32_t T, N, K, wpr, ng32, rows_per_slot;
    } pc{1u, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
         (uint32_t)(K / 64), (uint32_t)rows_per_slot};
    // grid (N, 8): WG (n, j) computes expert j's gate+up row n.
    batch_kernel_record("linear_nvfp4_gateup8_f32.comp.spv",
                        {resident_buf(h_arena_packed),
                         resident_buf(h_arena_scales),
                         resident_buf(h_x_fused),
                         resident_buf(h_silu_all),
                         resident_buf(h_ids), resident_buf(h_gs)},
                        {0, 0, 0, 1, 0, 0},  // W,S,X,Ids,Gs read; silu_all written
                        &pc, sizeof(pc), (uint32_t)N, 8u);
}

static void batch_nvfp4_acc8_io_vulkan(
    int64_t h_silu_all, int64_t h_arena_packed, int64_t h_arena_scales,
    int64_t h_y_acc, int64_t h_ids, int64_t h_gs,
    int64_t N, int64_t K, int64_t rows_per_slot, int64_t silu_N) {
    struct PC {
        uint32_t T, N, K, wpr, ng32, rows_per_slot, silu_N;
    } pc{1u, (uint32_t)N, (uint32_t)K, (uint32_t)(K / 8),
         (uint32_t)(K / 64), (uint32_t)rows_per_slot,
         (uint32_t)silu_N};
    // grid (N, 1): one 128-lane WG per output row computes all 8 dots.
    batch_kernel_record("linear_nvfp4_acc8_f32.comp.spv",
                        {resident_buf(h_arena_packed),
                         resident_buf(h_arena_scales),
                         resident_buf(h_silu_all),
                         resident_buf(h_y_acc),
                         resident_buf(h_ids), resident_buf(h_gs)},
                        {0, 0, 0, 1, 0, 0},  // W,S,X,Ids,Gs read; h_y_acc RMW
                        &pc, sizeof(pc), (uint32_t)N, 1u);
}

// Fused-MoE layer record: 2 fused dispatches + the unchanged shared-
// expert sequence (gate/up/gsc dense + swiglu fold + shared down + 3
// downloads) in one pybind crossing.  The shared tail is recorded in
// the exact same order as batch_moe_record_vulkan (bit-identical).
static void batch_moe_fused_record_vulkan(
    int64_t h_x_fused, int64_t h_y_acc,
    int64_t h_sg, int64_t h_su, int64_t h_gsc,
    int64_t h_x3, int64_t h_sdown,
    int64_t h_silu_all,
    int64_t h_arena_wf, int64_t h_arena_sf,
    int64_t h_arena_wd, int64_t h_arena_sd,
    int64_t h_ids, int64_t h_gs,
    int64_t gu_rows, int64_t gu_cols, int64_t rows_fused,
    int64_t dn_rows, int64_t dn_cols, int64_t rows_down,
    int64_t silu_N,
    int64_t sg_w, int64_t sg_rows, int64_t sg_cols,
    int64_t su_w, int64_t su_rows, int64_t su_cols,
    int64_t gsc_w, int64_t gsc_rows, int64_t gsc_cols,
    int64_t sdp_w, int64_t sdp_rows, int64_t sdp_cols,
    int64_t gate_rows,
    torch::Tensor acc_out, torch::Tensor sd_out, torch::Tensor gsc_out) {
    TORCH_CHECK(g_batch.active,
                "batch_moe_fused_record outside an open batch");
    batch_nvfp4_gateup8_io_vulkan(
        h_x_fused, h_arena_wf, h_arena_sf, h_silu_all, h_ids, h_gs,
        gu_rows, gu_cols, rows_fused);
    batch_nvfp4_acc8_io_vulkan(
        h_silu_all, h_arena_wd, h_arena_sd, h_y_acc, h_ids, h_gs,
        dn_rows, dn_cols, rows_down, silu_N);
    batch_bf16_io_vulkan(h_x_fused, sg_w, sg_w, h_sg, 1, sg_rows, sg_cols, false);
    batch_bf16_io_vulkan(h_x_fused, su_w, su_w, h_su, 1, su_rows, su_cols, false);
    batch_bf16_io_vulkan(h_x_fused, gsc_w, gsc_w, h_gsc, 1, gsc_rows, gsc_cols, false);
    batch_swiglu_io_vulkan(h_sg, h_su, h_x3, gate_rows);
    batch_bf16_io_vulkan(h_x3, sdp_w, sdp_w, h_sdown, 1, sdp_rows, sdp_cols, false);
    batch_download_vulkan(h_y_acc, acc_out, 0);
    batch_download_vulkan(h_sdown, sd_out, 0);
    batch_download_vulkan(h_gsc, gsc_out, 0);
}

// BF16 dense dispatch inside a batch.
static void batch_bf16_io_vulkan(int64_t x_handle, int64_t h_w, int64_t h_b,
                                 int64_t y_handle, int64_t T, int64_t N,
                                 int64_t K, bool use_bias) {
    struct PC {
        uint32_t T, N, K, use_bias;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, use_bias ? 1u : 0u};
    VkBuffer w = resident_buf(h_w);
    VkBuffer b = use_bias ? resident_buf(h_b) : w;
    batch_kernel_record("linear_bf16_resident_f32.comp.spv",
                        {w, b, resident_buf(x_handle), resident_buf(y_handle)},
                        {0, 0, 0, 1},
                        &pc, sizeof(pc), (uint32_t)N, (uint32_t)T);
}

// FP32 dense dispatch inside a batch (GPU-router task #22). Same
// signature + push layout as batch_bf16_io_vulkan but runs the FP32
// resident shader (linear_fp32_resident_f32) so FP32 router weights are
// NEVER cast to BF16 (oracle-3 R1: do NOT use batch_bf16_io for the
// router GEMM).
static void batch_fp32_io_vulkan(int64_t x_handle, int64_t h_w, int64_t h_b,
                                 int64_t y_handle, int64_t T, int64_t N,
                                 int64_t K, bool use_bias) {
    struct PC {
        uint32_t T, N, K, use_bias;
    } pc{(uint32_t)T, (uint32_t)N, (uint32_t)K, use_bias ? 1u : 0u};
    VkBuffer w = resident_buf(h_w);
    VkBuffer b = use_bias ? resident_buf(h_b) : w;
    batch_kernel_record("linear_fp32_resident_f32.comp.spv",
                        {w, b, resident_buf(x_handle), resident_buf(y_handle)},
                        {0, 0, 0, 1},
                        &pc, sizeof(pc), (uint32_t)N, (uint32_t)T);
}

// Zero-centered RMSNorm inside a batch (GPU-router task #22, oracle-3 R1):
// y = x * rsqrt(mean(x^2) + eps) * (1 + w)  — the Qwen3.5 host contract.
// Uses the new rmsnorm_1pw_f32 shader, NOT the plain rmsnorm_f32 (which
// computes `* w` without the (1+w) offset).  One workgroup per ROW
// (gx = rows); H is the hidden dim.
static void batch_rms1p_io_vulkan(int64_t x_handle, int64_t h_w,
                                  int64_t y_handle, int64_t H, double eps,
                                  int64_t rows) {
    struct PC {
        uint32_t H;
        float eps;
    } pc{(uint32_t)H, (float)eps};
    batch_kernel_record("rmsnorm_1pw_f32.comp.spv",
                        {resident_buf(x_handle), resident_buf(h_w),
                         resident_buf(y_handle)},
                        {0, 0, 1},
                        &pc, sizeof(pc), (uint32_t)rows, 1);
}

// GPU router tail inside a batch (oracle-3 R3/R5): softmax(256) -> topk-8
// -> renormalize -> canonical sort -> arena ids/gs write.  One
// 256-thread WG per token; E/K passed in push constants, shader
// hardcoded to 256-local workgroup + 8-wide arrays (do NOT reuse the
// E=128/K=4 topk_softmax_128 shader).
static void batch_topk256_io_vulkan(
    int64_t h_logits, int64_t h_expscale, int64_t h_map,
    int64_t h_topk_ids, int64_t h_topk_vals,
    int64_t h_arena_ids, int64_t h_arena_gs,
    int64_t E, int64_t K, int64_t tokens) {
    struct PC {
        uint32_t E, K, T;
    } pc{(uint32_t)E, (uint32_t)K, (uint32_t)tokens};
    batch_kernel_record("topk256_k8_f32.comp.spv",
                        {resident_buf(h_logits), resident_buf(h_expscale),
                         resident_buf(h_map), resident_buf(h_topk_ids),
                         resident_buf(h_topk_vals), resident_buf(h_arena_ids),
                         resident_buf(h_arena_gs)},
                        {0, 0, 0, 1, 1, 1, 1},  // logits/expscale/map read;
                                                // ids/vals/arena buffers written
                        &pc, sizeof(pc), (uint32_t)tokens, 1);
}

// Elementwise SwiGLU fold inside a batch: y = silu(gate) * up, all three
// resident fp32 buffers same length N. Lets the batched path compute the
// shared-expert silu(gate)*up fold on the GPU (task #12 opt: kills a
// submit+upload per layer by folding + shared down inside batch 3).
static void batch_swiglu_io_vulkan(int64_t h_gate, int64_t h_up,
                                   int64_t h_out, int64_t N) {
    struct PC {
        uint32_t N;
    } pc{(uint32_t)N};
    batch_kernel_record("swiglu_f32.comp.spv",
                        {resident_buf(h_gate), resident_buf(h_up),
                         resident_buf(h_out)},
                        {0, 0, 1},
                        &pc, sizeof(pc), (uint32_t)((N + 255) / 256), 1);
}

// GDN conv taps inside a batch: mixed = silu(taps . [buf, qkv]) with
// in-place conv history shift (task #13: GPU-resident GDN recurrence).
// h_qkv/h_buf/h_taps/h_mixed are resident fp32; taps is [C,4] fp32.
static void batch_gdn_conv_io_vulkan(int64_t h_qkv, int64_t h_buf,
                                     int64_t h_taps, int64_t h_mixed,
                                     int64_t C) {
    struct PC {
        uint32_t C;
    } pc{(uint32_t)C};
    batch_kernel_record("gdn_conv_f32.comp.spv",
                        {resident_buf(h_qkv), resident_buf(h_buf),
                         resident_buf(h_taps), resident_buf(h_mixed)},
                        {0, 1, 0, 1},  // buf is shifted in place (RMW)
                        &pc, sizeof(pc), (uint32_t)((C + 255) / 256), 1);
}

// GDN state recurrence inside a batch: one thread per value-head;
// reads mixed/z/a/b + per-layer params, updates resident state in place,
// writes pre_out (task #13). 9 bindings now fit the widened DescKey, so
// this hits the desc cache (30 stable per-layer keys, 1 alloc each on
// first decode after prefill; zero vkAllocate+Update on steady tokens).
static void batch_gdn_recur_io_vulkan(
    int64_t h_mixed, int64_t h_z, int64_t h_a, int64_t h_b,
    int64_t h_alog, int64_t h_dtb, int64_t h_normw,
    int64_t h_state, int64_t h_preout,
    int64_t C, int64_t key_dim, int64_t nv, int64_t hk, int64_t hv,
    int64_t rep, double scale, double l2eps, double rms_eps) {
    struct PC {
        uint32_t C, key_dim, nv, hk, hv, rep;
        float scale, l2eps, rms_eps;
    } pc{(uint32_t)C, (uint32_t)key_dim, (uint32_t)nv, (uint32_t)hk,
         (uint32_t)hv, (uint32_t)rep, (float)scale, (float)l2eps,
         (float)rms_eps};
    batch_kernel_record("gdn_recur_f32.comp.spv",
                        {resident_buf(h_mixed), resident_buf(h_z),
                         resident_buf(h_a), resident_buf(h_b),
                         resident_buf(h_alog), resident_buf(h_dtb),
                         resident_buf(h_normw), resident_buf(h_state),
                         resident_buf(h_preout)},
                        {0, 0, 0, 0, 0, 0, 0, 1, 1},  // state/preout written
                        &pc, sizeof(pc), (uint32_t)nv, 1);
}

// Elementwise add inside a batch: c = a + b (task #13: x1 = x + ao).
static void batch_vecadd_io_vulkan(int64_t h_a, int64_t h_b,
                                   int64_t h_out, int64_t N) {
    struct PC {
        uint32_t N;
    } pc{(uint32_t)N};
    batch_kernel_record("vector_add.comp.spv",
                        {resident_buf(h_a), resident_buf(h_b),
                         resident_buf(h_out)},
                        {0, 0, 1},
                        &pc, sizeof(pc), (uint32_t)((N + 255) / 256), 1);
}

// Read a resident buffer back into a caller-provided fp32 tensor. The
// staging copy is recorded now; the memcpy into the tensor happens after
// the batch fence in batch_submit.
static void batch_download_vulkan(int64_t handle, torch::Tensor out,
                                  int64_t offset) {
    TORCH_CHECK(g_batch.active, "batch_download outside an open batch");
    TORCH_CHECK(out.device().is_cpu(), "batch_download target must be CPU");
    auto oc = out.contiguous();
    const size_t bytes = (size_t)oc.numel() * oc.element_size();
    auto& src = g_resident.at(handle);
    TORCH_CHECK((size_t)offset + bytes <= src.size,
                "batch download exceeds resident capacity");
    batch_ensure_ordered({{src.buf, false}});
    auto& ctx = get_ctx();
    vku::Buffer staging = get_pool().acquire_host(bytes);
    VkBufferCopy c{};
    c.srcOffset = (VkDeviceSize)offset;
    c.size = (VkDeviceSize)bytes;
    vkCmdCopyBuffer(g_batch.cbuf, src.buf, staging.buf, 1, &c);
    g_batch.readbacks.push_back({staging, oc, bytes});
}

static double batch_submit_vulkan(
    const std::vector<int64_t>& wait_semaphores) {
    TORCH_CHECK(g_batch.active, "batch_submit with no open batch");
    auto& ctx = get_ctx();
    VKU_CHECK(vkEndCommandBuffer(g_batch.cbuf));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_batch.cbuf;
    // Cross-queue semaphore handoff (task #29): wait the queue-2 upload
    // batches this consumer depends on before executing.  Ordering of the
    // upload vs compute comes from the semaphore; memory visibility from
    // the cross-queue barrier the caller records at batch_begin.
    std::vector<VkSemaphore> waits;
    waits.reserve(wait_semaphores.size());
    for (int64_t c : wait_semaphores) {
        auto it = g_sem_by_cookie.find(c);
        TORCH_CHECK(it != g_sem_by_cookie.end(),
                    "unknown upload semaphore cookie: ", c);
        waits.push_back(it->second);
    }
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT
                                      | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (!waits.empty()) {
        si.waitSemaphoreCount = (uint32_t)waits.size();
        si.pWaitSemaphores = waits.data();
        si.pWaitDstStageMask = &wait_stage;
    }
    ++g_submit_total;
    ++g_submit_batch;
    auto t0 = std::chrono::steady_clock::now();
    VKU_CHECK(vkQueueSubmit(ctx.queue, 1, &si, g_batch.fence));
    // Blocking wait (not a vkGetFenceStatus spin): on a loaded box the
    // spin thread gets descheduled and scheduling latency masquerades as
    // GPU time; vkWaitForFences yields the core instead (task #12 opt).
    VKU_CHECK(vkWaitForFences(ctx.device, 1, &g_batch.fence, VK_TRUE,
                              UINT64_C(30'000'000'000)));
    VKU_CHECK(vkResetFences(ctx.device, 1, &g_batch.fence));
    auto t1 = std::chrono::steady_clock::now();
    for (auto& rb : g_batch.readbacks) {
        vku::download(ctx, rb.staging, rb.out.data_ptr<float>(), rb.bytes);
        get_pool().release_host(rb.staging);
    }
    g_batch.readbacks.clear();
    for (auto& b : g_batch.uploads) get_pool().release_host(b);
    g_batch.uploads.clear();
    // Consumed upload batches: the consumer fence completed, so the
    // queue-2 copies are done; recycle their staging + semaphores.
    for (int64_t c : wait_semaphores) {
        auto it = g_sem_by_cookie.find(c);
        if (it == g_sem_by_cookie.end()) continue;  // defensive
        VkSemaphore sem = it->second;
        g_sem_by_cookie.erase(it);
        auto u = g_uploads_inflight.begin();
        for (; u != g_uploads_inflight.end(); ++u)
            if (u->sem == sem) break;
        if (u != g_uploads_inflight.end()) {
            for (auto& b : u->staging) get_pool().release_host(b);
            g_uploads_inflight.erase(u);
        }
        sem_release(sem);
    }
    g_batch.active = false;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void linear_bf16_resident_io_vulkan(
    int64_t x_handle, int64_t h_w, int64_t h_b, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, bool use_bias) {
    auto& pipe=get_pipeline("linear_bf16_resident_f32.comp.spv",4,16);
    struct PC{uint32_t T,N,K,use_bias;}pc{(uint32_t)T,(uint32_t)N,(uint32_t)K,use_bias?1u:0u};
    VkBuffer w=resident_buf(h_w),b=use_bias?resident_buf(h_b):w;
    run_kernel(pipe,{w,b,resident_buf(x_handle),resident_buf(y_handle)},&pc,(uint32_t)N,(uint32_t)T,1);
}

void linear_fp16_resident_io_vulkan(
    int64_t x_handle, int64_t h_w, int64_t h_b, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, bool use_bias) {
    auto& pipe=get_pipeline("linear_fp16_resident_f32.comp.spv",4,16);
    struct PC{uint32_t T,N,K,use_bias;}pc{(uint32_t)T,(uint32_t)N,(uint32_t)K,use_bias?1u:0u};
    VkBuffer w=resident_buf(h_w),b=use_bias?resident_buf(h_b):w;
    run_kernel(pipe,{w,b,resident_buf(x_handle),resident_buf(y_handle)},&pc,(uint32_t)N,(uint32_t)T,1);
}

void linear_resident_io_vulkan(
    int64_t x_handle, int64_t h_w, int64_t h_b, int64_t y_handle,
    int64_t T, int64_t N, int64_t K, bool use_bias) {
    const size_t x_bytes = (size_t)T * K * sizeof(float);
    const size_t y_bytes = (size_t)T * N * sizeof(float);
    TORCH_CHECK(g_resident.at(x_handle).size >= x_bytes, "resident x too small");
    TORCH_CHECK(g_resident.at(y_handle).size >= y_bytes, "resident y too small");
    TORCH_CHECK(g_resident.at(h_w).size >= (size_t)N*K*sizeof(float),
                "resident weight too small");
    auto& pipe = get_pipeline("linear_fp32_resident_f32.comp.spv", 4, 16);
    struct PC { uint32_t T, N, K, use_bias; } pc = {
        (uint32_t)T, (uint32_t)N, (uint32_t)K,
        use_bias ? 1u : 0u
    };
    VkBuffer weight = resident_buf(h_w);
    VkBuffer bias = use_bias ? resident_buf(h_b) : weight;
    run_kernel(pipe, {weight, bias, resident_buf(x_handle),
                      resident_buf(y_handle)},
               &pc, (uint32_t)N, (uint32_t)T, 1);
}

torch::Tensor linear_resident_vulkan(
    torch::Tensor x,           // [T, K] fp32
    int64_t h_w,               // W handle [N, K] fp32
    int64_t h_b,               // B handle [N] fp32 (may be 0 if no bias)
    int64_t N, int64_t K, bool use_bias
) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x fp32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "x must be [T, K]");
    const int64_t T = x.size(0);

    auto xc = x.contiguous();
    auto y  = torch::empty({T, N}, x.options());

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("linear_fp32_resident_f32.comp.spv", 4, 12);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host((size_t)T * K * sizeof(float)));  // x
    cb.bufs.push_back(get_pool().acquire_host((size_t)T * N * sizeof(float)));  // y
    vku::upload(ctx, cb.bufs[0], xc.data_ptr<float>(), (size_t)T * K * sizeof(float));

    struct PC {
        uint32_t T; uint32_t N; uint32_t K; uint32_t use_bias;
    } pc = { (uint32_t)T, (uint32_t)N, (uint32_t)K, use_bias ? 1u : 0u };

    // Bindings: 0=W (resident), 1=B (resident or ignored), 2=x, 3=y
    VkBuffer w_buf = resident_buf(h_w);
    VkBuffer b_buf = use_bias ? resident_buf(h_b) : w_buf; // reuse if unused
    std::vector<VkBuffer> vbufs = { w_buf, b_buf, cb.bufs[0].buf, cb.bufs[1].buf };
    run_kernel(pipe, vbufs, &pc, (uint32_t)N, (uint32_t)T, 1);

    vku::download(ctx, cb.bufs[1], y.data_ptr<float>(), (size_t)T * N * sizeof(float));
    return y;
}

// ============================================================
// Op 6c: same MoE MLP but runs the kernel N times back-to-back in one
// call.  Used to measure per-call Python<->C++ overhead vs Vulkan
// overhead: if N calls in one Python transition are much faster than N
// separate Python transitions, the bottleneck is language boundary.
// ============================================================
void moe_mlp_gpt_oss_resident_bench_vulkan(
    torch::Tensor x, torch::Tensor indices, torch::Tensor weights,
    int64_t h_gu_blocks, int64_t h_gu_scales, int64_t h_gu_bias,
    int64_t h_d_blocks,  int64_t h_d_scales,  int64_t h_d_bias,
    int64_t E, int64_t D, int64_t Dff, int64_t N
) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x fp32");
    const int64_t T = x.size(0);
    const int64_t K = indices.size(1);

    auto x_c   = x.contiguous();
    auto idx32 = indices.contiguous().to(torch::kInt32);
    auto w_c   = weights.contiguous();

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("moe_mlp_mxfp4_gpt_oss_f32.comp.spv", 10, 20);

    const size_t sz_x   = (size_t)T * D * sizeof(float);
    const size_t sz_idx = (size_t)T * K * sizeof(uint32_t);
    const size_t sz_w   = (size_t)T * K * sizeof(float);
    const size_t sz_y   = sz_x;

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host(sz_x));
    cb.bufs.push_back(get_pool().acquire_host(sz_idx));
    cb.bufs.push_back(get_pool().acquire_host(sz_w));
    cb.bufs.push_back(get_pool().acquire_host(sz_y));

    vku::upload(ctx, cb.bufs[0], x_c.data_ptr<float>(), sz_x);
    vku::upload(ctx, cb.bufs[1], idx32.data_ptr<int32_t>(), sz_idx);
    vku::upload(ctx, cb.bufs[2], w_c.data_ptr<float>(), sz_w);

    struct PC { uint32_t T; uint32_t D; uint32_t Dff; uint32_t E; uint32_t K; } pc =
        { (uint32_t)T, (uint32_t)D, (uint32_t)Dff, (uint32_t)E, (uint32_t)K };

    std::vector<VkBuffer> vbufs = {
        resident_buf(h_gu_blocks), resident_buf(h_gu_scales), resident_buf(h_gu_bias),
        resident_buf(h_d_blocks),  resident_buf(h_d_scales),  resident_buf(h_d_bias),
        cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf,
    };
    for (int64_t i = 0; i < N; ++i) {
        run_kernel(pipe, vbufs, &pc, (uint32_t)T, 1, 1);
    }
    // Discard output - this is a benchmark.
}

// ============================================================
// Op 5: SwiGLU (elementwise: silu(gate) * up).
// ============================================================
torch::Tensor swiglu_vulkan(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.dtype() == torch::kFloat32 && up.dtype() == torch::kFloat32,
                "swiglu: fp32 only");
    TORCH_CHECK(gate.sizes() == up.sizes(), "swiglu: shapes must match");
    auto g_c = gate.contiguous(), u_c = up.contiguous();
    auto y = torch::empty_like(g_c);
    const int64_t N = g_c.numel();

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("swiglu_f32.comp.spv", 3, 4);

    const size_t bytes = (size_t)N * sizeof(float);
    CallBuffers cb;
    for (int i = 0; i < 3; ++i)
        cb.bufs.push_back(get_pool().acquire_host( bytes));
    vku::upload(ctx, cb.bufs[0], g_c.data_ptr<float>(), bytes);
    vku::upload(ctx, cb.bufs[1], u_c.data_ptr<float>(), bytes);

    uint32_t N_pc = (uint32_t)N;
    const uint32_t WG = 256;
    uint32_t groups = (N_pc + WG - 1) / WG;
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf}, &N_pc, groups, 1, 1);
    vku::download(ctx, cb.bufs[2], y.data_ptr<float>(), bytes);
    return y;
}

// ============================================================
// Op 6: MoE router - top-K softmax + renormalized weights.
// ============================================================
std::tuple<torch::Tensor, torch::Tensor>
moe_router_vulkan(torch::Tensor logits, int64_t K) {
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "router: fp32 only");
    TORCH_CHECK(logits.dim() == 2, "router: logits must be [T, E]");
    const int64_t T = logits.size(0), E = logits.size(1);
    auto l_c = logits.contiguous();

    // Output: indices [T, K] int32 (kept as int32 to match uint32 shader output).
    auto indices = torch::empty({T, K}, torch::TensorOptions().dtype(torch::kInt32));
    auto weights = torch::empty({T, K}, torch::TensorOptions().dtype(torch::kFloat32));

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("moe_router_f32.comp.spv", 3, 8);

    const size_t l_bytes = (size_t)T * E * sizeof(float);
    const size_t i_bytes = (size_t)T * K * sizeof(uint32_t);
    const size_t w_bytes = (size_t)T * K * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( l_bytes));
    cb.bufs.push_back(get_pool().acquire_host( i_bytes));
    cb.bufs.push_back(get_pool().acquire_host( w_bytes));
    vku::upload(ctx, cb.bufs[0], l_c.data_ptr<float>(), l_bytes);

    struct PC { uint32_t E; uint32_t K; } pc = { (uint32_t)E, (uint32_t)K };
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf}, &pc,
               (uint32_t)T, 1, 1);
    vku::download(ctx, cb.bufs[1], indices.data_ptr<int32_t>(), i_bytes);
    vku::download(ctx, cb.bufs[2], weights.data_ptr<float>(),   w_bytes);
    return {indices.to(torch::kInt64), weights};
}

// ============================================================
// Op 7: MoE MLP - fused per-token routed SwiGLU MLP.
// ============================================================
static torch::Tensor moe_mlp_impl(torch::Tensor x, torch::Tensor indices,
                                   torch::Tensor weights,
                                   torch::Tensor W_gate, torch::Tensor W_up,
                                   torch::Tensor W_down,
                                   const char* shader_name) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "moe_mlp: x must be float32");
    TORCH_CHECK(W_gate.dim() == 3 && W_up.dim() == 3 && W_down.dim() == 3,
                "moe_mlp: W_gate/W_up/W_down must be 3D [E, ..., ...]");

    const int64_t T = x.size(0), D = x.size(1);
    const int64_t K = indices.size(1);
    const int64_t E = W_gate.size(0), Dff = W_gate.size(1);
    TORCH_CHECK(W_gate.size(2) == D && W_up.size(2) == D,
                "moe_mlp: W_gate/W_up must be [E, Dff, D]");
    TORCH_CHECK(W_down.size(1) == D && W_down.size(2) == Dff,
                "moe_mlp: W_down must be [E, D, Dff]");

    auto x_c = x.contiguous();
    auto w_c = weights.contiguous();
    auto wg  = W_gate.contiguous(), wu = W_up.contiguous(), wd = W_down.contiguous();
    auto idx32 = indices.contiguous().to(torch::kInt32);
    auto y = torch::empty({T, D}, x.options());

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline(shader_name, 7, 20);

    const size_t x_bytes  = (size_t)T * D * sizeof(float);
    const size_t i_bytes  = (size_t)T * K * sizeof(uint32_t);
    const size_t w_bytes  = (size_t)T * K * sizeof(float);
    const size_t wg_bytes = (size_t)E * Dff * D * sizeof(float);
    const size_t wd_bytes = (size_t)E * D * Dff * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( x_bytes));
    cb.bufs.push_back(get_pool().acquire_host( i_bytes));
    cb.bufs.push_back(get_pool().acquire_host( w_bytes));
    cb.bufs.push_back(get_pool().acquire_host( wg_bytes));
    cb.bufs.push_back(get_pool().acquire_host( wg_bytes));
    cb.bufs.push_back(get_pool().acquire_host( wd_bytes));
    cb.bufs.push_back(get_pool().acquire_host( x_bytes));

    vku::upload(ctx, cb.bufs[0], x_c.data_ptr<float>(),   x_bytes);
    vku::upload(ctx, cb.bufs[1], idx32.data_ptr<int32_t>(), i_bytes);
    vku::upload(ctx, cb.bufs[2], w_c.data_ptr<float>(),   w_bytes);
    vku::upload(ctx, cb.bufs[3], wg.data_ptr<float>(),    wg_bytes);
    vku::upload(ctx, cb.bufs[4], wu.data_ptr<float>(),    wg_bytes);
    vku::upload(ctx, cb.bufs[5], wd.data_ptr<float>(),    wd_bytes);

    struct PC { uint32_t T; uint32_t D; uint32_t Dff; uint32_t E; uint32_t K; } pc =
        { (uint32_t)T, (uint32_t)D, (uint32_t)Dff, (uint32_t)E, (uint32_t)K };
    std::vector<VkBuffer> vbufs;
    for (auto& b : cb.bufs) vbufs.push_back(b.buf);
    run_kernel(pipe, vbufs, &pc, (uint32_t)T, 1, 1);

    vku::download(ctx, cb.bufs[6], y.data_ptr<float>(), x_bytes);
    return y;
}

// Small-shape MoE MLP (D <= 256, Dff <= 512).
torch::Tensor moe_mlp_vulkan(torch::Tensor x, torch::Tensor indices,
                              torch::Tensor weights,
                              torch::Tensor W_gate, torch::Tensor W_up,
                              torch::Tensor W_down) {
    return moe_mlp_impl(x, indices, weights, W_gate, W_up, W_down,
                        "moe_mlp_f32.comp.spv");
}

// Large-shape MoE MLP (D <= 4096, Dff up to any). Handles real Phi-3.5-MoE
// shape (D=4096, Dff=6400) via Dff blocking. See moe_mlp_lg_f32.comp.
torch::Tensor moe_mlp_lg_vulkan(torch::Tensor x, torch::Tensor indices,
                                 torch::Tensor weights,
                                 torch::Tensor W_gate, torch::Tensor W_up,
                                 torch::Tensor W_down) {
    return moe_mlp_impl(x, indices, weights, W_gate, W_up, W_down,
                        "moe_mlp_lg_f32.comp.spv");
}

// ============================================================
// Op: MXFP4 matvec - y[j] = sum_i W[j, i] * x[i]
// W stored as MXFP4 (blocks uint8 + scales uint8, block_size=32).
// Foundation kernel for gpt-oss MoE experts.
// ============================================================
torch::Tensor mxfp4_matvec_vulkan(torch::Tensor blocks, torch::Tensor scales,
                                   torch::Tensor x) {
    TORCH_CHECK(blocks.dtype() == torch::kUInt8, "blocks must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kUInt8, "scales must be uint8");
    TORCH_CHECK(x.dtype()      == torch::kFloat32, "x must be float32");
    TORCH_CHECK(blocks.dim() == 3 && blocks.size(2) == 16,
                "blocks must be [M, N_blocks, 16]");
    TORCH_CHECK(scales.dim() == 2, "scales must be [M, N_blocks]");
    TORCH_CHECK(x.dim() == 1, "x must be 1-D");

    const int64_t M  = blocks.size(0);
    const int64_t NB = blocks.size(1);
    const int64_t K  = NB * 32;
    TORCH_CHECK(scales.size(0) == M && scales.size(1) == NB, "scales shape mismatch");
    TORCH_CHECK(x.numel() == K, "x length must equal N_blocks * 32");

    auto blocks_c = blocks.contiguous();
    auto scales_c = scales.contiguous();
    auto x_c      = x.contiguous();
    auto y = torch::empty({M}, x.options());

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("mxfp4_matvec_f32.comp.spv", 4, 12);

    const size_t blocks_bytes = (size_t)M * NB * 16;
    const size_t scales_bytes = (size_t)M * NB;
    const size_t x_bytes      = (size_t)K * sizeof(float);
    const size_t y_bytes      = (size_t)M * sizeof(float);

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( blocks_bytes));
    cb.bufs.push_back(get_pool().acquire_host( scales_bytes));
    cb.bufs.push_back(get_pool().acquire_host( x_bytes));
    cb.bufs.push_back(get_pool().acquire_host( y_bytes));

    vku::upload(ctx, cb.bufs[0], blocks_c.data_ptr(), blocks_bytes);
    vku::upload(ctx, cb.bufs[1], scales_c.data_ptr(), scales_bytes);
    vku::upload(ctx, cb.bufs[2], x_c.data_ptr<float>(), x_bytes);

    struct PC { uint32_t M; uint32_t K; uint32_t N_blocks; } pc =
        { (uint32_t)M, (uint32_t)K, (uint32_t)NB };
    const uint32_t WG = 64;
    const uint32_t gx = ((uint32_t)M + WG - 1) / WG;
    run_kernel(pipe, {cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf},
               &pc, gx, 1, 1);

    vku::download(ctx, cb.bufs[3], y.data_ptr<float>(), y_bytes);
    return y;
}

// ============================================================
// Op: gpt-oss MoE MLP with MXFP4 expert weights.
// Signature matches transformers.GptOssExperts.forward semantics.
// ============================================================
torch::Tensor moe_mlp_gpt_oss_vulkan(
    torch::Tensor x,           // [T, D] float32
    torch::Tensor indices,     // [T, K] int64 or int32
    torch::Tensor weights,     // [T, K] float32
    torch::Tensor gu_blocks,   // [E, 2*Dff, NB_D, 16] uint8
    torch::Tensor gu_scales,   // [E, 2*Dff, NB_D]     uint8
    torch::Tensor gu_bias,     // [E, 2*Dff]           float32
    torch::Tensor d_blocks,    // [E, D, NB_Dff, 16]   uint8
    torch::Tensor d_scales,    // [E, D, NB_Dff]       uint8
    torch::Tensor d_bias)      // [E, D]               float32
{
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x fp32");
    TORCH_CHECK(gu_blocks.dtype() == torch::kUInt8, "gu_blocks uint8");
    TORCH_CHECK(gu_scales.dtype() == torch::kUInt8, "gu_scales uint8");
    TORCH_CHECK(gu_bias.dtype()   == torch::kFloat32, "gu_bias fp32");
    TORCH_CHECK(d_blocks.dtype()  == torch::kUInt8, "d_blocks uint8");
    TORCH_CHECK(d_scales.dtype()  == torch::kUInt8, "d_scales uint8");
    TORCH_CHECK(d_bias.dtype()    == torch::kFloat32, "d_bias fp32");

    const int64_t T = x.size(0), D = x.size(1);
    const int64_t K = indices.size(1);
    const int64_t E = gu_blocks.size(0);
    const int64_t two_Dff = gu_blocks.size(1);
    TORCH_CHECK(two_Dff % 2 == 0, "gate_up out dim must be 2*Dff");
    const int64_t Dff = two_Dff / 2;

    const int64_t NB_D   = gu_blocks.size(2);
    const int64_t NB_Dff = d_blocks.size(2);
    TORCH_CHECK(gu_blocks.size(3) == 16 && d_blocks.size(3) == 16,
                "packed block size must be 16 uint8s (block=32 FP4)");
    TORCH_CHECK(NB_D * 32 == D,     "D must be multiple of 32");
    TORCH_CHECK(NB_Dff * 32 == Dff, "Dff must be multiple of 32");
    TORCH_CHECK(d_blocks.size(1) == D,      "down_proj out dim mismatch");

    auto x_c   = x.contiguous();
    auto idx32 = indices.contiguous().to(torch::kInt32);
    auto w_c   = weights.contiguous();
    auto gu_b  = gu_blocks.contiguous();
    auto gu_s  = gu_scales.contiguous();
    auto gu_bi = gu_bias.contiguous();
    auto d_b   = d_blocks.contiguous();
    auto d_s   = d_scales.contiguous();
    auto d_bi  = d_bias.contiguous();
    auto y = torch::empty({T, D}, x.options());

    auto& ctx = get_ctx();
    auto& pipe = get_pipeline("moe_mlp_mxfp4_gpt_oss_f32.comp.spv", 10, 20);

    // Sizes in bytes.
    const size_t sz_x       = (size_t)T * D * sizeof(float);
    const size_t sz_idx     = (size_t)T * K * sizeof(uint32_t);
    const size_t sz_w       = (size_t)T * K * sizeof(float);
    const size_t sz_gu_b    = (size_t)E * two_Dff * NB_D   * 16;
    const size_t sz_gu_s    = (size_t)E * two_Dff * NB_D;
    const size_t sz_gu_bias = (size_t)E * two_Dff * sizeof(float);
    const size_t sz_d_b     = (size_t)E * D       * NB_Dff * 16;
    const size_t sz_d_s     = (size_t)E * D       * NB_Dff;
    const size_t sz_d_bias  = (size_t)E * D       * sizeof(float);
    const size_t sz_y       = sz_x;

    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( sz_gu_b));      // 0
    cb.bufs.push_back(get_pool().acquire_host( sz_gu_s));      // 1
    cb.bufs.push_back(get_pool().acquire_host( sz_gu_bias));   // 2
    cb.bufs.push_back(get_pool().acquire_host( sz_d_b));       // 3
    cb.bufs.push_back(get_pool().acquire_host( sz_d_s));       // 4
    cb.bufs.push_back(get_pool().acquire_host( sz_d_bias));    // 5
    cb.bufs.push_back(get_pool().acquire_host( sz_x));         // 6
    cb.bufs.push_back(get_pool().acquire_host( sz_idx));       // 7
    cb.bufs.push_back(get_pool().acquire_host( sz_w));         // 8
    cb.bufs.push_back(get_pool().acquire_host( sz_y));         // 9

    vku::upload(ctx, cb.bufs[0], gu_b.data_ptr(),   sz_gu_b);
    vku::upload(ctx, cb.bufs[1], gu_s.data_ptr(),   sz_gu_s);
    vku::upload(ctx, cb.bufs[2], gu_bi.data_ptr<float>(), sz_gu_bias);
    vku::upload(ctx, cb.bufs[3], d_b.data_ptr(),    sz_d_b);
    vku::upload(ctx, cb.bufs[4], d_s.data_ptr(),    sz_d_s);
    vku::upload(ctx, cb.bufs[5], d_bi.data_ptr<float>(), sz_d_bias);
    vku::upload(ctx, cb.bufs[6], x_c.data_ptr<float>(),  sz_x);
    vku::upload(ctx, cb.bufs[7], idx32.data_ptr<int32_t>(), sz_idx);
    vku::upload(ctx, cb.bufs[8], w_c.data_ptr<float>(),  sz_w);

    struct PC { uint32_t T; uint32_t D; uint32_t Dff; uint32_t E; uint32_t K; } pc =
        { (uint32_t)T, (uint32_t)D, (uint32_t)Dff, (uint32_t)E, (uint32_t)K };
    std::vector<VkBuffer> vbufs;
    for (auto& b : cb.bufs) vbufs.push_back(b.buf);
    run_kernel(pipe, vbufs, &pc, (uint32_t)T, 1, 1);

    vku::download(ctx, cb.bufs[9], y.data_ptr<float>(), sz_y);
    return y;
}


// ============================================================
// Op: gpt-oss MoE MLP with RESIDENT expert weights.
// The 6 MoE weight tensors are pre-uploaded once via upload_resident_vulkan
// and referenced by handles. This eliminates ~424 MB of per-call PCIe
// traffic per layer, which is the biggest per-layer cost currently.
//
// x, indices, weights, y still go through the normal per-call upload path.
// ============================================================
void moe_mlp_gpt_oss_twostage_io_vulkan(
    int64_t x_handle, int64_t indices_handle, int64_t weights_handle,
    int64_t hidden_handle, int64_t output_handle,
    int64_t h_gu_blocks, int64_t h_gu_scales, int64_t h_gu_bias,
    int64_t h_d_blocks, int64_t h_d_scales, int64_t h_d_bias,
    int64_t E, int64_t D, int64_t Dff, int64_t T, int64_t K) {
    const uint64_t stage1_count = (uint64_t)T*K*((Dff+63)/64);
    const uint64_t stage2_count = (uint64_t)T*((D+31)/32);
    TORCH_CHECK(stage1_count <= 65535 && stage2_count <= 65535,
                "resident MoE dispatch exceeds Vulkan X dimension; "
                "use transient/prefill path or add 2D dispatch");
    auto& ctx=get_ctx();
    auto& stage1=get_pipeline(
        "moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv",6,20);
    auto& stage2=get_pipeline(
        "moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv",7,20);
    VkDescriptorSet d1=make_descriptor_set(stage1,{
        resident_buf(h_gu_blocks),resident_buf(h_gu_scales),
        resident_buf(h_gu_bias),resident_buf(x_handle),
        resident_buf(indices_handle),resident_buf(hidden_handle)});
    VkDescriptorSet d2=make_descriptor_set(stage2,{
        resident_buf(h_d_blocks),resident_buf(h_d_scales),
        resident_buf(h_d_bias),resident_buf(hidden_handle),
        resident_buf(indices_handle),resident_buf(weights_handle),
        resident_buf(output_handle)});
    struct PC{uint32_t T,D,Dff,E,K;}pc{
        (uint32_t)T,(uint32_t)D,(uint32_t)Dff,(uint32_t)E,(uint32_t)K};
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,stage1.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,
            stage1.pipeline_layout,0,1,&d1,0,nullptr);
        vkCmdPushConstants(cmd,stage1.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,0,20,&pc);
        vkCmdDispatch(cmd,(uint32_t)stage1_count,1,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,
            0,nullptr,0,nullptr);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,stage2.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,
            stage2.pipeline_layout,0,1,&d2,0,nullptr);
        vkCmdPushConstants(cmd,stage2.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,0,20,&pc);
        vkCmdDispatch(cmd,(uint32_t)stage2_count,1,1);
    });
}

void topk_moe_residual_resident_vulkan(
    int64_t logits_handle, int64_t indices_handle, int64_t weights_handle,
    int64_t x_handle, int64_t hidden_handle, int64_t moe_output_handle,
    int64_t residual_handle, int64_t final_output_handle,
    int64_t h_gu_blocks, int64_t h_gu_scales, int64_t h_gu_bias,
    int64_t h_d_blocks, int64_t h_d_scales, int64_t h_d_bias,
    int64_t E, int64_t D, int64_t Dff) {
    auto& ctx=get_ctx();auto& topk=get_pipeline("topk_softmax_128_f32.comp.spv",3,4);auto& s1=get_pipeline("moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv",6,20);auto& s2=get_pipeline("moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv",7,20);auto& add=get_pipeline("vector_add.comp.spv",3,4);
    VkDescriptorSet dt=make_descriptor_set(topk,{resident_buf(logits_handle),resident_buf(indices_handle),resident_buf(weights_handle)});VkDescriptorSet d1=make_descriptor_set(s1,{resident_buf(h_gu_blocks),resident_buf(h_gu_scales),resident_buf(h_gu_bias),resident_buf(x_handle),resident_buf(indices_handle),resident_buf(hidden_handle)});VkDescriptorSet d2=make_descriptor_set(s2,{resident_buf(h_d_blocks),resident_buf(h_d_scales),resident_buf(h_d_bias),resident_buf(hidden_handle),resident_buf(indices_handle),resident_buf(weights_handle),resident_buf(moe_output_handle)});VkDescriptorSet da=make_descriptor_set(add,{resident_buf(residual_handle),resident_buf(moe_output_handle),resident_buf(final_output_handle)});
    struct TPC{uint32_t rows;}tpc{1u};struct PC{uint32_t T,D,Dff,E,K;}pc{1u,(uint32_t)D,(uint32_t)Dff,(uint32_t)E,4u};struct APC{uint32_t n;}apc{(uint32_t)D};
    vku::submit_and_wait(ctx,[&](VkCommandBuffer cmd){auto barrier=[&](){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,topk.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,topk.pipeline_layout,0,1,&dt,0,nullptr);vkCmdPushConstants(cmd,topk.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&tpc);vkCmdDispatch(cmd,1,1,1);barrier();vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s1.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s1.pipeline_layout,0,1,&d1,0,nullptr);vkCmdPushConstants(cmd,s1.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&pc);vkCmdDispatch(cmd,(uint32_t)(4*((Dff+63)/64)),1,1);barrier();vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s2.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s2.pipeline_layout,0,1,&d2,0,nullptr);vkCmdPushConstants(cmd,s2.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&pc);vkCmdDispatch(cmd,(uint32_t)((D+31)/32),1,1);barrier();vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,add.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,add.pipeline_layout,0,1,&da,0,nullptr);vkCmdPushConstants(cmd,add.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&apc);vkCmdDispatch(cmd,(uint32_t)((D+255)/256),1,1);});
}

torch::Tensor moe_mlp_gpt_oss_twostage_vulkan(
    torch::Tensor x, torch::Tensor indices, torch::Tensor weights,
    int64_t h_gu_blocks, int64_t h_gu_scales, int64_t h_gu_bias,
    int64_t h_d_blocks, int64_t h_d_scales, int64_t h_d_bias,
    int64_t E, int64_t D, int64_t Dff) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x fp32");
    const int64_t T = x.size(0);
    const int64_t K = indices.size(1);
    auto xc = x.contiguous();
    auto idx = indices.contiguous().to(torch::kInt32);
    auto routing = weights.contiguous();
    auto output = torch::empty({T, D}, x.options());
    auto& ctx = get_ctx();
    auto& stage1 = get_pipeline(
        "moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv", 6, 20);
    auto& stage2 = get_pipeline(
        "moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv", 7, 20);
    const size_t x_bytes = (size_t)T * D * sizeof(float);
    const size_t idx_bytes = (size_t)T * K * sizeof(uint32_t);
    const size_t weight_bytes = (size_t)T * K * sizeof(float);
    const size_t hidden_bytes = (size_t)T * K * Dff * sizeof(float);
    const size_t output_bytes = x_bytes;
    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host(x_bytes));
    cb.bufs.push_back(get_pool().acquire_host(idx_bytes));
    cb.bufs.push_back(get_pool().acquire_host(weight_bytes));
    cb.bufs.push_back(get_pool().acquire_host(hidden_bytes));
    cb.bufs.push_back(get_pool().acquire_host(output_bytes));
    vku::upload(ctx, cb.bufs[0], xc.data_ptr<float>(), x_bytes);
    vku::upload(ctx, cb.bufs[1], idx.data_ptr<int32_t>(), idx_bytes);
    vku::upload(ctx, cb.bufs[2], routing.data_ptr<float>(), weight_bytes);
    std::vector<VkBuffer> b1 = {
        resident_buf(h_gu_blocks), resident_buf(h_gu_scales),
        resident_buf(h_gu_bias), cb.bufs[0].buf, cb.bufs[1].buf,
        cb.bufs[3].buf};
    std::vector<VkBuffer> b2 = {
        resident_buf(h_d_blocks), resident_buf(h_d_scales),
        resident_buf(h_d_bias), cb.bufs[3].buf, cb.bufs[1].buf,
        cb.bufs[2].buf, cb.bufs[4].buf};
    VkDescriptorSet d1 = make_descriptor_set(stage1, b1);
    VkDescriptorSet d2 = make_descriptor_set(stage2, b2);
    struct PC { uint32_t T, D, Dff, E, K; } pc = {
        (uint32_t)T, (uint32_t)D, (uint32_t)Dff,
        (uint32_t)E, (uint32_t)K};
    const uint32_t stage1_groups =
        (uint32_t)(T * K * ((Dff + 63) / 64));
    const uint32_t stage2_groups =
        (uint32_t)(T * ((D + 31) / 32));
    vku::submit_and_wait(ctx, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          stage1.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                stage1.pipeline_layout, 0, 1, &d1, 0, nullptr);
        vkCmdPushConstants(cmd, stage1.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &pc);
        vkCmdDispatch(cmd, stage1_groups, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &barrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          stage2.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                stage2.pipeline_layout, 0, 1, &d2, 0, nullptr);
        vkCmdPushConstants(cmd, stage2.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &pc);
        vkCmdDispatch(cmd, stage2_groups, 1, 1);
    });
    vku::download(ctx, cb.bufs[4], output.data_ptr<float>(), output_bytes);
    return output;
}

torch::Tensor moe_mlp_gpt_oss_resident_vulkan(
    torch::Tensor x,           // [T, D] float32
    torch::Tensor indices,     // [T, K] int32 or int64
    torch::Tensor weights,     // [T, K] float32
    int64_t h_gu_blocks, int64_t h_gu_scales, int64_t h_gu_bias,
    int64_t h_d_blocks,  int64_t h_d_scales,  int64_t h_d_bias,
    int64_t E, int64_t D, int64_t Dff)
{
    TORCH_CHECK(x.dtype()      == torch::kFloat32, "x fp32");
    const int64_t T = x.size(0);
    TORCH_CHECK(x.size(1) == D, "x hidden dim mismatch");
    const int64_t K = indices.size(1);

    auto x_c   = x.contiguous();
    auto idx32 = indices.contiguous().to(torch::kInt32);
    auto w_c   = weights.contiguous();
    auto y = torch::empty({T, D}, x.options());

    auto& ctx = get_ctx();
    // Route K=4 to the parallel-experts kernel; else fall back.
    const char* spv = (indices.size(1) == 4)
        ? "moe_mlp_mxfp4_gpt_oss_par_f32.comp.spv"
        : "moe_mlp_mxfp4_gpt_oss_f32.comp.spv";
    auto& pipe = get_pipeline(spv, 10, 20);

    const size_t sz_x       = (size_t)T * D * sizeof(float);
    const size_t sz_idx     = (size_t)T * K * sizeof(uint32_t);
    const size_t sz_w       = (size_t)T * K * sizeof(float);
    const size_t sz_y       = sz_x;

    // Only 4 transient (host-allocated) buffers: x, indices, weights, y.
    CallBuffers cb;
    cb.bufs.push_back(get_pool().acquire_host( sz_x));   // 6 (x)
    cb.bufs.push_back(get_pool().acquire_host( sz_idx)); // 7 (indices)
    cb.bufs.push_back(get_pool().acquire_host( sz_w));   // 8 (weights)
    cb.bufs.push_back(get_pool().acquire_host( sz_y));   // 9 (y, output)

    vku::upload(ctx, cb.bufs[0], x_c.data_ptr<float>(),  sz_x);
    vku::upload(ctx, cb.bufs[1], idx32.data_ptr<int32_t>(), sz_idx);
    vku::upload(ctx, cb.bufs[2], w_c.data_ptr<float>(),  sz_w);

    // Assemble the 10-buffer descriptor set: 6 resident MoE weights first,
    // then 4 transient.
    std::vector<VkBuffer> vbufs = {
        resident_buf(h_gu_blocks), resident_buf(h_gu_scales),
        resident_buf(h_gu_bias),
        resident_buf(h_d_blocks),  resident_buf(h_d_scales),
        resident_buf(h_d_bias),
        cb.bufs[0].buf, cb.bufs[1].buf, cb.bufs[2].buf, cb.bufs[3].buf,
    };

    struct PC { uint32_t T; uint32_t D; uint32_t Dff; uint32_t E; uint32_t K; } pc =
        { (uint32_t)T, (uint32_t)D, (uint32_t)Dff, (uint32_t)E, (uint32_t)K };
    run_kernel(pipe, vbufs, &pc, (uint32_t)T, 1, 1);

    vku::download(ctx, cb.bufs[3], y.data_ptr<float>(), sz_y);
    return y;
}

// ============================================================
// pybind11 module.
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FreeToken Vulkan compute backend - full op set.";
    m.def("rmsnorm", &rmsnorm_vulkan, "Rowwise RMSNorm on Vulkan.",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6);
    m.def("matmul",  &matmul_vulkan,  "2D matmul on Vulkan (reg_tiled f32).",
          py::arg("A"), py::arg("B"));
    m.def("softmax", &softmax_vulkan, "Rowwise softmax over last dim.",
          py::arg("x"));
    m.def("flash_attention", &flash_attention_vulkan,
          "Single-head FlashAttention v1 on Vulkan.",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));
    m.def("flash_attention_mh", &flash_attention_mh_vulkan,
          "Multi-head FlashAttention v1 with optional causal masking. "
          "Q, K, V, O are [B, H, S, D] float32.",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"),
          py::arg("causal") = false);
    m.def("flash_attention_gpt_oss", &flash_attention_gpt_oss_vulkan,
          "gpt-oss FlashAttention: MH + GQA + causal + optional sliding "
          "window + optional attention sinks. Q [B, H_q, S, D], K/V "
          "[B, H_kv, S, D], sinks [H_q].",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("sinks"),
          py::arg("scale"),
          py::arg("sliding_window") = 0,
          py::arg("use_sinks") = true);
    m.def("flash_attention_gpt_oss_kv", &flash_attention_gpt_oss_kv_vulkan,
          "gpt-oss FlashAttention with KV cache support. Q [B, H_q, S_q, D], "
          "K/V [B, H_kv, S_kv, D]. past_len = absolute position of Q[0].",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("sinks"),
          py::arg("scale"), py::arg("past_len"),
          py::arg("sliding_window") = 0,
          py::arg("use_sinks") = true);
    m.def("rope_partial", &rope_partial_vulkan,
          "Partial rotary position embedding. Rotates the first `rotary_dim` "
          "of each head; leaves the remaining head_dim - rotary_dim unchanged. "
          "cos, sin: [S, rotary_dim] fp32 (precomputed with YARN/NTK on host). "
          "x: [B, H, S, D] fp32.",
          py::arg("x"), py::arg("cos"), py::arg("sin"), py::arg("rotary_dim"));
    m.def("swiglu", &swiglu_vulkan,
          "SwiGLU elementwise: silu(gate) * up.",
          py::arg("gate"), py::arg("up"));
    m.def("moe_router", &moe_router_vulkan,
          "Top-K softmax router. Returns (indices int64, weights float32).",
          py::arg("logits"), py::arg("K"));
    m.def("moe_mlp", &moe_mlp_vulkan,
          "Fused MoE MLP forward with per-token top-K routing + SwiGLU. "
          "Small shapes: D <= 256, Dff <= 512.",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("W_gate"), py::arg("W_up"), py::arg("W_down"));
    m.def("moe_mlp_lg", &moe_mlp_lg_vulkan,
          "Same as moe_mlp but Dff-blocked for large shapes. "
          "Handles D <= 4096 (e.g. Phi-3.5-MoE at D=4096 Dff=6400).",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("W_gate"), py::arg("W_up"), py::arg("W_down"));
    m.def("mxfp4_matvec", &mxfp4_matvec_vulkan,
          "MXFP4-weight x FP32-activation matvec: y[j] = <W[j, :], x>. "
          "blocks [M, NB, 16] uint8, scales [M, NB] uint8, x [K=NB*32] fp32, "
          "returns y [M] fp32.",
          py::arg("blocks"), py::arg("scales"), py::arg("x"));
    m.def("moe_mlp_gpt_oss", &moe_mlp_gpt_oss_vulkan,
          "gpt-oss MoE MLP with MXFP4 expert weights, interleaved gate/up, "
          "and the (up+1)*gate*sigmoid(gate*1.702) activation.",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("gu_blocks"), py::arg("gu_scales"), py::arg("gu_bias"),
          py::arg("d_blocks"),  py::arg("d_scales"),  py::arg("d_bias"));
    // Persistent VRAM residency for MoE weights.
    m.def("upload_resident", &upload_resident_vulkan,
          "Upload a CPU tensor to VRAM as a persistent buffer. Returns an "
          "int64 handle usable in the *_resident variants of ops.",
          py::arg("tensor"));
    m.def("allocate_resident", &allocate_resident_vulkan,
          "Allocate an uninitialized device-local resident buffer.",
          py::arg("bytes"));
    m.def("download_resident", &download_resident_vulkan,
          "Download a resident FP32 buffer into a CPU tensor.",
          py::arg("handle"), py::arg("shape"));
    m.def("download_resident_i32", &download_resident_i32_vulkan,
          "Download a resident int32 buffer into a CPU tensor.",
          py::arg("handle"), py::arg("shape"));
    m.def("topk_resident", &topk_resident_vulkan,
          "Top-4 plus softmax for resident 128-expert router logits.",
          py::arg("logits_handle"), py::arg("indices_handle"),
          py::arg("weights_handle"), py::arg("rows"));
    m.def("rmsnorm_resident", &rmsnorm_resident_vulkan,
          "RMSNorm from resident buffers into a resident output buffer.",
          py::arg("x_handle"), py::arg("weight_handle"),
          py::arg("y_handle"), py::arg("rows"), py::arg("hidden"),
          py::arg("eps") = 1e-5);
    m.def("transpose_resident", &transpose_resident_vulkan,
          "Transpose resident FP32 [S,H,D] and [H,S,D] layouts.",
          py::arg("src_handle"), py::arg("dst_handle"),
          py::arg("S"), py::arg("H"), py::arg("D"),
          py::arg("inverse") = false);
    m.def("swiglu_resident", &swiglu_resident_vulkan,
          "Resident FP32 SwiGLU activation.",
          py::arg("gate_handle"), py::arg("up_handle"),
          py::arg("out_handle"), py::arg("elements"));
    m.def("add_resident", &add_resident_vulkan,
          "Elementwise add of resident FP32 buffers.",
          py::arg("a_handle"), py::arg("b_handle"),
          py::arg("out_handle"), py::arg("elements"));
    m.def("update_resident", &update_resident_vulkan,
          "Update a resident buffer from one CPU tensor.",
          py::arg("handle"), py::arg("tensor"), py::arg("offset") = 0);
    m.def("upload_resident_batch", &upload_resident_batch_vulkan,
          "Upload CPU tensors into resident-buffer subranges in one submit.",
          py::arg("handles"), py::arg("tensors"), py::arg("offsets"));
    m.def("free_resident", &free_resident_vulkan,
          "Free the persistent buffer identified by handle. No-op if unknown.",
          py::arg("handle"));
    m.def("resident_bytes_total", &resident_bytes_total_vulkan,
          "Total bytes currently held in resident VRAM buffers across all "
          "outstanding handles.");
    m.def("desc_cache_stats", []() -> py::tuple {
        return py::make_tuple((uint64_t)g_desc_cache.size(),
                              (uint64_t)g_desc_pool_used,
                              (uint64_t)g_desc_cache_pool_used,
                              (uint64_t)g_desc_hits, (uint64_t)g_desc_miss,
                              (uint64_t)g_desc_clears);
    }, "Descriptor cache entries, transient pool used, cache pool used, "
       "cache hits, cache misses, cache clears.");
    m.def("qwen3_decode_layer_awq_twostage_resident",
          &qwen3_decode_layer_awq_twostage_resident_vulkan,
          "Fused two-stage fully resident AWQ Qwen layer.");
    m.def("qwen3_decode_layer_resident",
          &qwen3_decode_layer_resident_vulkan,
          "Fused fully resident Qwen3 single-token layer.");
    m.def("decode_pre_moe_all_resident",
          &decode_pre_moe_all_resident_vulkan,
          "Fused all-resident decode segment through router logits.",
          py::arg("x"), py::arg("input_norm"), py::arg("normalized"),
          py::arg("qw"), py::arg("qb"), py::arg("q"),
          py::arg("kw"), py::arg("kb"), py::arg("k"),
          py::arg("vw"), py::arg("vb"), py::arg("v"),
          py::arg("cos"), py::arg("sin"), py::arg("qrot"),
          py::arg("krot"), py::arg("kcache"), py::arg("vcache"),
          py::arg("sinks"), py::arg("attention_out"),
          py::arg("ow"), py::arg("ob"), py::arg("projected"),
          py::arg("residual"), py::arg("post_norm_weight"),
          py::arg("post_normalized"), py::arg("router_w"),
          py::arg("router_b"), py::arg("logits"),
          py::arg("hidden"), py::arg("Hq"), py::arg("Hkv"),
          py::arg("head_dim"), py::arg("experts"),
          py::arg("position"), py::arg("capacity"),
          py::arg("sliding_window"), py::arg("eps"),
          py::arg("scale"));
    m.def("topk_moe_residual_resident",
          &topk_moe_residual_resident_vulkan,
          "Fused top-K, all-resident MoE, and final residual.",
          py::arg("logits_handle"), py::arg("indices_handle"),
          py::arg("weights_handle"), py::arg("x_handle"),
          py::arg("hidden_handle"), py::arg("moe_output_handle"),
          py::arg("residual_handle"), py::arg("final_output_handle"),
          py::arg("h_gu_blocks"), py::arg("h_gu_scales"),
          py::arg("h_gu_bias"), py::arg("h_d_blocks"),
          py::arg("h_d_scales"), py::arg("h_d_bias"),
          py::arg("E"), py::arg("D"), py::arg("Dff"));
    m.def("moe_mlp_gpt_oss_twostage_io",
          &moe_mlp_gpt_oss_twostage_io_vulkan,
          "Two-stage MoE with resident activation and control I/O.",
          py::arg("x_handle"), py::arg("indices_handle"),
          py::arg("weights_handle"), py::arg("hidden_handle"),
          py::arg("output_handle"), py::arg("h_gu_blocks"),
          py::arg("h_gu_scales"), py::arg("h_gu_bias"),
          py::arg("h_d_blocks"), py::arg("h_d_scales"),
          py::arg("h_d_bias"), py::arg("E"), py::arg("D"),
          py::arg("Dff"), py::arg("T"), py::arg("K"));
    m.def("moe_mlp_gpt_oss_twostage",
          &moe_mlp_gpt_oss_twostage_vulkan,
          "Two-stage parallel resident gpt-oss MoE pipeline.",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("h_gu_blocks"), py::arg("h_gu_scales"), py::arg("h_gu_bias"),
          py::arg("h_d_blocks"), py::arg("h_d_scales"), py::arg("h_d_bias"),
          py::arg("E"), py::arg("D"), py::arg("Dff"));
    m.def("moe_mlp_gpt_oss_resident", &moe_mlp_gpt_oss_resident_vulkan,
          "Same as moe_mlp_gpt_oss but the 6 MoE weight tensors are "
          "referenced by resident-buffer handles instead of uploaded "
          "per-call. Shape metadata (E, D, Dff) is passed as ints.",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("h_gu_blocks"), py::arg("h_gu_scales"), py::arg("h_gu_bias"),
          py::arg("h_d_blocks"),  py::arg("h_d_scales"),  py::arg("h_d_bias"),
          py::arg("E"), py::arg("D"), py::arg("Dff"));
    m.def("rmsnorm_qkv_resident", &rmsnorm_qkv_resident_vulkan,
          "Fused resident RMSNorm followed by Q, K, and V projections.",
          py::arg("x_handle"), py::arg("norm_weight_handle"),
          py::arg("normalized_handle"),
          py::arg("q_weight_handle"), py::arg("q_bias_handle"),
          py::arg("q_handle"), py::arg("k_weight_handle"),
          py::arg("k_bias_handle"), py::arg("k_handle"),
          py::arg("v_weight_handle"), py::arg("v_bias_handle"),
          py::arg("v_handle"), py::arg("rows"), py::arg("hidden"),
          py::arg("q_size"), py::arg("kv_size"),
          py::arg("eps") = 1e-5);
    m.def("rope_kv_attention_resident",
          &rope_kv_attention_resident_vulkan,
          "Resident RoPE, KV-cache append, and capacity-strided attention.",
          py::arg("q_handle"), py::arg("k_handle"), py::arg("v_handle"),
          py::arg("cos_handle"), py::arg("sin_handle"),
          py::arg("q_rot_handle"), py::arg("k_rot_handle"),
          py::arg("k_cache_handle"), py::arg("v_cache_handle"),
          py::arg("sinks_handle"), py::arg("output_handle"),
          py::arg("B"), py::arg("Hq"), py::arg("Hkv"),
          py::arg("Snew"), py::arg("D"), py::arg("start"),
          py::arg("capacity"), py::arg("sliding_window"),
          py::arg("use_sinks"), py::arg("scale"));
    m.def("oproj_router_resident", &oproj_router_resident_vulkan,
          "Fused resident O projection, residual, RMSNorm, and router.",
          py::arg("attention_handle"), py::arg("o_weight_handle"),
          py::arg("o_bias_handle"), py::arg("projected_handle"),
          py::arg("residual_handle"), py::arg("residual_out_handle"),
          py::arg("norm_weight_handle"), py::arg("normalized_handle"),
          py::arg("router_weight_handle"), py::arg("router_bias_handle"),
          py::arg("logits_handle"), py::arg("rows"), py::arg("hidden"),
          py::arg("attention_size"), py::arg("experts"),
          py::arg("eps") = 1e-5);
    m.def("linear_awq4_twostage_resident_io",
          &linear_awq4_twostage_resident_io_vulkan,
          "Two-stage FP32 linear with resident packed AWQ 4-bit weights.",
          py::arg("x_handle"), py::arg("qweight_handle"),
          py::arg("qzeros_handle"), py::arg("scales_handle"),
          py::arg("partials_handle"), py::arg("y_handle"),
          py::arg("N"), py::arg("K"), py::arg("group_size"));
    m.def("linear_awq4_resident_io", &linear_awq4_resident_io_vulkan,
          "FP32 linear output with resident packed AWQ 4-bit weights.",
          py::arg("x_handle"), py::arg("qweight_handle"),
          py::arg("qzeros_handle"), py::arg("scales_handle"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"),
          py::arg("K"), py::arg("group_size"));
    m.def("linear_ao_affine_resident_io", &linear_ao_affine_resident_io_vulkan,
          "FP32 linear output with resident TorchAO-affine weights: U32 "
          "bit-packed payload plus BF16 scales and pre-negated biases.",
          py::arg("x_handle"), py::arg("packed_handle"),
          py::arg("scales_handle"), py::arg("biases_handle"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"),
          py::arg("K"), py::arg("bits"), py::arg("group_size"));
    m.def("linear_nvfp4_resident_io", &linear_nvfp4_resident_io_vulkan,
          "FP32 linear output with resident ModelOpt NVFP4 weights: U8 "
          "E2M1 payload (low nibble first) plus E4M3 block scales and an "
          "fp32 global scale.",
          py::arg("x_handle"), py::arg("packed_handle"),
          py::arg("scales_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("gscale"), py::arg("slot0") = 0);
    m.def("linear_nvfp4_gateup_silu_io", &linear_nvfp4_gateup_silu_io_vulkan,
          "Fused gate+up with SiLU fold (task #11): stacked bundle rows n "
          "(gate) and n+N (up) with a shared fp32 gscale; writes Y = "
          "silu(gate)*up, the down-proj input, resident-to-resident.",
          py::arg("x_handle"), py::arg("packed_handle"),
          py::arg("scales_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("gscale"), py::arg("slot0") = 0);
    // ggml K-quant resident matvecs: packed GGUF bytes uploaded verbatim,
    // unpacked in-shader (no host dequant). See KQuantPC above.
    m.def("linear_kquant_resident_io", &linear_kquant_resident_io_vulkan,
          "K-quant resident matvec (qtype = q4_k|q5_k|q8_0, kind = "
          "plain|gateup|acc).",
          py::arg("qtype"), py::arg("kind"), py::arg("x_handle"),
          py::arg("packed_handle"), py::arg("y_handle"), py::arg("T"),
          py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q4k_resident_io", &linear_q4k_resident_io_vulkan,
          "FP32 linear output with resident ggml Q4_K weights unpacked "
          "in-shader (144 B per 256 weights; no host dequant).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q5k_resident_io", &linear_q5k_resident_io_vulkan,
          "FP32 linear output with resident ggml Q5_K weights unpacked "
          "in-shader (176 B per 256 weights; no host dequant).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q8_0_resident_io", &linear_q8_0_resident_io_vulkan,
          "FP32 linear output with resident ggml Q8_0 weights unpacked "
          "in-shader (34 B per 32 weights; buffer needs 4 B of slack).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q4k_gateup_silu_io", &linear_q4k_gateup_silu_io_vulkan,
          "Fused Q4_K gate+up with SiLU fold: packed bundle rows n (gate) "
          "and n+N (up); writes Y = silu(gate)*up resident-to-resident.",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q5k_gateup_silu_io", &linear_q5k_gateup_silu_io_vulkan,
          "Fused Q5_K gate+up with SiLU fold (see "
          "linear_q4k_gateup_silu_io).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q4k_acc_resident_io", &linear_q4k_acc_resident_io_vulkan,
          "Q4_K resident matvec accumulating into Y (expert down-proj).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_q5k_acc_resident_io", &linear_q5k_acc_resident_io_vulkan,
          "Q5_K resident matvec accumulating into Y (expert down-proj).",
          py::arg("x_handle"), py::arg("packed_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("linear_fp8e4m3_resident_io", &linear_fp8e4m3_resident_io_vulkan,
          "FP32 linear output with resident block-scaled FP8 E4M3 weights.",
          py::arg("x_handle"), py::arg("w_handle"),
          py::arg("scales_handle"), py::arg("y_handle"),
          py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("scale_cols"));
    m.def("linear_bf16_resident_io", &linear_bf16_resident_io_vulkan,
          "FP32 linear output with resident BF16 weights and bias.",
          py::arg("x_handle"), py::arg("w_handle"), py::arg("b_handle"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("use_bias") = true);
    // Batched resident ops (task #12): record uploads/kernels/readbacks
    // into one command buffer, submit once. Collapses the ~3200
    // submit+fence roundtrips of a decode step to ~80.
    m.def("batch_begin", &batch_begin_vulkan,
          "Open a command-buffer batch.");
    m.def("batch_barrier_stats", []() -> std::tuple<int64_t, int64_t> {
              return {int64_t(g_barrier_total), int64_t(g_barrier_batch)};
          },
          "(cumulative record barriers, barriers in open batch).");
    m.def("batch_submit_count", []() -> std::tuple<int64_t, int64_t> {
              return {int64_t(g_submit_total), int64_t(g_submit_batch)};
          },
          "(cumulative submissions, submissions in open batch).");
    m.def("batch_upload", &batch_upload_vulkan,
          "Record a host->device copy into the open batch.",
          py::arg("handle"), py::arg("tensor"), py::arg("offset") = 0);
    m.def("batch_fill", &batch_fill_vulkan,
          "Record a float fill (vkCmdFillBuffer) into the open batch.",
          py::arg("handle"), py::arg("value"), py::arg("count"));
    m.def("batch_nvfp4_io", &batch_nvfp4_io_vulkan,
          "Record an NVFP4 matvec dispatch (kind = plain|gateup|acc).",
          py::arg("kind"), py::arg("h_packed"), py::arg("h_scales"),
          py::arg("x_handle"), py::arg("y_handle"), py::arg("T"),
          py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("batch_kquant_io", &batch_kquant_io_vulkan,
          "Record a K-quant matvec dispatch (qtype = q4_k|q5_k|q8_0, kind "
          "= plain|gateup|acc; no separate scale buffer).",
          py::arg("qtype"), py::arg("kind"), py::arg("h_packed"),
          py::arg("x_handle"), py::arg("y_handle"), py::arg("T"),
          py::arg("N"), py::arg("K"), py::arg("gscale"),
          py::arg("slot0") = 0);
    m.def("batch_moe_kquant_record", &batch_moe_kquant_record_vulkan,
          "Record 8 packed K-quant gateup-silu + 8 down-accumulate "
          "dispatches in one crossing (mirror of batch_moe_record for "
          "Q4_K/Q5_K experts; caller records the shared-expert tail).",
          py::arg("h_x_fused"), py::arg("h_y_acc"), py::arg("qtype"),
          py::arg("silu_slots"), py::arg("gu_w"), py::arg("gu_scale"),
          py::arg("gu_rows"), py::arg("gu_cols"), py::arg("dn_w"),
          py::arg("dn_scale"), py::arg("dn_rows"), py::arg("dn_cols"),
          py::arg("gu_slot0"), py::arg("dn_slot0"));
    m.def("batch_bf16_io", &batch_bf16_io_vulkan,
          "Record a BF16 dense matvec dispatch into the open batch.",
          py::arg("x_handle"), py::arg("h_w"), py::arg("h_b"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("use_bias"));
    m.def("batch_fp32_io", &batch_fp32_io_vulkan,
          "Record an FP32 dense matvec dispatch into the open batch "
          "(no BF16 cast; router GEMM).",
          py::arg("x_handle"), py::arg("h_w"), py::arg("h_b"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("use_bias"));
    m.def("batch_rms1p_io", &batch_rms1p_io_vulkan,
          "Record a zero-centered RMSNorm (y = x*rsqrt(mean(x^2)+eps)*(1+w)) "
          "dispatch into the open batch.",
          py::arg("x_handle"), py::arg("h_w"), py::arg("y_handle"),
          py::arg("H"), py::arg("eps"), py::arg("rows"));
    m.def("batch_topk256_io", &batch_topk256_io_vulkan,
          "Record the GPU router tail (softmax->topk-256x8->renorm->canonical "
          "sort->arena ids/gs write) into the open batch.",
          py::arg("h_logits"), py::arg("h_expscale"), py::arg("h_map"),
          py::arg("h_topk_ids"), py::arg("h_topk_vals"),
          py::arg("h_arena_ids"), py::arg("h_arena_gs"),
          py::arg("E"), py::arg("K"), py::arg("tokens"));
    m.def("batch_swiglu_io", &batch_swiglu_io_vulkan,
          "Record y = silu(gate)*up elementwise fold into the open batch.",
          py::arg("h_gate"), py::arg("h_up"), py::arg("h_out"),
          py::arg("N"));
    m.def("batch_gdn_conv_io", &batch_gdn_conv_io_vulkan,
          "Record GDN conv taps + silu + history shift into the open batch.",
          py::arg("h_qkv"), py::arg("h_buf"), py::arg("h_taps"),
          py::arg("h_mixed"), py::arg("C"));
    m.def("batch_gdn_recur_io", &batch_gdn_recur_io_vulkan,
          "Record GDN state recurrence (one thread/head) into the open batch.",
          py::arg("h_mixed"), py::arg("h_z"), py::arg("h_a"),
          py::arg("h_b"), py::arg("h_alog"), py::arg("h_dtb"),
          py::arg("h_normw"), py::arg("h_state"), py::arg("h_preout"),
          py::arg("C"), py::arg("key_dim"), py::arg("nv"),
          py::arg("hk"), py::arg("hv"), py::arg("rep"),
          py::arg("scale"), py::arg("l2eps"), py::arg("rms_eps"));
    m.def("batch_vecadd_io", &batch_vecadd_io_vulkan,
          "Record elementwise c = a + b into the open batch.",
          py::arg("h_a"), py::arg("h_b"), py::arg("h_out"),
          py::arg("N"));
    m.def("batch_moe_record", &batch_moe_record_vulkan,
          "Record a full MoE layer (8 gateup + 8 acc + shared gate/up/"
          "gate + swiglu fold + shared down + 3 downloads) in one call.",
          // Task b': symmetric GIL release (batch_submit has the same
          // guard) so the Python spec-pack daemon can build CPU tensors
          // while this records ~20 kernels into the command buffer. The
          // tensor args are already converted C++ objects at this point;
          // body-only torch/vulkan work is GIL-free.
          py::call_guard<py::gil_scoped_release>(),
          py::arg("h_x_fused"), py::arg("h_y_acc"),
          py::arg("h_sg"), py::arg("h_su"), py::arg("h_gsc"),
          py::arg("h_x3"), py::arg("h_sdown"),
          py::arg("silu_slots"),
          py::arg("gu_w"), py::arg("gu_s"), py::arg("gu_scale"),
          py::arg("gu_rows"), py::arg("gu_cols"),
          py::arg("dn_w"), py::arg("dn_s"), py::arg("dn_scale"),
          py::arg("dn_rows"), py::arg("dn_cols"),
          py::arg("sg_w"), py::arg("sg_rows"), py::arg("sg_cols"),
          py::arg("su_w"), py::arg("su_rows"), py::arg("su_cols"),
          py::arg("gsc_w"), py::arg("gsc_rows"), py::arg("gsc_cols"),
          py::arg("sdp_w"), py::arg("sdp_rows"), py::arg("sdp_cols"),
          py::arg("gate_rows"),
          py::arg("gu_slot0") = std::vector<int64_t>(),
          py::arg("dn_slot0") = std::vector<int64_t>(),
          py::arg("acc_out"), py::arg("sd_out"), py::arg("gsc_out"));
    m.def("batch_moe_fused_record", &batch_moe_fused_record_vulkan,
          "Record a full MoE layer with the fused MUL_MAT_ID-style "
          "dispatches (1 fused 8-expert gateup+silu + 1 fused 8-expert "
          "down-accumulate, indexed by the resident u32[8] ids buffer) "
          "plus the shared gate/up/gate + swiglu fold + shared down + 3 "
          "downloads in one call.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("h_x_fused"), py::arg("h_y_acc"),
          py::arg("h_sg"), py::arg("h_su"), py::arg("h_gsc"),
          py::arg("h_x3"), py::arg("h_sdown"),
          py::arg("h_silu_all"),
          py::arg("h_arena_wf"), py::arg("h_arena_sf"),
          py::arg("h_arena_wd"), py::arg("h_arena_sd"),
          py::arg("h_ids"), py::arg("h_gs"),
          py::arg("gu_rows"), py::arg("gu_cols"), py::arg("rows_fused"),
          py::arg("dn_rows"), py::arg("dn_cols"), py::arg("rows_down"),
          py::arg("silu_N"),
          py::arg("sg_w"), py::arg("sg_rows"), py::arg("sg_cols"),
          py::arg("su_w"), py::arg("su_rows"), py::arg("su_cols"),
          py::arg("gsc_w"), py::arg("gsc_rows"), py::arg("gsc_cols"),
          py::arg("sdp_w"), py::arg("sdp_rows"), py::arg("sdp_cols"),
          py::arg("gate_rows"),
          py::arg("acc_out"), py::arg("sd_out"), py::arg("gsc_out"));
    m.def("batch_download", &batch_download_vulkan,
          "Record a device->host readback; the tensor is filled after submit.",
          py::arg("handle"), py::arg("tensor"), py::arg("offset") = 0);
    m.def("batch_submit", &batch_submit_vulkan,
          "Submit the open batch once, wait the fence, flush readbacks."
          " Optional queue-2 upload semaphore cookies to wait before"
          " executing (task #29); consumed cookies also recycle their"
          " staging + semaphores after the fence.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("wait_semaphores") = std::vector<int64_t>());
    m.def("batch_upload_async", &batch_upload_async_vulkan,
          "Record one H2D copy on the async upload queue (queue 2).  Must"
          " run OUTSIDE the open main batch; the copies are flushed by"
          " batch_upload_async_flush and the returned cookie is consumed"
          " by the MoE batch's batch_submit(wait_semaphores=[cookie])."
          " Staging is pooled; recycled after the consumer fence.",
          py::arg("handle"), py::arg("tensor"), py::arg("offset") = 0);
    m.def("batch_upload_async_flush", &batch_upload_async_flush_vulkan,
          "End + submit the open queue-2 upload batch; signal a fresh"
          " binary semaphore and return its int64 cookie.  GIL-released"
          " so the queue-2 submit + fence bookkeeping never stalls the"
          " host record path.",
          py::call_guard<py::gil_scoped_release>());
    m.def("batch_cross_barrier", &batch_cross_barrier_vulkan,
          "Record the queue-2 -> main-queue memory barrier on the OPEN"
          " main batch (visibility of async upload writes).  Call once at"
          " batch_begin when the batch will batch_submit with the async"
          " cookie; separately counted from batch_barrier.");
    m.def("batch_upload_async_drain", &batch_upload_async_drain_vulkan,
          "Safety net: wait every in-flight queue-2 batch and recycle its"
          " staging + semaphores.  No-op if nothing is in flight.");
    m.def("batch_upload_async_stats", []() -> std::tuple<int64_t, int64_t> {
              return {int64_t(g_cross_total), int64_t(g_upload_submit_total)};
          },
          "(cumulative cross-queue barriers, cumulative queue-2 submits)."
          " Queue-2 submits are tracked separately from batch_submit_count"
          " (which counts MAIN-queue submissions only) so counter columns"
          " can unambiguously show async-upload traffic moved off the main"
          " queue without it rising.");
    m.def("linear_fp16_resident_io", &linear_fp16_resident_io_vulkan,
          "FP32 linear output with resident FP16 weights and bias.",
          py::arg("x_handle"), py::arg("w_handle"), py::arg("b_handle"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("use_bias") = true);
    m.def("linear_resident_io", &linear_resident_io_vulkan,
          "FP32 linear layer with resident input, weight, bias, and output.",
          py::arg("x_handle"), py::arg("w_handle"), py::arg("b_handle"),
          py::arg("y_handle"), py::arg("T"), py::arg("N"), py::arg("K"),
          py::arg("use_bias") = true);
    m.def("linear_resident", &linear_resident_vulkan,
          "FP32 linear layer y = x @ W.T + b where W [N,K] and b [N] live "
          "in resident VRAM. Suitable for LM head and Q/K/V/O projections.",
          py::arg("x"), py::arg("h_w"), py::arg("h_b"),
          py::arg("N"), py::arg("K"), py::arg("use_bias") = true);
    m.def("moe_mlp_gpt_oss_resident_bench",
          &moe_mlp_gpt_oss_resident_bench_vulkan,
          "Bench-only: run the resident MoE kernel N times back-to-back "
          "inside one C++ call. Used to isolate Python<->C++ overhead.",
          py::arg("x"), py::arg("indices"), py::arg("weights"),
          py::arg("h_gu_blocks"), py::arg("h_gu_scales"), py::arg("h_gu_bias"),
          py::arg("h_d_blocks"),  py::arg("h_d_scales"),  py::arg("h_d_bias"),
          py::arg("E"), py::arg("D"), py::arg("Dff"),
          py::arg("N") = 24);
}
