"""Safe ctypes probe of the local Vulkan instance/device.

Ported context: the fork's ``ext_module.cpp``/``vk_util.hpp`` abort the whole
process on any Vulkan error (``VKU_CHECK`` calls ``std::exit(1)``), so Python
must NEVER probe "is a Vulkan device usable?" by calling into the extension —
a missing driver would kill the interpreter instead of raising. This module
talks to ``libvulkan`` directly with plain ctypes, checks every ``VkResult``
itself, and ``probe()`` never raises: it returns ``None`` on any failure.

Everything here is read-only and allocation-light: one instance, enumerate
physical devices, read properties + memory properties, destroy.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
from typing import Any

# ---- Vulkan constants (subset) ---------------------------------------------
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_API_VERSION_1_2 = (1 << 22) | (2 << 12)
VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x2


def decode_version(v: int) -> tuple[int, int, int]:
    """Decode a VkVersion-style uint32 into (major, minor, patch)."""
    return ((v >> 22) & 0x7F, (v >> 12) & 0x3FF, v & 0xFFF)


def format_version(v: int) -> str:
    major, minor, patch = decode_version(v)
    return f"{major}.{minor}.{patch}"


# ---- ctypes structs ---------------------------------------------------------
class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class VkMemoryType(ctypes.Structure):
    _fields_ = [
        ("propertyFlags", ctypes.c_uint32),
        ("heapIndex", ctypes.c_uint32),
    ]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
    ]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * 16),
    ]


# Dispatchable handles are plain pointers.
VkInstance = ctypes.c_void_p
VkPhysicalDevice = ctypes.c_void_p

# VkPhysicalDeviceProperties has no sType/pNext; deviceName sits at byte
# offset 20 (4x u32 + deviceType u32). The full struct (limits + sparse
# props) is ~1.5 KiB; a 4 KiB scratch buffer is comfortably safe.
_PROPS_BUF_SIZE = 4096
_PROPS_API_VERSION_OFF = 0
_PROPS_DRIVER_VERSION_OFF = 4
_PROPS_VENDOR_ID_OFF = 8
_PROPS_DEVICE_ID_OFF = 12
_PROPS_DEVICE_TYPE_OFF = 16
_PROPS_DEVICE_NAME_OFF = 20
_PROPS_DEVICE_NAME_LEN = 256


def _load_libvulkan() -> ctypes.CDLL:
    path = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
    return ctypes.CDLL(path)


@functools.cache
def probe() -> dict[str, Any] | None:
    """Probe the primary Vulkan device. Returns None instead of raising.

    Keys (all best-effort, None when unknown):
      name, api_version (str), driver_version (str), vendor_id, device_id,
      device_type, vram_bytes, device_count
    """
    try:
        return _probe_impl()
    except Exception:  # noqa: BLE001 - probing must never raise
        return None


def _probe_impl() -> dict[str, Any] | None:
    lib = _load_libvulkan()

    lib.vkCreateInstance.argtypes = [
        ctypes.POINTER(VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(VkInstance),
    ]
    lib.vkCreateInstance.restype = ctypes.c_int
    lib.vkDestroyInstance.argtypes = [VkInstance, ctypes.c_void_p]
    lib.vkDestroyInstance.restype = None
    lib.vkEnumeratePhysicalDevices.argtypes = [
        VkInstance,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(VkPhysicalDevice),
    ]
    lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int
    lib.vkGetPhysicalDeviceProperties.argtypes = [
        VkPhysicalDevice,
        ctypes.c_void_p,
    ]
    lib.vkGetPhysicalDeviceProperties.restype = None
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [
        VkPhysicalDevice,
        ctypes.POINTER(VkPhysicalDeviceMemoryProperties),
    ]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None

    app = VkApplicationInfo()
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO
    app.pApplicationName = b"freetoken-vulkan-probe"
    app.apiVersion = VK_API_VERSION_1_2

    info = VkInstanceCreateInfo()
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    info.pApplicationInfo = ctypes.cast(
        ctypes.byref(app), ctypes.POINTER(VkApplicationInfo)
    )

    instance = VkInstance()
    rc = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if rc != VK_SUCCESS:
        return None
    try:
        count = ctypes.c_uint32(0)
        rc = lib.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(count), None
        )
        if rc != VK_SUCCESS or count.value == 0:
            return None
        devices = (VkPhysicalDevice * count.value)()
        rc = lib.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(count), devices
        )
        if rc != VK_SUCCESS:
            return None

        # Prefer a discrete GPU; otherwise take the first device.
        best = devices[0]
        best_type = None
        for i in range(count.value):
            raw = ctypes.create_string_buffer(_PROPS_BUF_SIZE)
            lib.vkGetPhysicalDeviceProperties(devices[i], raw)
            dtype = int.from_bytes(
                raw.raw[_PROPS_DEVICE_TYPE_OFF:_PROPS_DEVICE_TYPE_OFF + 4],
                "little",
            )
            if dtype == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                best, best_type = devices[i], dtype
                break
            if best_type is None:
                best, best_type = devices[i], dtype

        raw = ctypes.create_string_buffer(_PROPS_BUF_SIZE)
        lib.vkGetPhysicalDeviceProperties(best, raw)
        view = memoryview(raw.raw)
        api_version = int.from_bytes(
            view[_PROPS_API_VERSION_OFF:_PROPS_API_VERSION_OFF + 4], "little"
        )
        driver_version = int.from_bytes(
            view[_PROPS_DRIVER_VERSION_OFF:_PROPS_DRIVER_VERSION_OFF + 4],
            "little",
        )
        vendor_id = int.from_bytes(
            view[_PROPS_VENDOR_ID_OFF:_PROPS_VENDOR_ID_OFF + 4], "little"
        )
        device_id = int.from_bytes(
            view[_PROPS_DEVICE_ID_OFF:_PROPS_DEVICE_ID_OFF + 4], "little"
        )
        device_type = int.from_bytes(
            view[_PROPS_DEVICE_TYPE_OFF:_PROPS_DEVICE_TYPE_OFF + 4], "little"
        )
        name_bytes = bytes(
            view[_PROPS_DEVICE_NAME_OFF:_PROPS_DEVICE_NAME_OFF + _PROPS_DEVICE_NAME_LEN]
        )
        name = name_bytes.split(b"\x00", 1)[0].decode("utf-8", "replace")

        mem = VkPhysicalDeviceMemoryProperties()
        lib.vkGetPhysicalDeviceMemoryProperties(best, ctypes.byref(mem))

        # Real VRAM: among DEVICE_LOCAL heaps, prefer those whose memory
        # types are NOT host-visible (pure GDDR). On RADV the system-RAM
        # "smart memory" heap is also flagged DEVICE_LOCAL but its types
        # are HOST_VISIBLE, so it must not win. Fallback: largest
        # device-local heap.
        device_local = []
        for h in range(mem.memoryHeapCount):
            heap = mem.memoryHeaps[h]
            if not (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT):
                continue
            types = [
                mem.memoryTypes[t]
                for t in range(mem.memoryTypeCount)
                if mem.memoryTypes[t].heapIndex == h
            ]
            host_visible = any(
                t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                for t in types
            )
            device_local.append((host_visible, heap.size))
        vram_bytes = None
        pure = [size for host_vis, size in device_local if not host_vis]
        if pure:
            vram_bytes = max(pure)
        elif device_local:
            vram_bytes = max(size for _, size in device_local)

        return {
            "name": name,
            "api_version": format_version(api_version),
            "driver_version": format_version(driver_version),
            "vendor_id": vendor_id,
            "device_id": device_id,
            "device_type": device_type,
            "vram_bytes": vram_bytes,
            "device_count": int(count.value),
        }
    finally:
        lib.vkDestroyInstance(instance, None)
