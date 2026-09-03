# Memory monitoring

`MemoryMonitor` is used by the example applications to report GPU and CPU memory peaks in their text and JSON profiles. The implementation supports systems with discrete GPU memory as well as integrated and unified-memory systems, where a single "unified memory" value does not have a consistent meaning.

## Design goals

The monitor is designed to:

- select a usable GPU metric from runtime capabilities instead of product names;
- preserve process-level GPU attribution where the platform exposes it;
- provide an explicit fallback on platforms without a process-level API;
- report the GPU and CPU peaks independently; and
- identify the selected GPU metric so consumers can interpret the value correctly.

The monitor does not attempt to calculate a synchronized total CPU + GPU footprint. On unified-memory systems, CPU-visible residency and system GPU-memory pressure can overlap, so adding the two reported peaks can double-count memory.

## Capability-based selection

The monitor selects its backend from interfaces that are available and usable at runtime. This is necessary because memory-accounting capabilities and semantics differ across platforms:

- some integrated platforms provide reliable NVML current-process accounting;
- some platforms do not provide NVML process accounting at all;
- CUDA free memory can represent device-pool usage on one platform and broader system or unified-memory pressure on another; and
- Linux process RSS does not include ordinary device-only CUDA allocations.

No platform name or integrated-GPU flag is used in the selection.

## GPU backend selection

The current selection order is:

1. `nvml_process_allocation`
2. `cuda_free_memory_delta`

### NVML process allocation

NVML is loaded with `dlopen` so systems without NVML do not acquire a hard runtime dependency. The backend is selected only when NVML initialization, CUDA-device mapping, and the process query are supported.

The CUDA device is mapped to NVML by PCI bus ID. A CUDA ordinal is not used as an NVML index because `CUDA_VISIBLE_DEVICES` can remap CUDA ordinals.

NVML process lists can change between the count query and the data query. The implementation keeps a reusable process buffer, reserves extra entries, and retries if NVML reports that the buffer became too small. If the query succeeds but the current PID is not listed yet, the current allocation is treated as zero. This prevents an early probe, before the first real GPU allocation, from permanently selecting the fallback backend.

`nvml_process_allocation` is an absolute allocation value attributed to the current PID. It is not the process RSS and does not describe CPU-visible residency.

### CUDA free-memory delta

When NVML process accounting is unavailable, the monitor records a `cudaMemGetInfo` free-memory baseline at start and samples:

```text
max(baseline_free_memory - current_free_memory, 0)
```

The peak of this value is reported as `cuda_free_memory_delta`.

This fallback is a device- or system-level observation, not a process-attributed allocation. It has several known limitations:

- allocations made before the monitor baseline are not included;
- allocations from other processes can inflate the result;
- memory released by other processes can hide growth from the monitored process;
- allocator caching and platform reclaim behavior can affect free memory;
- short-lived peaks can be missed by the 50 ms sampling interval; and
- on unified-memory systems, the value can include CPU, host, managed, or other system-memory pressure.

Treat this metric as an approximation of the current process only when the device is otherwise idle. Consumers must not interpret it as exact process GPU allocation.

### Why nvmap is not a production backend

The DRIVE nvmap debugfs interface provides useful process-allocation evidence, but it requires root access on every tested DRIVE system. Reading the clients node repeatedly can also take a global nvmap lock and perturb the workload. `MemoryMonitor` therefore does not read nvmap. Users can instead monitor nvmap from a separate process with root permission.

To observe the current nvmap allocation of a running `llm_inference` process:

```bash
PID=$(pgrep -n -x llm_inference)
sudo env PID="${PID}" bash -c '
while kill -0 "${PID}" 2>/dev/null; do
    grep -E "[[:space:]]${PID}[[:space:]]" /sys/kernel/debug/nvmap/iovmm/clients
    sleep 0.1
done'
```

The final `SIZE` column is the current process allocation in KiB. This loop is intended for external diagnosis only; increase the interval when monitoring overhead is a concern.

### Failure behavior

If the selected backend fails during initial or periodic sampling, the monitor logs the first error and reports a zero GPU peak with `gpu_memory_metric=unavailable` while sampling remains unsuccessful. A later successful sample clears the failure state and monitoring continues, so a transient error does not invalidate the entire run.

## CPU metric

`peak_cpu_memory_bytes` comes from Linux `getrusage(RUSAGE_SELF).ru_maxrss`, converted from KiB to bytes. It is the process-lifetime high-water mark of CPU-visible resident memory.

The value is monotonic and does not decrease after memory is released. It does not include ordinary device-only `cudaMalloc` allocations, and it is not a complete ownership breakdown for pinned or managed memory. The existing `Peak CPU Memory` label is retained for output compatibility.

## Profile output

Text profiles report:

```text
Peak GPU Memory: <value>
Peak CPU Memory: <value>
GPU Memory Metric: nvml_process_allocation | cuda_free_memory_delta | unavailable
```

JSON profiles report:

```text
peak_gpu_memory_bytes
peak_gpu_memory_mb
peak_cpu_memory_bytes
peak_cpu_memory_mb
gpu_memory_metric
```

The `peak_gpu_memory_*` field name is stable across backends, but its meaning is defined by `gpu_memory_metric`. Profile consumers must inspect the metric before comparing results across platforms or interpreting the value as exact process allocation.
