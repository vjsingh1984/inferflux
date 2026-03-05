# StartupAdvisor Configurable Constants - Summary

**Date**: 2026-03-04
**Changes**: Replaced magic numbers with named constants and made key values configurable via environment variables

---

## What Changed

### Before: Magic Numbers

```cpp
// Hardcoded values - not flexible
rec.overhead_bytes = 1024ULL * 1024 * 1024;  // 1 GB - magic number!

constexpr double kTargetUtilization = 0.85;  // 85% - hardcoded

int min_slots = 16;  // 16 minimum - hardcoded
```

### After: Named Constants + Configurable

```cpp
// Named constant with configurable override
constexpr std::uint64_t kDefaultOverheadBytes = 1024ULL * 1024 * 1024;  // 1 GB
std::uint64_t overhead_bytes = kDefaultOverheadBytes;

if (const char* env_overhead = std::getenv("INFERFLUX_OVERHEAD_GB")) {
  try {
    int overhead_gb = std::stoi(env_overhead);
    if (overhead_gb >= 0 && overhead_gb <= 16) {
      overhead_bytes = static_cast<std::uint64_t>(overhead_gb) * 1024ULL * 1024 * 1024;
    }
  } catch (const std::exception&) {
    // Invalid value, use default
  }
}

// Same pattern for GPU utilization target
constexpr double kDefaultTargetUtilization = 0.85;
double target_utilization = kDefaultTargetUtilization;

if (const char* env_util = std::getenv("INFERFLUX_GPU_UTILIZATION_PCT")) {
  // ... parse and validate 50-98 range
}

// Reduced minimum slots from 16 to 10 per user request
int min_slots = 10;  // Changed from 16
```

---

## New Environment Variables

| Variable | Default | Range | Purpose |
|----------|---------|-------|---------|
| `INFERFLUX_GPU_UTILIZATION_PCT` | 85 | 50-98 | Target GPU memory utilization percentage |
| `INFERFLUX_OVERHEAD_GB` | 1 | 0-16 | Memory overhead for CUDA context, fragmentation, activation tensors (GB) |
| `INFERFLUX_MIN_SLOTS` | 10 | 4-256 | Minimum concurrent sequences (reduced from 16) |
| `INFERFLUX_MAX_SLOTS` | 256 | 16-512 | Maximum concurrent sequences |

---

## Usage Examples

### Conservative Production Setup

```bash
# 80% utilization, 2GB overhead for safety, high concurrency
INFERFLUX_GPU_UTILIZATION_PCT=80 \
INFERFLUX_OVERHEAD_GB=2 \
INFERFLUX_MIN_SLOTS=32 \
INFERFLUX_MAX_SLOTS=256 \
./build/inferfluxd
```

### Aggressive Development Setup

```bash
# 90% utilization, minimal overhead, maximize throughput
INFERFLUX_GPU_UTILIZATION_PCT=90 \
INFERFLUX_OVERHEAD_GB=0 \
INFERFLUX_MIN_SLOTS=10 \
INFERFLUX_MAX_SLOTS=512 \
./build/inferfluxd
```

### Large Context Model

```bash
# 75% utilization (extra safe), 2GB overhead, fewer slots
INFERFLUX_GPU_UTILIZATION_PCT=75 \
INFERFLUX_OVERHEAD_GB=2 \
INFERFLUX_MIN_SLOTS=8 \
INFERFLUX_MAX_SLOTS=64 \
./build/inferfluxd
```

### Small GPU (8 GB)

```bash
# 80% utilization, minimal overhead, fewer slots
INFERFLUX_GPU_UTILIZATION_PCT=80 \
INFERFLUX_OVERHEAD_GB=0 \
INFERFLUX_MIN_SLOTS=8 \
INFERFLUX_MAX_SLOTS=32 \
./build/inferfluxd
```

---

## Benefits

### 1. **No Magic Numbers**
- All constants have clear, descriptive names
- Intent is obvious from code
- Easy to maintain and modify

### 2. **Runtime Configurability**
- Adjust behavior without recompiling
- Different environments (dev/staging/prod) can use different values
- A/B testing of different configurations

### 3. **Safety**
- Input validation on all environment variables
- Safe fallback to defaults if invalid values provided
- Range limits prevent impossible configurations

### 4. **Flexibility**
- Production: Use 80% utilization for safety margin
- Development: Use 90% utilization to maximize capacity
- Large models: Increase overhead to 2GB for activation tensors
- Small GPUs: Reduce minimum slots to fit in memory

---

## Formula Reference

### Before (Hardcoded)

```
Max Slots = floor((GPU_VRAM × 0.85 - Model_Size - 1GB) / Per_Slot_KV)
Clamp to [16, 256]
```

### After (Configurable)

```
target_pct = INFERFLUX_GPU_UTILIZATION_PCT (default: 85)
overhead_gb = INFERFLUX_OVERHEAD_GB (default: 1)

Max Slots = floor((GPU_VRAM × (target_pct/100) - Model_Size - overhead_gb) / Per_Slot_KV)
Clamp to [INFERFLUX_MIN_SLOTS, INFERFLUX_MAX_SLOTS]
```

---

## Files Modified

1. **server/startup_advisor.cpp**
   - Replaced magic number `1024ULL * 1024 * 1024` with `kDefaultOverheadBytes`
   - Made overhead configurable via `INFERFLUX_OVERHEAD_GB`
   - Renamed `kTargetUtilization` to `kDefaultTargetUtilization`
   - Made GPU utilization configurable via `INFERFLUX_GPU_UTILIZATION_PCT`
   - Changed minimum slots from 16 to 10

2. **docs/STARTUP_ADVISOR_DYNAMIC_SLOTS_SUMMARY.md**
   - Updated environment variable table
   - Added new usage examples
   - Updated clamp value from 16 to 10

3. **docs/DYNAMIC_SLOT_ALLOCATION_STARTUP_ADVISOR.md**
   - Updated formula reference
   - Expanded environment variable documentation
   - Added new usage examples

4. **/tmp/test_dynamic_slot_allocation.py**
   - Updated formula reference section
   - Updated summary to mention new environment variables
   - Clarified that 85% is configurable

---

## Validation

All changes compiled successfully:

```bash
$ cmake --build build -j$(nproc) --target inferfluxd
[100%] Built target inferfluxd
```

---

## Next Steps

The dynamic slot allocation feature is now fully implemented with configurable constants. The next logical step would be to integrate it with the actual server initialization code so that `SequenceSlotManager` is automatically configured with the calculated values at startup.

This would involve:
1. Calling `CalculateOptimalSlotAllocation()` during server startup
2. Passing the recommendation to `SequenceSlotManager` constructor
3. Applying the recommended `max_parallel_sequences` to llama backend config
4. Testing with actual model loading

However, this integration should be done only if explicitly requested, as the current implementation provides complete recommendation logic that can be manually applied to configuration.
