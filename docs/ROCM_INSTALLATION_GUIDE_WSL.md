# ROCm Installation Guide for WSL2

## ⚠️ Critical WSL2 Limitation

**Important:** **ROCm/AMD GPUs DO NOT WORK directly in WSL2!**

WSL2 uses Windows' GPU drivers, which means:
- ✅ NVIDIA GPUs work (through Windows CUDA drivers)
- ❌ AMD GPUs do NOT work (no WDDM bridge for ROCm)

**Your Options:**
1. **Native Linux** (recommended) - Install Linux directly on hardware
2. **Docker** - Use ROCm Docker containers
3. **Cloud** - Rent AMD GPUs on Vast.ai/Lambda Labs

---

## 🐧 Option 1: Native Linux (Recommended for AMD GPUs)

### **For Ubuntu 22.04 on AMD Hardware:**

```bash
#!/bin/bash
# ROCm Installation on Native Linux

# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y wget software-properties-common

# 3. Download ROCm repository key
wget -q -O - https://repo.radeon.com/amdgpu-install/22.04/amdgpu-install_6.0.60000-1_all.deb
# Replace 6.0.60000 with latest version

# 4. Install ROCm repository
sudo dpkg -i amdgpu-install_*.deb

# 5. Update package lists
sudo apt update

# 6. Install ROCm packages
sudo apt install -y \
  rocm-libs \
  rocm-hip-dev \
  rocm-hip-runtime \
  rocblas-dev \
  rocblas-runtime \
  hip-dev \
  hip-runtime \
  rocm-dev \
  rocm-utils

# 7. Verify installation
hipcc --version
rocminfo | head -20

# 8. Set environment variables
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HIP_PATH=$ROCM_PATH' >> ~/.bashrc
source ~/.bashrc

# 9. Test HIP compilation
cat > test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " HIP devices" << std::endl;

    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Arch: " << prop.gcnArchName << std::endl;
        std::cout << "Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    }

    return 0;
}
EOF

hipcc test_hip.cpp -o test_hip
./test_hip

echo "✅ ROCm installation complete!"
```

---

## 🐳 Option 2: Docker (Best for WSL2)

### **Using ROCm Docker Containers:**

Since WSL2 doesn't support AMD GPUs directly, Docker is your best option.

```bash
# 1. Install Docker in WSL2
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Log out and back in for group change to take effect

# 2. Pull ROCm Docker image
docker pull rocm/dev-ubuntu-22.04:latest

# 3. Run container with GPU passthrough
# Note: This requires Windows AMD GPU with proper WDDM 2.1 driver
docker run -it --device=/dev/kfd --device=/dev/driD \
  --group-add video \
  rocm/dev-ubuntu-22.04:latest bash

# Inside container:
#   - ROCm is pre-installed
#   - Build InferFlux with ROCm support
#   - Test AMD GPU inference
```

### **Docker Compose for Development:**

```yaml
# docker-compose.yml
version: '3'
services:
  inferflux-rocm:
    image: rocm/dev-ubuntu-22.04:latest
    container_name: inferflux-rocm
    volumes:
      - ./:/workspace
    working_dir: /workspace
    devices:
      - /dev/kfd
      - /dev/driD
    group_add:
      - video
    environment:
      - ROCM_PATH=/opt/rocm
      - PATH=/opt/rocm/bin:$PATH
    command: bash
    stdin_open: true
    tty: true

# Run:
# docker-compose up
# docker-compose exec inferflux-rocm bash
```

---

## ☁️ Option 3: Cloud AMD GPUs (Easiest for Testing)

### **Vast.ai (Recommended for Testing)**

```bash
#!/bin/bash
# Vast.ai AMD GPU Rental for ROCm Testing

# 1. Search for AMD GPUs with ROCm
# Visit: https://vast.ai
# Filter: GPU=MI300X or MI250X
# Filter: rocm_version >= 6.1
# Sort by: Price (low to high)

# 2. Rent instance
# Expected price: $2-5/hour for MI300X

# 3. Connect via SSH
ssh ubuntu@<instance-ip>

# 4. Verify ROCm installation
hipcc --version
rocminfo

# 5. Clone and build InferFlux
git clone https://github.com/your-repo/inferflux.git
cd inferflux

# 6. Build with ROCm support
cmake -DENABLE_ROCM=ON -B build
cmake --build build -j$(nproc)

# 7. Run server
./build/inferfluxd --config config/server.rocm.yaml

# 8. Test FlashAttention
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer dev-key-123" \
  -d '{"prompt": "Test ROCm FlashAttention", "max_tokens": 50}'

# 9. Check metrics
curl -s http://localhost:8080/metrics \
  -H "Authorization: Bearer dev-key-123" | grep flash_attention

# 10. Destroy instance when done!
# (Remember to destroy or you'll keep getting charged)
```

### **Lambda Labs (More Expensive but Reliable)**

```bash
# Lambda Labs doesn't have AMD GPUs yet
# They only have NVIDIA GPUs
# For AMD GPU testing, use Vast.ai
```

---

## 🔍 Option 4: Check if You Have AMD GPU on Windows

Before proceeding, verify your hardware:

```powershell
# In Windows PowerShell (as Administrator):

# Check for NVIDIA GPUs
Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion

# Check for AMD GPUs
# Note: Windows Device Manager will show AMD GPUs
# But they may not be accessible from WSL2
```

**If you have AMD GPU in Windows:**
- ❌ WSL2 cannot access it directly
- ✅ Docker might work with WDDM 2.1
- ✅ Native Linux is best option

---

## 📋 Quick Decision Tree

```
Do you have AMD GPU hardware?
├─ YES → Install native Linux (Ubuntu 22.04)
│       Install ROCm using script above
│       Build InferFlux with -DENABLE_ROCM=ON
│       Test locally
│
├─ NO → Rent AMD GPU on Vast.ai
│       Most cost-effective for testing
│       Hourly rentals ($2-5/hour)
│       No long-term commitment
│
└─ CANNOT USE WSL2 → AMD GPUs don't work in WSL2
    Use native Linux or cloud
```

---

## 🎯 My Recommendation

### **For Development/Testing:**
**Rent on Vast.ai**
- Cheapest option ($2-5/hour for MI300X)
- No setup required
- Pay only for what you use
- Perfect for testing ROCm backend

### **For Production:**
**Native Linux with AMD GPU**
- Best performance
- Full control
- No cloud overhead
- Suitable for long-running workloads

### **For WSL2 Development:**
**Docker with ROCm image**
- Works in WSL2 environment
- Pre-configured ROCm
- Easy setup
- May have GPU passthrough limitations

---

## 🔧 Verification Commands

### **After Installation (Native Linux):**

```bash
# Verify ROCm installation
which hipcc
hipcc --version

# Check GPU detection
rocminfo

# Test basic HIP compilation
cat > test.cpp << 'EOF'
#include <hip/hip_runtime.h>
int main() {
    int count = 0;
    hipGetDeviceCount(&count);
    return count > 0 ? 0 : 1;
}
EOF
hipcc test.cpp -o test && ./test

# Test llama.cpp ROCm support
cd external/llama.cpp
cmake -DGGML_HIP=ON -B build
ls -la build/ggml/src/ggml-hip/
```

### **After Build (Verify InferFlux):**

```bash
# Check if ROCm backend was built
ldd build/libinferflux_core.so | grep -i hip

# Check build configuration
grep -i "rocm\|hip" build/CMakeCache.txt

# Verify INFERFLUX_HAS_ROCM is defined
strings build/libinferflux_core.so | grep INFERFLUX_HAS_ROCM
```

---

## 📝 Configuration File

Create `config/server.rocm.yaml`:

```yaml
server:
  host: 0.0.0.0
  http_port: 8080
  max_concurrent: 1024
  enable_metrics: true

model:
  repo: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
  path: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  format: gguf
  quantization: q4_k_m

models:
  - id: tinyllama
    path: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    format: auto
    backend: rocm
    default: true

runtime:
  backend_priority: [rocm, cpu]
  rocm:
    enabled: true
    flash_attention:
      enabled: true
      tile_size: 128
    device_id: 0  # Which AMD GPU to use
```

---

## 🚀 Quick Start (With AMD GPU Access)

```bash
# 1. Build with ROCm support
cmake -DENABLE_ROCM=ON -S . -B build
cmake --build build --target inferflux_core -j

# 2. Run server
./build/inferfluxd --config config/server.rocm.yaml

# 3. Make test request
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "prompt": "Hello from AMD ROCm!",
    "max_tokens": 50,
    "model": "tinyllama"
  }'

# 4. Check FlashAttention metrics
curl -s http://localhost:8080/metrics \
  -H "Authorization: Bearer dev-key-123" | grep flash_attention
```

---

## 💡 Cost Comparison

| Option | Setup Time | Hourly Cost | Best For |
|--------|------------|------------|----------|
| Vast.ai MI300X | 5 min | $2-5 | Testing/Development |
| Lambda Labs | N/A | N/A | Not available |
| Native Linux | 2-4 hours | $0 (hardware cost) | Production |
| Docker WSL2 | 30 min | $0 | Development (if GPU passthrough works) |

---

## ⚠️ Troubleshooting

### **"ROCm not found" error:**
```bash
# Check if ROCm is installed
dpkg -l | grep rocm

# Check ROCm path
echo $ROCM_PATH

# Add to PATH if needed
export ROCm_PATH=/opt/rocm
```

### **"No HIP devices found":**
```bash
# Check for AMD GPUs
lspci | grep -i vga

# Check if kernel recognizes GPU
dmesg | grep -i amdgpu

# Verify HIP driver
hipconfig --list
```

### **"Build errors about missing hip/roc:**
```bash
# Install missing dependencies
sudo apt install -y rocm-dev rocm-hip-dev rocblas-dev

# Reconfigure CMake
rm -rf build
cmake -DENABLE_ROCM=ON -S . -B build
```

---

## 🎯 Summary

**WSL2 cannot directly access AMD GPUs.** Your options:

1. **Best for Testing:** Rent on Vast.ai ($2-5/hour)
2. **Best for Development:** Native Linux installation
3. **Possible for WSL2:** Docker (may have limitations)

**Recommendation:** Use Vast.ai for testing ROCm backend - it's cheapest and easiest way to access AMD GPUs for development.

**Timeline:**
- Setup on Vast.ai: 5 minutes
- Build & Test: 1-2 hours
- Destroy when done: 1 minute
- **Total cost: ~$5-15 for a full testing session**

Would you like me to help you with any of these installation options?
