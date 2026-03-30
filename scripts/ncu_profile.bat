@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1
set INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === Starting inferfluxd under ncu (MMVQ kernels, skip model-load launches) ===
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat" ^
  --kernel-name "regex:inferflux|fused_dequant|mmvq|mmq|silu|ResidualAdd|RmsNorm|RoPE|FlashDecode|flash_attn" ^
  --launch-skip 0 ^
  --launch-count 100 ^
  --section SpeedOfLight ^
  --section MemoryWorkloadAnalysis ^
  --target-processes all ^
  -o C:\Users\vjsin\code\inferflux\ncu_report ^
  -f ^
  C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe ^
  --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml

echo === ncu session complete ===
