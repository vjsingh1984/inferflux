@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === Starting inferfluxd under nsys (CUDA+NVTX trace, 120s capture) ===
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile ^
  --trace cuda,nvtx ^
  --cuda-memory-usage true ^
  --duration 120 ^
  --output C:\Users\vjsin\code\inferflux\nsys_inferflux_report ^
  --force-overwrite true ^
  C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe ^
  --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml

echo === nsys session complete ===
