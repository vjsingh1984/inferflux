@echo off
setlocal

set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1
set INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === nsys profiling first_token_probe ===
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile ^
  --trace=cuda,nvtx ^
  --cuda-memory-usage=true ^
  --stats=true ^
  --output=C:\Users\vjsin\code\inferflux\nsys_probe_report ^
  --force-overwrite=true ^
  C:\Users\vjsin\code\inferflux\build\Release\inferflux_first_token_probe.exe ^
  --backend inferflux_cuda ^
  --model C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf ^
  --prompt "The capital of France is" ^
  --top-n 5 ^
  --max-tokens 1

echo === nsys exit: %ERRORLEVEL% ===
