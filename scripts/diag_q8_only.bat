@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION_LIMIT=200
REM Enable Q8_1 but disable fused gate_up_silu
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=0
set INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === Diagnostic: Q8_1 ON, fused gate_up OFF ===
C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml 2> C:\Users\vjsin\code\inferflux\server_stderr.txt

echo === Server exited: %ERRORLEVEL% ===
