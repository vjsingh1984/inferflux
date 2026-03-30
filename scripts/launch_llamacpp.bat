@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=0
set INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === Launch with llama.cpp CUDA backend ===
C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml
echo Exit code: %ERRORLEVEL%
