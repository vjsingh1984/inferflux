@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_CUDA_TIMING_SAMPLE_RATE=1
set INFERFLUX_CUDA_SYNC_TRACE=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION_LIMIT=200
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === InferFlux CUDA Profiling Server ===
echo Model: %INFERFLUX_MODEL_PATH%
echo Timing sample rate: every batch
echo Sync trace: enabled
echo Operator selection debug: enabled (limit 200)
echo.
echo Starting server... (Ctrl+C to stop)
echo Stderr will go to server_stderr.txt
echo.

C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml 2> C:\Users\vjsin\code\inferflux\server_stderr.txt

echo.
echo === Server exited with code: %ERRORLEVEL% ===
