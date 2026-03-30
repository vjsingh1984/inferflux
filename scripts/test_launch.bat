@echo off
setlocal

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin;%PATH%

echo === Testing inferfluxd launch ===
echo Model path: %INFERFLUX_MODEL_PATH%
echo Checking exe exists...
if exist C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe (
    echo inferfluxd.exe found
) else (
    echo ERROR: inferfluxd.exe not found!
    exit /b 1
)

echo Checking model exists...
if exist %INFERFLUX_MODEL_PATH% (
    echo Model file found
) else (
    echo ERROR: Model file not found!
    exit /b 1
)

echo Checking config exists...
if exist C:\Users\vjsin\code\inferflux\config\server.cuda.yaml (
    echo Config file found
) else (
    echo ERROR: Config file not found!
    exit /b 1
)

echo.
echo === Launching inferfluxd (will run for 30s max) ===
C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml
echo.
echo === inferfluxd exited with code %ERRORLEVEL% ===
