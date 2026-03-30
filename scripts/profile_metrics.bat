@echo off
setlocal

echo === InferFlux CUDA Profiling via Built-in Metrics ===

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_CUDA_TIMING_SAMPLE_RATE=1
set INFERFLUX_CUDA_SYNC_TRACE=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION=1
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION_LIMIT=200
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
REM Put Windows system dirs first so we get Windows timeout, not MinGW timeout
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo Starting inferfluxd with CUDA timing enabled...
start /B "" C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml 2> C:\Users\vjsin\code\inferflux\server_stderr.txt

echo Waiting for server to be ready...
set attempts=0
:waitloop
set /a attempts+=1
if %attempts% gtr 40 (
    echo ERROR: Server did not start after 40 attempts
    goto cleanup
)
REM Use ping for delay (6 pings ~5 seconds) since MinGW overrides timeout
ping -n 6 127.0.0.1 >nul
curl -s --connect-timeout 2 http://localhost:8080/healthz 2>nul | findstr "ok" >nul
if errorlevel 1 (
    echo Attempt %attempts%/40 - not ready yet
    goto waitloop
)
echo Server is up after %attempts% attempts!

echo.
echo === Sending warm-up request ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":5}"
echo.

echo === Sending profiling request 1 (long decode, 200 tokens) ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain the theory of relativity in detail\"}],\"max_tokens\":200}"
echo.

echo === Sending profiling request 2 (long decode, 200 tokens) ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short story about a robot learning to paint\"}],\"max_tokens\":200}"
echo.

echo === Sending profiling request 3 (150 tokens) ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"What are the fundamental laws of thermodynamics?\"}],\"max_tokens\":150}"
echo.

echo.
echo === Capturing Prometheus Metrics ===
curl -s -H "Authorization: Bearer dev-key-123" http://localhost:8080/metrics > C:\Users\vjsin\code\inferflux\metrics_output.txt 2>&1
echo Metrics saved to metrics_output.txt

echo.
echo === Capturing CUDA-specific metrics ===
curl -s -H "Authorization: Bearer dev-key-123" http://localhost:8080/metrics 2>nul | findstr /C:"inferflux_cuda" > C:\Users\vjsin\code\inferflux\cuda_metrics.txt
echo CUDA metrics saved to cuda_metrics.txt

:cleanup
echo.
echo === Stopping server ===
taskkill /IM inferfluxd.exe /F >nul 2>&1
ping -n 4 127.0.0.1 >nul

echo.
echo === Server stderr (sync trace, operator selection) ===
if exist C:\Users\vjsin\code\inferflux\server_stderr.txt (
    type C:\Users\vjsin\code\inferflux\server_stderr.txt
) else (
    echo No stderr file found
)

echo.
echo === DONE ===
