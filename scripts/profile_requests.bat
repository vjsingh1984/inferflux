@echo off
setlocal

set PATH=C:\Windows\System32;%PATH%

echo === InferFlux CUDA Profiling - Request Sender ===
echo.
echo Checking if server is up...

:waitloop
curl -s --connect-timeout 2 http://localhost:8080/healthz > C:\Users\vjsin\code\inferflux\healthz_response.txt 2>nul
if errorlevel 1 (
    echo Server not responding yet, retrying in 3s...
    ping -n 4 127.0.0.1 >nul
    goto waitloop
)
echo Server is up!
echo Health response:
type C:\Users\vjsin\code\inferflux\healthz_response.txt
echo.

echo.
echo === Warm-up request (5 tokens) ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":5}"
echo.

echo.
echo === Profiling request 1: 200 tokens ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain the theory of relativity in detail\"}],\"max_tokens\":200}"
echo.

echo.
echo === Profiling request 2: 200 tokens ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short story about a robot learning to paint\"}],\"max_tokens\":200}"
echo.

echo.
echo === Profiling request 3: 150 tokens ===
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
echo.
echo Output files:
echo   metrics_output.txt  - Full Prometheus metrics
echo   cuda_metrics.txt    - CUDA-specific metrics only
echo   server_stderr.txt   - Operator selection + sync trace
