@echo off
echo === Checking server health ===
:loop
curl -s --connect-timeout 2 http://localhost:8080/healthz 2>nul | findstr "ok" >nul
if errorlevel 1 (
    echo Waiting for server...
    timeout /t 10 /nobreak >nul
    goto loop
)
echo Server is up!
echo === Sending inference request ===
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dev-key-123" -d "{\"model\":\"qwen2.5-3b-instruct-q4_k_m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":10}"
echo.
echo === Done ===
