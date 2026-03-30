@echo off
setlocal

set LOGFILE=C:\Users\vjsin\code\inferflux\launch_test_output.txt

echo === Test Launch Log === > %LOGFILE%
echo Time: %date% %time% >> %LOGFILE%

set INFERFLUX_MODEL_PATH=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_BACKEND_PREFER_INFERFLUX=1
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin;%PATH%

echo Checking files... >> %LOGFILE%
if exist C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe (
    echo EXE: found >> %LOGFILE%
) else (
    echo EXE: NOT FOUND >> %LOGFILE%
)

if exist %INFERFLUX_MODEL_PATH% (
    echo MODEL: found >> %LOGFILE%
) else (
    echo MODEL: NOT FOUND >> %LOGFILE%
)

if exist C:\Users\vjsin\code\inferflux\config\server.cuda.yaml (
    echo CONFIG: found >> %LOGFILE%
) else (
    echo CONFIG: NOT FOUND >> %LOGFILE%
)

echo. >> %LOGFILE%
echo Launching server... >> %LOGFILE%
C:\Users\vjsin\code\inferflux\build\Release\inferfluxd.exe --config C:\Users\vjsin\code\inferflux\config\server.cuda.yaml >> %LOGFILE% 2>&1
echo Exit code: %ERRORLEVEL% >> %LOGFILE%
echo === Done === >> %LOGFILE%
