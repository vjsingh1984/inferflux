@echo off
setlocal

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_PATH_V13_2=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
set LIB=%CUDA_PATH%\lib\x64;%LIB%
set PATH=C:\Windows\System32;C:\Program Files\CMake\bin;%CUDA_PATH%\bin;%PATH%

echo === Building inferflux_first_token_probe (Release) ===
cmake --build C:\Users\vjsin\code\inferflux\build --config Release --target inferflux_first_token_probe -- /m /p:CudaToolkitDir="%CUDA_PATH%" /p:CudaToolkitLibDir="%CUDA_PATH%\lib\x64"
echo.
echo === Build exit code: %ERRORLEVEL% ===
