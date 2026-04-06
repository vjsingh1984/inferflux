@echo off
setlocal

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_PATH_V13_2=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;C:\Program Files\CMake\bin;%CUDA_PATH%\bin;%PATH%

echo === Reconfiguring CMake with CUDA toolkit path ===
cmake -S C:\Users\vjsin\code\inferflux -B C:\Users\vjsin\code\inferflux\build -G "Visual Studio 17 2022" -DCUDAToolkit_ROOT="%CUDA_PATH%" -DENABLE_CUDA=ON -DENABLE_ROCM=OFF -DENABLE_MPS=OFF -DENABLE_VULKAN=OFF -DENABLE_MTMD=OFF
if errorlevel 1 (
    echo CMake configure failed!
    exit /b 1
)

echo.
echo === Building inferfluxd (Release) ===
cmake --build C:\Users\vjsin\code\inferflux\build --config Release --target inferfluxd -- /m
echo.
echo === Build exit code: %ERRORLEVEL% ===
