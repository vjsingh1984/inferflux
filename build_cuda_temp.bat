@echo off
setlocal enabledelayedexpansion

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"

echo Configuring CMake...
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b %ERRORLEVEL%
)

echo Building InferFlux...
"C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-opt --config Release -j

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!

endlocal
