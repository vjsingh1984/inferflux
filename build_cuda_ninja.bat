@echo off
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
set "PATH=%PATH%;C:\Program Files\Ninja"
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"

echo Configuring CMake with Ninja generator...
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" -DCMAKE_CUDA_ARCHITECTURES=89 -G Ninja

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b %ERRORLEVEL%
)

echo Building InferFlux with CUDA optimizations (Ninja)...
"C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-opt -j

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!
echo Binaries should be in: build-cuda-opt\

endlocal
