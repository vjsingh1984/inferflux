@echo off
REM Workaround for CUDA 13.2 + VS2022 + CMake 4.2 integration issue

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%PATH%;%CUDA_PATH%\bin;C:\Program Files\CMake\bin"
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"

echo CUDA_PATH: %CUDA_PATH%
echo.
echo Attempting workaround: disabling CUDA language detection...
echo.

REM Try building without explicit CUDA language, let llama.cpp handle it
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt ^
    -DENABLE_CUDA=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=FALSE ^
    -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CMake configuration failed!
    echo.
    echo Trying alternative: using pre-built llama.cpp CUDA...
    "C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt2 ^
        -DENABLE_CUDA=OFF ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" ^
        -G "Visual Studio 17 2022" -A x64
)

pause
