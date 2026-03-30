@echo off
REM Build script for CUDA 12.1 on Windows
REM This version uses CUDA 12.1 which has better VS2022 integration

echo ========================================
echo InferFlux CUDA 12.1 Build Script
echo ========================================
echo.

REM Refresh environment variables from registry
call refreshenv 2>nul

REM If refreshenv not available, manually reload PATH
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CUDA_PATH 2^>nul') do set CUDA_PATH=%%b

echo CUDA_PATH detected: %CUDA_PATH%
echo.

REM Check if CUDA_PATH is set
if "%CUDA_PATH%"=="" (
    echo ERROR: CUDA_PATH environment variable not set!
    echo Please install CUDA 12.1 and reboot.
    pause
    exit /b 1
)

REM Add CUDA 12.1 and CMake to PATH for this session
set "PATH=%PATH%;%CUDA_PATH%\bin"
set "PATH=%PATH%;C:\Program Files\CMake\bin"
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"

echo CUDA bin: %CUDA_PATH%\bin
echo CMake bin: C:\Program Files\CMake\bin
echo OpenSSL: %OPENSSL_ROOT_DIR%
echo.
echo ========================================
echo.

echo Step 1: Cleaning previous build...
if exist build-cuda-12 (
    rmdir /s /q build-cuda-12
)

echo.
echo Step 2: Configuring CMake with CUDA 12.1...
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-12 ^
    -DENABLE_CUDA=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo CMake configuration failed!
    echo ========================================
    echo.
    echo Common issues:
    echo 1. CUDA 12.1 not installed properly
    echo 2. Need to reboot after CUDA installation
    echo 3. Visual Studio 2022 not installed
    echo 4. CUDA_PATH not set correctly
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Step 3: Building InferFlux with CUDA optimizations...
echo ========================================
echo.

"C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-12 --config Release -j

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Binaries location: build-cuda-12\Release\
echo Main executables:
echo   - build-cuda-12\Release\inferfluxd.exe (server)
echo   - build-cuda-12\Release\inferctl.exe (CLI client)
echo   - build-cuda-12\Release\inferflux_tests.exe (unit tests)
echo.
echo Next steps:
echo   1. Run unit tests: build-cuda-12\Release\inferflux_tests.exe
echo   2. Run throughput benchmark: bash scripts/benchmark.sh throughput-gate
echo ========================================
echo.
pause
