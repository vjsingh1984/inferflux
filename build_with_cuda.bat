@echo off
REM Refresh environment variables from registry
call refreshenv 2>nul

REM If refreshenv not available, manually reload PATH
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CUDA_PATH 2^>nul') do set CUDA_PATH=%%b

echo CUDA_PATH is now: %CUDA_PATH%

REM Add CUDA and CMake to PATH for this session
set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
set "PATH=%PATH%;C:\Program Files\CMake\bin"
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"

echo.
echo Configuring CMake with CUDA...
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT_DIR%" -DCMAKE_CUDA_ARCHITECTURES=89 -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Building InferFlux with CUDA optimizations...
"C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-opt --config Release -j

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Build completed successfully!
echo Binaries location: build-cuda-opt\Release\
echo ========================================
pause
