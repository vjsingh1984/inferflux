# Set up environment for Ninja + MSVC + CUDA
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$vsDevCmd = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
$env:PATH += ";C:\Program Files\Ninja"
$env:OPENSSL_ROOT_DIR = "C:\Program Files\OpenSSL-Win64"

Write-Host "Configuring CMake with Ninja generator..."
& "C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt `
    -DENABLE_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release `
    -DOPENSSL_ROOT_DIR="$env:OPENSSL_ROOT_DIR" `
    -DCMAKE_CUDA_ARCHITECTURES="89" `
    -G Ninja

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nBuilding InferFlux with CUDA optimizations (Ninja)..."
& "C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-opt -j

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nBuild completed successfully!"
Write-Host "Binaries should be in: build-cuda-opt\"
