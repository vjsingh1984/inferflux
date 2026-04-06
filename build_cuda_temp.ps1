$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
$env:OPENSSL_ROOT_DIR = "C:\Program Files\OpenSSL-Win64"

$nvccPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"

Write-Host "Configuring CMake with explicit CUDA compiler..."
& "C:\Program Files\CMake\bin\cmake.exe" -S . -B build-cuda-opt `
    -DENABLE_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release `
    -DOPENSSL_ROOT_DIR="$env:OPENSSL_ROOT_DIR" `
    -DCMAKE_CUDA_COMPILER="$nvccPath" `
    -DCMAKE_CUDA_ARCHITECTURES="89" `
    -G "Visual Studio 17 2022" -A x64

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nBuilding InferFlux with CUDA optimizations..."
& "C:\Program Files\CMake\bin\cmake.exe" --build build-cuda-opt --config Release -j

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nBuild completed successfully!"
Write-Host "Binaries should be in: build-cuda-opt\Release\"
