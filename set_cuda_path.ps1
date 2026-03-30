# Set CUDA_PATH as a system environment variable (requires Administrator)
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2', [System.EnvironmentVariableTarget]::Machine)

Write-Host "CUDA_PATH set to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
Write-Host ""
Write-Host "IMPORTANT: You must restart your terminal/shell for the changes to take effect."
Write-Host "After restarting, verify with: echo %CUDA_PATH% (cmd) or $env:CUDA_PATH (PowerShell)"
