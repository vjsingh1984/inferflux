# Windows Packaging (winget + WiX/MSI)

## WiX-based MSI

1. Install the WiX Toolset (v3.14 or later) and ensure `candle`/`light` are on `PATH`.
2. Build InferFlux:
   ```powershell
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_WEBUI=ON
   cmake --build build --config Release
   ```
3. Generate the installer:
   ```powershell
   cpack --config build/CPackConfig.cmake -G WIX
   ```
   The MSI appears under `build/`.

## winget manifest

`installers/winget/inferencial.inferflux.yaml` is the template manifest. Update `Version`, `InstallerSha256`, and URLs once the MSI is published, then submit a PR to `microsoft/winget-pkgs`.
