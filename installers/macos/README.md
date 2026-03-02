# macOS Packaging (pkgbuild + Homebrew)

## Homebrew tap

The formula lives in `installers/homebrew/inferflux.rb`. Update the `url`, `sha256`, and `version` whenever a new tag is cut, then push it to your tap:

```bash
brew tap inferencial/cli
brew install inferencial/cli/inferflux
```

## Standalone `.pkg`

1. Build InferFlux:
   ```bash
   cmake -S . -B build -DENABLE_WEBUI=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(sysctl -n hw.ncpu)
   ```
2. Generate the installer using Apple’s productbuild generator:
   ```bash
   cpack --config build/CPackConfig.cmake -G productbuild
   ```
   The signed (or ad-hoc) `.pkg` will be emitted into `build/`.

3. Notarize/sign the `.pkg` as required, then distribute it or publish via GitHub Releases.
