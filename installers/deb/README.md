# Ubuntu / Debian Packaging

InferFlux uses CPack to generate `.deb` artifacts directly from the build tree, which means we can ship a native package without Docker images or containerized build roots.

## Build steps

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_WEBUI=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
cpack --config build/CPackConfig.cmake -G DEB
```

The resulting `inferflux-<version>-Linux-<arch>.deb` lives under `build/`. Install it with `sudo apt install ./inferflux-*.deb`.

## control template (for Launchpad / apt repo)

`control` provides the metadata expected by Launchpad/aptly. Replace the placeholders before uploading:

```
Source: inferflux
Section: utils
Priority: optional
Maintainer: Inferencial Labs <support@inferencial.ai>
Build-Depends: debhelper (>= 12), cmake, ninja-build, libssl-dev, libyaml-cpp-dev
Standards-Version: 4.5.0
Homepage: https://github.com/inferencial/InferFlux

Package: inferflux
Architecture: amd64
Depends: ${shlibs:Depends}, ${misc:Depends}, libssl3, libyaml-cpp0.7
Description: InferFlux inference server and CLI
 InferFlux ships the `inferfluxd` server and `inferctl` CLI with admin tooling.
```
