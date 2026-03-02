# Red Hat / RPM Packaging

Use the same build artifacts and CPack to produce `.rpm` files for Fedora, RHEL, and other RPM-based distributions—no Docker images required.

## Build steps

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_WEBUI=ON
cmake --build build -j$(nproc)
cpack --config build/CPackConfig.cmake -G RPM
```

Install locally with `sudo rpm -i inferflux-<version>-Linux-<arch>.rpm`.

## SPEC template (for Copr / OBS)

The snippet below mirrors the metadata in `cpack` and can be used when submitting to Copr/OBS:

```
Name:           inferflux
Version:        0.1.0
Release:        1%{?dist}
Summary:        InferFlux inference server and CLI
License:        Apache-2.0
URL:            https://github.com/inferencial/InferFlux
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cmake, ninja-build, gcc-c++, openssl-devel, yaml-cpp-devel
Requires:       openssl, yaml-cpp

%description
InferFlux ships the `inferfluxd` inference server and `inferctl` CLI with admin tooling.

%build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_WEBUI=ON
cmake --build build --target inferfluxd inferctl

%install
DESTDIR=%{buildroot} cmake --install build

%files
%license README.md
%{_bindir}/inferfluxd
%{_bindir}/inferctl
%config(noreplace) %{_sysconfdir}/inferflux/inferflux.yaml
%doc %{_datadir}/doc/inferflux/*
```
