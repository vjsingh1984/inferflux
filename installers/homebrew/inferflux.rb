class Inferflux < Formula
  desc "InferFlux inference server and CLI"
  homepage "https://github.com/inferencial/InferFlux"
  url "https://github.com/inferencial/InferFlux/archive/refs/heads/main.tar.gz"
  version "0.1.0"
  sha256 "SKIP"
  license "Apache-2.0"

  depends_on "cmake" => :build
  depends_on "openssl@3"
  depends_on "yaml-cpp"

  def install
    system "cmake", "-S", ".", "-B", "build", "-DENABLE_WEBUI=ON", *std_cmake_args
    system "cmake", "--build", "build", "-j"
    bin.install "build/inferfluxd", "build/inferctl"
    pkgshare.install "docs/Quickstart.md"
  end

  test do
    assert_match "Usage", shell_output("#{bin}/inferctl status --help", 1)
  end
end
