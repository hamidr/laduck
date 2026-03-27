{
  description = "LaDuck — LLM inference extension for DuckDB via llama.cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    duckdb = {
      url = "github:hamidr/duckdb/nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    extension-ci-tools = {
      url = "github:duckdb/extension-ci-tools";
      flake = false;
    };
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = { pkgs, lib, system, self', ... }: let
        duckdb-pkg = inputs.duckdb.packages.${system}.duckdb;
        duckdb-src = inputs.duckdb;
        ext-ci-tools = inputs.extension-ci-tools;
      in {
        packages = {
          laduck = pkgs.stdenv.mkDerivation {
            pname = "duckdb-laduck";
            version = "0.1.0-dev";
            src = lib.cleanSource ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
            ];

            buildInputs = with pkgs; [
              openssl
            ] ++ lib.optionals stdenv.isDarwin [
              darwin.apple_sdk.frameworks.Metal
              darwin.apple_sdk.frameworks.MetalKit
              darwin.apple_sdk.frameworks.Foundation
              darwin.apple_sdk.frameworks.Accelerate
            ];

            postUnpack = ''
              ln -s ${duckdb-src} $sourceRoot/duckdb
              ln -s ${ext-ci-tools} $sourceRoot/extension-ci-tools
            '';

            buildPhase = ''
              runHook preBuild
              GEN=ninja make release
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/lib
              cp build/release/extension/laduck/laduck.duckdb_extension $out/lib/
              runHook postInstall
            '';

            dontConfigure = true;

            meta = {
              description = "LLM inference extension for DuckDB powered by llama.cpp";
              license = lib.licenses.mit;
            };
          };

          # CUDA variant (Linux only, requires NVIDIA GPU)
          laduck-cuda = pkgs.stdenv.mkDerivation {
            pname = "duckdb-laduck-cuda";
            version = "0.1.0-dev";
            src = lib.cleanSource ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
            ] ++ lib.optionals stdenv.isLinux [
              cudaPackages.cuda_nvcc
            ];

            buildInputs = with pkgs; [
              openssl
            ] ++ lib.optionals stdenv.isLinux [
              cudaPackages.cuda_cudart
              cudaPackages.libcublas
            ];

            postUnpack = ''
              ln -s ${duckdb-src} $sourceRoot/duckdb
              ln -s ${ext-ci-tools} $sourceRoot/extension-ci-tools
            '';

            cmakeFlags = [ "-DGGML_CUDA=ON" ];

            buildPhase = ''
              runHook preBuild
              GEN=ninja EXTRA_CMAKE_VARIABLES="-DGGML_CUDA=ON" make release
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/lib
              cp build/release/extension/laduck/laduck.duckdb_extension $out/lib/
              runHook postInstall
            '';

            dontConfigure = true;

            meta = {
              description = "LLM inference extension for DuckDB (CUDA GPU)";
              license = lib.licenses.mit;
              platforms = [ "x86_64-linux" "aarch64-linux" ];
            };
          };

          # Vulkan variant (cross-platform GPU — AMD, NVIDIA, Intel)
          laduck-vulkan = pkgs.stdenv.mkDerivation {
            pname = "duckdb-laduck-vulkan";
            version = "0.1.0-dev";
            src = lib.cleanSource ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
            ];

            buildInputs = with pkgs; [
              openssl
              vulkan-headers
              vulkan-loader
            ];

            postUnpack = ''
              ln -s ${duckdb-src} $sourceRoot/duckdb
              ln -s ${ext-ci-tools} $sourceRoot/extension-ci-tools
            '';

            buildPhase = ''
              runHook preBuild
              GEN=ninja EXTRA_CMAKE_VARIABLES="-DGGML_VULKAN=ON" make release
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/lib
              cp build/release/extension/laduck/laduck.duckdb_extension $out/lib/
              runHook postInstall
            '';

            dontConfigure = true;

            meta = {
              description = "LLM inference extension for DuckDB (Vulkan GPU)";
              license = lib.licenses.mit;
            };
          };

          # ROCm/HIP variant (Linux only, AMD GPUs)
          laduck-rocm = pkgs.stdenv.mkDerivation {
            pname = "duckdb-laduck-rocm";
            version = "0.1.0-dev";
            src = lib.cleanSource ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
            ] ++ lib.optionals stdenv.isLinux [
              rocmPackages.clr
            ];

            buildInputs = with pkgs; [
              openssl
            ] ++ lib.optionals stdenv.isLinux [
              rocmPackages.clr
              rocmPackages.hipblas
              rocmPackages.rocblas
            ];

            postUnpack = ''
              ln -s ${duckdb-src} $sourceRoot/duckdb
              ln -s ${ext-ci-tools} $sourceRoot/extension-ci-tools
            '';

            buildPhase = ''
              runHook preBuild
              GEN=ninja EXTRA_CMAKE_VARIABLES="-DGGML_HIP=ON" make release
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/lib
              cp build/release/extension/laduck/laduck.duckdb_extension $out/lib/
              runHook postInstall
            '';

            dontConfigure = true;

            meta = {
              description = "LLM inference extension for DuckDB (AMD ROCm GPU)";
              license = lib.licenses.mit;
              platforms = [ "x86_64-linux" ];
            };
          };

          default = self'.packages.laduck;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            python3
            ccache
            clang-tools
            openssl
          ] ++ lib.optionals stdenv.isLinux [
            gdb
            valgrind
            linuxPackages.perf
          ] ++ lib.optionals stdenv.isDarwin [
            lldb
          ] ++ [
            duckdb-pkg
          ];

          shellHook = ''
            if [ ! -e duckdb ]; then
              ln -sf ${duckdb-src} duckdb
              echo "Linked duckdb/ → ${duckdb-src}"
            fi
            if [ ! -e extension-ci-tools ]; then
              ln -sf ${ext-ci-tools} extension-ci-tools
              echo "Linked extension-ci-tools/ → ${ext-ci-tools}"
            fi

            echo ""
            echo "LaDuck extension dev shell ready."
            echo "  Build:  GEN=ninja make release"
            echo "  Test:   make test"
            echo "  DuckDB: duckdb -cmd \"LOAD 'build/release/extension/laduck/laduck.duckdb_extension'\""
            echo ""
          '';
        };
      };
    };
}
