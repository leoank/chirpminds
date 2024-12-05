{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    nixpkgs_ank.url = "github:leoank/nixpkgs/cuda";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs = { self, nixpkgs, flake-utils, systems, ... } @ inputs:
      flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = import nixpkgs {
              system = system;
              config.allowUnfree = true;
              config.cudaSupport = true;
            };

            apkgs = import inputs.nixpkgs_ank {
              system = system;
              config.allowUnfree = true;
              config.cudaSupport = true;
            };
 
            apple = pkgs.darwin.apple_sdk.frameworks;

            libList = [
                # Add needed packages here
                pkgs.stdenv.cc.cc
                pkgs.libGL
                pkgs.glib
              ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with apkgs.cudaPackages_12_4; [
                # Only needed on linux env
                # pkgs.cudaPackages.cudatoolkit
                libcublas
                libcurand
                pkgs.cudaPackages.cudnn
                libcufft
                cuda_cudart

                # This is required for most app that uses graphics api
                pkgs.linuxPackages.nvidia_x11
              ]);

            # mpkgs = import inputs.nixpkgs_master {
            #   system = system;
            #   config.allowUnfree = true;
            # };
          in
          with pkgs;
        {
          devShells = {
              default = let 
                python_with_pkgs = (pkgs.python311.withPackages(pp: [
                  pp.torch
                  pp.torchvision
                  pp.onnx
                  pp.scikit-image
                  # https://github.com/NixOS/nixpkgs/issues/323965
                  # https://github.com/NixOS/nixpkgs/issues/329378#issuecomment-2247861665
                  # Why tensorflow have to do this??
                  # Also for numa node issue: https://github.com/tensorflow/tensorflow/issues/42738#issuecomment-922422874
                  (pp.tensorflow-bin.overrideAttrs {
                    postFixup =
                      let
                        cudaSupport = true;
                        # rpaths we only need to add if CUDA is enabled.
                        cudapaths = [
                          cudaPackages.cudnn
                          apkgs.cudaPackages_12_4.cudatoolkit.out
                          apkgs.cudaPackages_12_4.cudatoolkit.lib
                        ];

                        libpaths = [
                          stdenv.cc.cc.lib
                          zlib
                        ];

                        rpath = lib.makeLibraryPath (libpaths ++ cudapaths);
                        pythonSitePackages = "lib/python3.11/site-packages";
                      in
                      lib.optionalString stdenv.isLinux ''
                        # This is an array containing all the directories in the tensorflow2
                        # package that contain .so files.
                        #
                        # TODO: Create this list programmatically, and remove paths that aren't
                        # actually needed.
                        rrPathArr=(
                          "$out/${pythonSitePackages}/tensorflow/"
                          "$out/${pythonSitePackages}/tensorflow/core/kernels"
                          "$out/${pythonSitePackages}/tensorflow/compiler/mlir/stablehlo/"
                          "$out/${pythonSitePackages}/tensorflow/compiler/tf2tensorrt/"
                          "$out/${pythonSitePackages}/tensorflow/compiler/tf2xla/ops/"
                          "$out/${pythonSitePackages}/tensorflow/include/external/ml_dtypes/"
                          "$out/${pythonSitePackages}/tensorflow/lite/experimental/microfrontend/python/ops/"
                          "$out/${pythonSitePackages}/tensorflow/lite/python/analyzer_wrapper/"
                          "$out/${pythonSitePackages}/tensorflow/lite/python/interpreter_wrapper/"
                          "$out/${pythonSitePackages}/tensorflow/lite/python/metrics/"
                          "$out/${pythonSitePackages}/tensorflow/lite/python/optimize/"
                          "$out/${pythonSitePackages}/tensorflow/python/"
                          "$out/${pythonSitePackages}/tensorflow/python/autograph/impl/testing"
                          "$out/${pythonSitePackages}/tensorflow/python/client"
                          "$out/${pythonSitePackages}/tensorflow/python/data/experimental/service"
                          "$out/${pythonSitePackages}/tensorflow/python/framework"
                          "$out/${pythonSitePackages}/tensorflow/python/grappler"
                          "$out/${pythonSitePackages}/tensorflow/python/lib/core"
                          "$out/${pythonSitePackages}/tensorflow/python/lib/io"
                          "$out/${pythonSitePackages}/tensorflow/python/platform"
                          "$out/${pythonSitePackages}/tensorflow/python/profiler/internal"
                          "$out/${pythonSitePackages}/tensorflow/python/saved_model"
                          "$out/${pythonSitePackages}/tensorflow/python/util"
                          "$out/${pythonSitePackages}/tensorflow/tsl/python/lib/core"
                          "$out/${pythonSitePackages}/tensorflow.libs/"
                          "${rpath}"
                        )

                        # The the bash array into a colon-separated list of RPATHs.
                        rrPath=$(IFS=$':'; echo "''${rrPathArr[*]}")
                        echo "about to run patchelf with the following rpath: $rrPath"

                        find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
                          echo "about to patchelf $lib..."
                          chmod a+rx "$lib"
                          patchelf --set-rpath "$rrPath" "$lib"
                          ${lib.optionalString cudaSupport ''
                            addOpenGLRunpath "$lib"
                          ''}
                        done
                      '';
                  })
                  pp.packaging
                ]));
              in mkShell {
                    NIX_LD = runCommand "ld.so" {} ''
                        ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
                      '';
                    NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
                    packages = [
                      python_with_pkgs
                      python311Packages.venvShellHook
                      uv
                      ffmpeg
                      duckdb
                      gcc
                    ]
                    ++ libList
                    ++ lib.optionals stdenv.isDarwin [
                      darwin.libobjc
                      apple.Security
                      apple.CoreServices
                      apple.CoreFoundation
                      apple.AppKit
                      apple.Foundation
                      apple.ApplicationServices
                      apple.CoreGraphics
                      apple.CoreVideo
                      apple.Carbon
                      apple.IOKit
                      apple.CoreAudio
                      apple.AudioUnit
                      apple.QuartzCore
                      apple.Metal
                    ];
                    venvDir = "./.venv";
                    postVenvCreation = ''
                        unset SOURCE_DATE_EPOCH
                      '';
                    postShellHook = ''
                        unset SOURCE_DATE_EPOCH
                      '';
                    shellHook = ''
                        export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                        export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                        export CUDA_HOME=${apkgs.cudaPackages_12_4.cudatoolkit}
                        runHook venvShellHook
                        export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                    '';
                  };
              };
        }
      );
}

