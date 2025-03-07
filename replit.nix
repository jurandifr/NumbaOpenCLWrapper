{pkgs}: {
  deps = [
    pkgs.libcxx
    pkgs.opencl-headers
    pkgs.ocl-icd
    pkgs.mesa_drivers
  ];
}
