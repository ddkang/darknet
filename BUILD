package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library"
)

tf_kernel_library(
    name = "yolo",
    copts = ["-DGPU", "-DCUDNN"],
    srcs = glob(["src/*.c"]),
    gpu_srcs = glob(["src/*.cc"]) + glob(["src/*.h"]),
    hdrs = glob(["src/*.h"]),
    deps = [
        "//tensorflow/core:tensorflow"
    ],
)
