import logging
import tempfile

import numpy as np
import pytest
import tvm
import sys
import time
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable
import os
from tvm import te, runtime, topi, tir
from tvm.meta_schedule import Database
from tvm.meta_schedule.database import JSONDatabase


import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
parser.add_argument("--only_show", action="store_true")
parser.add_argument("--only_run", action="store_true")
args = parser.parse_args()

@I.ir_module
class bgemv_16_64_256:
    @T.prim_func
    def main(A: T.Buffer((16, 64, 256), "int32"), B: T.Buffer((16, 256), "int32"), C: T.Buffer((16, 64), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(64, annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 16, 1, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, n_0_i_0_n_1_i_1_fused // 8 * 2 + n_0_i_0_n_1_i_1_fused % 4 // 2 + n_2_init + n_3_init)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 8 // 4 * 32 + n_0_i_0_n_1_i_1_fused % 2 * 16 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 16, 1, 1, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(16, n_0_i_0_n_1_i_1_fused // 8 * 2 + n_0_i_0_n_1_i_1_fused % 4 // 2 + n_2 + n_3)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 8 // 4 * 32 + n_0_i_0_n_1_i_1_fused % 2 * 16 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_16_128_256:
    @T.prim_func
    def main(A: T.Buffer((16, 128, 256), "int32"), B: T.Buffer((16, 256), "int32"), C: T.Buffer((16, 128), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((16, 128), "int32")
        for n_0_i_0_fused in T.parallel(32, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(1, 8):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(2, 4, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(16, n_0_i_0_fused // 4 * 2 + n_1 * 2 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 4 * 32 + i_1 * 4 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 2, 4, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, n_0_i_0_fused // 4 * 2 + n_1 * 2 + n_2 + n_3)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 4 * 32 + i_1 * 4 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
            for ax0 in range(2):
                for ax1_fused in T.vectorized(32):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(16, n_0_i_0_fused // 4 * 2 + ax0)
                        v1 = T.axis.spatial(128, n_0_i_0_fused % 4 * 32 + ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_16_256_256:
    @T.prim_func
    def main(A: T.Buffer((16, 256, 256), "int32"), B: T.Buffer((16, 256), "int32"), C: T.Buffer((16, 256), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((16, 256), "int32")
        for n_0_i_0_fused in T.parallel(128, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(4, 4):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 2, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(16, n_0_i_0_fused // 32 * 4 + n_1 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(256, n_0_i_0_fused % 32 * 8 + i_1 * 2 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 2, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, n_0_i_0_fused // 32 * 4 + n_1 + n_2 + n_3)
                        v_i = T.axis.spatial(256, n_0_i_0_fused % 32 * 8 + i_1 * 2 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
            for ax0 in range(4):
                for ax1_fused in T.vectorized(8):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(16, n_0_i_0_fused // 32 * 4 + ax0)
                        v1 = T.axis.spatial(256, n_0_i_0_fused % 32 * 8 + ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_16_512_256:
    @T.prim_func
    def main(A: T.Buffer((16, 512, 256), "int32"), B: T.Buffer((16, 256), "int32"), C: T.Buffer((16, 512), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_fused in T.parallel(128, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for i_1 in range(8):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 8, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(16, n_0_i_0_n_1_fused // 32 * 4 + n_0_i_0_n_1_fused % 4 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(512, n_0_i_0_n_1_fused % 32 // 4 * 64 + i_1 * 8 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 8, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, n_0_i_0_n_1_fused // 32 * 4 + n_0_i_0_n_1_fused % 4 + n_2 + n_3)
                        v_i = T.axis.spatial(512, n_0_i_0_n_1_fused % 32 // 4 * 64 + i_1 * 8 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_28_64_256:
    @T.prim_func
    def main(A: T.Buffer((28, 64, 256), "int32"), B: T.Buffer((28, 256), "int32"), C: T.Buffer((28, 64), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(896, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 2, 1, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(28, n_0_i_0_n_1_i_1_fused // 448 * 14 + n_0_i_0_n_1_i_1_fused % 112 // 8 + n_2_init + n_3_init)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 448 // 112 * 16 + n_0_i_0_n_1_i_1_fused % 8 * 2 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 2, 1, 1, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(28, n_0_i_0_n_1_i_1_fused // 448 * 14 + n_0_i_0_n_1_i_1_fused % 112 // 8 + n_2 + n_3)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 448 // 112 * 16 + n_0_i_0_n_1_i_1_fused % 8 * 2 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_28_128_256:
    @T.prim_func
    def main(A: T.Buffer((28, 128, 256), "int32"), B: T.Buffer((28, 256), "int32"), C: T.Buffer((28, 128), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(448, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 8, 1, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(28, n_0_i_0_n_1_i_1_fused // 32 * 2 + n_0_i_0_n_1_i_1_fused % 4 // 2 + n_2_init + n_3_init)
                    v_i = T.axis.spatial(128, n_0_i_0_n_1_i_1_fused % 32 // 4 * 16 + n_0_i_0_n_1_i_1_fused % 2 * 8 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 8, 1, 1, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(28, n_0_i_0_n_1_i_1_fused // 32 * 2 + n_0_i_0_n_1_i_1_fused % 4 // 2 + n_2 + n_3)
                    v_i = T.axis.spatial(128, n_0_i_0_n_1_i_1_fused % 32 // 4 * 16 + n_0_i_0_n_1_i_1_fused % 2 * 8 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_28_256_256:
    @T.prim_func
    def main(A: T.Buffer((28, 256, 256), "int32"), B: T.Buffer((28, 256), "int32"), C: T.Buffer((28, 256), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_fused in T.parallel(224, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for i_1 in range(2):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(2, 8, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(28, n_0_i_0_n_1_fused % 14 * 2 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(256, n_0_i_0_n_1_fused // 14 * 16 + i_1 * 8 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 2, 8, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(28, n_0_i_0_n_1_fused % 14 * 2 + n_2 + n_3)
                        v_i = T.axis.spatial(256, n_0_i_0_n_1_fused // 14 * 16 + i_1 * 8 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_28_512_256:
    @T.prim_func
    def main(A: T.Buffer((28, 512, 256), "int32"), B: T.Buffer((28, 256), "int32"), C: T.Buffer((28, 512), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((28, 512), "int32")
        for n_0_i_0_fused in T.parallel(448, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(2, 1):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(2, 8, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(28, n_0_i_0_fused // 64 * 4 + n_1 * 2 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(512, n_0_i_0_fused % 64 * 8 + i_1 * 8 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 2, 8, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(28, n_0_i_0_fused // 64 * 4 + n_1 * 2 + n_2 + n_3)
                        v_i = T.axis.spatial(512, n_0_i_0_fused % 64 * 8 + i_1 * 8 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
            for ax0 in range(4):
                for ax1_fused in T.vectorized(8):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(28, n_0_i_0_fused // 64 * 4 + ax0)
                        v1 = T.axis.spatial(512, n_0_i_0_fused % 64 * 8 + ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_64_64_256:
    @T.prim_func
    def main(A: T.Buffer((64, 64, 256), "int32"), B: T.Buffer((64, 256), "int32"), C: T.Buffer((64, 64), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((64, 64), "int32")
        for n_0_i_0_fused in T.parallel(32, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(1, 8):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(2, 8, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(64, n_0_i_0_fused * 2 + n_1 * 2 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(64, i_1 * 8 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 2, 8, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(64, n_0_i_0_fused * 2 + n_1 * 2 + n_2 + n_3)
                        v_i = T.axis.spatial(64, i_1 * 8 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
            for ax0 in range(2):
                for ax1_fused in T.vectorized(64):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(64, n_0_i_0_fused * 2 + ax0)
                        v1 = T.axis.spatial(64, ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_64_128_256:
    @T.prim_func
    def main(A: T.Buffer((64, 128, 256), "int32"), B: T.Buffer((64, 256), "int32"), C: T.Buffer((64, 128), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((64, 128), "int32")
        for n_0_i_0_fused in T.parallel(2048, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(1, 4):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 1, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(64, n_0_i_0_fused // 32 + n_1 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 32 * 4 + i_1 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 1, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(64, n_0_i_0_fused // 32 + n_1 + n_2 + n_3)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 32 * 4 + i_1 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
                for ax0, ax1 in T.grid(1, 1):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(64, n_0_i_0_fused // 32 + ax0)
                        v1 = T.axis.spatial(128, n_0_i_0_fused % 32 * 4 + i_1 + ax1)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_64_256_256:
    @T.prim_func
    def main(A: T.Buffer((64, 256, 256), "int32"), B: T.Buffer((64, 256), "int32"), C: T.Buffer((64, 256), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(1024, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 16, 1, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused // 64 * 4 + n_0_i_0_n_1_i_1_fused % 4 + n_2_init + n_3_init)
                    v_i = T.axis.spatial(256, n_0_i_0_n_1_i_1_fused % 64 // 4 * 16 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 16, 1, 1, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused // 64 * 4 + n_0_i_0_n_1_i_1_fused % 4 + n_2 + n_3)
                    v_i = T.axis.spatial(256, n_0_i_0_n_1_i_1_fused % 64 // 4 * 16 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_64_512_256:
    @T.prim_func
    def main(A: T.Buffer((64, 512, 256), "int32"), B: T.Buffer((64, 256), "int32"), C: T.Buffer((64, 512), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_fused in T.parallel(32, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for i_0, n_1, i_1 in T.grid(1, 1, 64):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(2, 8, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(64, n_0_fused * 2 + n_1 * 2 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(512, i_0 * 512 + i_1 * 8 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 2, 8, 1, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(64, n_0_fused * 2 + n_1 * 2 + n_2 + n_3)
                        v_i = T.axis.spatial(512, i_0 * 512 + i_1 * 8 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_112_64_256:
    @T.prim_func
    def main(A: T.Buffer((112, 64, 256), "int32"), B: T.Buffer((112, 256), "int32"), C: T.Buffer((112, 64), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(1792):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 2, 2, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(112, n_0_i_0_n_1_i_1_fused // 64 * 4 + n_0_i_0_n_1_i_1_fused % 8 // 4 * 2 + n_2_init * 2 + n_3_init)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 64 // 8 * 8 + n_0_i_0_n_1_i_1_fused % 4 * 2 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 2, 1, 2, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(112, n_0_i_0_n_1_i_1_fused // 64 * 4 + n_0_i_0_n_1_i_1_fused % 8 // 4 * 2 + n_2 * 2 + n_3)
                    v_i = T.axis.spatial(64, n_0_i_0_n_1_i_1_fused % 64 // 8 * 8 + n_0_i_0_n_1_i_1_fused % 4 * 2 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_112_128_256:
    @T.prim_func
    def main(A: T.Buffer((112, 128, 256), "int32"), B: T.Buffer((112, 256), "int32"), C: T.Buffer((112, 128), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((112, 128), "int32")
        for n_0_i_0_fused in T.parallel(1792, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(2, 1):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 2, 2, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(112, n_0_i_0_fused // 64 * 4 + n_1 * 2 + n_2_init * 2 + n_3_init)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 64 * 2 + i_1 * 2 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 2, 1, 2, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(112, n_0_i_0_fused // 64 * 4 + n_1 * 2 + n_2 * 2 + n_3)
                        v_i = T.axis.spatial(128, n_0_i_0_fused % 64 * 2 + i_1 * 2 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
            for ax0 in range(4):
                for ax1_fused in T.vectorized(2):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(112, n_0_i_0_fused // 64 * 4 + ax0)
                        v1 = T.axis.spatial(128, n_0_i_0_fused % 64 * 2 + ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

@I.ir_module
class bgemv_112_256_256:
    @T.prim_func
    def main(A: T.Buffer((112, 256, 256), "int32"), B: T.Buffer((112, 256), "int32"), C: T.Buffer((112, 256), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n_0_i_0_n_1_i_1_fused in T.parallel(1792, annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 16, 1, 1):
                with T.block("C_init"):
                    v_n = T.axis.spatial(112, n_0_i_0_n_1_i_1_fused // 128 * 8 + n_0_i_0_n_1_i_1_fused % 64 // 8 + n_2_init + n_3_init)
                    v_i = T.axis.spatial(256, n_0_i_0_n_1_i_1_fused % 128 // 64 * 128 + n_0_i_0_n_1_i_1_fused % 8 * 16 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = 0
            for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(256, 1, 16, 1, 1, 1):
                with T.block("C_update"):
                    v_n = T.axis.spatial(112, n_0_i_0_n_1_i_1_fused // 128 * 8 + n_0_i_0_n_1_i_1_fused % 64 // 8 + n_2 + n_3)
                    v_i = T.axis.spatial(256, n_0_i_0_n_1_i_1_fused % 128 // 64 * 128 + n_0_i_0_n_1_i_1_fused % 8 * 16 + i_2 + i_3)
                    v_k = T.axis.reduce(256, k_0 + k_1)
                    T.reads(C[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                    T.writes(C[v_n, v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

@I.ir_module
class bgemv_112_512_256:
    @T.prim_func
    def main(A: T.Buffer((112, 512, 256), "int32"), B: T.Buffer((112, 256), "int32"), C: T.Buffer((112, 512), "int32")):
        T.func_attr({"global_symbol": "batched_gemv", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((112, 512), "int32")
        for n_0_i_0_fused in T.parallel(28672, annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for n_1, i_1 in T.grid(2, 1):
                for n_2_init, i_2_init, n_3_init, i_3_init in T.grid(1, 1, 1, 1):
                    with T.block("C_init"):
                        v_n = T.axis.spatial(112, n_0_i_0_fused // 512 * 2 + n_1 + n_2_init + n_3_init)
                        v_i = T.axis.spatial(512, n_0_i_0_fused % 512 + i_1 + i_2_init + i_3_init)
                        T.reads()
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = 0
                for k_0, n_2, i_2, k_1, n_3, i_3 in T.grid(4, 1, 1, 64, 1, 1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(112, n_0_i_0_fused // 512 * 2 + n_1 + n_2 + n_3)
                        v_i = T.axis.spatial(512, n_0_i_0_fused % 512 + i_1 + i_2 + i_3)
                        v_k = T.axis.reduce(256, k_0 * 64 + k_1)
                        T.reads(C_global[v_n, v_i], A[v_n, v_i, v_k], B[v_n, v_k])
                        T.writes(C_global[v_n, v_i])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_n, v_i] = C_global[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
                for ax0, ax1 in T.grid(1, 1):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(112, n_0_i_0_fused // 512 * 2 + n_1 + ax0)
                        v1 = T.axis.spatial(512, n_0_i_0_fused % 512 + ax1)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]

dims = [
    (16, 64, 256, bgemv_16_64_256),
    # (16, 128, 256, bgemv_16_128_256),
    # (16, 256, 256, bgemv_16_256_256),
    # (16, 512, 256, bgemv_16_512_256),
    # (28, 64, 256, bgemv_28_64_256),
    # (28, 128, 256, bgemv_28_128_256),
    # (28, 256, 256, bgemv_28_256_256),
    # (28, 512, 256, bgemv_28_512_256),
    # (64, 64, 256, bgemv_64_64_256),
    # (64, 128, 256, bgemv_64_128_256),
    # (64, 256, 256, bgemv_64_256_256),
    # (64, 512, 256, bgemv_64_512_256),
    # (112, 64, 256, bgemv_112_64_256),
    # (112, 128, 256, bgemv_112_128_256),
    # (112, 256, 256, bgemv_112_256_256),
    # (112, 512, 256, bgemv_112_512_256),
]

target = Target("llvm --num-cores=96")

for m, n, k, cl in dims:
    sch = tvm.tir.Schedule(cl)
    func = tvm.build(sch.mod, target=target, name="batched_gemv")

    A = tvm.nd.array(np.random.randint(0, 100, (m, n, k), dtype="int32"))
    B = tvm.nd.array(np.random.randint(0, 100, (m, k), dtype="int32"))
    C = tvm.nd.array(np.zeros((m, n), dtype="int32"))

    evaluator = func.time_evaluator(
        func.entry_name,
        dev=tvm.device("cpu", 0),
        number=10,
        repeat=10,
    )
    profile_result = evaluator(A, B, C)
    print(profile_result)

# def query(workdir: str, only_show: False, only_run: False) -> None:
#     parsed = workdir.split("_")
#     dtype = "int32"
#     if "red" in workdir or "dot" in workdir:
#         dtype = "int64"

#     database = JSONDatabase(work_dir=workdir)
#     all_records = database.get_all_tuning_records()
#     top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
#     assert len(top_record.run_secs) == 1
#     mod = top_record.workload.mod
#     schedule = lambda *args, **kwargs: ms.tir_integration.compile_tir(database, mod, target)
#     if not only_run:
#         schedule().trace.show()
#     if only_show:
#         return

#     op_class = get_module(
#         parsed[1],
#     )
#     workload = op_class(
#         repeat=100,
#         warmup=10,
#         verbose=False,
#         compile_only=False,
#         output_format="tab"
#     )
#     workload.test(
#         schedule,
#         M=int(parsed[2]),
#         N=int(parsed[3]),
#         K=int(parsed[4]),
#         n_xb=-1,
#         n_yb=-1,
#         n_t=-1,
#         n_rt=-1,
#         n_cache=-1,
#         dtype=dtype,
#     )


# if __name__ == "__main__":
#     query(args.workdir, args.only_show, args.only_run)