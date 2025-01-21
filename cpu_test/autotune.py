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
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable


def bgemv_factory(N: int, M: int, K: int, dtype="int32"):
    @T.prim_func
    def batched_gemv(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (N, M, K), dtype=dtype)
        B = T.match_buffer(b, (N, K), dtype=dtype)
        C = T.match_buffer(c, (N, M), dtype=dtype)

        for n, i, k in T.grid(N, M, K):
            with T.block("C"):
                v_n, v_i, v_k = T.axis.remap("SSR", [n, i, k])
                with T.init():
                    C[v_n, v_i] = 0
                C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]
    return batched_gemv


# mod_mv = [matvec_factory(163840, 4096, dtype="int32")]


# layers = {6: 16, 13: 20, 30: 28, 175: 48}
tuple_bmv = {
    # "params6B_batch1_token64": (16, 64, 256),
    # "params6B_batch1_token128": (16, 128, 256),
    # "params6B_batch1_token256": (16, 256, 256),
    # "params6B_batch1_token512": (16, 512, 256),
    # "params6B_batch16_token64": (256, 64, 256),
    # "params6B_batch16_token128": (256, 128, 256),
    # "params6B_batch16_token256": (256, 256, 256),
    # "params6B_batch16_token512": (256, 512, 256),
    # "params13B_batch1_token64": (20, 64, 256),
    # "params13B_batch1_token128": (20, 128, 256),
    # "params13B_batch1_token256": (20, 256, 256),
    # "params13B_batch1_token512": (20, 512, 256),
    # "params13B_batch16_token64": (320, 64, 256),
    # "params13B_batch16_token128": (320, 128, 256),
    # "params13B_batch16_token256": (320, 256, 256),
    # "params13B_batch16_token512": (320, 512, 256),
    # "params30B_batch1_token64": (28, 64, 256),
    # "params30B_batch1_token128": (28, 128, 256),
    # "params30B_batch1_token256": (28, 256, 256),
    # "params30B_batch1_token512": (28, 512, 256),
    # "params30B_batch16_token64": (448, 64, 256),
    # "params30B_batch16_token128": (448, 128, 256),
    # "params30B_batch16_token256": (448, 256, 256),
    # "params30B_batch16_token512_1": (448, 512, 256),
    # "params30B_batch16_token512_2": (448, 512, 256),
    # "params30B_batch16_token512_3": (448, 512, 256),
    "params175B_batch1_token64": (48, 64, 256),
    "params175B_batch1_token128": (48, 128, 256),
    "params175B_batch1_token256": (48, 256, 256),
    "params175B_batch1_token512": (48, 512, 256),
    "params175B_batch16_token64": (768, 64, 256),
    "params175B_batch16_token128": (768, 128, 256),
    "params175B_batch16_token256": (768, 256, 256),
    "params175B_batch16_token512": (768, 512, 256),
}

tuple_bmv = [(k, j, 256) for j in [64, 128, 256, 512] for k in [16, 64, 28, 112]]

target = Target("llvm --num-cores=96")

for (N, M, K) in tuple_bmv:
    start = time.time()
    name = f"bgemv_{N}_{M}_{K}"
    with open(f"./autotuner_result/bgemv_{N}_{M}_{K}.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        mod = bgemv_factory(N, M, K, dtype="int32")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir="./autotuner_result",
            max_trials_global=1000,
            num_trials_per_iter=64,
            # num_tuning_cores=1,  # to prevent dpu allocation error
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.trace.show()
            print("######################################################")
            sch.mod.show(
                black_format=False,
                name=name,
            )

        sys.stdout = original_stdout
    end = time.time()
    print("DONE ", name, " in ", end - start, " seconds")