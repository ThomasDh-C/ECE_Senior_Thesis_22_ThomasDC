from helper_funcs import *
from helper_classes import dla_processor_group
import numpy as np
from time import time_ns

all_macros = retrieve_all_macros()


def dla_sdp_set_producer(group_id: int, rdma_group_id: int):
    # set producer pointer for all sub-modules
    shift_by = all_macros[shift(reg="SDP_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    sdp_reg_write("S_POINTER", reg)

    shift_by = all_macros[shift(reg="SDP_RDMA_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    sdp_rdma_reg_write("S_POINTER", reg)

# enable 1. perf registers if engine desires 2. rdma if data needed 3. sdp
def dla_sdp_enable(group: dla_processor_group):
    # enable performance counting registers
    enable_stats = False
    if enable_stats:
        # enable performance counting registers
        l1 = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_DMA_EN", "YES")]
        r1 = all_macros[shift(reg="SDP_D_PERF_ENABLE_0", field="PERF_DMA_EN")]
        l2 = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_LUT_EN", "YES")]
        r2 = all_macros[shift(reg="SDP_D_PERF_ENABLE_0", field="PERF_LUT_EN")]
        l3 = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_SAT_EN", "YES")]
        r3 = all_macros[shift(reg="SDP_D_PERF_ENABLE_0", field="PERF_SAT_EN")]
        l4 = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_NAN_INF_COUNT_EN", "YES")]
        r4 = all_macros[shift(reg="SDP_D_PERF_ENABLE_0",
                              field="PERF_NAN_INF_COUNT_EN")]
        perf_reg = (l1 << r1) | (l2 << r2) | (l3 << r3) | (l4 << r4)

        sdp_reg_write("D_PERF_ENABLE", perf_reg)
        group.start_time = np.uint64(time_ns()//1000) # store time in us

    # enable all sub-modules
    if group.is_rdma_needed:
        reg = all_macros[field_enum("SDP_RDMA_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
        sdp_rdma_reg_write("D_OP_ENABLE", reg)
    reg = all_macros[field_enum("SDP_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
    sdp_reg_write("D_OP_ENABLE", reg)

# assign to group if rdma is needed by any of the ops
# void dla_sdp_rdma_check(struct dla_processor_group *group)

# ignored for the moment as longggg
# static int32_t processor_sdp_program(struct dla_processor_group *group)

# Look up table (LUT) checking ready (if needed) as sigmoid and tanh need them
# int dla_sdp_is_ready(struct dla_processor *processor,
#                      struct dla_processor_group *group)

# ignored for the moment as dumping
# void dla_sdp_dump_config(struct dla_processor_group *group)

#  highest level function
def dla_sdp_program(group: dla_processor_group):
    print("Enter SDP")
    mask1 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK1")]
    mask2 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK0")]
    dla_enable_intr(mask1 | mask2)

    ret = processor_sdp_program(group)

    print("Exit SDP")
