from helper_funcs import *
from helper_classes import Dla_processor_group
import numpy as np
from time import time_ns

all_macros = retrieve_all_macros()


def dla_sdp_set_producer(group_id: int, rdma_group_id: int):
    """Set producer pointer for all sub-modules"""
    shift_by = all_macros[shift(reg="SDP_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    sdp_reg_write("S_POINTER", reg)

    shift_by = all_macros[shift(reg="SDP_RDMA_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(rdma_group_id, shift_by, dtype=np.int32)
    sdp_rdma_reg_write("S_POINTER", reg)


def dla_sdp_enable(group: Dla_processor_group):
    """Enable
    1. perf registers if engine desires 
    2. rdma if data needed 
    3. SDP"""

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
        group.start_time = np.uint64(time_ns()//1000)  # store time in us

    # enable all sub-modules
    if group.is_rdma_needed:
        reg = all_macros[field_enum(
            "SDP_RDMA_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
        sdp_rdma_reg_write("D_OP_ENABLE", reg)
    reg = all_macros[field_enum("SDP_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
    sdp_reg_write("D_OP_ENABLE", reg)


def dla_sdp_rdma_check(group: Dla_processor_group):
    """Mark rdma as needed in group if x1,x2,y _op's enable is set and their mode!=0 OR data isn't on NVDLA according to sdp_surface"""
    sdp_op = group.operation_desc.sdp_op
    sdp_surface = group.surface_desc.sdp_surface

    # TODO:convert these to boolean so less dumb
    x1_rdma_ena = sdp_op.x1_op.enable  # uint8s
    x2_rdma_ena = sdp_op.x2_op.enable
    y_rdma_ena = sdp_op.y_op.enable

    # TODO:ops per layer here doesn't make a load of sense ... need to understand modes
    sdp_op_per_layer = np.uint8(0)  # SDP_OP_PER_LAYER from dla_interface.h
    f, t = np.uint8(0), np.uint8(1)
    x1_rdma_ena &= (f, t)[sdp_op.x1_op.mode != sdp_op_per_layer]
    x2_rdma_ena &= (f, t)[sdp_op.x2_op.mode != sdp_op_per_layer]
    y_rdma_ena &= (f, t)[sdp_op.y_op.mode != sdp_op_per_layer]

    dla_mem_hw = np.uint16(2)  # DLA_MEM_HW from dla_interface.h
    # fly means data on DLA sub-module
    fly = (sdp_surface.src_data.type == dla_mem_hw)

    a, b, c = x1_rdma_ena > 1
    group.is_rdma_needed = (not fly) | (x1_rdma_ena > 1) | (
        x2_rdma_ena > 1) | (y_rdma_ena > 1)

# ignored for the moment as longggg
# static int32_t processor_sdp_program(struct dla_processor_group *group)

# Look up table (LUT) checking ready (if needed) as sigmoid and tanh need them
# int dla_sdp_is_ready(struct dla_processor *processor,
#                      struct dla_processor_group *group)

# ignored for the moment as dumping
# void dla_sdp_dump_config(struct dla_processor_group *group)


def dla_sdp_program(group: Dla_processor_group):
    """Highest level function"""
    print("Enter SDP")
    mask1 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK1")]
    mask2 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK0")]
    dla_enable_intr(mask1 | mask2)

    ret = processor_sdp_program(group)

    print("Exit SDP")
