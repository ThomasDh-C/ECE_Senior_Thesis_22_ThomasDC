from helper_funcs import *
from helper_classes import Dla_processor_group
import numpy as np
from time import time_ns
from copy import copy
import json

all_macros = retrieve_all_macros()


def dla_sdp_set_producer(group_id: int, rdma_group_id: int, acceleratorIR, MMIO):
    """Set producer pointer for all sub-modules"""

    shift_by = all_macros[shift(reg="SDP_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': 'SDP_S_POINTER',
                          'group_id': group_id,
                          'group_id_shift': shift_by})
    sdp_reg_write("S_POINTER", reg, all_macros, MMIO)

    shift_by = all_macros[shift(reg="SDP_RDMA_S_POINTER_0", field="PRODUCER")]
    reg = np.left_shift(rdma_group_id, shift_by, dtype=np.int32)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': 'SDPRDMA_S_POINTER',
                          'rdma_group_id': rdma_group_id,
                          'rdma_group_id_shift': shift_by})
    sdp_rdma_reg_write("S_POINTER", reg, all_macros, MMIO)


def dla_sdp_enable(group: Dla_processor_group, acceleratorIR, MMIO):
    """Enable
    1. perf registers if engine desires 
    2. rdma if data needed 
    3. SDP"""

    # enable performance counting registers
    enable_stats = False
    if enable_stats:
        # enable performance counting registers
        dma_stats_enable = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_DMA_EN", "YES")]
        dma_stats_enable_shift = all_macros[shift(
            reg="SDP_D_PERF_ENABLE_0", field="PERF_DMA_EN")]
        lut_stats_enable = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_LUT_EN", "YES")]
        lut_stats_enable_shift = all_macros[shift(
            reg="SDP_D_PERF_ENABLE_0", field="PERF_LUT_EN")]
        sat_stats_enable = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_SAT_EN", "YES")]
        sat_stats_enable_shift = all_macros[shift(
            reg="SDP_D_PERF_ENABLE_0", field="PERF_SAT_EN")]
        nan_stats_enable = all_macros[field_enum(
            "SDP_D_PERF_ENABLE_0", "PERF_NAN_INF_COUNT_EN", "YES")]
        nan_stats_enable_shift = all_macros[shift(reg="SDP_D_PERF_ENABLE_0",
                                                  field="PERF_NAN_INF_COUNT_EN")]
        perf_reg = (dma_stats_enable << dma_stats_enable_shift) | (lut_stats_enable << lut_stats_enable_shift) | (
            sat_stats_enable << sat_stats_enable_shift) | (nan_stats_enable << nan_stats_enable_shift)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': 'SDP_D_PERF_ENABLE',
                              'dma_stats_enable': dma_stats_enable,
                              'dma_stats_enable_shift': dma_stats_enable_shift,
                              'lut_stats_enable': lut_stats_enable,
                              'lut_stats_enable_shift': lut_stats_enable_shift,
                              'sat_stats_enable': sat_stats_enable,
                              'sat_stats_enable_shift': sat_stats_enable_shift,
                              'nan_stats_enable': nan_stats_enable,
                              'nan_stats_enable_shift': nan_stats_enable_shift
                              })
        sdp_reg_write("D_PERF_ENABLE", perf_reg, all_macros, MMIO)
        group.start_time = np.uint64(time_ns()//1000)  # store time in us

    # enable all sub-modules
    if group.is_rdma_needed:
        reg = all_macros[field_enum(
            "SDP_RDMA_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': 'SDPRDMA_D_OP_ENABLE',
                              'enable': reg})
        sdp_rdma_reg_write("D_OP_ENABLE", reg, all_macros, MMIO)

    reg = all_macros[field_enum("SDP_D_OP_ENABLE_0", "OP_EN", "ENABLE")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': 'SDP_D_OP_ENABLE',
                          'enable': reg})
    sdp_reg_write("D_OP_ENABLE", reg, all_macros, MMIO)


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


def config_source_info(xy_rdma_ena: np.uint8, xy_op: Dla_sdp_op, xy_data: Dla_data_cube, xy_addr, xy_name: str, acceleratorIR, MMIO):
    """Config source info for xy = x1, x2 or y"""
    # for shifts have to use shift for correct mem and for address if it is enabled
    str_to_letters = {"x1": ("BRDMA", "BS"), "x2": (
        "NRDMA", "BN"), "y": ("ERDMA", "EW")}
    rdma, bias_data_cube = str_to_letters[xy_name]
    rdma_disable = all_macros[field_enum(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DISABLE", ("YES", "NO")[xy_rdma_ena])]
    rdma_disable_shift = all_macros[shift(
        f"SDP_RDMA_D_{rdma}_CFG_0", f"{rdma}_DISABLE")]
    rdma_to_which_engine = all_macros[field_enum(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DATA_USE", ("MUL", "MUL", "ALU", "BOTH")[xy_op.type])]
    rdma_to_which_engine_shift = all_macros[shift(
        f"SDP_RDMA_D_{rdma}_CFG_0", f"{rdma}_DATA_USE")]
    prec = all_macros[field_enum("SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DATA_SIZE", (
        "ONE_BYTE", "TWO_BYTE", "TWO_BYTE")[xy_op.precision])]
    prec_shift = all_macros[shift(
        f"SDP_RDMA_D_{rdma}_CFG_0", f"{rdma}_DATA_SIZE")]
    mode = all_macros[field_enum("SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DATA_MODE", (
        "PER_ELEMENT", "PER_KERNEL", "PER_ELEMENT")[xy_op.mode])]
    mode_shift = all_macros[shift(
        f"SDP_RDMA_D_{rdma}_CFG_0", f"{rdma}_DATA_MODE")]
    ram_type = all_macros[field_enum(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_RAM_TYPE", ("MC", "CV", "CV")[xy_data.type])]  # last CV is a bodge as insufficient documentaion
    ram_type_shift = all_macros[shift(
        f"SDP_RDMA_D_{rdma}_CFG_0", f"{rdma}_RAM_TYPE")]

    reg = (rdma_disable << rdma_disable_shift) | (rdma_to_which_engine << rdma_to_which_engine_shift) | (
        prec << prec_shift) | (mode << mode_shift) | (ram_type << ram_type_shift)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f'SDPRDMA_D_{rdma}_CFG',
                          'rdma_disable': rdma_disable,
                          'rdma_disable_shift': rdma_disable_shift,
                          'rdma_to_which_engine': rdma_to_which_engine,
                          'rdma_to_which_engine_shift': rdma_to_which_engine_shift,
                          'prec': prec,
                          'prec_shift': prec_shift,
                          'mode': mode,
                          'mode': mode_shift,
                          'ram_type': ram_type,
                          'ram_type_shift': ram_type_shift
                          })
    sdp_rdma_reg_write(f"D_{rdma}_CFG", reg, all_macros, MMIO)

    if xy_rdma_ena:
        high = high32bits(xy_addr)
        low = low32bits(xy_addr)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDPRDMA_D_{bias_data_cube}_BASE_ADDR_LOW",
                              'low': pad_hex(low)})
        sdp_rdma_reg_write(
            f"D_{bias_data_cube}_BASE_ADDR_LOW", low, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDPRDMA_D_{bias_data_cube}_BASE_ADDR_HIGH",
                              'high': pad_hex(high)})
        sdp_rdma_reg_write(
            f"D_{bias_data_cube}_BASE_ADDR_HIGH", high, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDPRDMA_D_{bias_data_cube}_LINE_STRIDE",
                              'line_stride': xy_data.line_stride})
        sdp_rdma_reg_write(
            f"D_{bias_data_cube}_LINE_STRIDE", xy_data.line_stride, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDPRDMA_D_{bias_data_cube}_SURFACE_STRIDE",
                              'surf_stride': xy_data.surf_stride})
        sdp_rdma_reg_write(
            f"D_{bias_data_cube}_SURFACE_STRIDE", xy_data.surf_stride, all_macros, MMIO)


def config_bias_data_cubes(xy_op: Dla_sdp_op, xy_name: str, acceleratorIR, MMIO):
    str_to_letters = {"x1": "BS", "x2":  "BN", "y":  "EW"}
    # bias data cube = bdc
    bdc = str_to_letters[xy_name]
    yt = xy_name == "y"  # yt = true we are doing y
    bypass = all_macros[field_enum(
        "SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[xy_op.enable])]
    bypass_shift = all_macros[shift(f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_BYPASS")]
    sdp_op_both, sdp_op_add, sdp_op_mul, sdp_op_none = 3, 2, 1, 0
    ab_idx = xy_op.type != sdp_op_mul & xy_op.type != sdp_op_none
    alu_bypass = all_macros[field_enum(
        "SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[bool(ab_idx)])]
    alu_bypass_shift = all_macros[shift(
        f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_ALU_BYPASS")]
    alu_algo = all_macros[field_enum(
        "SDP_D_DP_EW_CFG_0", "EW_ALU_ALGO", ("MAX", "MIN", "SUM", "EQL")[xy_op.alu_type])]
    alu_algo_shift = all_macros[shift(
        f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_ALU_ALGO")]
    mb_idx = (xy_op.type != sdp_op_add) & (xy_op.type != sdp_op_none)
    mul_bypass = all_macros[field_enum(
        "SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[bool(mb_idx)])]
    mul_bypass_shift = all_macros[shift(
        f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_MUL_BYPASS")]
    activation_prelu, activation_lut, activation_relu = 3, 2, 1
    mul_prelu = all_macros[field_enum("SDP_D_DP_BS_CFG_0", "BS_MUL_PRELU", ("NO", "YES")[
                                      bool(xy_op.act == activation_prelu)])]
    mul_prelu_shift = all_macros[shift(
        f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_MUL_PRELU")]
    relu_bypass = all_macros[field_enum("SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[
                                        bool(xy_op.act == (activation_relu, activation_lut)[yt])])]
    relulut = ("RELU", "LUT")[yt]
    relu_bypass_shift = all_macros[shift(
        f"SDP_D_DP_{bdc}_CFG_0", f"{bdc}_{relulut}_BYPASS")]
    reg = (bypass << bypass_shift) | (alu_bypass << alu_bypass_shift) | (alu_algo << alu_algo_shift) | (
        mul_bypass << mul_bypass_shift) | (mul_prelu << mul_prelu_shift) | (relu_bypass << relu_bypass_shift)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DP_{bdc}_CFG",
                          'bypass': bypass,
                          'bypass_shift': bypass_shift,
                          'alu_bypass': alu_bypass,
                          'alu_bypass_shift': alu_bypass,
                          'alu_algo': alu_algo,
                          'alu_algo_shift': alu_algo_shift,
                          'mul_bypass': mul_bypass,
                          'mul_bypass_shift': mul_bypass_shift,
                          'mul_prelu': mul_prelu,
                          'mul_prelu_shift': mul_prelu_shift,
                          'relu_bypass': relu_bypass,
                          'relu_bypass_shift': relu_bypass_shift
                          })
    sdp_reg_write(f"D_DP_{bdc}_CFG", reg, all_macros, MMIO)

    # TODO: check all types on if statements without value checking
    if xy_op.enable:
        sdp_op_per_layer = 0
        if (xy_op.type == sdp_op_add | xy_op.type == sdp_op_both):
            alu_src_str = field_enum("SDP_D_DP_BS_ALU_CFG_0", "BS_ALU_SRC", ("MEM", "REG")[
                                     xy_op.mode == sdp_op_per_layer])
            alu_src = all_macros[alu_src_str]
            alu_src_shift = all_macros[shift(
                f"SDP_D_DP_{bdc}_ALU_CFG_0", f"{bdc}_ALU_SRC")]
            op_shift = xy_op.shift_value
            op_shift_shift = all_macros[shift(
                f"SDP_D_DP_{bdc}_ALU_CFG_0", f"{bdc}_ALU_SHIFT_VALUE")]
            if yt:
                op_shift = all_macros[field_enum(
                    "SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[xy_op.cvt.alu_cvt.enable])]
                op_shift_shift = all_macros[shift(
                    f"SDP_D_DP_{bdc}_ALU_CFG_0", f"{bdc}_ALU_CVT_BYPASS")]
            reg = (alu_src << alu_src_shift) | (op_shift << op_shift_shift)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDP_D_DP_{bdc}_ALU_CFG",
                                  'alu_src': alu_src,
                                  'alu_src_shift': alu_src_shift,
                                  'op_shift': op_shift,
                                  'op_shift_shift': op_shift_shift
                                  })
            sdp_reg_write(f"D_DP_{bdc}_ALU_CFG", reg, all_macros, MMIO)

            if yt:
                if xy_op.mode == sdp_op_per_layer:
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_ALU_SRC_VALUE",
                                          'alu_op': xy_op.alu_operand
                                          })
                    sdp_reg_write("D_DP_EW_ALU_SRC_VALUE",
                                  xy_op.alu_operand, all_macros, MMIO)
                else:
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_ALU_CVT_OFFSET_VALUE",
                                          'offset': xy_op.cvt.alu_cvt.offset
                                          })
                    sdp_reg_write("D_DP_EW_ALU_CVT_OFFSET_VALUE",
                                  xy_op.cvt.alu_cvt.offset, all_macros, MMIO)
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_ALU_CVT_SCALE_VALUE",
                                          'scale': xy_op.cvt.alu_cvt.scale
                                          })
                    sdp_reg_write("D_DP_EW_ALU_CVT_SCALE_VALUE",
                                  xy_op.cvt.alu_cvt.scale, all_macros, MMIO)
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_ALU_CVT_TRUNCATE_VALUE",
                                          'truncate': xy_op.cvt.alu_cvt.truncate
                                          })
                    sdp_reg_write("D_DP_EW_ALU_CVT_TRUNCATE_VALUE",
                                  xy_op.cvt.alu_cvt.truncate, all_macros, MMIO)
        if not yt:
            if xy_op.mode == sdp_op_per_layer:
                acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                      'name': f"SDP_D_DP_{bdc}_ALU_SRC_VALUE",
                                      'alu_op': xy_op.alu_operand
                                      })
                sdp_reg_write(f"D_DP_{bdc}_ALU_SRC_VALUE",
                              xy_op.alu_operand, all_macros, MMIO)
                acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                      'name': f"SDP_D_DP_{bdc}_MUL_SRC_VALUE",
                                      'mul_op': xy_op.mul_operand
                                      })
                sdp_reg_write(f"D_DP_{bdc}_MUL_SRC_VALUE",
                              xy_op.mul_operand, all_macros, MMIO)

            # MUL truncate will take effect no matter whether MUL is bypassed or not
            mul_src_str = field_enum("SDP_D_DP_BS_ALU_CFG_0", "BS_ALU_SRC", ("MEM", "REG")[
                                     bool(xy_op.mode == sdp_op_per_layer)])
            mul_src = all_macros[mul_src_str]
            mul_src_shift = all_macros[shift(
                f"SDP_D_DP_{bdc}_MUL_CFG_0", f"{bdc}_MUL_SRC")]
            trunc = xy_op.truncate
            trunc_shift = all_macros[shift(
                f"SDP_D_DP_{bdc}_MUL_CFG_0", f"{bdc}_MUL_SHIFT_VALUE")]

            reg = (mul_src << mul_src_shift) | (trunc << trunc_shift)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDP_D_DP_{bdc}_MUL_CFG",
                                  'mul_src': mul_src,
                                  'mul_src_shift': mul_src_shift,
                                  'trunc': trunc,
                                  'trunc_shift': trunc_shift
                                  })
            sdp_reg_write(f"D_DP_{bdc}_MUL_CFG", reg, all_macros, MMIO)
        else:
            if (xy_op.type == sdp_op_mul | xy_op.type == sdp_op_both):
                mul_src = all_macros[field_enum("SDP_D_DP_BS_ALU_CFG_0", "BS_ALU_SRC", ("MEM", "REG")[
                                                xy_op.mode == sdp_op_per_layer])]
                mul_src_shift = all_macros[shift(
                    "SDP_D_DP_EW_MUL_CFG_0", "EW_MUL_SRC")]
                mul_cvt_bypass = all_macros[field_enum(
                    "SDP_D_DP_BS_CFG_0", "BS_BYPASS", ("YES", "NO")[xy_op.cvt.mul_cvt.enable])]
                mul_cvt_bypass_shift = all_macros[shift(
                    "SDP_D_DP_EW_MUL_CFG_0", "EW_MUL_CVT_BYPASS")]
                reg = (mul_src << mul_src_shift) | (
                    mul_cvt_bypass << mul_cvt_bypass_shift)
                acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                      'name': f"SDP_D_DP_EW_MUL_CFG",
                                      'mul_src': mul_src,
                                      'mul_src_shift': mul_src_shift,
                                      'mul_cvt_bypass': mul_cvt_bypass,
                                      'mul_cvt_bypass_shift': mul_cvt_bypass_shift
                                      })
                sdp_reg_write("D_DP_EW_MUL_CFG", reg, all_macros, MMIO)

                if xy_op.mode == sdp_op_per_layer:
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_MUL_SRC_VALUE",
                                          'mul_op': xy_op.mul_operand
                                          })
                    sdp_reg_write("D_DP_EW_MUL_SRC_VALUE",
                                  xy_op.mul_operand, all_macros, MMIO)
                else:
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_MUL_CVT_OFFSET_VALUE",
                                          'offset': xy_op.cvt.mul_cvt.offset
                                          })
                    sdp_reg_write("D_DP_EW_MUL_CVT_OFFSET_VALUE",
                                  xy_op.cvt.mul_cvt.offset, all_macros, MMIO)
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_MUL_CVT_SCALE_VALUE",
                                          'scale': xy_op.cvt.mul_cvt.scale
                                          })
                    sdp_reg_write("D_DP_EW_MUL_CVT_SCALE_VALUE",
                                  xy_op.cvt.mul_cvt.scale, all_macros, MMIO)
                    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                          'name': f"SDP_D_DP_EW_MUL_CVT_TRUNCATE_VALUE",
                                          'truncate': xy_op.cvt.mul_cvt.truncate
                                          })
                    sdp_reg_write("D_DP_EW_MUL_CVT_TRUNCATE_VALUE",
                                  xy_op.cvt.mul_cvt.truncate, all_macros, MMIO)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDP_D_DP_EW_TRUNCATE_VALUE",
                                  'truncate': xy_op.truncate
                                  })
            sdp_reg_write("D_DP_EW_TRUNCATE_VALUE", xy_op.truncate, all_macros)


def processor_sdp_program(group: Dla_processor_group, acceleratorIR, MMIO):
    """Program all config registers as needed"""
    # atom_size = engine->config_data->atom_size # Work out how to do this

    sdp_op = group.operation_desc.sdp_op
    sdp_surface = group.surface_desc.sdp_surface

    dla_mem_hw = 2  # DLA_MEM_HW
    f, t = np.uint8(0), np.uint8(1)  # TODO: move away from silly c logic

    fly: bool = sdp_surface.src_data.type == dla_mem_hw
    # if output is to move off nvdla need dma
    out_dma_ena = sdp_surface.dst_data.type != dla_mem_hw
    x1_op = sdp_op.x1_op
    x2_op = sdp_op.x2_op
    y_op = sdp_op.y_op

    sdp_op_none = 0  # SDP_OP_NONE
    x1_rdma_ena = x1_op.enable & (f, t)[bool(x1_op.type != sdp_op_none)]
    x2_rdma_ena = x2_op.enable & (f, t)[bool(x2_op.type != sdp_op_none)]
    y_rdma_ena = y_op.enable & (f, t)[bool(y_op.type != sdp_op_none)]

    # load addresses
    src_addr, x1_addr, x2_addr, y_addr, dst_addr = np.uint64(
        -1), np.uint64(-1), np.uint64(-1),  np.uint64(-1), np.uint64(-1)
    if not fly:
        src_addr = dla_read_input_address(
            sdp_surface.src_data, group.op_desc.index, group.roi_index, 1)
        # check_align(src_addr, atom_size, debug=True)

    lut = Dla_lut_param()
    if sdp_op.lut_index >= 0:
        group.lut_index = sdp_op.lut_index  # load lut index from current op
        lut = dla_read_lut()
        # debug lut by printing it if desired

    sdp_op_per_layer = 0
    x1_rdma_ena &= (x1_op.mode != sdp_op_per_layer)
    x2_rdma_ena &= (x2_op.mode != sdp_op_per_layer)
    y_rdma_ena &= (y_op.mode != sdp_op_per_layer)
    if x1_rdma_ena > 0:
        x1_addr = dla_get_dma_cube_address()
        # old params were
        # engine.driver_context, engine.task.task_data, sdp_surface.x1_data.address, sdp_surface.x1_data.offset
        # check_align
    if x2_rdma_ena > 0:
        x2_addr = dla_get_dma_cube_address()
        # old params were
        # engine.driver_context, engine.task.task_data, sdp_surface.x2_data.address, sdp_surface.x2_data.offset
        # check_align
    if y_rdma_ena > 0:
        y_addr = dla_get_dma_cube_address()
        # old params were
        # engine.driver_context, engine.task.task_data, sdp_surface.y_data.address, sdp_surface.y_data.offset
        # check_align

    if out_dma_ena:
        dst_addr = dla_get_dma_cube_address()
        # check_align(dst_addr, atom_size, debug=True)

    # flying mode
    fly_mode = all_macros[field_enum(
        "SDP_D_FEATURE_MODE_CFG_0", "FLYING_MODE", "OFF")]
    fly_mode_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "FLYING_MODE")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDPRDMA_D_FEATURE_MODE_CFG",
                          'fly_mode': fly_mode,
                          'fly_mode_shift': fly_mode_shift,
                          })
    sdp_rdma_reg_write("D_FEATURE_MODE_CFG", fly_mode <<
                       fly_mode_shift, all_macros, MMIO)

    # bdma, nrdma, erdma (for x1, x2 and y respectively)
    # turn off disable for all
    disable_brdma = all_macros[field_enum(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DISABLE", "NO")]
    disable_brdma_shift = all_macros[shift(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_DISABLE")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDPRDMA_D_BRDMA_CFG",
                          'disable': disable_brdma_shift,
                          'disable_shift': disable_brdma_shift,
                          })
    sdp_rdma_reg_write("D_BRDMA_CFG", disable_brdma <<
                       disable_brdma_shift, all_macros, MMIO)

    disable_nrdma_shift = all_macros[shift(
        "SDP_RDMA_D_NRDMA_CFG_0", "NRDMA_DISABLE")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDPRDMA_D_NRDMA_CFG",
                          'disable': disable_brdma_shift,
                          'disable_shift': disable_nrdma_shift,
                          })
    sdp_rdma_reg_write("D_NRDMA_CFG", disable_brdma <<
                       disable_nrdma_shift, all_macros, MMIO)

    disable_erdma_shift = all_macros[shift(
        "SDP_RDMA_D_ERDMA_CFG_0", "ERDMA_DISABLE")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDPRDMA_D_ERDMA_CFG",
                          'disable': disable_brdma_shift,
                          'disable_shift': disable_erdma_shift,
                          })
    sdp_rdma_reg_write("D_ERDMA_CFG", disable_brdma <<
                       disable_erdma_shift, all_macros, MMIO)

    fly_mode = all_macros[field_enum(
        "SDP_D_FEATURE_MODE_CFG_0", "FLYING_MODE", ("OFF", "ON")[bool(fly)])]
    fly_mode_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "FLYING_MODE")]
    win_grad = all_macros[field_enum(
        "SDP_D_FEATURE_MODE_CFG_0", "FLYING_MODE", ("OFF", "ON")[bool(sdp_op.conv_mode == 1)])]
    win_grad_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "WINOGRAD")]
    src_precision = all_macros[field_enum(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION", ("INT8", "INT16", "FP16")[sdp_op.src_precision])]
    src_precision_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION")]
    dest_precision = all_macros[field_enum(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION", ("INT8", "INT16", "FP16")[sdp_op.dst_precision])]
    dest_precision_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "OUT_PRECISION")]
    proc_map = [["INT8", "INT8", "FP16"], [
        "INT8", "INT16", "FP16"], ["INT8", "INT16", "FP16"]]
    proc_precision = all_macros[field_enum(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION", proc_map[sdp_op.dst_precision][sdp_op.src_precision])]
    proc_precision_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "PROC_PRECISION")]
    batch_num = sdp_op.batch_num - 1
    batch_num_shift = all_macros[shift(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "BATCH_NUMBER")]
    reg = (fly_mode << fly_mode_shift) | (win_grad << win_grad_shift) | (src_precision << src_precision_shift) | (
        dest_precision << dest_precision_shift) | (proc_precision << proc_precision_shift) | (batch_num << batch_num_shift)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDPRDMA_D_FEATURE_MODE_CFG",
                          'fly_mode': fly_mode,
                          'fly_mode_shift': fly_mode_shift,
                          'win_grad': win_grad,
                          'win_grad_shift': win_grad_shift,
                          'src_precision': src_precision,
                          'src_precision_shift': src_precision_shift,
                          'dest_precision': dest_precision,
                          'dest_precision_shift': dest_precision_shift,
                          'proc_precision': proc_precision,
                          'proc_precision_shift': proc_precision_shift,
                          'batch_num': batch_num,
                          'batch_num_shift': batch_num_shift
                          })
    sdp_rdma_reg_write("D_FEATURE_MODE_CFG", reg, all_macros, MMIO)

    # If RDMA needed
    if group.is_rdma_needed:
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DATA_CUBE_WIDTH",
                              'width': sdp_surface.src_data.width - 1
                              })
        sdp_rdma_reg_write("D_DATA_CUBE_WIDTH",
                           sdp_surface.src_data.width - 1, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DATA_CUBE_HEIGHT",
                              'height': sdp_surface.src_data.height - 1
                              })
        sdp_rdma_reg_write("D_DATA_CUBE_HEIGHT",
                           sdp_surface.src_data.height - 1, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DATA_CUBE_CHANNEL",
                              'channel': sdp_surface.src_data.channel - 1
                              })
        sdp_rdma_reg_write("D_DATA_CUBE_CHANNEL",
                           sdp_surface.src_data.channel - 1, all_macros, MMIO)

        # config SDP source info
        if not fly:
            # if not on-the-fly, we have to config
            # the source cube info
            high = high32bits(src_addr)
            low = low32bits(src_addr)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDPRDMA_D_SRC_BASE_ADDR_LOW",
                                  'low': pad_hex(low)
                                  })
            sdp_rdma_reg_write("D_SRC_BASE_ADDR_LOW", low, all_macros, MMIO)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDPRDMA_D_SRC_BASE_ADDR_HIGH",
                                  'high': pad_hex(high)
                                  })
            sdp_rdma_reg_write("D_SRC_BASE_ADDR_HIGH", high, all_macros, MMIO)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDPRDMA_D_SRC_LINE_STRIDE",
                                  'line_stride': sdp_surface.src_data.line_stride
                                  })
            sdp_rdma_reg_write("D_SRC_LINE_STRIDE",
                               sdp_surface.src_data.line_stride, all_macros, MMIO)
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDPRDMA_D_SRC_SURFACE_STRIDE",
                                  'surf_stride': sdp_surface.src_data.surf_stride
                                  })
            sdp_rdma_reg_write("D_SRC_SURFACE_STRIDE",
                               sdp_surface.src_data.surf_stride, all_macros, MMIO)
            # bodge extra last CV in as otherwise get out of bounds
            ram_type = all_macros[field_enum(
                "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_RAM_TYPE", ("MC", "CV", "CV")[sdp_surface.src_data.type])]
            acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                                  'name': f"SDPRDMA_D_SRC_DMA_CFG",
                                  'ram_type': ram_type
                                  })
            sdp_rdma_reg_write("D_SRC_DMA_CFG", ram_type, all_macros, MMIO)

        # config x1, x2, y source info
        config_source_info(x1_rdma_ena, x1_op,
                           sdp_surface.x1_data, x1_addr, "x1", acceleratorIR, MMIO)
        config_source_info(x2_rdma_ena, x2_op,
                           sdp_surface.x2_data, x2_addr, "x2", acceleratorIR, MMIO)
        config_source_info(y_rdma_ena, y_op, sdp_surface.y_data,
                           y_addr, "y", acceleratorIR, MMIO)

    if sdp_op.lut_index >= 0:
        update_lut("SDP_S_LUT_ACCESS_CFG_0", lut, sdp_op.src_precision)

    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DATA_CUBE_WIDTH",
                          'width': sdp_surface.src_data.width - 1
                          })
    sdp_reg_write("D_DATA_CUBE_WIDTH",
                  sdp_surface.src_data.width - 1, all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DATA_CUBE_HEIGHT",
                          'height': sdp_surface.src_data.height - 1
                          })
    sdp_reg_write("D_DATA_CUBE_HEIGHT",
                  sdp_surface.src_data.height - 1, all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DATA_CUBE_CHANNEL",
                          'channel': sdp_surface.src_data.channel - 1
                          })
    sdp_reg_write("D_DATA_CUBE_CHANNEL",
                  sdp_surface.src_data.channel - 1, all_macros, MMIO)

    if out_dma_ena:
        high = high32bits(dst_addr)
        low = low32bits(dst_addr)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DST_BASE_ADDR_HIGH",
                              'high': pad_hex(high)
                              })
        sdp_reg_write("D_DST_BASE_ADDR_HIGH", high, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DST_BASE_ADDR_LOW",
                              'low': pad_hex(low)
                              })
        sdp_reg_write("D_DST_BASE_ADDR_LOW", low, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DST_LINE_STRIDE",
                              'line_stride': sdp_surface.dst_data.line_stride
                              })
        sdp_reg_write("D_DST_LINE_STRIDE",
                      sdp_surface.dst_data.line_stride, all_macros, MMIO)
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DST_SURFACE_STRIDE",
                              'surf_stride': sdp_surface.dst_data.surf_stride
                              })
        sdp_reg_write("D_DST_SURFACE_STRIDE",
                      sdp_surface.dst_data.surf_stride, all_macros, MMIO)

    # Config bias modules BS, BN, EW for X1, X2 and Y
    config_bias_data_cubes(x1_op, "x1", acceleratorIR, MMIO)
    config_bias_data_cubes(x2_op, "x2", acceleratorIR, MMIO)
    config_bias_data_cubes(y_op, "y", acceleratorIR, MMIO)

    fly_mode = all_macros[field_enum("SDP_D_FEATURE_MODE_CFG_0", "FLYING_MODE", ("OFF", "ON")[
                                     bool(sdp_surface.src_data.type == dla_mem_hw)])]
    fly_mode_shift = all_macros[shift(
        "SDP_D_FEATURE_MODE_CFG_0", "FLYING_MODE")]
    out_dst = all_macros[field_enum("SDP_D_FEATURE_MODE_CFG_0", "OUTPUT_DST", ("MEM", "PDP")[
                                    bool(sdp_surface.dst_data.type == dla_mem_hw)])]
    out_dst_shift = all_macros[shift("SDP_D_FEATURE_MODE_CFG_0", "OUTPUT_DST")]
    conv_mode_winograd = 1
    mode_winograd = all_macros[field_enum("SDP_D_FEATURE_MODE_CFG_0", "WINOGRAD", ("OFF", "ON")[
                                          bool(sdp_op.conv_mode == conv_mode_winograd)])]
    mode_winograd_shift = all_macros[shift(
        "SDP_D_FEATURE_MODE_CFG_0", "WINOGRAD")]
    batch_num = sdp_op.batch_num - 1
    batch_num_shift = all_macros[shift(
        "SDP_D_FEATURE_MODE_CFG_0", "BATCH_NUMBER")]
    reg = (fly_mode << fly_mode_shift) | (out_dst << out_dst_shift) | (
        mode_winograd << mode_winograd_shift) | (batch_num << batch_num_shift)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_FEATURE_MODE_CFG",
                          'fly_mode': fly_mode,
                          'fly_mode_shift': fly_mode_shift,
                          'out_dst': out_dst,
                          'out_dst_shift': out_dst_shift,
                          'mode_winograd': mode_winograd,
                          'mode_winograd_shift': mode_winograd_shift,
                          'batch_num': batch_num,
                          'batch_num_shift': batch_num_shift
                          })
    sdp_reg_write("D_FEATURE_MODE_CFG", reg, all_macros, MMIO)
    # bodge last CV in ... will find solution later but correct as of c code
    dst_ram_type_str = field_enum(
        "SDP_RDMA_D_BRDMA_CFG_0", "BRDMA_RAM_TYPE", ("MC", "CV", "CV")[sdp_surface.dst_data.type])
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DST_DMA_CFG",
                          'ram_type': all_macros[dst_ram_type_str]
                          })
    sdp_reg_write("D_DST_DMA_CFG",
                  all_macros[dst_ram_type_str], all_macros, MMIO)

    if sdp_op.batch_num > 1:
        acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                              'name': f"SDP_D_DST_BATCH_STRIDE",
                              'batch_stride': sdp_op.batch_stride
                              })
        sdp_reg_write("D_DST_BATCH_STRIDE",
                      sdp_op.batch_stride, all_macros, MMIO)

    proc_precision = all_macros[field_enum(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION", proc_map[sdp_op.dst_precision][sdp_op.src_precision])]
    proc_precision_shift = all_macros[shift(
        "SDP_D_DATA_FORMAT_0", "PROC_PRECISION")]
    dest_precision = all_macros[field_enum(
        "SDP_RDMA_D_FEATURE_MODE_CFG_0", "IN_PRECISION", ("INT8", "INT16", "FP16")[sdp_op.dst_precision])]
    dest_precision_shift = all_macros[shift(
        "SDP_D_DATA_FORMAT_0", "OUT_PRECISION")]
    reg = (proc_precision << proc_precision_shift) | (
        dest_precision << dest_precision_shift)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_DATA_FORMAT",
                          'proc_precision': proc_precision,
                          'proc_precision_shift': proc_precision_shift,
                          'dest_precision': dest_precision,
                          'dest_precision_shift': dest_precision_shift
                          })
    sdp_reg_write("D_DATA_FORMAT", reg, all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_CVT_OFFSET",
                          'offset': sdp_op.out_cvt.offset
                          })
    sdp_reg_write("D_CVT_OFFSET", sdp_op.out_cvt.offset, all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_CVT_SCALE",
                          'scale': sdp_op.out_cvt.scale
                          })
    sdp_reg_write("D_CVT_SCALE", sdp_op.out_cvt.scale, all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': f"SDP_D_CVT_SHIFT",
                          'truncate': sdp_op.out_cvt.truncate
                          })
    # this seems wrong but matches their c code
    sdp_reg_write("D_CVT_SHIFT", sdp_op.out_cvt.truncate, all_macros, MMIO)


# Look up table (LUT) checking ready (if needed) as sigmoid and tanh need them
# int dla_sdp_is_ready(struct dla_processor *processor,
#                      struct dla_processor_group *group)


def dla_sdp_dump_config(group: Dla_processor_group):
    """Print sdp_surface with all attributes of input and output data cubes
    Print sdp_op with all parameters"""
    # note surface_desc only contains sdp_surface ... it is just a common container
    # dla_debug_sdp_surface_desc
    print("*********************************************************\n")
    print("NVDLA FW ROI[{:d}]: dla_sdp_surface_desc\n".format(group.roi_index))
    print("---------------------------------------------------------\n")
    print(group.surface_desc.sdp_surface)

    # note operation_desc only contains sdp_op ... it is just a common container
    # dla_debug_sdp_op_desc(sdp_op, group->roi_index)
    print("*********************************************************\n")
    print("NVDLA FW ROI[{:d}]: dla_sdp_op_desc\n".format(group.roi_index))
    print("---------------------------------------------------------\n")
    print(group.operation_desc.sdp_op)


def dla_sdp_program(group: Dla_processor_group, acceleratorIR, MMIO):
    """Highest level function"""

    dla_enable_intr(all_macros, acceleratorIR, MMIO)
    processor_sdp_program(group, acceleratorIR, MMIO)
