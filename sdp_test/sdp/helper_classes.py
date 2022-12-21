import numpy as np
from time import time_ns


class Dla_processor_group:
    def __init__(self):
        # TODO:switch uint8 to bool
        self.id = np.uint8(0)                       # unused
        self.rdma_id = np.uint8(0)                  # unused
        self.active = np.uint8(0)                   # unused
        self.events = np.uint8(0)                   # unused
        self.roi_index = np.uint8(0)                # unused
        self.is_rdma_needed = True                  # unused
        self.pending = np.uint8(0)                  # unused

        # we load lut index from current op (group.operation_desc.sdp_op)
        self.lut_index = np.int32(0)
        self.programming = np.uint8(0)              # unused
        self.start_time = np.uint64(time_ns()//1000)  # store time in us

        # used if not fly and fly = (sdp_surface.src_data.type == dla_mem_hw)
        # fly is what will use so probs unused
        self.op_desc = Dla_common_op_desc()

        self.consumers = [Dla_common_op_desc()      # unused
                          for i in range(6)]  # 6 for 6 processors (DLA_OP_NUM)
        self.fused_parent = Dla_common_op_desc()    # unused
        self.operation_desc = Dla_operation_container()  # only use sdp_op inside this
        self.surface_desc = Dla_surface_container()  # only use sdp_surface inside this


# Notes: __attribute__ ((packed, aligned(4)));
class Dla_common_op_desc:
    # all current values are arbitrary
    def __init__(self):
        # set by ucode ... couldn't find any info on where this is really set
        self.index = np.int16(0)
        self.roi_index = np.int8(0)
        self.op_type = np.uint8(0)
        self.dependency_count = np.uint8(0)
        self.reserved0 = np.zeros(3, dtype='uint8')  # reserved0 is not a typo
        # 6 for 6 processors (DLA_OP_NUM)
        self.consumers = [Dla_consumer() for i in range(6)]
        self.fused_parent = Dla_consumer()


class Dla_operation_container:
    # all current values are arbitrary
    def __init__(self):
        # normally this is a union object
        # so the values of sdp rn are the vals of this
        # class normally
        self.sdp_op = Dla_sdp_op_desc()

        # all union items
        # struct dla_bdma_op_desc bdma_op;
        # struct dla_conv_op_desc conv_op;
        # DONE struct dla_sdp_op_desc sdp_op;
        # struct dla_pdp_op_desc pdp_op;
        # struct dla_cdp_op_desc cdp_op;
        # struct dla_rubik_op_desc rubik_op;


class Dla_sdp_op_desc:
    # Notes: __packed __aligned(4);
    # all current values are arbitrary
    def __init__(self):
        # Precision parameters
        # dla_precision
        self.src_precision = np.uint8(0)  # "INT8", "INT16", "FP16"
        self.dst_precision = np.uint8(0)  # "INT8", "INT16", "FP16"
        self.lut_index = np.int16(0)  # dig into dla_read_lut

        self.out_cvt = Dla_cvt_param()  # CVT_OFFSET, CVT_SCALE and CVT_SHIFT regs

        # Performance parameters

        # 0 = regular conv mode, 1 = winograd, for brdma, nrdma, erdma setup and also some bias stuff
        self.conv_mode = np.uint8(0)
        self.batch_num = np.uint8(1)     # next will be used if batch_num > 1
        self.batch_stride = np.uint32(0)  # used for reg D_DST_BATCH_STRIDE
        self.reserved0 = np.uint16(0)    # unused

        # Algorithm parameters
        self.x1_op = Dla_sdp_op()
        self.x2_op = Dla_sdp_op()
        self.y_op = Dla_sdp_op()

    def __str__(self):
        ret = ""
        ret += "src_precision    = {:u}\n".format(self.src_precision)
        ret += "dst_precision    = {:u}\n".format(self.dst_precision)
        ret += "lut_index        = {:d}\n".format(self.lut_index)
        ret += "out_cvt          =\n"
        ret += str(self.out_cvt)
        ret += "conv_mode        = {:u}\n".format(self.conv_mode)
        ret += "batch_num        = {:u}\n".format(self.batch_num)
        ret += "batch_stride     = {:u}\n".format(self.batch_stride)
        ret += "x1_op            = [ dla_sdp_op =>\n"
        ret += str(self.x1_op)
        ret += "x2_op            = [ dla_sdp_op =>\n"
        ret += str(self.x2_op)
        ret += "y_op             = [ dla_sdp_op =>\n"
        ret += str(self.y_op)
        return ret


class Dla_cvt_param:
    # all current values are arbitrary
    def __init__(self):
        self.scale = np.int16(0)
        self.truncate = np.uint8(0)
        self.enable = np.uint8(0)
        self.offset = np.int32(0)

    def __str__(self):
        return "[ scale = {:d}, truncate = {:u}, enable = {:u}, offset = {:d} ]\n".format(
            self.scale, self.truncate, self.enable, self.offset)


class Dla_sdp_op:
    # __packed __aligned(4);
    # all current values are arbitrary
    def __init__(self):
        self.enable = np.uint8(0)       # used

        self.alu_type = np.uint8(0)  # ("MAX", "MIN", "SUM", "EQL")?
        self.type = np.uint8(0)  # (SDP_OP_NONE, MUL, ADD, BOTH)

        self.mode = np.uint8(0)  # SDP_OP_PER_LAYER, KERNEL, POINT

        self.act = np.uint8(0)  # dla_act_type
        self.shift_value = np.uint8(0)  # left shift
        self.truncate = np.uint8(0)  # ACTIVATION_NONE, RELU, LUT, PRELU
        # RDMA precision "ONE_BYTE", "TWO_BYTE", "TWO_BYTE" (for int8, int16, fp16)
        self.precision = np.uint8(0)

        # write val to reg DP_??_ALU_SRC_VALUE ... operand value of alu ... maybe this is the value we are adding?
        self.alu_operand = np.int32(0)
        self.mul_operand = np.int32(0)  # value we are multiplying by

        self.cvt = Dla_sdp_cvt()  # scale, truncate, enable, offset for alu and mul

    def __str__(self):
        ret = ""
        ret += "    enable         = {:u}\n".format(self.enable)
        ret += "    alu_type       = {:u}\n".format(self.alu_type)
        ret += "    type           = {:u}\n".format(self.type)
        ret += "    mode           = {:u}\n".format(self.mode)
        ret += "    act            = {:u}\n".format(self.act)
        ret += "    shift_value    = {:u}\n".format(self.shift_value)
        ret += "    truncate       = {:u}\n".format(self.truncate)
        ret += "    precision      = {:u}\n".format(self.precision)
        ret += "    alu_operand    = {:d}\n".format(self.alu_operand)
        ret += "    mul_operand    = {:d}\n".format(self.mul_operand)
        ret += "cvt.alu_cvt          =\n"
        ret += str(self.cvt.alu_cvt)
        ret += "cvt.mul_cvt          =\n"
        ret += str(self.cvt.mul_cvt)
        ret += "]\n"
        return ret


class Dla_sdp_cvt:
    # __packed __aligned(4)
    # all current values are arbitrary
    def __init__(self):
        self.alu_cvt = Dla_cvt_param()
        self.mul_cvt = Dla_cvt_param()


class Dla_sdp_stat_desc:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # normally this is a union object
        # so the values of param rn are the vals of this
        # class normally
        self.nan_input_num = np.uint32(0)
        self.inf_input_num = np.uint32(0)
        self.nan_output_num = np.uint32(0)
        self.wdma_write_stall = np.uint32(0)
        self.lut_underflow = np.uint32(0)
        self.lut_overflow = np.uint32(0)
        self.lut_hybrid = np.uint32(0)
        self.lut_le_hit = np.uint32(0)
        self.lut_lo_hit = np.uint32(0)
        self.saturation_count = np.uint32(0)
        self.runtime = np.uint32(0)


class Dla_surface_container:
    # all current values are arbitrary
    def __init__(self):
        # normally this is a union object
        # so the values of sdp_surface rn are the vals of this
        # class normally
        self.sdp_surface = Dla_sdp_surface_desc()

        # all union items
        # struct dla_bdma_surface_desc bdma_surface;
        # struct dla_conv_surface_desc conv_surface;
        # DONE struct dla_sdp_surface_desc sdp_surface;
        # struct dla_pdp_surface_desc pdp_surface;
        # struct dla_cdp_surface_desc cdp_surface;
        # struct dla_rubik_surface_desc rubik_surface;


class Dla_sdp_surface_desc:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # source input, available when SDP working on offline mode
        self.src_data = Dla_data_cube()
        self.x1_data = Dla_data_cube()  # input
        self.x2_data = Dla_data_cube()  # input
        self.y_data = Dla_data_cube()  # input
        self.dst_data = Dla_data_cube()  # output

    def __str__(self):
        ret = ""
        ret += "src_data            =   dla_data_cube =>\n"
        ret += str(self.src_data)
        ret += "x1_data             =   dla_data_cube =>\n"
        ret += str(self.x1_data)
        ret += "x2_data             =   dla_data_cube =>\n"
        ret += str(self.x2_data)
        ret += "y_data              =   dla_data_cube =>\n"
        ret += str(self.y_data)
        ret += "dst_data            =   dla_data_cube =>\n"
        ret += str(self.dst_data)
        return ret


class Dla_data_cube:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # dla engine can read from [External DRAM, CV-SRAM, DLA sub-module]
        # if you would like to write to computer use 2
        self.type = np.uint16(2)

        # set up of these 2 could occur in dla_get_dma_cube_address but just set manually?
        # offset to the actual IOVA in task.address_list
        self.address = np.int16(0)      # high and low used for rdma
        self.offset = np.uint32(0)      # offset within address  .. unused?

        # unused ... precision in xy_op used for precision so not sure what this is
        self.size = np.uint32(0)

        # Cube dimensions
        self.width = np.uint16(0)       # used
        self.height = np.uint16(0)      # used
        self.channel = np.uint16(0)     # used
        self.reserved0 = np.uint16(0)   # unused

        # Stride information
        self.line_stride = np.uint32(0)  # used
        # http://nvdla.org/hw/format.html is useful
        self.surf_stride = np.uint32(0)  # surface stride

        # For Rubik only
        self.plane_stride = np.uint32(0)  # used for rubik only

    def __str__(self):
        ret = "["
        ret += "    type          = {:u}\n".format(self.type)
        ret += "    address       = {:d}\n".format(self.address)
        ret += "    width         = {:x}\n".format(self.width)
        ret += "    height        = {:x}\n".format(self.height)
        ret += "    channel       = {:x}\n".format(self.channel)
        ret += "    size          = {:u}\n".format(self.size)
        ret += "    line_stride   = {:u}\n".format(self.line_stride)
        ret += "    surf_stride   = {:u}\n".format(self.surf_stride)
        ret += "    plane_stride  = {:u}\n".format(self.plane_stride)
        ret += "]"
        return ret


class Dla_consumer:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # the index of dla_common_op_desc in dep_graph_addr
        self.index = np.int16(0)
        self.event = np.uint8(0)
        self.res = np.uint8(0)

# LUT classes


class Dla_lut_param:
    # all current values are arbitrary
    def __init__(self):
        LUT_LINEAR_EXP_TABLE_ENTRY_LOG2 = 6
        LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2 = 8
        self.linear_exp_table = np.zeros(
            (1 << LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1, dtype='int16')
        self.linear_only_table = np.zeros(
            (1 << LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1, dtype='int16')

        self.linear_exp_offset = Dla_lut_offset()
        self.linear_only_offset = Dla_lut_offset()

        # The start and end point of raw table,
        # valid when raw_method=LINEAR only
        self.linear_exp_start = np.uint64()
        self.linear_exp_end = np.uint64()
        self.linear_only_start = np.uint64()
        self.linear_only_end = np.uint64()

        self.linear_exp_underflow_slope = Dla_slope()
        self.linear_exp_overflow_slope = Dla_slope()
        self.linear_only_underflow_slope = Dla_slope()
        self.linear_only_overflow_slope = Dla_slope()

        # dla_lut_priority, when both lut are hit (or one overflow,
        # the other underflow), which one should be selected as output
        self.hybrid_priority = np.uint8(0)
        self.underflow_priority = np.uint8(0)
        self.overflow_priority = np.uint8(0)
        self.method = np.uint8(0)  # dla_lut_method


class Dla_lut_offset:
    # all current values are arbitrary
    # used to be a union
    def __init__(self):
        # Number should be substracted on log domain before look up
        # exponetial table it has the same definition as hardware
        # thus input scaling should also take into account when
        # set this field.
        self.exp_offset = np.int8(0)

        # Number of bits should be right shift before looking
        # up linear table
        self.frac_bits = np.int8(0)
        self.reserved0 = np.uint16(0)


class Dla_float_data:
    """This struct is used to represent floating point values by INT.\n
    Suppose we have a float point number fp_x, it will be represented
    as: fp_x = scale_int_x>>(shifter_x)
    This is very useful for INT pipeline"""
    # all current values are arbitrary
    # __packed __aligned(4);

    def __init__(self) -> None:
        self.scale = np.int16(0)
        self.shifter = np.int8(0)
        self.reserved0 = np.uint8(0)


class Dla_slope:
    """ For INT pipeline, we use the struct above to represent a floating number\n
    For FP16 pipeline, we should store the FP16 encoded value into a uint16_t\n
    container"""
    # all current values are arbitrary
    # used to be a union

    def __init__(self):
        self.data_i = Dla_float_data()
        self.data_f = np.uint16()
