import numpy as np


class Dla_processor_group:
    def __init__(self):
        self.id = np.uint8(0)
        self.rdma_id = np.uint8(0)
        self.active = np.uint8(0)
        self.events = np.uint8(0)
        self.roi_index = np.uint8(0)
        self.is_rdma_needed = True
        self.pending = np.uint8(0)  # TODO:switch to bool
        self.lut_index = np.int32(0)
        self.programming = np.uint8(0)
        self.start_time = np.uint64(0)  # store time in us

        self.op_desc = Dla_common_op_desc()
        self.consumers = [Dla_common_op_desc()
                          for i in range(6)]  # 6 for 6 processors (DLA_OP_NUM)
        self.fused_parent = Dla_common_op_desc()
        self.operation_desc = Dla_operation_container()
        self.surface_desc = Dla_surface_container()


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
        self.src_precision = np.uint8(0)
        self.dst_precision = np.uint8(0)
        self.lut_index = np.int16(0)

        self.out_cvt = Dla_cvt_param()

        # Performance parameters
        self.conv_mode = np.uint8(0)
        self.batch_num = np.uint8(0)
        self.reserved0 = np.uint16(0)

        self.batch_stride = np.uint32(0)  # will be used when batch_num > 1

        # Algorithm parameters
        self.x1_op = Dla_sdp_op()
        self.x2_op = Dla_sdp_op()
        self.y_op = Dla_sdp_op()


class Dla_cvt_param:
    # all current values are arbitrary
    def __init__(self):
        self.scale = np.int16(0)
        self.truncate = np.uint8(0)
        self.enable = np.uint8(0)
        self.offset = np.int32(0)


class Dla_sdp_op:
    # __packed __aligned(4);
    # all current values are arbitrary
    def __init__(self):
        self.enable = np.uint8(0)
        self.alu_type = np.uint8(0)  # dla_sdp_alu_op_type
        self.type = np.uint8(0)  # dla_sdp_op_type
        self.mode = np.uint8(0)  # dla_sdp_op_mode

        self.act = np.uint8(0)  # dla_act_type
        self.shift_value = np.uint8(0)  # left shift
        self.truncate = np.uint8(0)
        self.precision = np.uint8(0)

        self.alu_operand = np.int32(0)
        self.mul_operand = np.int32(0)

        self.cvt = Dla_sdp_cvt()


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


class Dla_data_cube:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # dla engine can read from [External DRAM, CV-SRAM, DLA sub-module]
        self.type = np.uint16(0)
        # offset to the actual IOVA in task.address_list
        self.address = np.int16(0)
        self.offset = np.uint32(0)  # offset within address

        self.size = np.uint32(0)
        # Cube dimensions
        self.width = np.uint16(0)
        self.height = np.uint16(0)
        self.channel = np.uint16(0)
        self.reserved0 = np.uint16(0)

        # Stride information
        self.line_stride = np.uint32(0)
        self.surf_stride = np.uint32(0)

        # For Rubik only
        self.plane_stride = np.uint32(0)


class Dla_consumer:
    # __attribute__ ((packed, aligned(4)));
    # all current values are arbitrary
    def __init__(self):
        # the index of dla_common_op_desc in dep_graph_addr
        self.index = np.int16(0)
        self.event = np.uint8(0)
        self.res = np.uint8(0)
