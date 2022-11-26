import numpy as np


class dla_processor_group:
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

        self.op_desc = dla_common_op_desc()
        self.consumers = [dla_common_op_desc()
                          for i in range(6)]  # 6 for 6 processors (DLA_OP_NUM)
        self.fused_parent = dla_common_op_desc()
        self.dla_operation_container = dla_operation_container()
        self.dla_surface_container = dla_surface_container()


# Notes: __attribute__ ((packed, aligned(4)));
class dla_common_op_desc:
    # all current values are arbitrary
    def __init__(self):
        # set by ucode ... couldn't find any info on where this is really set
        self.index = np.int16(0)
        self.roi_index = np.int8(0)
        self.op_type = np.uint8(0)
        self.dependency_count = np.uint8(0)
        self.reserved0 = np.zeros(3, dtype='uint8')  # reserved0 is not a typo
        # 6 for 6 processors (DLA_OP_NUM)
        self.consumers = [dla_consumer() for i in range(6)]
        self.fused_parent = dla_consumer()


class dla_operation_container:
    # all current values are arbitrary
    def __init__(self):
        # normally this is a union object
        # so the values of sdp rn are the vals of this
        # class normally
        self.param = dla_sdp_stat_desc()

        # all union items
        # struct dla_bdma_stat_desc bdma_stat;
        # struct dla_conv_stat_desc conv_stat;
        # DONE - struct dla_sdp_stat_desc sdp_stat;
        # struct dla_pdp_stat_desc pdp_stat;
        # struct dla_cdp_stat_desc cdp_stat;
        # struct dla_rubik_stat_desc rubik_stat;

# __attribute__ ((packed, aligned(4)));


class dla_sdp_stat_desc:
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


class dla_surface_container:
    # all current values are arbitrary
    def __init__(self):
        # normally this is a union object
        # so the values of sdp_surface rn are the vals of this
        # class normally
        self.sdp_surface = dla_sdp_surface_desc()

        # all union items
        # struct dla_bdma_surface_desc bdma_surface;
        # struct dla_conv_surface_desc conv_surface;
        # DONE struct dla_sdp_surface_desc sdp_surface;
        # struct dla_pdp_surface_desc pdp_surface;
        # struct dla_cdp_surface_desc cdp_surface;
        # struct dla_rubik_surface_desc rubik_surface;

# __attribute__ ((packed, aligned(4)));


class dla_sdp_surface_desc:
    # all current values are arbitrary
    def __init__(self):
        # source input, available when SDP working on offline mode
        self.src_data = dla_data_cube()
        self.x1_data = dla_data_cube()  # input
        self.x2_data = dla_data_cube()  # input
        self.y_data = dla_data_cube()  # input
        self.dst_data = dla_data_cube()  # output

# __attribute__ ((packed, aligned(4)));


class dla_data_cube:
    # all current values are arbitrary
    def __init__(self):
        self.type = np.uint16(0)  # dla_mem_type
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

# __attribute__ ((packed, aligned(4)));


class dla_consumer:
    # all current values are arbitrary
    def __init__(self):
        # the index of dla_common_op_desc in dep_graph_addr
        self.index = np.int16(0)
        self.event = np.uint8(0)
        self.res = np.uint8(0)
