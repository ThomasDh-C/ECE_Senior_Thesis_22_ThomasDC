import numpy as np


def retrieve_all_macros():
    all_macros = {}
    # test = []

    with open("./opendla_small.h", 'r') as h:
        # skip ifndef statements - only define statements
        all_lines = [l[8:][:-1] for l in h.readlines() if l[:7] == '#define']

        for l in all_lines:
            split_up = l.split('\t')
            # split_up lengths are 1, 2, 4
            if len(split_up) == 1:
                # first line where define small dla
                continue
            elif len(split_up) == 2:
                # last lines where define base addr for cores
                base_adr_name, base_adr_value_str = split_up
                all_macros[base_adr_name] = base_adr_value_str
            elif len(split_up) == 4:
                # rest of lines have length 4 - [name, '', '', val]
                # val is always a function of form func(a) or func(a,b)
                macro_name, _, _, macro_value = split_up
                # test.append(macro_value.count(','))
                all_macros[macro_name] = macro_value

    return all_macros

# don't understand ? thing as think I need a : after a ? (normally contidional)
# def bits(num: np.int32, range: np.int32):
#     return ((((0xFFFFFFFF >> (31 - (1 ? range))) & (0xFFFFFFFF << (0 ? range))) & num) >> (0 ? range))

# TODO: returns a int32 not a uint32 ... fix me


def high32bits(val64bit: np.int64):
    return np.right_shift(val64bit, 32, dtype=np.int32)


def low32bits(val64bit: np.int64):
    return np.uint32(val64bit)


def check_align(val, align, debug=False):
    if debug:
        assert ((val & (align-1)) == 0)


def mask(reg: str, field: str):
    return f'{reg}_{field}_FIELD'


def field_enum(r: str, f: str, e: str):
    return f'{r}_{f}_{e}'


def shift(reg: str, field: str):
    return f'{reg}_{field}_SHIFT'


def glb_reg(name: str, all_macros):
    return all_macros[f'GLB_{name}_0']


def mcif_reg(name: str, all_macros):
    return all_macros[f'MCIF_{name}_0']


def cvif_reg(name: str, all_macros):
    return all_macros[f'CVIF_{name}_0']


def bdma_reg(name: str, all_macros):
    return all_macros[f'BDMA_{name}_0']


def cdma_reg(name: str, all_macros):
    return all_macros[f'CDMA_{name}_0']


def csc_reg(name: str, all_macros):
    return all_macros[f'CSC_{name}_0']


def cmac_a_reg(name: str, all_macros):
    return all_macros[f'CMAC_A_{name}_0']


def cmac_b_reg(name: str, all_macros):
    return all_macros[f'CMAC_B_{name}_0']


def cacc_reg(name: str, all_macros):
    return all_macros[f'CACC_{name}_0']


def sdp_rdma_reg(name: str, all_macros):
    return all_macros[f'SDP_RDMA_{name}_0']


def sdp_reg(name: str, all_macros):
    return all_macros[f'SDP_{name}_0']


def pdp_rdma_reg(name: str, all_macros):
    return all_macros[f'PDP_RDMA_{name}_0']


def pdp_reg(name: str, all_macros):
    return all_macros[f'PDP_{name}_0']


def cdp_rdma_reg(name: str, all_macros):
    return all_macros[f'CDP_RDMA_{name}_0']


def cdp_reg(name: str, all_macros):
    return all_macros[f'CDP_{name}_0']


def rbk_reg(name: str, all_macros):
    return all_macros[f'RBK_{name}_0']


# Functions below use these general reg write and read


def reg_read(addr: np.uint32) -> np.uint32:
    # TODO: FIND MY DEF BUT RET uint32

    return


def reg_write(addr: np.uint32, reg: np.uint32):
    # TODO: FIND MY DEF BUT RET VOID
    return


# alias for register read for each sub-module - REMOVE IN FUTURE
def glb_reg_read(reg: str):
    return reg_read(glb_reg(reg))


def bdma_reg_read(reg: str):
    return reg_read(bdma_reg(reg))


def cdma_reg_read(reg: str):
    return reg_read(cdma_reg(reg))


def csc_reg_read(reg: str):
    return reg_read(csc_reg(reg))


def cmac_a_reg_read(reg: str):
    return reg_read(cmac_a_reg(reg))


def cmac_b_reg_read(reg: str):
    return reg_read(cmac_b_reg(reg))


def cacc_reg_read(reg: str):
    return reg_read(cacc_reg(reg))


def sdp_rdma_reg_read(reg: str):
    return reg_read(sdp_rdma_reg(reg))


def sdp_reg_read(reg: str):
    return reg_read(sdp_reg(reg))


def pdp_rdma_reg_read(reg: str):
    return reg_read(pdp_rdma_reg(reg))


def pdp_reg_read(reg: str):
    return reg_read(pdp_reg(reg))


def cdp_rdma_reg_read(reg: str):
    return reg_read(cdp_rdma_reg(reg))


def cdp_reg_read(reg: str):
    return reg_read(cdp_reg(reg))


def rubik_reg_read(reg: str):
    return reg_read(rbk_reg(reg))


# alias for register write for each sub-module
def glb_reg_write(reg: str, val: np.uint32):
    return reg_write(glb_reg(reg), val)


def bdma_reg_write(reg: str, val: np.uint32):
    return reg_write(bdma_reg(reg), val)


def cdma_reg_write(reg: str, val: np.uint32):
    return reg_write(cdma_reg(reg), val)


def csc_reg_write(reg: str, val: np.uint32):
    return reg_write(csc_reg(reg), val)


def cmac_a_reg_write(reg: str, val: np.uint32):
    return reg_write(cmac_a_reg(reg), val)


def cmac_b_reg_write(reg: str, val: np.uint32):
    return reg_write(cmac_b_reg(reg), val)


def cacc_reg_write(reg: str, val: np.uint32):
    return reg_write(cacc_reg(reg), val)


def sdp_rdma_reg_write(reg: str, val: np.uint32):
    return reg_write(sdp_rdma_reg(reg), val)


def sdp_reg_write(reg: str, val: np.uint32):
    return reg_write(sdp_reg(reg), val)


def pdp_rdma_reg_write(reg: str, val: np.uint32):
    return reg_write(pdp_rdma_reg(reg), val)


def pdp_reg_write(reg: str, val: np.uint32):
    return reg_write(pdp_reg(reg), val)


def cdp_rdma_reg_write(reg: str, val: np.uint32):
    return reg_write(cdp_rdma_reg(reg), val)


def cdp_reg_write(reg: str, val: np.uint32):
    return reg_write(cdp_reg(reg), val)


def rubik_reg_write(reg: str, val: np.uint32):
    return reg_write(rbk_reg(reg), val)

# AFTER LINE 120 in kmd/firmware/dla_engine_internal.h not implemented

# ... note that from line 233 this may be useful
# /**
#  * SDP operations
#  */
# void
# dla_sdp_set_producer(int32_t group_id, int32_t rdma_group_id);
# int
# dla_sdp_enable(struct dla_processor_group *group);
# int
# dla_sdp_program(struct dla_processor_group *group);
# int
# dla_sdp_is_ready(struct dla_processor *processor,
# 			   struct dla_processor_group *group);
# void
# dla_sdp_dump_config(struct dla_processor_group *group);
# void
# dla_sdp_rdma_check(struct dla_processor_group *group);

# #if STAT_ENABLE
# void
# dla_sdp_stat_data(struct dla_processor *processor,
# 				struct dla_processor_group *group);
# void
# dla_sdp_dump_stat(struct dla_processor *processor);

# #else
# static inline void
# dla_sdp_stat_data(struct dla_processor *processor,
# 				struct dla_processor_group *group) {}
# static inline void
# dla_sdp_dump_stat(struct dla_processor *processor) {}
# #endif


# From engine.c
def dla_enable_intr(mask: np.uint32):
    reg = glb_reg_read("S_INTR_MASK")
    reg = reg & (~mask)
    glb_reg_write("S_INTR_MASK", reg)


def dla_disable_intr(mask: np.uint32):
    reg = glb_reg_read("S_INTR_MASK")
    reg = reg | mask
    glb_reg_write("S_INTR_MASK", reg)