import numpy as np
from helper_classes import *
import json


def retrieve_all_macros():
    all_macros = {}
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
                base_adr_value_int = int(base_adr_value_str, 0)
                all_macros[base_adr_name] = np.uint32(base_adr_value_int)

            elif len(split_up) == 4:
                # rest of lines have length 4 - [name, '', '', val]
                # val is always a function of form func(a) or func(a,b)
                # deal with func(a) on this pass
                macro_name, _, _, macro_value_str = split_up
                if macro_value_str.count(',') > 0:
                    continue
                start_paren = macro_value_str.find('(')
                end_paren = macro_value_str.find(')')
                macro_value_str = macro_value_str[start_paren+1:end_paren]
                macro_value_int = int(macro_value_str, 0)
                all_macros[macro_name] = macro_value_int

        # deal with func(a,b) on this pass
        for l in all_lines:
            split_up = l.split('\t')
            # _MK_FIELD_CONST are only thing left and we have to do
            # _MK_FIELD_CONST(_mask_, _shift_) (_mask_ << _shift_)
            if len(split_up) == 4 and split_up[3].count(',') > 0:
                macro_name, _, _, macro_value_str = split_up
                start_paren = macro_value_str.find('(')
                end_paren = macro_value_str.find(')')
                macro_value_str = macro_value_str[start_paren+1:end_paren]
                l, r = macro_value_str.split(', ')
                if l in all_macros:
                    l = all_macros[l]
                if r in all_macros:
                    r = all_macros[r]
                if type(l) == str:
                    l = int(l, 0)
                if type(r) == str:
                    r = int(r, 0)
                res = np.right_shift(l, r, dtype=np.int32)
                all_macros[macro_name] = np.uint32(res)

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


def glb_reg(name: str):
    return f'GLB_{name}_0'


def mcif_reg(name: str):
    return f'MCIF_{name}_0'


def cvif_reg(name: str):
    return f'CVIF_{name}_0'


def bdma_reg(name: str):
    return f'BDMA_{name}_0'


def cdma_reg(name: str):
    return f'CDMA_{name}_0'


def csc_reg(name: str):
    return f'CSC_{name}_0'


def cmac_a_reg(name: str):
    return f'CMAC_A_{name}_0'


def cmac_b_reg(name: str):
    return f'CMAC_B_{name}_0'


def cacc_reg(name: str):
    return f'CACC_{name}_0'


def sdp_rdma_reg(name: str):
    return f'SDP_RDMA_{name}_0'


def sdp_reg(name: str):
    return f'SDP_{name}_0'


def pdp_rdma_reg(name: str):
    return f'PDP_RDMA_{name}_0'


def pdp_reg(name: str):
    return f'PDP_{name}_0'


def cdp_rdma_reg(name: str):
    return f'CDP_RDMA_{name}_0'


def cdp_reg(name: str):
    return f'CDP_{name}_0'


def rbk_reg(name: str):
    return f'RBK_{name}_0'


# Functions below use these general reg write and read


def reg_read(reg_name, all_macros, MMIO) -> np.uint32:
    # TODO: FIND MY DEF BUT RET uint32
    addr = all_macros[reg_name]
    data = 0  # arbitrary
    MMIO.append({"instr_No.": len(MMIO)+1,
                 "reg_name": reg_name,
                 "addr": hex(addr),
                 "mode": "R"})
    return data


def pad_hex(val: np.uint32):
    cleaned_hexed_val = hex(np.uint32(val))[2:]
    return '0x' + (8-len(cleaned_hexed_val))*'0' + cleaned_hexed_val


def reg_write(reg_name, val: np.uint32, all_macros, MMIO):
    addr = all_macros[reg_name]
    # I know reg_name shouldn't be in there but otherwise hard to match up quickly
    MMIO.append({"instr_No.": len(MMIO)+1,
                 "reg_name": reg_name,
                 "addr": hex(addr),
                 "data": pad_hex(val),
                 "mode": "W"})


# alias for register read for each sub-module - REMOVE IN FUTURE
def glb_reg_read(reg: str, all_macros, MMIO):
    return reg_read(glb_reg(reg), all_macros, MMIO)


def bdma_reg_read(reg: str, all_macros, MMIO):
    return reg_read(bdma_reg(reg), all_macros, MMIO)


def cdma_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cdma_reg(reg), all_macros, MMIO)


def csc_reg_read(reg: str, all_macros, MMIO):
    return reg_read(csc_reg(reg), all_macros, MMIO)


def cmac_a_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cmac_a_reg(reg), all_macros, MMIO)


def cmac_b_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cmac_b_reg(reg), all_macros, MMIO)


def cacc_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cacc_reg(reg), all_macros, MMIO)


def sdp_rdma_reg_read(reg: str, all_macros, MMIO):
    return reg_read(sdp_rdma_reg(reg), all_macros, MMIO)


def sdp_reg_read(reg: str, all_macros, MMIO):
    return reg_read(sdp_reg(reg), all_macros, MMIO)


def pdp_rdma_reg_read(reg: str, all_macros, MMIO):
    return reg_read(pdp_rdma_reg(reg), all_macros, MMIO)


def pdp_reg_read(reg: str, all_macros, MMIO):
    return reg_read(pdp_reg(reg), all_macros, MMIO)


def cdp_rdma_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cdp_rdma_reg(reg), all_macros, MMIO)


def cdp_reg_read(reg: str, all_macros, MMIO):
    return reg_read(cdp_reg(reg), all_macros, MMIO)


def rubik_reg_read(reg: str, all_macros, MMIO):
    return reg_read(rbk_reg(reg), all_macros, MMIO)


# alias for register write for each sub-module
def glb_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(glb_reg(reg), val, all_macros, MMIO)


def bdma_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(bdma_reg(reg), val, all_macros, MMIO)


def cdma_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cdma_reg(reg), val, all_macros, MMIO)


def csc_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(csc_reg(reg), val, all_macros, MMIO)


def cmac_a_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cmac_a_reg(reg), val, all_macros, MMIO)


def cmac_b_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cmac_b_reg(reg), val, all_macros, MMIO)


def cacc_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cacc_reg(reg), val, all_macros, MMIO)


def sdp_rdma_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(sdp_rdma_reg(reg), val, all_macros, MMIO)


def sdp_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(sdp_reg(reg), val, all_macros, MMIO)


def pdp_rdma_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(pdp_rdma_reg(reg), val, all_macros, MMIO)


def pdp_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(pdp_reg(reg), val, all_macros, MMIO)


def cdp_rdma_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cdp_rdma_reg(reg), val, all_macros, MMIO)


def cdp_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(cdp_reg(reg), val, all_macros, MMIO)


def rubik_reg_write(reg: str, val: np.uint32, all_macros, MMIO):
    return reg_write(rbk_reg(reg), val, all_macros, MMIO)

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
def dla_enable_intr(all_macros, acceleratorIR, MMIO):
    acceleratorIR.append(
        {"irfunc_No.": len(acceleratorIR)+1, 'name': 'INTR_MASK_READ'})
    reg = glb_reg_read("S_INTR_MASK", all_macros, MMIO)

    mask1 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK1")]
    mask2 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK0")]
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': 'GLB_INTR_MASK',
                          'sdp_done_mask1': pad_hex(mask1),
                          'sdp_done_mask2': pad_hex(mask2),
                          'curr': pad_hex(reg)})
    mask_var = mask1 | mask2
    reg = reg & (~mask_var)
    glb_reg_write("S_INTR_MASK", reg, all_macros, MMIO)

# unused


def dla_disable_intr(mask: np.uint32, all_macros, acceleratorIR, MMIO):
    acceleratorIR.append(
        {"irfunc_No.": len(acceleratorIR)+1, 'name': 'INTR_MASK_READ'})
    reg = glb_reg_read("S_INTR_MASK", all_macros, MMIO)
    acceleratorIR.append({"irfunc_No.": len(acceleratorIR)+1,
                          'name': 'GLB_INTR_MASK',
                          'mask': mask,
                          'curr': pad_hex(reg)})
    reg = reg | mask
    glb_reg_write("S_INTR_MASK", reg, all_macros, MMIO)


def dla_read_input_address(data: Dla_data_cube, op_index: np.int16, roi_index: np.uint8, bpp: np.uint8):
    """NOT IMPLEMENTED"""
    # TODO: implement
    return np.uint64(-1)


def dla_get_dma_cube_address():
    """NOT IMPLEMENTED"""
    # TODO: implement
    return np.uint64(-1)


def dla_read_lut():
    """NOT IMPLEMENTED"""
    # TODO: implement
    return Dla_lut_param()


def update_lut(reg_base: str, lut: Dla_lut_param, precision: np.uint8):
    """NOT IMPLEMENTED"""
    # TODO: implement

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.uint8, np.int16, np.uint32)):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


def mmio_print(MMIO):
    ret = "{\n\"ops\": ["
    ret += ',\n'.join([json.dumps(m, cls=NumpyEncoder) for m in MMIO])
    ret += "]\n}\n"
    return ret
