from helper_funcs import *
import numpy as np

# all_macros = retrieve_all_macros()
all_macros = {}
test = []
test2 = []
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


def dla_sdp_set_producer(group_id: int, rdma_group_id: int):
    # set producer pointer for all sub-modules
    reg, field = "SDP_S_POINTER_0", "PRODUCER"
    shift_by = all_macros[f'{reg}_{field}_SHIFT']
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    sdp_reg_write("S_POINTER", reg)

    reg, field = "SDP_RDMA_S_POINTER_0", "PRODUCER"
    shift_by = all_macros[f'{reg}_{field}_SHIFT']
    reg = np.left_shift(group_id, shift_by, dtype=np.int32)
    sdp_rdma_reg_write("S_POINTER", reg)

# enable perf registers if engine desires, then rdma if group says yes, then defo SDP
# int dla_sdp_enable(struct dla_processor_group *group)

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
# TODO: make a class for struct dla_processor_group * group


def dla_sdp_program(group):
    print("Enter SDP")
    mask1 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK1")]
    mask2 = all_macros[mask("GLB_S_INTR_MASK_0", "SDP_DONE_MASK0")]
    dla_enable_intr(mask1 | mask2)

    ret = processor_sdp_program(group)

    print("Exit SDP")
