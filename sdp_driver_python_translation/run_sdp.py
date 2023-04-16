import numpy as np
import argparse
import copy
from helper_classes import *
from helper_funcs import *
from sdp import *


inp_data = []
with open('../data/add/driver_command.txt') as f:
    inp_data = f.readlines()
_, inp_data = inp_data[0].split('.py ')
inp_data = inp_data.split(' ')

parser = argparse.ArgumentParser(description='Add Parameters')
parser.add_argument('--in_size_lhs', nargs='+', type=int)
parser.add_argument('--in_size_rhs', nargs='+', type=int)
parser.add_argument('--out_size', nargs='+', type=int)
parser.add_argument("--op_name")
args = parser.parse_args(inp_data)

lhs = np.fromfile("../data/add/lhs.txt",
                  sep="\n").astype("float16")
rhs = np.fromfile("../data/add/rhs.txt",
                  sep="\n").astype("float16")
print('Operation is:', args.op_name)
print('LHS:\n', lhs.reshape(args.in_size_lhs))
print('RHS:\n', rhs.reshape(args.in_size_rhs))


example_process_group = Dla_processor_group()
# set up special params for this sdp op
sdp_op = example_process_group.operation_desc.sdp_op
sdp_op.src_precision = 2  # "FP16"
sdp_op.dst_precision = 2  # "FP16"
sdp_op.x1_op.enable = 1
sdp_op.x1_op.alu_type = 2
sdp_op.x1_op.type = 2
sdp_op.x1_op.precision = 2
sdp_op.x1_op.mode = 2
sdp_op.x1_op.alu_operand = 42  # don't know if correct
# set up data cubes
sdp_surface = example_process_group.surface_desc.sdp_surface
basic_data_cube = Dla_data_cube()
basic_data_cube.address = 0
# not necessarily going to be passed in right but doesn't matter for current op
basic_data_cube.width = 2
basic_data_cube.height = 2
basic_data_cube.channel = 1
basic_data_cube.line_stride = 2
tot_elems, prec = 2*2, 16
basic_data_cube.surf_stride = tot_elems
sdp_surface.src_data = copy(basic_data_cube)
sdp_surface.x1_data = copy(basic_data_cube)
sdp_surface.x1_data.address = tot_elems * \
    prec  # has to be 32-bit aligned but is
sdp_surface.dst_data = copy(basic_data_cube)
sdp_surface.dst_data.address = tot_elems * \
    prec * 2  # has to be 32-bit aligned but is

acceleratorIR, MMIO = [], []
for i in range(2):
    a, b = lhs[i*2].tobytes().hex(), lhs[i*2+1].tobytes().hex()
    a, b = '0'*(4-len(a))+a, '0'*(4-len(b))+b
    combined = f'0x{b}{a}'  # high significant bits ... low
    acceleratorIR.append(
        {"irfunc_No.": len(acceleratorIR)+1, 'name': 'VirMemWr', 'addr': hex(i*32), 'data': combined})
    MMIO.append({"instr_No.": len(MMIO)+1,
                 "reg_name": 'na',
                 "addr": hex(i*32),
                 "data": combined,
                 "mode": "W"})
for i in range(2):
    # https://www.appsloveworld.com/numpy/100/135/convert-16-bit-hex-value-to-fp16-in-python
    a, b = rhs[i*2].tobytes().hex(), rhs[i*2+1].tobytes().hex()
    a, b = '0'*(4-len(a))+a, '0'*(4-len(b))+b
    combined = f'0x{b}{a}'  # high significant bits ... low
    acceleratorIR.append(
        {"irfunc_No.": len(acceleratorIR)+1, 'name': 'VirMemWr', 'addr': hex((i+2)*32), 'data': combined})
    MMIO.append({"instr_No.": len(MMIO)+1,
                 "reg_name": 'na',
                 "addr": hex((i+2)*32),
                 "data": combined,
                 "mode": "W"})
dla_sdp_program(example_process_group, acceleratorIR, MMIO)

for i in range(2):
    acceleratorIR.append(
        {"irfunc_No.": len(acceleratorIR)+1, 'name': 'VirMemRd', 'addr': hex((i+4)*32)})
    MMIO.append({"instr_No.": len(MMIO)+1,
                 "addr": hex((i+4)*32),
                 "mode": "R"})

with open('add_ila_asm.json', 'w') as f:
    f.write(mmio_print(acceleratorIR))
with open('add_prog_frag.json', 'w') as f:
    f.write(mmio_print(MMIO))
