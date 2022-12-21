import numpy as np
import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilanvdla

import sys


def test_add():
    in_chan = 1
    in_row = 2
    in_col = 2
    # compilerIR/ compiler intermediate representation
    x = relay.Var("x", type_annotation=relay.TensorType(
        (1, in_chan, in_row, in_col)))  # 1, 2px,2px 1ch
    y = relay.Var("y", type_annotation=relay.TensorType(
        (1, in_chan, in_row, in_col)))  # 1, 2px,2px 1ch
    add_func = relay.Function([x, y], relay.add(x, y))

    # with nvdla acceleration
    # mod = module
    mod = tvm.IRModule()
    mod["main"] = add_func
    pattern_table = ilanvdla.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilanvdla"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('!!heyyyho!!')
    with open("./test/add/mod.tvmscript", 'w') as fout:
        print(mod.astext(), file=fout)

    # no acceleration
    mod_wo_acc = tvm.IRModule()
    mod_wo_acc["main"] = add_func
    with open("./test/add/mod_wo_acc.tvmscript", 'w') as fout:
        print(mod_wo_acc.astext(), file=fout)

    inp1 = np.random.uniform(-1, 1, (1, in_chan, in_row,
                                     in_col)).astype("float32")
    inp2 = np.random.uniform(-1, 1, (1, in_chan, in_row,
                                     in_col)).astype("float32")

    with open("./test/add/inputs.log", 'w') as fout:
        print('input array 1:\n{}\n'.format(inp1), file=fout)
        print('input array 2:\n{}\n'.format(inp2), file=fout)

    # without nvdla backend
    print('--- USING CPU ---')
    with tvm.transform.PassContext():
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod_wo_acc, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp1, inp2]
        ret = vm.invoke("main", *args)
        ref_out = ret.asnumpy()

    print('--- USING NVDLA BACKGROUND ---')
    # use nvdla backend
    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp1, inp2]
        ret = vm.invoke("main", *args)
        ila_out = ret.asnumpy()

    print("reference output: \n{}".format(ref_out))
    print("ila output: \n{}".format(ila_out))


def test_bias_add():
    in_chan = 1
    in_row = 2
    in_col = 2
    # compilerIR/ compiler intermediate representation
    x = relay.Var("x", type_annotation=relay.TensorType(
        (1, in_chan, in_row, in_col)))  # 1, 2px,2px 1ch
    y = relay.Var("y", type_annotation=relay.TensorType(
        (1,)))  # 1, 2px,2px 1ch
    bias_add_func = relay.Function([x, y], relay.nn.bias_add(x, y))

    # with nvdla acceleration
    # mod = module
    mod = tvm.IRModule()
    mod["main"] = bias_add_func
    pattern_table = ilanvdla.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilanvdla"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    with open("./test/bias_add/mod.tvmscript", 'w') as fout:
        print(mod.astext(), file=fout)

    # no acceleration
    mod_wo_acc = tvm.IRModule()
    mod_wo_acc["main"] = bias_add_func
    with open("./test/bias_add/mod_wo_acc_biasadd.tvmscript", 'w') as fout:
        print(mod_wo_acc.astext(), file=fout)

    inp1 = np.random.uniform(-1, 1, (1, in_chan, in_row,
                                     in_col)).astype("float32")
    inp2 = np.random.uniform(-1, 1, (1,)).astype("float32")

    with open("./test/bias_add/inputs.log", 'w') as fout:
        print('input array 1:\n{}\n'.format(inp1), file=fout)
        print('input array 2:\n{}\n'.format(inp2), file=fout)

    # without nvdla backend
    print('--- USING CPU ---')
    with tvm.transform.PassContext():
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod_wo_acc, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp1, inp2]
        ret = vm.invoke("main", *args)
        ref_out = ret.asnumpy()

    print('--- USING NVDLA BACKGROUND ---')
    # use nvdla backend
    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp1, inp2]
        ret = vm.invoke("main", *args)
        ila_out = ret.asnumpy()

    print("reference output: \n{}".format(ref_out))
    print("ila output: \n{}".format(ila_out))


if __name__ == "__main__":
    test_add()
    # test_bias_add()
