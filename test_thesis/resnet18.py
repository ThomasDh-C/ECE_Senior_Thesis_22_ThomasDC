import tvm.relay.testing
from tvm import relay
from tvm import runtime
import numpy as np
from tvm.contrib import graph_executor
from tvm.runtime.ndarray import cpu
from tvm.relay import *
from tvm.relay.op.contrib import ilanvdla
from tvm.relay.dataflow_pattern import wildcard, is_op, is_tuple_get_item, rewrite, DFPatternCallback

mod_nvdla, params_nvdla = tvm.relay.testing.resnet.get_workload(
    1, 10, 18, image_shape=(3, 32, 32), layout='NCHW', dtype='int16')
mod_cpu, params_cpu = tvm.relay.testing.resnet.get_workload(
    1, 10, 18, image_shape=(3, 32, 32), layout='NCHW', dtype='int16')

# inp1 = np.zeros((1, 3, 32, 32), 'int16')  # only int16 supported by sim
inp1 = np.random.randint(size=(1, 3, 32, 32), low=-5,
                         high=5, dtype='int16')


class BatchnormCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once=rewrite_once)
        self.x = wildcard()
        self.var = wildcard()
        self.mean = wildcard()
        self.beta = wildcard()
        self.gamma = wildcard()
        bn_node = is_op('nn.batch_norm')(
            self.x, self.gamma, self.beta, self.mean, self.var)
        tuple_get_item_node = is_tuple_get_item(bn_node, 0)

        self.pattern = tuple_get_item_node

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        var = node_map[self.var][0]
        mean = node_map[self.mean][0]
        beta = node_map[self.beta][0]
        gamma = node_map[self.gamma][0]
        x_f = cast(x, 'float32')
        var_f = cast(var, 'float32')
        mean_f = cast(mean, 'float32')
        beta_f = cast(beta, 'float32')
        gamma_f = cast(gamma, 'float32')

        expr = relay.op.nn.batch_norm(
            x_f, gamma_f, beta_f, mean_f, var_f, epsilon=0.0)[0]
        expr = clip(expr, -32768, 32767)
        expr = cast(expr, 'int16')
        return expr


def compile_and_run(module, input, parameters, with_nvdla=True, print_output=True):
    if with_nvdla:
        pattern_table = ilanvdla.pattern_table()
        module = tvm.relay.transform.MergeComposite(pattern_table)(module)
        module = tvm.relay.transform.AnnotateTarget(["ilanvdla"])(module)
        module = tvm.relay.transform.PartitionGraph()(module)

    module['main'] = rewrite(BatchnormCallback(), module['main'])
    module = tvm.relay.transform.InferType()(module)
    if with_nvdla:
        with open("./test/mod.tvmscript", 'w') as fout:
            print(module.astext(), file=fout)
    else:
        with open("./test/mod_wo_acc.tvmscript", 'w') as fout:
            print(module.astext(), file=fout)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(module, target)
        vm = runtime.vm.VirtualMachine(exe, device)
        ret = vm.invoke("main", input, **parameters)
        ila_out = ret.asnumpy()

    if print_output:
        print("ila output: \n{}".format(ila_out))

    return ila_out


true_out = compile_and_run(
    mod_cpu, inp1, params_cpu, with_nvdla=False, print_output=True)
nvdla_out = compile_and_run(
    mod_nvdla, inp1, params_nvdla, with_nvdla=True, print_output=True)

if not np.equal(nvdla_out, true_out).all():
    print("NVDLA output DIFFERENT from true output")
else:
    print("NVDLA output SAME as true output")

# ----------------------------------------
# NVDLA output
# pre-softmax: [[-2798 -2044 -5613 -2710 -5523   376 -1378 -3192  6997  5068]]
# post-softmax: [[0 0 0 0 0 0 0 0 1 0]]
# ----------------------------------------
# CPU output
# pre-softmax: [[-2798 -2044 -5613 -2710 -5523   376 -1378 -3192  6997  5068]]
# post-softmax: [[0 0 0 0 0 0 0 0 1 0]]
