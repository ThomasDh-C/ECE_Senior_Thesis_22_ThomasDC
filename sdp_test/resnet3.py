import tvm.relay.testing
from tvm import relay
from tvm import runtime
import numpy as np
from tvm.contrib import graph_executor
from tvm.runtime.ndarray import cpu
from tvm.relay import *
from tvm.relay.op.contrib import ilanvdla

mod, params = tvm.relay.testing.resnet.get_workload(
    1, 10, 18, image_shape=(3, 32, 32), layout='NCHW', dtype='int16')
mod = relay.transform.InferType()(mod)

inp1 = np.zeros((1, 3, 32, 32), 'int16')  # only int16 supported by sim


# https://pages.dogdog.run/tvm/tvm_user_pass.html


@relay.transform.function_pass(opt_level=1)
class TransformSoftmax:
    def transform_function(self, func, mod, ctx):
        class SimpleTransform(relay.ExprMutator):
            def infer_type(self, node):
                mod = tvm.IRModule.from_expr(node)
                mod = relay.transform.InferType()(mod)
                entry = mod["main"]
                return entry if isinstance(node, relay.Function) else entry.body

            def visit_call(self, call):
                op = call.op
                if op.name not in {'nn.softmax', 'sqrt'}:
                    # return relay.Call(call.op, list(map(self.visit, call.args)), call.attrs, type_args=call.type_args, span=call.span)
                    return super().visit_call(call)
                args = [self.visit(x) for x in call.args]
                if op.name == 'nn.softmax':
                    data = args[0]
                    expr = cast(data, 'float32')
                    expr = relay.nn.softmax(expr)
                    expr = cast(expr, 'int16')
                    return expr
                if op.name == 'sqrt':
                    data = args[0]
                    expr = cast(data, 'float32')
                    expr = relay.sqrt(expr)
                    expr = cast(expr, 'int16')
                    return expr
        return SimpleTransform().visit(func)


def compile_and_run(module, input, parameters, with_nvdla=True, print_output=True):
    if with_nvdla:
        module = TransformSoftmax()(module)
        pattern_table = ilanvdla.pattern_table()
        module = tvm.relay.transform.MergeComposite(pattern_table)(module)
        module = tvm.relay.transform.AnnotateTarget(["ilanvdla"])(module)
        module = tvm.relay.transform.PartitionGraph()(module)
        print(module)
        with open("./test/mod.tvmscript", 'w') as fout:
            print(module.astext(), file=fout)
    else:
        module = tvm.relay.transform.SimplifyInference()(module)
        module = TransformSoftmax()(module)
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


ila_out = compile_and_run(
    mod, inp1, params, with_nvdla=True, print_output=True)
# mod = TransformSoftmax()(mod)
# with tvm.transform.PassContext(opt_level=3):
#     device = tvm.cpu()
#     target = "llvm"
#     exe = relay.vm.compile(mod, target)
#     vm = runtime.vm.VirtualMachine(exe, device)

#     ret = vm.invoke("main", inp1, **params)
#     ila_out = ret.asnumpy()
