import numpy as np
import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilanvdla

import sys


class bcolors:
    # https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def bold(inp_string):
    return f'{bcolors.BOLD}{inp_string}{bcolors.ENDC}'


def green(inp_string):
    return f'{bcolors.OKGREEN}{inp_string}{bcolors.ENDC}'


def red(inp_string):
    return f'{bcolors.FAIL}{inp_string}{bcolors.ENDC}'


def compile_and_run(relay_func, args, with_nvdla=True, print_output=True):
    # with nvdla acceleration
    # mod = module
    mod = tvm.IRModule()
    mod["main"] = relay_func
    if with_nvdla:
        # mod["main"] = ilanvdla.convert_to_nhwc(relay_func)
        pattern_table = ilanvdla.pattern_table()
        mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
        mod = tvm.relay.transform.AnnotateTarget(["ilanvdla"])(mod)
        mod = tvm.relay.transform.PartitionGraph()(mod)

    # with open("./test/mod.tvmscript", 'w') as fout:
    #     print(mod.astext(), file=fout)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        ret = vm.invoke("main", *args)
        ila_out = ret.asnumpy()

    if print_output:
        print("ila output: \n{}".format(ila_out))

    return ila_out


def test_correctness(relay_func, args):
    nvdla_out = compile_and_run(
        relay_func, args, with_nvdla=True, print_output=False)
    true_out = compile_and_run(
        relay_func, args, with_nvdla=False, print_output=False)
    caller_func = sys._getframe(1).f_code.co_name
    if caller_func[-6:] == 'tester':
        caller_func = sys._getframe(2).f_code.co_name
    if nvdla_out.shape != true_out.shape:
        print('ERROR - NVDLA output has wrong shape')
    if not np.equal(nvdla_out, true_out).all():
        print('------------------------------------')
        print(red('ERROR - NVDLA output is incorrect:'))
        print('CORRECT:\n', true_out)
        print('-----')
        print('NVDLA:\n', nvdla_out)
        print(red(f'Test failed for function: {bold(caller_func)}'))
        print('------------------------------------')
    else:
        print('------------------------------------')
        print('CORRECT:\n', true_out)
        print('-----')
        print('NVDLA:\n', nvdla_out)
        print(green(f'Test passed for function: {bold(caller_func)}'))
        print('------------------------------------')


def layer_relu(with_nvdla=True):
    # compilerIR/ compiler intermediate representation
    n, h, w, c = 1, 2, 3, 2
    x = relay.Var("x", type_annotation=relay.TensorType(
        shape=(n, h, w, c), dtype='int16'))  # 1, 2px, 3px 2ch - N * H * W * C
    relu_func = relay.Function([x], relay.nn.relu(
        x))
    inp1 = np.zeros((n, h, w, c), 'int16')  # only int16 supported by sim
    idx = 0
    for n_c in range(c):
        for n_h in range(h):
            for n_w in range(w):
                inp1[0][n_h][n_w][n_c] = idx
                idx += 1

    test_correctness(relu_func, [inp1])


def channel_tester(elemwise_relay_func, with_nvdla, n, h, w, c):
    inp1 = np.zeros((n, h, w, c), 'int16')  # only int16 supported by sim
    idx = 0
    for n_c in range(c):
        for n_h in range(h):
            for n_w in range(w):
                inp1[0][n_h][n_w][n_c] = idx
                idx += 1
    inp2 = np.arange(c, dtype='int16')
    test_correctness(elemwise_relay_func, [inp1, inp2])


def channel_bias_add(with_nvdla=True):
    # compilerIR/ compiler intermediate representation
    n, h, w, c = 1, 2, 3, 2
    x_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x = relay.Var("x", x_type)  # 1, 2px, 3px 2ch - N * H * W * C
    y_type = relay.TensorType((c,), dtype='int16')
    y = relay.Var("y", y_type)  # 2ch, - C
    bias_add_func = relay.Function([x, y], relay.nn.bias_add(x, y, axis=3))
    channel_tester(bias_add_func, with_nvdla, n, h, w, c)


def elemwise_tester(elemwise_relay_func, with_nvdla, n, h, w, c):
    inp1 = np.zeros((n, h, w, c), 'int16')  # only int16 supported by sim
    inp2 = np.zeros((n, h, w, c), 'int16')  # only int16 supported by sim
    idx = 0
    for n_c in range(c):
        for n_h in range(h):
            for n_w in range(w):
                inp1[0][n_h][n_w][n_c] = idx
                # inp2[0][n_h][n_w][n_c] = idx+100
                inp2[0][n_h][n_w][n_c] = n_c+1
                idx += 1
    test_correctness(elemwise_relay_func, [inp1, inp2])


def elemwise_max():
    n, h, w, c = 1, 2, 3, 2
    elemwise_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x, y = relay.Var("x", elemwise_type), relay.Var(
        "y", elemwise_type)  # 1, 2px,2px 1ch
    max_func = relay.Function([x, y], relay.maximum(x, y))
    elemwise_tester(max_func, with_nvdla=True, n=n, h=h, w=w, c=c)


def elemwise_min():
    n, h, w, c = 1, 2, 3, 2
    elemwise_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x, y = relay.Var("x", elemwise_type), relay.Var(
        "y", elemwise_type)  # 1, 2px,2px 1ch
    min_func = relay.Function([x, y], relay.minimum(x, y))
    elemwise_tester(min_func, with_nvdla=True, n=n, h=h, w=w, c=c)


def elemwise_equal():
    n, h, w, c = 1, 2, 3, 2
    elemwise_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x, y = relay.Var("x", elemwise_type), relay.Var(
        "y", elemwise_type)  # 1, 2px,2px 1ch
    equal_func = relay.Function([x, y], relay.equal(x, y))
    elemwise_tester(equal_func, with_nvdla=True, n=n, h=h, w=w, c=c)


def elemwise_mul():
    n, h, w, c = 1, 2, 3, 2
    elemwise_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x, y = relay.Var("x", elemwise_type), relay.Var(
        "y", elemwise_type)  # 1, 2px,2px 1ch
    multiply_func = relay.Function([x, y], relay.multiply(x, y))
    elemwise_tester(multiply_func, with_nvdla=True, n=n, h=h, w=w, c=c)


def channel_prelu(with_nvdla=True):
    # compilerIR/ compiler intermediate representation
    n, h, w, c = 1, 2, 3, 2
    x_type = relay.TensorType(shape=(n, h, w, c), dtype='int16')
    x = relay.Var("x", x_type)  # 1, 2px, 3px 2ch - N * H * W * C
    y_type = relay.TensorType((c,), dtype='int16')
    y = relay.Var("y", y_type)  # 2ch, - C
    prelu_func = relay.Function([x, y], relay.nn.prelu(x, y, axis=3))
    channel_tester(prelu_func, with_nvdla, n, h, w, c)


def channel_batch_norm(with_nvdla=True):
    n, h, w, c = 1, 2, 3, 2
    input_type = relay.TensorType(shape=(n, h, w, c), dtype='float32')
    data = relay.Var("data", input_type)  # 1, 2px, 3px 2ch - N * H * W * C

    # 1. norm data
    channel_type = relay.TensorType(shape=(c,), dtype='int16')
    channel_type_float = relay.TensorType(shape=(c,), dtype='float32')
    # 2ch, - C - 2. multiply after norm
    gamma = relay.Var("gamma", channel_type_float)
    # 2ch, - C -   3. add after multiply
    beta = relay.Var("beta", channel_type_float)
    moving_mean = relay.Var("moving_mean", channel_type_float)  # 2ch, - C
    moving_var = relay.Var("moving_var", channel_type_float)  # 2ch, - C

    func_params = [data, gamma, beta, moving_mean, moving_var]
    interior_func = relay.nn.batch_norm(
        data, gamma, beta, moving_mean, moving_var, axis=3, epsilon=0)[0]

    print(type(interior_func))
    batch_norm_func = relay.Function(
        func_params, body=interior_func)

    data_inp = np.zeros((n, h, w, c), 'float32')  # only int16 supported by sim
    idx = 0
    for n_h in range(h):
        for n_w in range(w):
            for n_c in range(c):
                data_inp[0][n_h][n_w][n_c] = idx
                idx += 10
    gamma_inp = np.zeros(c, dtype='float32') + 1
    beta_inp = np.zeros(c, dtype='float32')
    moving_mean_inp = np.zeros((c,), dtype='float32')
    moving_var_inp = np.zeros((c,), dtype='float32')
    test_correctness(batch_norm_func, [
        data_inp, gamma_inp, beta_inp, moving_mean_inp, moving_var_inp])


def conv2d(with_nvdla=True):
    # compilerIR/ compiler intermediate representation
    n, c, h, w = 1, 1, 4, 4
    kern_n, kern_c, kern_h, kern_w = 1, c, 2, 2  # kernel_nb, n_channels, h, w
    x_type = relay.TensorType(shape=(n, c, h, w), dtype='int16')
    x = relay.Var("x", x_type)  # 1, 2px, 3px 2ch - N * H * W * C
    y_type = relay.TensorType(
        shape=(kern_n, kern_c, kern_h, kern_w), dtype='int16')
    # 1 kernel - 2 channel, width 2, height 2 -> output h=2, w = 1
    y = relay.Var("y", y_type)
    conv2d_func = relay.Function([x, y], relay.nn.conv2d(x, y, strides=(
        1, 1), padding=(0, 0), dilation=(2, 2), data_layout='NCHW', kernel_layout="OIHW", kernel_size=(kern_h, kern_w), channels=kern_n))

    # compilerIR/ compiler intermediate representation
    inp1 = np.zeros((n, c, h, w), 'int16')  # only int16 supported by sim
    idx = 0
    for n_c in range(c):
        for n_h in range(h):
            for n_w in range(w):
                inp1[0][n_c][n_h][n_w] = idx
                idx += 1
    inp2 = np.zeros((kern_n, kern_c, kern_h, kern_w), 'int16') - 2
    # print("inp:\n", inp1)
    # print("kernel:\n", inp2)
    # compile_and_run(conv2d_func, [inp1, inp2], with_nvdla=True)
    test_correctness(conv2d_func, [inp1, inp2])


def avgpool2d(with_nvdla=True):
    # compilerIR/ compiler intermediate representation
    n, c, h, w = 1, 1, 3, 2
    x_type = relay.TensorType(shape=(n, c, h, w), dtype='int32')
    x = relay.Var("x", x_type)  # 1, 2px, 3px 2ch - N * H * W * C
    # int16 not supported by official relay
    avgpool2d_func = relay.Function([x], relay.nn.avg_pool2d(x, pool_size=(2, 2), strides=(2, 2),
                                                             dilation=(1, 1), padding=(1, 1), layout='NCHW',
                                                             out_layout='', ceil_mode=False, count_include_pad=True))
    # ceil mode = use ceiling to calculate output shape (default is floor)
    # count_include_pad = include padding to compute the average (default false)

    inp1 = np.zeros((n, c, h, w), 'int32')  # only int16 supported by sim
    idx = 0
    for n_c in range(c):
        for n_h in range(h):
            for n_w in range(w):
                inp1[0][n_c][n_h][n_w] = idx*10
                idx += 1
    compile_and_run(avgpool2d_func, [inp1], with_nvdla)
    # test_correctness(avgpool2d_func, [inp1])


if __name__ == "__main__":
    # layer_relu()
    # channel_bias_add()
    # elemwise_max()
    # elemwise_min()
    # elemwise_equal() # broken - error in size of elements
    # elemwise_mul()
    # channel_prelu()
    # channel_batch_norm(with_nvdla=False) # not fully implemented in c++
    conv2d(with_nvdla=True)
    # avgpool2d(with_nvdla=True)
