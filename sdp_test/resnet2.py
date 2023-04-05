# https://discuss.tvm.apache.org/t/check-failed-pval-nullptr-false-cannot-allocate-memory-symbolic-tensor-shape/8646/20
import tvm
from tvm.relay.frontend import from_onnx
import onnx
from onnx import shape_inference, helper
import onnx_tool
onnx_model_with_shape_and_data = onnx.load(
    "cifar_resnet20_float32q_shapes_data.onnx")
onnx_model_with_shape_and_data.graph.node[0].input[0] = 'input'
onnx_model_with_shape_and_data.graph.input[0].name = 'input'
graph = onnx_model_with_shape_and_data.graph
nodes = list(graph.input) + list(graph.value_info) + list(graph.output)
for node in nodes:
    if not len(node.type.tensor_type.shape.dim):
        continue
    if node.type.tensor_type.shape.dim[0].dim_value == 1:
        node.type.tensor_type.shape.dim[0].dim_param = "batch"
mod, params = from_onnx(onnx_model_with_shape_and_data, shape={"input": (
    1, 32, 32, 3)}, dtype='float32', opset=18, freeze_params=False, convert_config=None)
mod = tvm.relay.transform.InferType()(mod)
mod = tvm.relay.qnn.transform.CanonicalizeOps()(mod)
with open("cifar_resnet20_float32q.tvmscript", 'w') as fout:
    print(mod.astext(show_meta_data=False), file=fout)
