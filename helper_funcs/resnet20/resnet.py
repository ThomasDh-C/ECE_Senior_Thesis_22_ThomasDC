# https://tvm.apache.org/docs/how_to/compile_models/from_tflite.html
import tflite
# import tensorflow as tf
# import tensorflow_datasets as tfds
from tvm import relay, transform
import tvm
from tvm.contrib import graph_executor as runtime
import numpy as np

# ds = tfds.load('cifar10', as_supervised=True)
# std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
# mean = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))


# def valid_prep(x, y):
#     x = tf.cast(x, tf.float32)/255.
#     x = (x - mean) / std
#     return x, y
# ds['test'] = ds['test'].map(valid_prep)

from tvm.relay.frontend import from_onnx
import onnx
from onnx import shape_inference, helper
import onnx_tool
from mlprodict.onnx_tools.onnx_tools import ensure_topological_order
# python -m tf2onnx.convert --opset 13 --tflite cifar_resnet20_float32q.tflite --output cifar_resnet20_float32q.onnx
onnx_model_no_shape = onnx.load("cifar_resnet20_float32q.onnx")
onnx_tool.model_shape_infer(onnx_model_no_shape, None,
                            saveshapesmodel='cifar_resnet20_float32q_shapes.onnx', shapesonly=True, verbose=True)
onnx_tool.model_shape_infer(onnx_model_no_shape, None,
                            saveshapesmodel='cifar_resnet20_float32q_shapes_data.onnx', shapesonly=False, verbose=True)
onnx_model_with_shape = onnx.load("cifar_resnet20_float32q_shapes.onnx")
onnx_model_with_shape_and_data = onnx.load(
    "cifar_resnet20_float32q_shapes_data.onnx")


all_val_infos = []
all_val_infos_strs = []
transform_name = {1: 'float32', 3: 'int8', 6: 'int32', 7: 'int64'}
for node in onnx_model_with_shape_and_data.graph.initializer:
    temp_dims = node.dims
    if len(temp_dims) == 0:
        temp_dims = [1]
    temp = helper.make_value_info(name=node.name,
                                  type_proto=helper.make_tensor_type_proto(elem_type=node.data_type, shape=temp_dims))
    all_val_infos.append(temp)

onnx_model_graph = helper.make_graph(nodes=onnx_model_with_shape.graph.node,
                                     name='resnet20graph',
                                     inputs=list(
                                         onnx_model_with_shape.graph.input) + all_val_infos,
                                     outputs=onnx_model_with_shape.graph.output)
onnx_model_new = helper.make_model(onnx_model_graph, functions=[])

onnx.checker.check_model(onnx_model_new)
onnx_model_new = shape_inference.infer_shapes(onnx_model_new)
# onnx_model_with_shape_and_data
mod, params = from_onnx(onnx_model_new, shape={"serving_default_input_1:0": (
    1, 32, 32, 3)}, dtype='float32', opset=18, freeze_params=True, convert_config=None)


# onnx.save(onnx_model_new, "cifar_resnet20_float32q_shapes2.onnx")
# https://github.com/dmlc/tensorboard/blob/master/tensorboard/src/onnx.proto


# onnx_model.graph.node[0].input[0] = 'input'
# onnx_model.graph.input[0].name = 'input'


# onnx.checker.check_model(onnx_model)
# onnx_model = shape_inference.infer_shapes(onnx_model)
# mod, params = from_onnx(onnx_model, shape={"serving_default_input_1:0": (
#     1, 32, 32, 3)}, dtype='float32', opset=15, freeze_params=True, convert_config=None)
# mod = tvm.relay.transform.InferType()(mod)
# mod = tvm.relay.qnn.transform.CanonicalizeOps()(mod)


# tflite_model_buf = open("cifar_resnet20_float32q.tflite", "rb").read()
# tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

# input_tensor = "serving_default_input_1:0"
# # (serving_default_input_1:0) shape_signature:[-1, 32, 32, 3], type:INT8
# input_shape = (1, 32, 32, 3)
# input_dtype = "float32"

# mod, params = relay.frontend.from_tflite(
#     tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
# )

mod = tvm.relay.transform.FoldScaleAxis()(mod)
mod = tvm.relay.transform.FoldExplicitPadding()(mod)
mod = tvm.relay.qnn.transform.CanonicalizeOps()(mod)
mod = tvm.relay.transform.DefuseOps()(mod)
# mod = tvm.relay.transform.FoldConstant()(mod) # makes shorter but could confuse flexmatch
mod = tvm.relay.transform.SimplifyInference()(mod)

with open("cifar_resnet20_float32q.tvmscript", 'w') as fout:
    print(mod.astext(show_meta_data=False), file=fout)

funcs_called = set()
for l in mod.astext().split('\n'):
    if len(l) >= 3 and l[2] == '%':
        l = l.split(' = ')[1]
        funcs_called.add(l.split('(')[0])
