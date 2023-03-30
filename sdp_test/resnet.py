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

tflite_model_buf = open("cifar_resnet20_float32q.tflite", "rb").read()
tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

input_tensor = "serving_default_input_1:0"
# (serving_default_input_1:0) shape_signature:[-1, 32, 32, 3], type:INT8
input_shape = (1, 32, 32, 3)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

mod = tvm.relay.transform.FoldScaleAxis()(mod)
mod = tvm.relay.transform.FoldExplicitPadding()(mod)
mod = tvm.relay.qnn.transform.CanonicalizeOps()(mod)
mod = tvm.relay.transform.SimplifyInference()(mod)

with open("cifar_resnet20_float32q.tvmscript", 'w') as fout:
    print(mod.astext(), file=fout)


# image_data = np.random.uniform(size=input_shape).astype("int8")
# target = "llvm"
# with transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target, params=params)

# # Create a runtime executor module
# module = runtime.GraphModule(lib["default"](tvm.cpu()))

# # Feed input data
# module.set_input(input_tensor, tvm.nd.array(image_data))

# # Run
# module.run()

# # Get output
# tvm_output = module.get_output(0).numpy()
# print(tvm_output)
# class_names = ['plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
