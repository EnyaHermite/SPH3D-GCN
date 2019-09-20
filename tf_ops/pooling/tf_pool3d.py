import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
pool3d_module = tf.load_op_library(os.path.join(base_dir, 'tf_pool3d_so.so'))

def max_pool3d(input, nn_index, nn_count):
    return pool3d_module.max_pool3d(input, nn_index, nn_count)
@ops.RegisterGradient('MaxPool3d')
def _max_pool3d_grad(op, grad_output, grad_index):
    input = op.inputs[0]
    max_index = op.outputs[1]
    #grad_output = grads[0]  # grads[1] = None (gradient to indices of the max-pooled locations)
    grad_input = pool3d_module.max_pool3d_grad(input, grad_output, max_index)
    return [grad_input, None, None]


def avg_pool3d(input, nn_index, nn_count):
    return pool3d_module.avg_pool3d(input, nn_index, nn_count)
@ops.RegisterGradient('AvgPool3d')
def _avg_pool3d_grad(op, grad_output):
    input = op.inputs[0]
    nn_index = op.inputs[1]
    nn_count = op.inputs[2]
    grad_input = pool3d_module.avg_pool3d_grad(input, grad_output, nn_index, nn_count)
    return [grad_input, None, None]

