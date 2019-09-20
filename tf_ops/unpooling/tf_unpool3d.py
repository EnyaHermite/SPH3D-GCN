import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
unpool3d_module = tf.load_op_library(os.path.join(base_dir, 'tf_unpool3d_so.so'))

def mean_interpolate(input, nn_index, nn_count):
    return unpool3d_module.mean_interpolate(input, nn_index, nn_count)
@ops.RegisterGradient("MeanInterpolate")
def _mean_interpolate_grad(op, grad_output):
    input = op.inputs[0]
    nn_index = op.inputs[1]
    nn_count = op.inputs[2]
    grad_input = unpool3d_module.mean_interpolate_grad(input, grad_output, nn_index, nn_count)
    return [grad_input, None, None]

def weighted_interpolate(input, weight, nn_index, nn_count):
    return unpool3d_module.weighted_interpolate(input, weight, nn_index, nn_count)
@ops.RegisterGradient("WeightedInterpolate")
def _weighted_interpolate_grad(op, grad_output):
    input = op.inputs[0]
    weight = op.inputs[1]
    nn_index = op.inputs[2]
    nn_count = op.inputs[3]
    grad_input = unpool3d_module.weighted_interpolate_grad(input, grad_output, weight, nn_index, nn_count)
    return [grad_input, None, None, None]
