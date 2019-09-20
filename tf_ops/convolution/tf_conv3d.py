import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
conv3d_module = tf.load_op_library(os.path.join(base_dir, 'tf_conv3d_so.so'))


def depthwise_conv3d(input, filter, nn_index, nn_count, bin_index):
    '''
    Input:
        input:   (batch, npoint, in_channels) float32 array, input point features
        filter: (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        bin_index: (batch, mpoint, nnsample), filtet bins' indices
    Output:
        output: (batch, mpoint, out_channels) float32 array, output point features
    '''
    return conv3d_module.depthwise_conv3d(input, filter, nn_index, nn_count, bin_index)

@ops.RegisterGradient("DepthwiseConv3d")
def _depthwise_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    nn_index = op.inputs[2]
    nn_count = op.inputs[3]
    bin_index = op.inputs[4]
    grad_input, grad_filter = conv3d_module.depthwise_conv3d_grad(input, filter, grad_output, nn_index,
                                                                  nn_count, bin_index)
    return [grad_input, grad_filter, None, None, None]





