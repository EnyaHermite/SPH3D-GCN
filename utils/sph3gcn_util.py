import tensorflow as tf
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/buildkernel'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/convolution'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nnquery'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/pooling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/unpooling'))

import tf_conv3d, tf_pool3d, tf_unpool3d
from tf_nnquery import build_sphere_neighbor, build_cube_neighbor
from tf_sample import farthest_point_sample, inverse_density_sample, random_sample
from tf_buildkernel import spherical_kernel
neighbor_fn = build_sphere_neighbor # default nn search method


def build_global_graph(xyz, query, radius):
    nn_uplimit = xyz.get_shape().as_list()[1]
    nn_idx, nn_cnt, nn_dst = neighbor_fn(xyz, query, radius=radius,
                                         nnsample=nn_uplimit)

    return nn_idx, nn_cnt, nn_dst


def build_graph(xyz, radius, nn_uplimit, num_sample, sample_method=None):
    intra_idx, intra_cnt, intra_dst = neighbor_fn(xyz, xyz, radius=radius,
                                                  nnsample=nn_uplimit)

    if num_sample is not None:
        if sample_method == 'random':
            sample_index = random_sample(num_sample, xyz)
        elif sample_method == 'FPS':
            sample_index = farthest_point_sample(num_sample, xyz)
        elif sample_method == 'IDS':
            prob = tf.divide(tf.reduce_sum(intra_dst, axis=-1), tf.cast(intra_cnt,dtype=tf.float32))
            sample_index = inverse_density_sample(num_sample, prob)
        else:
            raise ValueError('Unknown sampling method.')

        batch_size = tf.shape(xyz)[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, num_sample, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(sample_index, axis=2)], axis=2)
    else:
        indices = None

    return intra_idx, intra_cnt, intra_dst, indices


def build_graph_deconv(xyz, xyz_unpool, radius, nn_uplimit):
    intra_idx, intra_cnt, intra_dst = neighbor_fn(xyz, xyz, radius=radius,
                                                  nnsample=nn_uplimit)
    inter_idx, inter_cnt, inter_dst = neighbor_fn(xyz, xyz_unpool, radius=radius,
                                                  nnsample=nn_uplimit)

    return intra_idx, intra_cnt, intra_dst, inter_idx, inter_cnt, inter_dst


def _variable_with_weight_decay(name, shape, stddev, with_decay, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        with_decay: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
        use_xavier: bool, whether to use the Xavier(Glorot) normal initializer

    Returns:
        Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    if with_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), with_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def separable_conv3d(inputs,
                     num_out_channels,
                     kernel_size,
                     depth_multiplier,
                     scope,
                     nn_index,
                     nn_count,
                     filt_index,
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.elu,
                     with_bn=False,
                     with_bias=False,
                     reuse=None,
                     is_training=None):
    """ 3D separable convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxNxC
        num_out_channels: int
        kernel_size: int
        depth_multiplier: int
        scope: string
        nn_index: int32 array, neighbor indices
        nn_count: int32 array, number of neighbors
        filt_index: int32 array, filter bin indices
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        with_bn: bool, whether to use batch norm
        is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope,reuse=reuse) as sc:
        num_in_channels = inputs.get_shape().as_list()[-1]
        depthwise_kernel_shape = [kernel_size, num_in_channels,
                                  depth_multiplier]
        depthwise_kernel = _variable_with_weight_decay('depthwise_weights',
                                                       shape=depthwise_kernel_shape,
                                                       use_xavier=use_xavier,
                                                       stddev=stddev,
                                                       with_decay=weight_decay)
        outputs = tf_conv3d.depthwise_conv3d(inputs, depthwise_kernel, nn_index,
                                             nn_count, filt_index)

        batch_size = outputs.get_shape().as_list()[0]
        num_in_channels = outputs.get_shape().as_list()[-1]
        kernel_shape = [num_in_channels, num_out_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             with_decay=weight_decay)

        # pointwise convolution with tf.matmul,
        # it takes less memory, and is also more efficient
        outputs = tf.reshape(outputs, [-1,num_in_channels])
        outputs = tf.matmul(outputs, kernel)
        outputs = tf.reshape(outputs,[batch_size,-1,num_out_channels])

        if with_bias:
            biases = tf.get_variable('biases', [num_out_channels], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if with_bn:
            outputs = batch_normalization(outputs, is_training, name='bn', reuse=reuse)

        return outputs


def pointwise_conv3d(inputs,
                     num_out_channels,
                     scope,
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.elu,
                     with_bn=False,
                     with_bias=False,
                     reuse=None,
                     is_training=None):
    """ pointwise convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxNxC
        num_out_channels: int
        scope: string
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        with_bn: bool, whether to use batch norm
        is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope,reuse=reuse) as sc:
        input_shape = inputs.get_shape().as_list()
        batch_size = input_shape[0]
        num_in_channels = input_shape[-1]
        kernel_shape = [num_in_channels, num_out_channels]
        kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           with_decay=weight_decay)

        inputs = tf.reshape(inputs, [-1, num_in_channels])
        outputs = tf.matmul(inputs, kernel)
        outputs = tf.reshape(outputs, [batch_size,-1,num_out_channels])

        if with_bias:
            print('has bias')
            biases = tf.get_variable('biases', [num_out_channels], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            print(biases)
            outputs = tf.nn.bias_add(outputs, biases)
            print(outputs)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if with_bn:
            outputs = batch_normalization(outputs, is_training, name='bn', reuse=reuse)

        return outputs


def fully_connected(inputs,
                    num_out_channels,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.elu,
                    with_bn=False,
                    with_bias=False,
                    reuse=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
        inputs: 2-D tensor variable BxC
        num_out_channels: int
        scope: string
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        with_bn: bool, whether to use batch norm
        is_training: bool Tensor variable

    Returns:
      Variable tensor of size B x num_out_channels
    """
    with tf.variable_scope(scope,reuse=reuse) as sc:
        num_in_channels = inputs.get_shape().as_list()[-1]
        kernel_shape = [num_in_channels, num_out_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             with_decay=weight_decay)
        outputs = tf.matmul(inputs, kernel)

        if with_bias:
            biases = tf.get_variable('biases', [num_out_channels], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if with_bn:
            outputs = batch_normalization(outputs, is_training, name='bn', reuse=reuse)

        return outputs


def pool3d(inputs, nn_index, nn_count, scope, method):
    """ 3D pooling.

    Args:
        inputs: 3-D tensor BxNxC
        nn_index: int32 array, neighbor and filter bin indices
        nn_count: int32 array, number of neighbors
        scope: string
        method: string, the pooling method

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        if method == 'max':
            outputs, max_index = tf_pool3d.max_pool3d(inputs, nn_index, nn_count)
        elif method == 'avg':
            outputs = tf_pool3d.avg_pool3d(inputs, nn_index, nn_count)
        else:
            raise ValueError("Unknow pooling method %s." % method)

        return outputs


def unpool3d(inputs, nn_index, nn_count, nn_dist, scope, method):
    """ 3D unpooling

    Args:
        inputs: 3-D tensor BxNxC
        nn_index: int32 array, neighbor indices
        nn_count: int32 array, number of neighbors
        nn_dist: float32 array, neighbor (sqrt) distances for weight computation
        scope: string
        method: string, the unpooling method

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        if method == 'mean':
            outputs = tf_unpool3d.mean_interpolate(inputs, nn_index, nn_count)
        elif method == 'weighted':
            sum_nn_dist = tf.reduce_sum(nn_dist, axis=-1, keepdims=True)
            epsilon = 1e-7
            weight = tf.divide(nn_dist+epsilon, sum_nn_dist+epsilon)
            outputs = tf_unpool3d.weighted_interpolate(inputs, weight, nn_index, nn_count)
        else:
            raise ValueError("Unknow unpooling method %s." % method)

        return outputs


def batch_normalization(data, is_training, name, reuse=None):
    return tf.layers.batch_normalization(data, momentum=0.99, training=is_training,
                                         beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         reuse=reuse, name=name)


