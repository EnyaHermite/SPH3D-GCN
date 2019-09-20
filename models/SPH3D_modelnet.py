import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import sph3gcn_util as s3g_util


def normalize_xyz(points):
    points -= tf.reduce_mean(points,axis=1,keepdims=True)
    scale = tf.reduce_max(tf.reduce_sum(tf.square(points),axis=-1,keepdims=True),axis=1,keepdims=True)
    scale = tf.sqrt(scale,name='normalize')
    points /= scale

    return points


def _separable_conv3d_block(net, list_channels, bin_size, nn_index, nn_count, filt_idx,
                            name, depth_multiplier=None, weight_decay=None, reuse=None,
                            with_bn=True, with_bias=True, is_training=None):
    for l, num_out_channels in enumerate(list_channels):
        scope = name + '_' + str(l+1) # number from 1, not 0
        net = s3g_util.separable_conv3d(net, num_out_channels, bin_size,
                                        depth_multiplier[l], scope, nn_index,
                                        nn_count, filt_idx, weight_decay=weight_decay,
                                        with_bn=with_bn, with_bias=with_bias,
                                        reuse=reuse, is_training=is_training)
    return net


def get_model(points, is_training, config=None):
    """ Classification Network, input is BxNx3, output Bx40 """
    batch_size = points.get_shape()[0].value
    num_point = points.get_shape()[1].value
    end_points = {}

    assert(num_point==config.num_input)

    if config.normalize:
        points = normalize_xyz(points)

    xyz = points
    query = tf.reduce_mean(xyz, axis=1, keepdims=True)  # the global viewing point
    reuse = None
    net = s3g_util.pointwise_conv3d(xyz, config.mlp, 'mlp1',
                                    weight_decay=config.weight_decay,
                                    with_bn=config.with_bn, with_bias=config.with_bias,
                                    reuse=reuse, is_training=is_training)

    global_feat = []
    for l in range(len(config.radius)):
        if config.use_raw:
            net = tf.concat([net, xyz], axis=-1)

        # the neighbor information is the same within xyz_pose_1 and xyz_pose_2.
        # Therefore, we compute it with xyz_pose_1, and apply it to xyz_pose_2 as well
        intra_idx, intra_cnt, \
        intra_dst, indices = s3g_util.build_graph(xyz, config.radius[l], config.nn_uplimit[l],
                                                  config.num_sample[l], sample_method=config.sample)
        filt_idx = s3g_util.spherical_kernel(xyz, xyz, intra_idx, intra_cnt,
                                             intra_dst, config.radius[l],
                                             kernel=config.kernel)

        net = _separable_conv3d_block(net, config.channels[l], config.binSize, intra_idx, intra_cnt,
                                      filt_idx, 'conv'+str(l+1), config.multiplier[l], reuse=reuse,
                                      weight_decay=config.weight_decay, with_bn=config.with_bn,
                                      with_bias=config.with_bias, is_training=is_training)

        if config.num_sample[l]>1:
            # ==================================gather_nd====================================
            xyz = tf.gather_nd(xyz, indices)
            inter_idx = tf.gather_nd(intra_idx, indices)
            inter_cnt = tf.gather_nd(intra_cnt, indices)
            inter_dst = tf.gather_nd(intra_dst, indices)
            # =====================================END=======================================

            net = s3g_util.pool3d(net, inter_idx, inter_cnt,
                                  method=config.pool_method, scope='pool'+str(l+1))

        global_maxpool = tf.reduce_max(net, axis=1, keepdims=True)
        global_feat.append(global_maxpool)

    # =============================global feature extraction in the final layer=============================
    global_radius = 100.0 # global_radius(>=2.0) should connect all points to each point in the cloud
    nn_idx, nn_cnt, nn_dst = s3g_util.build_global_graph(xyz, query, global_radius)
    filt_idx = s3g_util.spherical_kernel(xyz, query, nn_idx, nn_cnt, nn_dst,
                                         global_radius, kernel=[8,2,1])
    net = s3g_util.separable_conv3d(net, config.global_channels, 17, config.global_multiplier,
                                    'global_conv', nn_idx, nn_cnt, filt_idx, reuse=reuse,
                                    weight_decay=config.weight_decay, with_bn=config.with_bn,
                                    with_bias=config.with_bias, is_training=is_training)
    global_feat.append(net)
    net = tf.concat(global_feat,axis=2)
    # =====================================================================================================

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = s3g_util.fully_connected(net, 512, scope='fc1', weight_decay=config.weight_decay,
                                   with_bn=config.with_bn, with_bias=config.with_bias, is_training=is_training)
    net = tf.layers.dropout(net, 0.5, training=is_training, name='fc1_dp')
    net = s3g_util.fully_connected(net, 256, scope='fc2', weight_decay=config.weight_decay,
                                   with_bn=config.with_bn, with_bias=config.with_bias, is_training=is_training)
    net = tf.layers.dropout(net, 0.5, training=is_training, name='fc2_dp')
    net = s3g_util.fully_connected(net, config.num_cls, scope='logits', with_bn=False, with_bias=config.with_bias,
                                   activation_fn=None, is_training=is_training)

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

