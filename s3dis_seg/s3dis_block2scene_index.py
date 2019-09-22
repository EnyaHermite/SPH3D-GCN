import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import scipy.io as sio

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
sys.path.append(baseDir)
sys.path.append(os.path.join(rootDir, 'models'))
sys.path.append(os.path.join(rootDir, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--test_area', type=int, default=5, help='which Area is the test fold')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu

LOG_DIR = os.path.join(rootDir,'log_s3dis_Area_%d'%FLAGS.test_area)

if not os.path.exists(os.path.join(LOG_DIR,'block_index')):
    os.mkdir(os.path.join(LOG_DIR,'block_index'))

dataDir = os.path.join(rootDir, 'data/s3dis_3cm_overlap')


def parse_block_scene(datapath, areaidx):
    f = open(os.path.join(datapath,'log_block.txt'), 'r')
    blocklist = f.read().splitlines()
    block2scene = []
    testlist = []
    for line in blocklist:
        str = line.split(', ')
        if str[0]==('Area_%d'%areaidx):
            block2scene.append(tuple(str))
            tfrecord_name = os.path.join(datapath,'Area_%d_%s.tfrecord'%(areaidx,str[1]))
            if not tfrecord_name in testlist:
                testlist.append(tfrecord_name)

    return block2scene, testlist


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'rel_xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'rgb_raw': tf.FixedLenFeature([], dtype=tf.string),
            'seg_label': tf.FixedLenFeature([], dtype=tf.string),
            'inner_label': tf.FixedLenFeature([], dtype=tf.string),
            'index_label':tf.FixedLenFeature([], dtype=tf.string),
            'scene_label': tf.FixedLenFeature([], dtype=tf.int64)})

    index_label = tf.decode_raw(features['index_label'], tf.int64)
    index_label = tf.reshape(index_label, [-1, 1])
    index_label = tf.cast(index_label,tf.float32)

    return index_label


def input_fn(filelist, batch_size=16):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(None,1),
                                   padding_values=-1.0, drop_remainder=False)

    return dataset


if __name__ == "__main__":
    BLOCK2SCENE, TESTLIST = parse_block_scene(dataDir, FLAGS.test_area)
    print('block2scenes num:%d, testlist size:%d'%(len(BLOCK2SCENE), len(TESTLIST)))

    # ===============================Prepare the Dataset===============================
    testset = input_fn(TESTLIST, BATCH_SIZE)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        sess.run(test_iterator.initializer)

        batch_idx = 0
        while True:
            try:
                padded_all = sess.run(next_test_element)
                bsize = padded_all.shape[0]

                for b in range(bsize):
                    # to write out the block data, and its predictions
                    num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                    temp_out_data = padded_all[b, 0:num, -1]
                    print(temp_out_data.shape)
                    scene_name = BLOCK2SCENE[batch_idx*BATCH_SIZE+b][0]
                    block_name = '%s_%d.mat'%(scene_name, batch_idx*BATCH_SIZE+b)
                    print(os.path.join(LOG_DIR, 'block_index', block_name))
                    sio.savemat(os.path.join(LOG_DIR, 'block_index', block_name), {'index':temp_out_data})
                batch_idx += 1
            except tf.errors.OutOfRangeError:
                break
    # =====================================The End=====================================


