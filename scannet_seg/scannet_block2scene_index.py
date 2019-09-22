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
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu

LOG_DIR = os.path.join(rootDir,'log_scannet')


def parse_block_scene(datapath, scene_names):
    f = open(os.path.join(datapath,'log_block.txt'), 'r')
    blocklist = f.read().splitlines()
    block2scene = []
    ordered_testlist = []
    for line in blocklist:
        str = line.split(', ')
        if str[0] in  scene_names:
            block2scene.append(tuple(str))
            tfrecord_name = os.path.join(datapath,'%s.tfrecord'%str[0])
            if not tfrecord_name in ordered_testlist:
                ordered_testlist.append(tfrecord_name)

    return block2scene, ordered_testlist


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={'index_label':tf.FixedLenFeature([], dtype=tf.string)})

    index_label = tf.decode_raw(features['index_label'], tf.int32)
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
    dataDir = os.path.join(rootDir, 'data/scannet_3cm')
    blockindexDir = os.path.join(LOG_DIR, 'block_index')
    if not os.path.exists(blockindexDir):
        os.mkdir(blockindexDir)

    testlist = [line.rstrip() for line in open(os.path.join(dataDir, 'val_files.txt'))]
    SCENE_NAMES = [os.path.basename(item).replace('.tfrecord', '') for item in testlist]
    BLOCK2SCENE, TESTLIST = parse_block_scene(dataDir, SCENE_NAMES)
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

                print(padded_all.shape)

                for b in range(bsize):
                    # to write out the block data, and its predictions
                    num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                    temp_out_data = padded_all[b, 0:num, -1]
                    scene_name = BLOCK2SCENE[batch_idx*BATCH_SIZE+b][0]
                    block_name = '%s_%d.mat'%(scene_name, batch_idx*BATCH_SIZE+b)
                    print(os.path.join(blockindexDir, block_name))
                    sio.savemat(os.path.join(blockindexDir, block_name), {'index':temp_out_data})
                batch_idx += 1
            except tf.errors.OutOfRangeError:
                break
    # =====================================The End=====================================


