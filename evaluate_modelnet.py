import argparse
import time
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import scipy.io as sio
import data_util

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
print(baseDir)
print(rootDir)
sys.path.append(baseDir)
sys.path.append(os.path.join(rootDir, 'models'))
sys.path.append(os.path.join(rootDir, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='SPH3D_modelnet', help='Model name [default: SPH3D_modelnet]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
LOG_DIR = os.path.join(rootDir,FLAGS.log_dir)
MODEL_NAME = FLAGS.model_name
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

spec = importlib.util.spec_from_file_location('',os.path.join(LOG_DIR,FLAGS.model+'.py'))
MODEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODEL)


LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


spec = importlib.util.spec_from_file_location('',os.path.join(LOG_DIR,'modelnet_config.py'))
net_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_config)

dataDir = os.path.join(rootDir, 'data/modelnet40')
testlist = [line.rstrip() for line in open(os.path.join(dataDir, 'test_files.txt'))]
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(rootDir, 'data/modelnet40_shape_names.txt'))]

NUM_POINT = net_config.num_input
NUM_CLASSES = net_config.num_cls
HOSTNAME = socket.gethostname()


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def placeholder_inputs(batch_size, num_point):
    xyz_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size))

    return xyz_pl, label_pl


def augment_fn(batch_xyz):
    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augment_xyz = data_util.rotate_point_cloud(batch_xyz)
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)

    return augment_xyz


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], dtype=tf.int64)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    xyz = tf.reshape(xyz,[-1,3])

    return xyz, label


def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


def evaluate(num_votes):
    # ===============================Prepare the Dataset===============================
    testset = input_fn(testlist, BATCH_SIZE, 10000)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    # =================================Define the Graph================================
    with tf.device('/gpu:' + str(GPU_INDEX)):
        xyz_pl, label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(xyz_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    # =====================================The End=====================================


    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        # Load the model
        saver.restore(sess, os.path.join(LOG_DIR, MODEL_NAME))
        log_string("Model restored.")

        ops = {'xyz_pl': xyz_pl,
               'label_pl': label_pl,
               'training_pl': training_pl,
               'pred': pred,
               'loss': total_loss}

        sess.run(test_iterator.initializer)
        eval_one_epoch(sess, ops, next_test_element, num_votes)
    # =====================================The End=====================================


def eval_one_epoch(sess, ops, next_test_element, num_votes=1, topk=1):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Make sure batch data is of same size
    cur_batch_xyz = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    pred_votes = np.zeros((2468, num_votes, NUM_CLASSES))
    pred_label = np.zeros((2468, ), np.int32)

    test_time = 0.0
    while True:
        try:
            batch_xyz, batch_label = sess.run(next_test_element)
            bsize = batch_xyz.shape[0]

            batch_xyz = batch_xyz[:, :, [0, 2, 1]]  # xzy to xyz

            print('Batch: %03d, batch size: %d' % (batch_idx, bsize))

            cur_batch_xyz[0:bsize, ...] = batch_xyz
            cur_batch_label[0:bsize] = batch_label

            batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES))  # score for classes
            for vote_idx in range(num_votes):
                augment_xyz = batch_xyz.copy()
                if vote_idx>0:
                    augment_xyz = augment_fn(augment_xyz)

                feed_dict = {ops['xyz_pl']: augment_xyz,
                             ops['label_pl']: cur_batch_label,
                             ops['training_pl']: is_training}

                now = time.time()
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                test_time += (time.time() - now)

                pred_votes[batch_idx*BATCH_SIZE:(batch_idx*BATCH_SIZE+bsize),vote_idx,:] = pred_val[0:bsize,:]
                pred_label[batch_idx*BATCH_SIZE:(batch_idx*BATCH_SIZE+bsize)] = cur_batch_label[0:bsize]
                batch_pred_sum += pred_val

            pred_val = np.argmax(batch_pred_sum, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val
            batch_idx += 1
            for i in range(0, bsize):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)
        except tf.errors.OutOfRangeError:
            break

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))

    log_string("testing one batch require %.2f milliseconds" % (1000 * test_time / batch_idx))
    sio.savemat('pred_votes',{'pred':pred_votes,'label':pred_label})

    return


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
