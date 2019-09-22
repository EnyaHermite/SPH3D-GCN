import argparse
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import scipy.io as sio

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
sys.path.append(baseDir)
sys.path.append(os.path.join(rootDir, 'models'))
sys.path.append(os.path.join(rootDir, 'utils'))
import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='SPH3D_ruemonge2014', help='Model name [default: SPH3D_ruemonge2014]')
parser.add_argument('--config', default='ruemonge2014_config', help='Model name [default: ruemonge2014_config]')
parser.add_argument('--log_dir', default='log_ruemonge2014', help='Log dir [default: log_ruemonge2014]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu

MODEL_FILE = os.path.join(rootDir, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(rootDir,FLAGS.log_dir)

LOG_FOUT = open(os.path.join(LOG_DIR,'log_evaluate.txt'), 'a+')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

# import network module
spec = importlib.util.spec_from_file_location('',os.path.join(LOG_DIR,FLAGS.model+'.py'))
MODEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODEL)

# import network config
spec = importlib.util.spec_from_file_location('',os.path.join(LOG_DIR,FLAGS.config+'.py'))
net_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_config)

NUM_POINT = net_config.num_input
NUM_CLASSES = net_config.num_cls
INPUT_DIM = 9

dataDir = os.path.join(rootDir, 'data/ruemonge2014')
trainlist = [line.rstrip() for line in open(os.path.join(dataDir, 'train_files.txt'))]
testlist =  [line.rstrip() for line in open(os.path.join(dataDir, 'test_files.txt'))]


classes = {'roof':0,
           'shop':1,
           'balcony':2,
           'sky':3,
           'window':4,
           'door':5,
           'wall':6}


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def placeholder_inputs(batch_size, num_point):
    input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return input_pl, label_pl


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'normal_raw':tf.FixedLenFeature([], dtype=tf.string),
            'rgb_raw': tf.FixedLenFeature([], dtype=tf.string),
            'seg_label':tf.FixedLenFeature([], dtype=tf.string)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    rgb = tf.decode_raw(features['rgb_raw'], tf.float32)
    normal = tf.decode_raw(features['normal_raw'], tf.float32)
    seg_label = tf.decode_raw(features['seg_label'], tf.int32)

    xyz = tf.reshape(xyz, [-1, 3])
    rgb = tf.reshape(rgb, [-1, 3])
    normal = tf.reshape(normal, [-1, 3])
    seg_label = tf.reshape(seg_label, [-1, 1])
    all_in_one = tf.concat((xyz, normal, rgb, tf.cast(seg_label,tf.float32)), axis=-1)
    # all_in_one = tf.concat((xyz, rgb, tf.cast(seg_label, tf.float32)), axis=-1)

    return all_in_one


def augment_fn(batch_input, batch_label):
    if INPUT_DIM==6:
        augment_xyz = batch_input[:, :, 0:3]
        augment_xyz = data_util.rotate_point_cloud(augment_xyz)
        augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
        batch_input[:, :, 0:3] = augment_xyz
    elif INPUT_DIM==9:
        augment_xyz = batch_input[:, :, 0:6]
        augment_xyz = data_util.rotate_point_cloud_with_normal(augment_xyz)
        augment_xyz = data_util.rotate_perturbation_point_cloud_with_normal(augment_xyz)
        batch_input[:, :, 0:6] = augment_xyz

    return batch_input, batch_label


def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    # dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(None,INPUT_DIM+1),
                                   padding_values=-1.0, drop_remainder=False)

    return dataset


def train():
    # ===============================Prepare the Dataset===============================
    testset = input_fn(testlist, BATCH_SIZE, 10000)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    with tf.device('/gpu:0'):
        # =================================Define the Graph================================
        input_pl, label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        pred, end_points = MODEL.get_model(input_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # =====================================The End=====================================

    n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("*****************The Graph has %d nodes*****************"%(n))

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    ops = {'input_pl':input_pl,
           'label_pl':label_pl,
           'training_pl':training_pl,
           'pred':pred,
           'loss':total_loss}

    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(LOG_DIR, 'model.ckpt-%d'%FLAGS.which_epoch))

        sess.run(test_iterator.initializer)
        OA, mAcc, mIoU = eval_one_epoch(sess, ops, next_test_element)
    #=====================================The End=====================================


def eval_one_epoch(sess, ops, next_test_element):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Make sure batch data is of same size
    cur_batch_input = np.zeros((BATCH_SIZE, NUM_POINT, INPUT_DIM))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_union_class = [0 for _ in range(NUM_CLASSES)]

    class_names = list(classes.keys())
    class_iou = {cat:0.0 for cat in class_names}
    class_acc = {cat:0.0 for cat in class_names}

    test_time = 0.0
    while True:
        try:
            padded_all = sess.run(next_test_element)
            bsize = padded_all.shape[0]
            #print(padded_all.shape)

            batch_gt_label = []
            batch_pred_sum = []
            batch_sample_count = []
            batch_sample_index = []

            batch_point_size = np.zeros((bsize,), np.int32)
            batch_point_covered = np.zeros((bsize,), np.int32)
            for b in range(bsize):
                loc = np.where(padded_all[b, :, -1]<0)
                if len(loc[0])==0:
                    num = padded_all.shape[1]
                else:
                    num = loc[0][0]

                if num==0:
                    print(loc,padded_all[b, 0:10, :])
                    print('problem of eval')
                    exit()

                batch_point_size[b] = num
                batch_gt_label.append(padded_all[b, 0:num, -1])
                batch_pred_sum.append(np.zeros((num, NUM_CLASSES), dtype=np.float32))

                batch_sample_count.append(np.zeros((num,), dtype=np.int32))
                batch_sample_index.append(np.zeros((num,), dtype=np.int32))

            print(batch_point_size)

            # remove the padded data, select NUM_POINT point using np.random.choice
            batch_input = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            while any(batch_point_covered<batch_point_size):
                for b in range(bsize):
                    num = batch_point_size[b]

                    if num<NUM_POINT:
                        sample_index = np.random.choice(num, NUM_POINT, replace=True)
                    else:
                        sample_index = np.random.choice(num, NUM_POINT, replace=False)
                    batch_input[b, ...] = padded_all[b, sample_index, 0:-1]
                    batch_label[b, :] = padded_all[b, sample_index, -1]

                    batch_sample_count[b][sample_index] += 1
                    batch_sample_index[b] = sample_index
                    batch_point_covered[b] = np.sum(batch_sample_count[b]>10)

                batch_input, batch_label = augment_fn(batch_input, batch_label)

                cur_batch_input[0:bsize, ...] = batch_input
                cur_batch_label[0:bsize, :] = batch_label

                feed_dict = {ops['input_pl']: cur_batch_input,
                             ops['label_pl']: cur_batch_label,
                             ops['training_pl']: is_training}

                now = time.time()
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                test_time += (time.time() - now)

                for b in range(bsize):
                    batch_pred_sum[b][batch_sample_index[b]] += pred_val[b, ...]

            for b in range(bsize):
                # print(batch_pred_sum[b].shape)
                pred_label = np.argmax(batch_pred_sum[b], 1)
                correct = np.sum(pred_label==batch_gt_label[b])
                total_correct += correct
                total_seen += batch_point_size[b]

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_gt_label[b]==l)
                    total_correct_class[l] += (np.sum((pred_label==l) \
                                                    &(batch_gt_label[b]==l)))
                    total_union_class[l] += (np.sum((pred_label==l) \
                                                   |(batch_gt_label[b]==l)))

            loss_sum += loss_val
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break

    for cat in class_names:
        l = classes[cat]
        class_iou[cat] = total_correct_class[l]/(float(total_union_class[l])+np.finfo(float).eps)
        class_acc[cat] = total_correct_class[l]/(float(total_seen_class[l]) +np.finfo(float).eps)

    OA = total_correct/float(total_seen)
    mAcc = np.mean(list(class_acc.values()))
    mIoU = np.mean(list(class_iou.values()))
    log_string('eval mean loss: %f'%(loss_sum/batch_idx))
    log_string('eval overall accuracy: %f'%(total_correct/float(total_seen)))
    log_string('eval avg class acc: %f'%(np.mean(list(class_acc.values()))))
    for i in range(len(classes)):
        cat = list(classes.keys())[list(classes.values()).index(i)]
        log_string('eval mIoU of %s:\t %f'%(cat, class_iou[cat]))
    log_string('eval mIoU of all classes: %f'%(np.mean(list(class_iou.values()))))
    log_string("training one batch require %.2f milliseconds"%(1000*test_time/batch_idx))

    return OA, mAcc, mIoU


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()


