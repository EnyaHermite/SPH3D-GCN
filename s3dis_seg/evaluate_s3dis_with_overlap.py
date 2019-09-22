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
parser.add_argument('--model', default='SPH3D_s3dis', help='Model name [default: SPH3D_s3dis]')
parser.add_argument('--config', default='s3dis_config', help='Model name [default: s3dis_config]')
parser.add_argument('--test_area', type=int, default=5, help='which Area is the test fold')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu

MODEL_FILE = os.path.join(rootDir, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(rootDir,'log_s3dis_Area_%d'%FLAGS.test_area)

resultsFolder = 'results'
if not os.path.exists(os.path.join(LOG_DIR,resultsFolder)):
    os.mkdir(os.path.join(LOG_DIR,resultsFolder))

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
INPUT_DIM = 6

dataDir = os.path.join(rootDir, 'data/s3dis_3cm_overlap')

classes = {'ceiling':0,
           'floor':1,
           'wall':2,
           'beam':3,
           'column':4,
           'window':5,
           'door':6,
           'table':7,
           'chair':8,
           'sofa':9,
           'bookcase':10,
           'board':11,
           'clutter':12}

scenes = {'office':0,
          'conferenceroom':1,
          'hallway':2,
          'auditorium':3,
          'openspace':4,
          'lobby':5,
          'lounge':6,
          'pantry':7,
          'copyroom':8,
          'storage':9,
          'wc':10}


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


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


BLOCK2SCENE, TESTLIST = parse_block_scene(dataDir, FLAGS.test_area)
print('block2scenes num:%d, testlist size:%d'%(len(BLOCK2SCENE),len(TESTLIST)))


def placeholder_inputs(batch_size, num_point):
    data_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    inner_label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return data_pl, label_pl, inner_label_pl


def augment_fn(batch_input):
    augment_xyz = batch_input[:,:,0:3]
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)
    augment_xyz = data_util.jitter_point_cloud(augment_xyz)
    batch_input[:,:,0:3] = augment_xyz

    return batch_input


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            # 'rel_xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'rgb_raw': tf.FixedLenFeature([], dtype=tf.string),
            'seg_label': tf.FixedLenFeature([], dtype=tf.string),
            'inner_label': tf.FixedLenFeature([], dtype=tf.string)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    # rel_xyz = tf.decode_raw(features['rel_xyz_raw'], tf.float32)
    rgb = tf.decode_raw(features['rgb_raw'], tf.float32)
    seg_label = tf.decode_raw(features['seg_label'], tf.int32)
    inner_label = tf.decode_raw(features['inner_label'], tf.int32)

    xyz = tf.reshape(xyz, [-1, 3])
    # rel_xyz = tf.reshape(rel_xyz, [-1, 3])
    rgb = tf.reshape(rgb, [-1, 3])
    seg_label = tf.reshape(seg_label, [-1, 1])
    inner_label = tf.reshape(inner_label, [-1, 1])
    all_in_one = tf.concat((xyz, rgb, tf.cast(seg_label,tf.float32),
                            tf.cast(inner_label,tf.float32)), axis=-1)

    return all_in_one


def input_fn(filelist, batch_size=16):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(None,INPUT_DIM+2),
                                   padding_values=-1.0, drop_remainder=False)

    return dataset


def train():
    # ===============================Prepare the Dataset===============================
    testset = input_fn(TESTLIST, BATCH_SIZE)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    with tf.device('/gpu:0'):
        # =================================Define the Graph================================
        input_pl, label_pl, inner_label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        pred, end_points = MODEL.get_model(input_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points, inner_label_pl)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # 'saver' to restore all the variables.
        saver = tf.train.Saver()
        # =====================================The End=====================================

    n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("*****************The Graph has %d nodes*****************"%(n))

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        ops = {'input_pl': input_pl,
               'label_pl': label_pl,
               'inner_label_pl': inner_label_pl,
               'training_pl': training_pl,
               'pred': pred,
               'loss': total_loss}

        saver.restore(sess, os.path.join(LOG_DIR, FLAGS.model_name))

        sess.run(test_iterator.initializer)
        eval_one_epoch(sess, ops, next_test_element)
    # =====================================The End=====================================


def eval_one_epoch(sess, ops, next_test_element):
    """ ops: dict mapping from string to tf ops """
    global loss_val
    is_training = False

    # Make sure batch data is of same size
    cur_batch_input = np.zeros((BATCH_SIZE, NUM_POINT, INPUT_DIM))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_inner = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)

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

            batch_gt_label = []
            batch_pred_sum = []
            batch_inner_label = []
            batch_sample_count = []
            batch_sample_index = []

            batch_inner_size = np.zeros((bsize,),np.int32)
            batch_inner_covered = np.zeros((bsize,),np.int32)
            for b in range(bsize):
                num_in = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-2])
                num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                batch_gt_label.append(padded_all[b, 0:num, -2])
                batch_pred_sum.append(np.zeros((num,NUM_CLASSES),dtype=np.float32))

                batch_inner_size[b] = num_in
                batch_inner_label.append(padded_all[b, 0:num, -1])

                batch_sample_count.append(np.zeros((num,),dtype=np.int32))
                batch_sample_index.append(np.zeros((num,),dtype=np.int32))

            batch_input = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_inner = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            while any(batch_inner_covered<batch_inner_size):
                # ignore the padded data, and select NUM_POINT point using np.random.choice
                for b in range(bsize):
                    num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])

                    if num<NUM_POINT:
                        sample_index = np.random.choice(num, NUM_POINT, replace=True)
                    else:
                        sample_index = np.random.choice(num, NUM_POINT, replace=False)
                    batch_input[b, ...] = padded_all[b, sample_index, 0:-2]
                    batch_label[b, :] = padded_all[b, sample_index, -2]
                    batch_inner[b, :] = padded_all[b, sample_index, -1]

                    batch_sample_count[b][sample_index] += 1
                    batch_sample_index[b] = sample_index
                    inIdx = (batch_inner_label[b]==1)
                    batch_inner_covered[b] = np.sum(batch_sample_count[b][inIdx]>0)

                cur_batch_input[0:bsize, ...] = batch_input
                cur_batch_label[0:bsize, :] = batch_label
                cur_batch_inner[0:bsize, :] = batch_inner

                feed_dict = {ops['input_pl']: cur_batch_input,
                             ops['label_pl']: cur_batch_label,
                             ops['inner_label_pl']: cur_batch_inner,
                             ops['training_pl']: is_training}

                now = time.time()
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                test_time += (time.time() - now)

                for b in range(bsize):
                    batch_pred_sum[b][batch_sample_index[b]] += pred_val[b,...]

            for b in range(bsize):
                # to write out the block data, and its predictions
                num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                temp_out_data = np.concatenate((padded_all[b,0:num,:],batch_pred_sum[b]), axis=1)
                scene_name = BLOCK2SCENE[batch_idx*BATCH_SIZE+b][1]
                block_name = '%s_%d.mat'%(scene_name,batch_idx*BATCH_SIZE+b)
                print(os.path.join(LOG_DIR,resultsFolder,block_name))

                sio.savemat(os.path.join(LOG_DIR,resultsFolder,block_name), {'data':temp_out_data})

                pred_label = np.argmax(batch_pred_sum[b], 1)
                inIdx = (batch_inner_label[b]==1)
                correct = np.sum(pred_label[inIdx]==batch_gt_label[b][inIdx])
                total_correct += correct
                total_seen += batch_inner_size[b]

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_gt_label[b][inIdx]==l)
                    total_correct_class[l] += (np.sum((pred_label[inIdx]==l) \
                                                    &(batch_gt_label[b][inIdx]==l)))
                    total_union_class[l] += (np.sum((pred_label[inIdx]==l) \
                                           |(batch_gt_label[b][inIdx]==l)))
            loss_sum += loss_val
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break

    for cat in class_names:
        l = classes[cat]
        class_iou[cat] = total_correct_class[l]/(float(total_union_class[l])+np.finfo(float).eps)
        class_acc[cat] = total_correct_class[l]/(float(total_seen_class[l]) +np.finfo(float).eps)

    log_string('eval mean loss: %f'%(loss_sum/batch_idx))
    log_string('eval overall accuracy: %f'%(total_correct/float(total_seen)))
    log_string('eval avg class acc: %f'%(np.mean(list(class_acc.values()))))
    for i in range(len(classes)):
        cat = list(classes.keys())[list(classes.values()).index(i)]
        log_string('eval mIoU of %s:\t %f'%(cat, class_iou[cat]))
    log_string('eval mIoU of all classes: %f'%(np.mean(list(class_iou.values()))))
    log_string("testing one batch require %.2f milliseconds"%(1000*test_time/batch_idx))

    return


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()


