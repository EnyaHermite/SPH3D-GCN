import argparse
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os, sys
import scipy.io as sio

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
sys.path.append(baseDir)
sys.path.append(os.path.join(rootDir, 'models'))
sys.path.append(os.path.join(rootDir, 'utils'))
import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='SPH3D_scannet', help='Model name [default: SPH3D_scannet]')
parser.add_argument('--config', default='scannet_config', help='Model name [default: scannet_config]')
parser.add_argument('--test_folder', default='scannet_3cm', help='which folder')
parser.add_argument('--log_dir', default='log_scannet', help='Log dir [default: log_scannet]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(rootDir, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(rootDir, FLAGS.log_dir)

resultsFolder = 'results'
if not os.path.exists(os.path.join(LOG_DIR,resultsFolder)):
    os.mkdir(os.path.join(LOG_DIR,resultsFolder))

LOG_FOUT = open(os.path.join(LOG_DIR,'log_evaluate.txt'), 'a+')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

net_config = importlib.import_module(FLAGS.config)
NUM_POINT = net_config.num_input
NUM_CLASSES = net_config.num_cls
INPUT_DIM = 6

dataDir = os.path.join(rootDir, 'data/scannet_3cm')
testlist = [line.rstrip() for line in open(os.path.join(dataDir, 'val_files.txt'))]

sub20_classids = np.arange(1,20)
classes = {'other20':0,'wall':1,'floor':2,'cabinet':3,'bed':4,'chair':5,\
           'sofa':6,'table':7,'door':8,'window':9,'bookshelf':10,\
           'picture':11,'counter':12,'desk':13,'curtain':14,'refridgerator':15,\
           'shower curtain':16,'toilet':17,'sink':18,'bathtub':19,'otherfurniture':20}


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


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


SCENE_NAMES = [os.path.basename(item).replace('.tfrecord','') for item in testlist]
BLOCK2SCENE, TESTLIST = parse_block_scene(dataDir, SCENE_NAMES)
print('block2scenes num:%d, testlist size:%d'%(len(BLOCK2SCENE),len(TESTLIST)))


def placeholder_inputs(batch_size, num_point):
    input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    inner_label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return input_pl, label_pl, inner_label_pl


def augment_fn1(batch_input):
    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augment_xyz = batch_input[:,:,0:3]
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)
    augment_xyz = data_util.jitter_point_cloud(augment_xyz)
    batch_input[:,:,0:3] = augment_xyz

    return batch_input


def augment_fn2(batch_input):
    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augment_xyz = batch_input[:,:,0:3]
    augment_xyz = data_util.rotate_point_cloud(augment_xyz)
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
            'rgb_raw': tf.FixedLenFeature([], dtype=tf.string),
            'seg_label':tf.FixedLenFeature([], dtype=tf.string),
            'inner_label':tf.FixedLenFeature([], dtype=tf.string)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    rgb = tf.decode_raw(features['rgb_raw'], tf.float32)
    seg_label = tf.decode_raw(features['seg_label'], tf.int32)
    inner_label = tf.decode_raw(features['inner_label'], tf.int32)

    xyz = tf.reshape(xyz, [-1, 3])
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


def evaluate():
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
        pred, end_points = MODEL.get_model(input_pl, NUM_CLASSES, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points, inner_label_pl)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name, l)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=500)
        # =====================================The End=====================================

    n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("*****************The Graph has %d nodes*****************"%(n))

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

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

    sub20_class_names = [name for name in class_names if name is not 'other20']
    sub20_class_iou = {cat:0.0 for cat in sub20_class_names}
    sub20_class_acc = {cat:0.0 for cat in sub20_class_names}

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

            batch_inner_size = np.zeros((bsize,), np.int32)
            batch_inner_covered = np.zeros((bsize,), np.int32)
            for b in range(bsize):
                num_in = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-2])
                num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                batch_gt_label.append(padded_all[b, 0:num, -2])
                batch_pred_sum.append(np.zeros((num, NUM_CLASSES), dtype=np.float32))

                batch_inner_size[b] = num_in
                batch_inner_label.append(padded_all[b, 0:num, -1])

                batch_sample_count.append(np.zeros((num,), dtype=np.int32))
                batch_sample_index.append(np.zeros((num,), dtype=np.int32))

            batch_input = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_inner = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            while any(batch_inner_covered<batch_inner_size):
                # ignore the padded data, and select NUM_POINT point using np.random.choice
                for b in range(bsize):
                    num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                    print(num_in, num, NUM_POINT)

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
                    batch_inner_covered[b] = np.sum(batch_sample_count[b][inIdx]>1)

                for augType in ['none', 'aug1', 'aug2']:
                    if augType is 'aug1':
                        batch_input = augment_fn1(batch_input)
                    elif augType is 'aug2':
                        batch_input = augment_fn2(batch_input)

                    cur_batch_input[0:bsize, ...] = batch_input
                    cur_batch_label[0:bsize, :] = batch_label
                    cur_batch_inner[0:bsize, :] = batch_inner

                    feed_dict = {ops['input_pl']:cur_batch_input,
                                 ops['label_pl']:cur_batch_label,
                                 ops['inner_label_pl']:cur_batch_inner,
                                 ops['training_pl']:is_training}

                    now = time.time()
                    loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                    test_time += (time.time()-now)

                    for b in range(bsize):
                        batch_pred_sum[b][batch_sample_index[b]] += pred_val[b, ...]

            for b in range(bsize):
                # to write out the block data, and its predictions
                num = np.int32(BLOCK2SCENE[batch_idx*BATCH_SIZE+b][-1])
                temp_out_data = np.concatenate((padded_all[b, 0:num, :], batch_pred_sum[b]), axis=1)
                scene_name = BLOCK2SCENE[batch_idx*BATCH_SIZE+b][0]
                block_name = '%s_%d.mat'%(scene_name, batch_idx*BATCH_SIZE+b)
                print(os.path.join(LOG_DIR, resultsFolder, block_name))

                sio.savemat(os.path.join(LOG_DIR, resultsFolder, block_name), {'data':temp_out_data})

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

    sub20_total_seen_class = [total_seen_class[l] for l in sub20_classids]
    sub20_total_correct_class = [total_correct_class[l] for l in sub20_classids]
    total_correct = np.sum(sub20_total_correct_class)
    total_seen = np.sum(sub20_total_seen_class)
    for i in sub20_classids:
        cat = list(classes.keys())[list(classes.values()).index(i)]
        sub20_class_iou[cat] = class_iou[cat]
        sub20_class_acc[cat] = class_iou[cat]

    log_string('eval mean loss: %f'%(loss_sum/batch_idx))
    log_string('eval overall accuracy: %f'%(total_correct/float(total_seen)))
    log_string('eval avg class acc: %f'%(np.mean(list(sub20_class_acc.values()))))
    for i in sub20_classids:
        cat = list(classes.keys())[list(classes.values()).index(i)]
        log_string('eval mIoU of %s:\t %f'%(cat, class_iou[cat]))
    log_string('eval mIoU of all classes: %f'%(np.mean(list(sub20_class_iou.values()))))
    log_string("testing one batch require %.2f milliseconds"%(1000*test_time/batch_idx))

    return


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()


