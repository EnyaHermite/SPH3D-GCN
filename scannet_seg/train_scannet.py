import argparse
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

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
parser.add_argument('--log_dir', default='log_scannet', help='Log dir [default: log_scannet]')
parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=500000, help='Decay step for lr decay [default: 1000000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(rootDir, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(rootDir, FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_scannet.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp %s.py %s' % (FLAGS.config, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

net_config = importlib.import_module(FLAGS.config)
NUM_POINT = net_config.num_input
NUM_CLASSES = net_config.num_cls
INPUT_DIM = 6

dataDir = os.path.join(rootDir, 'data/scannet_3cm')
trainlist = [line.rstrip() for line in open(os.path.join(dataDir, 'train_val_files.txt'))]
vallist =  [line.rstrip() for line in open(os.path.join(dataDir, 'val_files.txt'))]


classes = {'other20':0,'wall':1,'floor':2,'cabinet':3,'bed':4,'chair':5,\
         'sofa':6,'table':7,'door':8,'window':9,'bookshelf':10,\
         'picture':11,'counter':12,'desk':13,'curtain':14,'refridgerator':15,\
         'shower curtain':16,'toilet':17,'sink':18,'bathtub':19,'otherfurniture':20}


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate


def placeholder_inputs(batch_size, num_point):
    input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    inner_label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return input_pl, label_pl, inner_label_pl


def augment_fn(batch_input, batch_label, batch_inner):
    bsize, num_point, _ = batch_input.shape

    # shuffle the orders of samples in a batch
    idx = np.arange(bsize)
    np.random.shuffle(idx)
    batch_input = batch_input[idx,:,:]
    batch_label = batch_label[idx,:]
    batch_inner = batch_inner[idx,:]

    # shuffle the point orders of each sample
    idx = np.arange(num_point)
    np.random.shuffle(idx)
    batch_input = batch_input[:,idx,:]
    batch_label = batch_label[:,idx]
    batch_inner = batch_inner[:,idx]

    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augSize = np.int32(1/3.0 * bsize)
    augment_xyz = batch_input[0:augSize,:,0:3]
    augment_xyz = data_util.rotate_point_cloud(augment_xyz)
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)
    augment_xyz = data_util.jitter_point_cloud(augment_xyz)
    batch_input[0:augSize,:,0:3] = augment_xyz

    augment_xyz = batch_input[augSize:2*augSize,:,0:3]
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)
    augment_xyz = data_util.jitter_point_cloud(augment_xyz)
    batch_input[augSize:2*augSize,:,0:3] = augment_xyz

    return batch_input, batch_label, batch_inner


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


def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(None,INPUT_DIM+2),
                                   padding_values=-1.0, drop_remainder=False)

    return dataset


def train():
    # ===============================Prepare the Dataset===============================
    trainset = input_fn(trainlist, BATCH_SIZE, 10000)
    train_iterator = trainset.make_initializable_iterator()
    next_train_element = train_iterator.get_next()

    testset = input_fn(vallist, BATCH_SIZE, 10000)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    with tf.device('/gpu:0'):
        # =================================Define the Graph================================
        input_pl, label_pl, inner_label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get model and loss
        pred, end_points = MODEL.get_model(input_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points, inner_label_pl)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name, l)

        correct = tf.equal(tf.argmax(pred, 2, output_type=tf.int32), tf.cast(label_pl,tf.int32))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
        tf.summary.scalar('accuracy', accuracy)

        print("--- Get training operator")
        # Get training operator
        learning_rate = get_learning_rate(global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=True)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

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

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with tf.Session(config=config) as sess:
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        sess.run(init)  # Init variables

        # Load the model
        latest_ckpt = tf.train.latest_checkpoint(LOG_DIR)
        print(FLAGS.load_ckpt)
        if FLAGS.load_ckpt is not None:
            saver.restore(sess, FLAGS.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), FLAGS.load_ckpt))
        else:
            if latest_ckpt:
                print('{}-Found checkpoint {}'.format(datetime.now(), latest_ckpt))
                saver.restore(sess, latest_ckpt)
                print('{}-Checkpoint loaded from {} (Iter {})'.format(
                    datetime.now(), latest_ckpt, sess.run(global_step)))

        ops = {'input_pl': input_pl,
               'label_pl': label_pl,
               'inner_label_pl': inner_label_pl,
               'training_pl': training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'global_step': global_step,
               'end_points': end_points}

        if latest_ckpt:
            checkpoint_epoch = int(latest_ckpt.split('-')[-1])+1
        elif FLAGS.load_ckpt is not None:
            checkpoint_epoch = 0

        for epoch in range(checkpoint_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            # sess.run(global_step.initializer)

            print('learning rate:',sess.run(learning_rate))
            print('global step:', sess.run(global_step))

            sess.run(train_iterator.initializer)
            train_one_epoch(sess, ops, next_train_element, train_writer)

            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----' %(epoch))

            sess.run(test_iterator.initializer)
            eval_one_epoch(sess, ops, next_test_element, test_writer)

            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
            log_string("Model saved in file: %s" % save_path)
    # =====================================The End=====================================


def train_one_epoch(sess, ops, next_train_element, train_writer):
    """ ops: dict mapping from string to tf ops """
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_input = np.zeros((BATCH_SIZE, NUM_POINT, INPUT_DIM))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_inner = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    train_time = 0.0
    while True:
        try:
            padded_all = sess.run(next_train_element)
            bsize = padded_all.shape[0]
            #print(padded_all.shape)

            # remove the padded data, select NUM_POINT point using np.random.choice
            batch_input = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_inner = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            for b in range(bsize):
                loc = np.where(padded_all[b,:,-1]<0)
                if len(loc[0])==0:
                    num = padded_all.shape[1]
                else:
                    num = loc[0][0]

                if num==0:
                    print(loc, padded_all[b, 0:10, :])
                    print('problem of train')
                    exit()

                if num<NUM_POINT:
                    sample_index = np.random.choice(num, NUM_POINT, replace=True)
                else:
                    sample_index = np.random.choice(num, NUM_POINT, replace=False)
                batch_input[b,...] = padded_all[b,sample_index,0:-2]
                batch_label[b,:] = padded_all[b,sample_index,-2]
                batch_inner[b,:] = padded_all[b,sample_index,-1]
                # print(np.sum(batch_inner[b,:]))

            # training augmentation on the fly
            batch_input, batch_label, batch_inner = augment_fn(batch_input, batch_label,
                                                               batch_inner)

            cur_batch_input[0:bsize,...] = batch_input
            cur_batch_label[0:bsize,:] = batch_label
            cur_batch_inner[0:bsize,:] = batch_inner

            feed_dict = {ops['input_pl']: cur_batch_input,
                         ops['label_pl']: cur_batch_label,
                         ops['inner_label_pl']: cur_batch_inner,
                         ops['training_pl']: True}

            now = time.time()
            summary, global_step, _, loss_val, pred_val = sess.run([ops['merged'], ops['global_step'],
                                     ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_time += (time.time() - now)

            train_writer.add_summary(summary, global_step)
            pred_val = np.argmax(pred_val, 2)
            for b in range(bsize):
                inIdx = (batch_inner[b]==1)
                correct = np.sum(pred_val[b,inIdx] == batch_label[b,inIdx])
                total_correct += correct
                total_seen += sum(inIdx)
            loss_sum += loss_val
            if (batch_idx+1)%50 == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx+1))
                log_string('mean loss: %f' % (loss_sum / 10))
                log_string('accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break
    log_string("training one batch require %.2f milliseconds" %(1000*train_time/batch_idx))

    return


def eval_one_epoch(sess, ops, next_test_element, test_writer):
    """ ops: dict mapping from string to tf ops """
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
            #print(padded_all.shape)

            # remove the padded data, select NUM_POINT point using np.random.choice
            batch_input = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_inner = np.zeros((bsize, NUM_POINT), dtype=np.int32)
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

                if num<NUM_POINT:
                    sample_index = np.random.choice(num, NUM_POINT, replace=True)
                else:
                    sample_index = np.random.choice(num, NUM_POINT, replace=False)
                batch_input[b, ...] = padded_all[b, sample_index, 0:-2]
                batch_label[b, :] = padded_all[b, sample_index, -2]
                batch_inner[b, :] = padded_all[b, sample_index, -1]

            cur_batch_input[0:bsize, ...] = batch_input
            cur_batch_label[0:bsize, :] = batch_label
            cur_batch_inner[0:bsize, :] = batch_inner

            feed_dict = {ops['input_pl']: cur_batch_input,
                         ops['label_pl']: cur_batch_label,
                         ops['inner_label_pl']: cur_batch_inner,
                         ops['training_pl']: is_training}

            now = time.time()
            summary, global_step, loss_val, pred_val = sess.run([ops['merged'], ops['global_step'],
                                                       ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_time += (time.time() - now)

            test_writer.add_summary(summary, global_step)
            pred_label = np.argmax(pred_val, 2)

            for b in range(bsize):
                inIdx = (batch_inner[b]==1)
                correct = np.sum(pred_label[b, inIdx]==batch_label[b, inIdx])
                total_correct += correct
                total_seen += sum(batch_inner[b])

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label[b, inIdx]==l)
                    total_correct_class[l] += (np.sum((pred_label[b, inIdx]==l) \
                                                    &(batch_label[b, inIdx]==l)))
                    total_union_class[l] += (np.sum((pred_label[b, inIdx]==l) \
                                                   |(batch_label[b, inIdx]==l)))

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


