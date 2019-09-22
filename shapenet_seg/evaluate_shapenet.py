import argparse
import time
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
parser.add_argument('--model', default='SPH3D_shapenet', help='Model name [default: SPH3D_shapenet]')
parser.add_argument('--config', default='shapenet_config', help='Model name [default: shapenet_config]')
parser.add_argument('--log_dir', default='log_shapenet', help='Log dir [default: log_shapenet]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')
parser.add_argument('--shape_name', default=None, help='Which class to perform segment on')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu

MODEL_FILE = os.path.join(rootDir, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(rootDir, FLAGS.log_dir,FLAGS.shape_name)

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

resultDir = 'shapenet_seg/%s_results'%(FLAGS.shape_name)
if not os.path.exists(resultDir):
    os.mkdir(resultDir)

NUM_POINT = net_config.num_input
INPUT_DIM = 3

dataDir = os.path.join(rootDir, 'data/shapenet/%s'%FLAGS.shape_name)
seg_info = [int(line.rstrip().split('\t')[-1])
            for line in open(os.path.join(os.path.dirname(dataDir), 'class_info_all.txt'))]
seg_info.append(50)
shape_names = [line.rstrip().split('\t')[0]
               for line in open(os.path.join(os.path.dirname(dataDir), 'class_info_all.txt'))]

trainlist = [line.rstrip() for line in open(os.path.join(dataDir, 'train_files.txt'))]
testlist =  [line.rstrip() for line in open(os.path.join(dataDir, 'test_files.txt'))]

shape_name = FLAGS.shape_name
cls = shape_names.index(shape_name)
NUM_CLASSES = seg_info[cls+1] - seg_info[cls]
print('class %s: (class ID #%d, part number #%d)'%(shape_name,cls,NUM_CLASSES))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def placeholder_inputs(batch_size, num_point):
    xyz_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return xyz_pl, label_pl


def augment_fn(batch_xyz):
    augment_xyz = batch_xyz
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)
    augment_xyz = data_util.jitter_point_cloud(augment_xyz)
    batch_xyz = augment_xyz

    return batch_xyz


def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'part_label':tf.FixedLenFeature([], dtype=tf.string)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    seg_label = tf.decode_raw(features['part_label'], tf.int32)

    xyz = tf.reshape(xyz,[-1,3])
    seg_label = tf.reshape(seg_label, [-1, 1])
    all_in_one = tf.concat((xyz, tf.cast(seg_label, tf.float32)), axis=-1)

    return all_in_one


def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(None, INPUT_DIM+1),
                                   padding_values=-1.0, drop_remainder=False)

    return dataset


def evaluate():
    # ===============================Prepare the Dataset===============================
    testset = input_fn(testlist, BATCH_SIZE, 10000)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    with tf.device('/gpu:0'):
        # =================================Define the Graph================================
        xyz_pl, label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        pred, end_points = MODEL.get_model(xyz_pl, NUM_CLASSES, training_pl, config=net_config)
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

    with tf.Session(config=config) as sess:
        ops = {'xyz_pl': xyz_pl,
               'label_pl': label_pl,
               'training_pl': training_pl,
               'pred': pred,
               'loss': total_loss}

        saver.restore(sess, os.path.join(LOG_DIR, FLAGS.model_name))

        sess.run(test_iterator.initializer)
        eval_one_epoch(sess, ops, next_test_element)
    # =====================================The End=====================================


def eval_one_epoch(sess, ops, next_test_element):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Make sure batch data is of same size
    cur_batch_xyz = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_cls_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    shape_ious = []

    test_time = 0.0
    while True:
        try:
            padded_all = sess.run(next_test_element)
            bsize = padded_all.shape[0]

            batch_xyz = np.zeros((bsize, NUM_POINT, INPUT_DIM))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)

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
                    print(loc, padded_all[b, 0:10, :])
                    print('problem of eval')
                    exit()

                batch_point_size[b] = num
                batch_gt_label.append(padded_all[b, 0:num, -1])
                batch_pred_sum.append(np.zeros((num, NUM_CLASSES), dtype=np.float32))

                batch_sample_count.append(np.zeros((num,), dtype=np.int32))
                batch_sample_index.append(np.zeros((num,), dtype=np.int32))

            # print(batch_point_size,batch_cls_label)

            # remove the padded data, select NUM_POINT point using np.random.choice
            while any(batch_point_covered<batch_point_size):
                for b in range(bsize):
                    num = batch_point_size[b]

                    if num<NUM_POINT:
                        sample_index = np.random.choice(num, NUM_POINT, replace=True)
                    else:
                        sample_index = np.random.choice(num, NUM_POINT, replace=False)
                    batch_xyz[b, ...] = padded_all[b, sample_index, 0:-1]
                    batch_label[b, :] = padded_all[b, sample_index, -1]

                    batch_sample_count[b][sample_index] += 1
                    batch_sample_index[b] = sample_index
                    batch_point_covered[b] = np.sum(batch_sample_count[b]>10)

                cur_batch_label[0:bsize, :] = batch_label

                for augType in ['none','augment']:
                    if augType is 'augment':
                        batch_xyz = augment_fn(batch_xyz)

                    cur_batch_xyz[0:bsize, ...] = batch_xyz
                    feed_dict = {ops['xyz_pl']:cur_batch_xyz,
                                 ops['label_pl']:cur_batch_label,
                                 ops['training_pl']:is_training}

                    now = time.time()
                    loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                                  feed_dict=feed_dict)
                    test_time += (time.time()-now)

                    for b in range(bsize):
                        batch_pred_sum[b][batch_sample_index[b]] += pred_val[b, ...]

            pred_logits = batch_pred_sum[0:bsize]
            for b in range(bsize):
                logits = pred_logits[b]
                pred_label = np.argmax(logits, 1)
                correct = np.sum(pred_label==batch_gt_label[b])
                total_correct += correct
                total_seen += batch_point_size[b]
                # print('check if size match:', batch_point_size[b], logits.shape[0])

                pred_gt = np.concatenate([np.reshape(pred_label, (-1, 1)),
                                          np.reshape(batch_gt_label[b], (-1, 1))], axis=1)
                np.savetxt('shapenet_seg/%s_results/%d.txt'%(shape_name,batch_idx*BATCH_SIZE+b),
                           pred_gt, fmt='%d')

                part_ious = [0.0 for _ in range(NUM_CLASSES)]
                for l in range(NUM_CLASSES):
                    union = (pred_label==l)|(batch_gt_label[b]==l)
                    intersect = (pred_label==l)&(batch_gt_label[b]==l)

                    total_seen_class[l] += np.sum(batch_gt_label[b]==l)
                    total_correct_class[l] += np.sum(intersect)

                    if np.sum(union)==0:  # part is not present, no prediction as well
                        part_ious[l] = 1.0
                    else:
                        part_ious[l] = np.sum(intersect)/float(
                            np.sum(union))
                shape_ious.append(np.mean(part_ious))

            loss_sum += loss_val
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break

    print(total_correct_class, total_seen_class)
    log_string('eval mean loss: %f'%(loss_sum/batch_idx))
    log_string('eval accuracy: %f'%(total_correct/float(total_seen)))
    log_string('eval avg class acc: %f'%(
        np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))
    log_string('eval mIoU of %s:\t %f'%(shape_name, np.mean(shape_ious)))
    log_string("testing one batch with augmentation require %.2f milliseconds"%(1000*test_time/batch_idx))

    return


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()


