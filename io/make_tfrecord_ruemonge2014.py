import numpy as np
import tensorflow as tf
import os, sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='path to the directory of the point cloud dataset')
dataDir = parser.parse_args()

rootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(rootDir,'tf_ops/sampling'))


def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecord_seg(itemPath, splitName, store_folder="", verbose=True):

    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    import tensorflow as tf

    data = np.loadtxt(itemPath,dtype=np.float32,delimiter=',')
    assert (data.shape[1]==10)  # the input point cloud has xyz+rgb

    xyz = data[:,0:3]
    center = np.mean(xyz,axis=0)
    center[2] = np.amin(xyz[:,2],axis=0)
    xyz -= center
    rgb = data[:,3:6]
    normal = data[:,6:9]
    seg_label = np.int32(data[:,9])

    # =================color processing/normalization==================
    rgb = 2*rgb/255.0-1  # normalize to [-1,1] range
    # =================================================================
    print('min_xyz:', np.amin(xyz,axis=0), 'max_xyz:', np.amax(xyz,axis=0))
    print('min_rgb:', np.amin(rgb,axis=0), 'max_rgb:', np.amax(rgb,axis=0))
    print('min_normal:', np.amin(normal,axis=0), 'max_normal:', np.amax(normal,axis=0))
    print(np.amin(seg_label), np.amax(seg_label))

    filename = os.path.join(store_folder, '%s.tfrecord'%(splitName))
    if verbose:
        print("start to make %s.tfrecord:"%(splitName))
    if not os.path.exists(filename):
        writer = tf.io.TFRecordWriter(filename)

    xyz_raw = xyz.tostring()
    rgb_raw = rgb.tostring()
    normal_raw = normal.tostring()
    seg_label_raw = seg_label.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
            'rgb_raw':_bytes_feature(rgb_raw),
            'normal_raw':_bytes_feature(normal_raw),
            'seg_label':_bytes_feature(seg_label_raw),
            'xyz_raw':_bytes_feature(xyz_raw)}))
    writer.write(example.SerializeToString())

    writer.close()

    return data.shape[0]


if __name__=='__main__':
    store_folder = os.path.join(rootDir, 'data/ruemonge2014')
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    for phase in ['train','test']:
        LOG_FOUT = open(os.path.join(store_folder, 'log_block.txt'), 'a')

        splits = glob.glob(os.path.join(dataDir,phase,'*.txt'))
        print("========================make tfrecords of ruemonge2014 %sset======================="%phase)
        for itemPath in splits:
            splitName = itemPath.split('/')[-1][0:-4]
            print(itemPath,splitName)
            dataNum = make_tfrecord_seg(itemPath, splitName, store_folder=store_folder)

            log_string(LOG_FOUT, '%s, %d'%(splitName, dataNum))
        print("===================================The End====================================")

    for phase in ['train', 'test']:
        files = glob.glob(os.path.join(store_folder,'%s*.tfrecord'%phase))
        file = open(os.path.join(store_folder, '%s_files.txt'%phase), 'w')

        for filepath in files:
            file.write("%s\n"%filepath)
        file.close()