import numpy as np
import tensorflow as tf
import os, sys
import glob
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='path to the directory of the point cloud dataset')
parser.add_argument('--store_folder', required=True, help='name of the store folder')
INFO = parser.parse_args()
dataDir = INFO.data_path

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
store_folder = os.path.join(rootDir,'data',INFO.store_folder)

if not os.path.exists(store_folder):
    os.mkdir(store_folder)

print(INFO,dataDir,store_folder)

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


def make_tfrecord_seg(buildPath):

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

    AreaID = buildPath.split('/')[-2]
    BuildID = buildPath.split('/')[-1]
    dataset = glob.glob(os.path.join(buildPath,'Annotations','*.txt'))

    xyz = []
    rgb = []
    seg_label = []
    # print(len(dataset),dataset)
    for filepath in dataset:
        # print(filepath)
        data = np.loadtxt(filepath,dtype=np.float32,delimiter=' ')

        assert (data.shape[1]==6)  # the input point cloud has xyz+rgb

        filename = os.path.basename(filepath)
        key = filename.split('_')[0]
        if key in classes:
            seg_label.append(np.zeros((data.shape[0],), np.int32)+np.int32(classes[key]))
        else:
            # print('not exist class %s'%key)
            # continue
            seg_label.append(np.zeros((data.shape[0],), np.int32)+np.int32(classes['clutter']))

        xyz.append(data[:, 0:3])
        rgb.append(data[:, 3:])

    xyz = np.concatenate(xyz, axis=0)
    rgb = np.concatenate(rgb, axis=0)
    seg_label = np.concatenate(seg_label, axis=0)
    print(xyz.shape, rgb.shape, seg_label.shape)

    # =================color processing/normalization==================
    rgb = 2*rgb/255.0 - 1 # normalize to [-1,1] range
    # =================================================================

    # =====================location normalization======================
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min+xyz_max)/2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz = xyz - xyz_center  # align to room bottom center

    ptCloud = np.concatenate((xyz,rgb,np.reshape(seg_label,(-1,1))),axis=1)
    sio.savemat(os.path.join('%s/%s_%s.mat'%(store_folder, AreaID, BuildID)),{'ptCloud':ptCloud})

    return


if __name__=='__main__':
    Area = ['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6']

    for currArea in Area:
        buildings = glob.glob(os.path.join(dataDir,currArea,'*'))
        for buildPath in buildings:
            currBuild = os.path.basename(buildPath)

            print("========================make tfrecords of s3dis %s-%s======================="%(currArea,currBuild))
            make_tfrecord_seg(buildPath)
            print("===================================The End====================================")
