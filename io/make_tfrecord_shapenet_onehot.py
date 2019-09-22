import numpy as np
import tensorflow as tf
import os, sys
import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='path to the directory of the point cloud dataset')
INFO = parser.parse_args()
dataDir = INFO.data_path
print(INFO,dataDir)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_dir)
sys.path.append(os.path.join(root_dir,'tf_ops/sampling'))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecord_seg(dataDir, filelist, store_folder="",
                      chunksize=1024, verbose=True, debug=True):
    phase = filelist.split('_')[1]
    dataset = [line.rstrip() for line in open(os.path.join(dataDir, 'train_test_split', filelist  + '.json'))]
    dataset = json.loads(dataset[0])

    class_names = [line.rstrip().split('\t')[0] for line in
                   open(os.path.join(dataDir, 'synsetoffset2category.txt'))]
    class_folders = [line.rstrip().split('\t')[1] for line in
                     open(os.path.join(dataDir, 'synsetoffset2category.txt'))]

    print("number of samples: %d, number of classes: %d"%(len(dataset),len(class_names)))
    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    if debug:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        dataset = dataset[0:10] # test with the first element

    class_size = np.zeros((len(class_names),),np.int32)
    for i, filepath in enumerate(dataset):
        _, folder, filename = filepath.split('/')

        filepath = os.path.join(dataDir, folder, filename+'.txt')
        data = np.loadtxt(filepath, dtype=np.float32, delimiter=',')

        assert (data.shape[1]==5)  # the input point cloud has xyz+normals

        xyz = data[:, 0:3]
        xyz = xyz[:,[0,2,1]]
        seg_label = np.int32(data[:, -1]) - 1 # to make index start from 0
        cls_label = class_folders.index(folder)
        print(cls_label)
        class_size[cls_label] += 1

        if debug:
            print(class_names)
            print(class_folders)
            print(folder)
            print(filepath, cls_label)
            print("original data size:")
            print(data.shape, xyz.shape)
            print('mean and scale info before processing')
            print(np.mean(xyz, axis=0), np.sqrt(np.amax(np.sum(np.square(xyz), axis=1))))

        xyz = xyz - np.mean(xyz, axis=0)
        scale = np.sqrt(np.amax(np.sum(np.square(xyz), axis=1)))
        xyz /= scale # sphere centered at (0,0,0)

        # print(xyz.shape,seg_label.shape)

        if debug:
            print("resampled data size:")
            print(xyz.shape)
            print('mean and scale info after processing')
            print(np.mean(xyz,axis=0), np.sqrt(np.amax(np.sum(np.square(xyz),axis=1))))
            plt.figure(0)
            ax = plt.axes(projection='3d')
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c='green')
            plt.suptitle(class_names[cls_label])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.show()
        else:
            filename = os.path.join(store_folder, 'data_%s%d.tfrecord'%(phase,i//chunksize))
            if not os.path.exists(filename):
                if i>0:
                    print(i)
                    writer.close()
                writer = tf.io.TFRecordWriter(filename)
                if verbose:
                    print("start to make data_%s%d.tfrecords of the %sset:"%(phase, i//chunksize, phase))

            xyz_raw = xyz.tostring()
            seg_label_raw = seg_label.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'seg_label': _bytes_feature(seg_label_raw),
                'cls_label':_int64_feature(cls_label),
                'xyz_raw': _bytes_feature(xyz_raw)}))
            writer.write(example.SerializeToString())

    return class_size, i


if __name__=='__main__':
    rootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    trainlist = 'shuffled_train_file_list'
    vallist = 'shuffled_val_file_list'
    testlist = 'shuffled_test_file_list'

    store_folder = os.path.join(rootDir, 'data/shapenet_onehot')
    print("========================make tfrecords of shapenet data=======================")
    train_cls_size, train_num = make_tfrecord_seg(dataDir,trainlist,store_folder=store_folder,debug=False)
    print(train_cls_size, train_num)

    val_cls_size, val_num = make_tfrecord_seg(dataDir,vallist,store_folder=store_folder,debug=False)
    print(val_cls_size, val_num)

    test_cls_size, test_num = make_tfrecord_seg(dataDir,testlist,store_folder=store_folder,debug=False)
    print(test_cls_size, test_num)
    print("===================================The End====================================")

    for phase in ["train", "val", "test"]:
        files = glob.glob(os.path.join(store_folder,'*%s*.tfrecord' % phase))
        if phase=="train":
            txtfile = open(os.path.join(store_folder, 'train_files.txt'), 'w')
        elif phase=="val":
            txtfile = open(os.path.join(store_folder,'train_files.txt'), 'a')
        else:
            txtfile = open(os.path.join(store_folder, 'test_files.txt'), 'w')

        for i in range(len(files)):
            txtfile.write("%s\n" % files[i])
        txtfile.close()