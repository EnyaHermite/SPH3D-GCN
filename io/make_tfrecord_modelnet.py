import numpy as np
import tensorflow as tf
import os, sys
import glob
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


def make_tfrecord_cls(dataDir, filelist, classlist, num_point=10000,
                      store_folder="", chunksize=1024, verbose=True, debug=True):

    phase = filelist.split('_')[-1]
    dataset = [line.rstrip() for line in open(os.path.join(dataDir, filelist + '.txt'))]
    classes = [line.rstrip() for line in open(os.path.join(dataDir, classlist + '.txt'))]

    print("number of samples: %d, number of classes: %d"%(len(dataset),len(classes)))
    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    # dataset = dataset[0:10]
    import tensorflow as tf
    if debug:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        dataset = dataset[0:1] # test with the first element

    for i, filename in enumerate(dataset):
        classname = '_'.join(filename.split('_')[0:-1])

        filepath = os.path.join(dataDir, classname, filename + '.txt')
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)

        label = classes.index(classname)

        assert(data.shape[1]==6) # the input point cloud has xyz+normals

        xyz = data[:,0:3]
        normal = data[:,3:6]

        if debug:
            print(classname, filepath, label)
            print("original data size:")
            print(data.shape, xyz.shape, normal.shape)
            print('mean and scale info before processing')
            print(np.mean(xyz, axis=0), np.sqrt(np.amax(np.sum(np.square(xyz), axis=1))))
            print(np.sqrt(np.amax(np.sum(np.square(normal),axis=1))),np.sqrt(np.amin(np.sum(np.square(normal),axis=1))))

        if num_point < xyz.shape[0]:
            from tf_sample import farthest_point_sample
            import tensorflow as tf

            index = farthest_point_sample(num_point, tf.expand_dims(xyz, axis=0)) # batch_size=1
            index = tf.squeeze(index)

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                sample_index = sess.run(index)

            xyz = xyz[sample_index,:]
            normal = normal[sample_index,:]
        elif num_point > xyz.shape[0]:
            print("The original pointcloud size %d must be no less than the expected %d"%(xyz.shape[0],num_point))
            exit()

        xyz = xyz - np.mean(xyz, axis=0)
        scale = np.sqrt(np.amax(np.sum(np.square(xyz), axis=1)))
        xyz /= scale # sphere centered at (0,0,0)

        if debug:
            print("resampled data size:")
            print(xyz.shape, normal.shape)
            print('mean and scale info after processing')
            print(np.mean(xyz,axis=0), np.sqrt(np.amax(np.sum(np.square(xyz),axis=1))))
            plt.figure(0)
            ax = plt.axes(projection='3d')
            ax.scatter(data[:,0], data[:,1], data[:,2], c='green')
            plt.show()
        else:
            if i%chunksize==0:
                filename = os.path.join(store_folder, 'data_%s%d.tfrecord'%(phase,i//chunksize))
                if verbose:
                    print("start to make data_%s%d.tfrecords of the %sset:" %(phase,i//chunksize,phase))
                if i>0:
                    writer.close()
                writer = tf.io.TFRecordWriter(filename)

            xyz_raw = xyz.tostring()
            normal_raw = normal.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'normal_raw': _bytes_feature(normal_raw),
                'label': _int64_feature(label),
                'xyz_raw': _bytes_feature(xyz_raw)}))
            writer.write(example.SerializeToString())

            if verbose:
                print(i, label)


if __name__=='__main__':
    rootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(rootDir)

    trainlist = 'modelnet40_train'
    testlist = 'modelnet40_test'
    classlist = 'modelnet40_shape_names'

    num_point = 10000

    store_folder = os.path.join(rootDir, 'data/modelnet40')

    print("===================make tfrecords of modelnets: 10K points===================")
    make_tfrecord_cls(dataDir, trainlist, classlist, store_folder=store_folder, num_point=num_point, debug=False)
    make_tfrecord_cls(dataDir, testlist, classlist, store_folder=store_folder, num_point=num_point, debug=False)
    print("===================================The End====================================")

    for phase in ["train", "test"]:
        files = glob.glob(os.path.join(store_folder,'*%s*.tfrecord' % phase))
        txtfile = open(os.path.join(store_folder,'%s_files.txt' % phase), 'w')
        for i in range(len(files)):
            txtfile.write("%s\n" % files[i])
        txtfile.close()