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


def make_tfrecord_seg(buildPath, block_point_num_thresh=10000,
                      block_size=2.5, context_size=0.3, interval = 0.5,
                      store_folder="", verbose=True, debug=False):
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

    # note the scene types are stored, but not used
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

    LOG_FOUT = open(os.path.join(store_folder, 'log_block.txt'), 'a')

    AreaID = buildPath.split('/')[-2]
    BuildID = buildPath.split('/')[-1]
    dataset = glob.glob(os.path.join(buildPath,'Annotations','*.txt'))

    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    import tensorflow as tf
    if debug:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        dataset = dataset[0:1] # test with the first element

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

    scene_label = scenes[BuildID.split('_')[0].lower()]
    scene_idx = np.int32(BuildID.split('_')[1])

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

    rel_xyz = np.zeros(xyz.shape,dtype=np.float32)
    rel_xyz[:,0] = 2*xyz[:,0]/(xyz_max[0,0]-xyz_min[0,0])
    rel_xyz[:,1] = 2*xyz[:,1]/(xyz_max[0,1]-xyz_min[0,1])
    rel_xyz[:,2] = 2*xyz[:,2]/(xyz_max[0,2]-xyz_min[0,2]) - 1.0
    print('min rel_xyz:', np.amin(rel_xyz, axis=0, keepdims=True))
    print('max rel_xyz:', np.amax(rel_xyz, axis=0, keepdims=True))
    print('min rgb:', np.amin(rgb, axis=0, keepdims=True))
    print('max rgb:', np.amax(rgb, axis=0, keepdims=True))
    # =================================================================

    filename = os.path.join(store_folder, '%s_%s.tfrecord'%(AreaID, BuildID))
    if verbose:
        print("start to make %s_%s.tfrecords:"%(AreaID, BuildID))
    if not os.path.exists(filename):
        writer = tf.io.TFRecordWriter(filename)

    minXYZ = np.min(xyz,axis=0)
    maxXYZ = np.max(xyz,axis=0)

    if interval<block_size:
        print('generating blocks with overlap %.2f'%(block_size-interval))
    else:
        interval = block_size # force all room spaces to be used
        print('generating blocks without overlap')

    # compute the block start point
    xLeft = np.arange(minXYZ[0], maxXYZ[0]-block_size, interval)
    yBack = np.arange(minXYZ[1], maxXYZ[1]-block_size, interval)

    if not xLeft.size:
        # print('xLeft Before',xLeft)
        xLeft = np.append(xLeft, minXYZ[0])
        # print('xLeft After', xLeft)
    if not yBack.size:
        # print('yBack Before', yBack)
        yBack = np.append(yBack, minXYZ[1])
        # print('yBack After', yBack)

    if xLeft[-1]<(maxXYZ[0]-block_size):
        xLeft = np.append(xLeft,maxXYZ[0]-block_size)
    if yBack[-1]<(maxXYZ[1]-block_size):
        yBack = np.append(yBack,maxXYZ[1]-block_size)
    # print(xLeft,yBack)

    for x in xLeft:
        for y in yBack:
            # ===============================Inner Points============================
            # only use the inner point to compute the loss function, as well as to
            # make the predictions
            inner = (xyz[:, 0]>=x)&(xyz[:, 0]<=(x+block_size))& \
                    (xyz[:, 1]>=y)&(xyz[:, 1]<=(y+block_size))
            inner = np.int32(inner)
            # =======================================================================

            if np.sum(inner) < block_point_num_thresh: # merge small blocks into oen of their big neighbor block
                coord = [(x-block_size, x+block_size,   y,            y+block_size),   \
                         (x,            x+2*block_size, y,            y+block_size),   \
                         (x,            x+block_size,   y-block_size, y+block_size),   \
                         (x,            x+block_size,   y,            y+2*block_size), \
                         (x-block_size, x+block_size,   y-block_size, y+block_size),   \
                         (x-block_size, x+block_size,   y,            y+2*block_size), \
                         (x,            x+2*block_size, y-block_size, y+block_size),   \
                         (x,            x+2*block_size, y,            y+2*block_size)]

                nbr_idx = -1
                for nnId in range(len(coord)):
                    inner = (xyz[:, 0]>=coord[nnId][0])&(xyz[:, 0]<=coord[nnId][1])& \
                            (xyz[:, 1]>=coord[nnId][2])&(xyz[:, 1]<=coord[nnId][3])
                    inner = np.int32(inner)
                    if np.sum(inner)>=block_point_num_thresh:
                        nbr_idx = nnId
                        break

                if nbr_idx==-1:
                    continue
                else:
                    min_x, max_x, min_y, max_y = coord[nbr_idx]
            else:
                min_x, max_x, min_y, max_y = (x, x+block_size, y, y+block_size)


            # ==========================With Context Padding=========================
            index = (xyz[:,0]>=(min_x-context_size)) & \
                    (xyz[:,0]<=(max_x+context_size)) & \
                    (xyz[:,1]>=(min_y-context_size)) & \
                    (xyz[:,1]<=(max_y+context_size))
            # =======================================================================
            points = xyz[index, :]
            rel_points = rel_xyz[index, :]
            color = rgb[index, :]
            label = seg_label[index]
            # ===============================Inner Points============================
            # only use the inner point to compute the loss function, as well as to
            # make the predictions
            inner = (points[:, 0]>=min_x)&(points[:, 0]<=max_x) & \
                    (points[:, 1]>=min_y)&(points[:, 1]<=max_y)
            inner = np.int32(inner)
            # =======================================================================
            log_string(LOG_FOUT, '%s, %s, %d, %d'%(AreaID, BuildID, np.sum(inner), np.sum(index)))

            index, = np.where(index)
            index = np.int32(index)

            xyz_raw = points.tostring()
            rel_xyz_raw = rel_points.tostring()
            rgb_raw = color.tostring()
            seg_label_raw = label.tostring()
            index_label = index.tostring()
            inner_label = inner.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'rgb_raw':_bytes_feature(rgb_raw),
                    'seg_label':_bytes_feature(seg_label_raw),
                    'inner_label':_bytes_feature(inner_label),
                    'index_label':_bytes_feature(index_label),
                    'scene_label':_int64_feature(scene_label),
                    'scene_idx':_int64_feature(scene_idx),
                    'rel_xyz_raw':_bytes_feature(rel_xyz_raw),
                    'xyz_raw':_bytes_feature(xyz_raw)}))
            writer.write(example.SerializeToString())

    writer.close()

    return


if __name__=='__main__':
    block_size = 1.5
    interval = block_size/2
    store_folder = os.path.join(rootDir, 'data/s3dis_3cm')
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    Area = ['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6']

    for currArea in Area:
        buildings = glob.glob(os.path.join(dataDir,currArea,'*'))
        for buildPath in buildings:
            currBuild = os.path.basename(buildPath)

            print("========================make tfrecords of s3dis %s-%s======================="%(currArea,currBuild))
            make_tfrecord_seg(buildPath, block_point_num_thresh=10000, block_size=block_size,
                              interval=interval, store_folder=store_folder, debug=False)
            print("===================================The End====================================")

    for i, currArea in enumerate(Area):
        files = glob.glob(os.path.join(store_folder,'*.tfrecord'))
        testfile = open(os.path.join(store_folder, 'test_files_fold%d.txt'%(i+1)), 'w')
        trainfile = open(os.path.join(store_folder, 'train_files_fold%d.txt'%(i+1)), 'w')

        for filepath in files:
            if currArea in filepath:
                testfile.write("%s\n" %filepath)
            else:
                trainfile.write("%s\n"%filepath)
        testfile.close()
        trainfile.close()