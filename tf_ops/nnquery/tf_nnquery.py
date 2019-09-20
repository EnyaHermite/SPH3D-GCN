import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
nnquery_module = tf.load_op_library(os.path.join(base_dir, 'tf_nnquery_so.so'))

def build_sphere_neighbor(database,
                          query,
                          radius=0.1,
                          dilation_rate=None,
                          nnsample=100):
    '''
    Input:
        database: (batch, npoint, 3+x) float32 array, database points
        query:    (batch, mpoint, 3) float32 array, query points
        radius:   float32, range search radius
        dilation_rate: float32, dilation rate of range search
        nnsample: int32, maximum number of neighbors to be sampled
    Output:
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist(optional): (batch, mpoint, nnsample) float32, sqrt distance array
    '''
    database = database[:,:,0:3]
    query = query[:,:,0:3]

    if dilation_rate is not None:
        radius = dilation_rate * radius

    return nnquery_module.build_sphere_neighbor(database, query, radius, nnsample)
ops.NoGradient('BuildSphereNeighbor')


def build_cube_neighbor(database,
                        query,
                        length=0.1,
                        dilation_rate=None,
                        nnsample=100,
                        gridsize=3):
    '''
    Input:
        database: (batch, npoint, 3) float32 array, database points
        query:    (batch, mpoint, 3) float32 array, query points
        length:   float32, cube search length
        dilation_rate: float32, dilation rate of cube search
        nnsample: int32, maximum number of neighbors to be sampled
        gridsize: int32 , cubical kernel size
    Output:
        nn_index: (batch, mpoint, nnsample, 2) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
    '''

    database = database[:, :, 0:3]
    query = query[:, :, 0:3]
    if dilation_rate is not None:
        length = dilation_rate * length
    return nnquery_module.build_cube_neighbor(database, query, length, nnsample, gridsize)
ops.NoGradient('BuildCubeNeighbor')
