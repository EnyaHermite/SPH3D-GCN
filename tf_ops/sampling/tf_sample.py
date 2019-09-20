''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_sample_so.so'))

def farthest_point_sample(neursize, database):
    '''
    input:
        neursize: int32, the number of neurons/points to be sampled
        database: (batch, npoint, 3) float32 array, database points
    returns:
        neuron_index: (batch_size, neursize) int32 array, index of sampled neurons in the database
    '''
    return sampling_module.farthest_point_sample(database, neursize)
ops.NoGradient('FarthestPointSample')


def inverse_density_sample(neursize, probability):
    '''
    input:
        neursize: int32, the number of neurons/points to be sampled
        probability: the inverse density of each point
    returns:
        neuron_index: (batch_size, neursize) int32 array, index of sampled neurons in the database
    '''
    # Refs for fast sampling without using tf.py_func and np.random.choice:
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    # http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    logits = tf.log(probability)  # required by the Gumbel-max trick
    z = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits), 0, 1)))  # work as a Gumbel random variate generator
    _, neuron_index = tf.nn.top_k(logits + z, neursize)
    return neuron_index


def random_sample(neursize, database):
    # NOTE: randomSampling applies a random sampling with uniform probability
    batch_size = tf.shape(database)[0]
    num_points = tf.shape(database)[1]
    neuron_index = tf.random.uniform((batch_size, neursize), minval=0, maxval=num_points, dtype=tf.int32)
    return neuron_index