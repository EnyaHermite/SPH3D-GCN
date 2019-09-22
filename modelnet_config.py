import numpy as np

num_input = 10000
num_cls = 40

mlp = 32
num_sample = [num_input//4**(i+1) for i in range(10) if (num_input//4**(i+1))>100]
print(num_sample)

radius = [0.1, 0.2, 0.4]
nn_uplimit = [64, 64, 64]
channels = [[64, 64], [64, 128], [128, 128]]
multiplier = [[2, 1], [1, 2], [1, 1]]

assert(len(num_sample)==len(radius))
assert(len(num_sample)==len(nn_uplimit))
assert(len(num_sample)==len(channels))
assert(len(num_sample)==len(multiplier))

# =====================for final layer convolution=====================
global_channels = 512
global_multiplier = 2
# =====================================================================

weight_decay = 1e-5

kernel=[8,2,2]
binSize = np.prod(kernel)+1

normalize = True
pool_method = 'max'
nnsearch = 'sphere'
sample = 'FPS' #{'FPS','IDS','random'}

use_raw = True
with_bn = True
with_bias = False