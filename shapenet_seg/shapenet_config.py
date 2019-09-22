import numpy as np

num_input = 2048
mlp = 64
num_sample = [1024, 768, 384, 128]
print(num_sample)
radius = [0.08, 0.16, 0.32, 0.64]
nn_uplimit = [64, 64, 64, 64]
channels = [[128,128], [256,256], [256,256], [512,512]]
multiplier = [[2,2], [2,2], [2,2], [2,2]]

weight_decay = None

kernel=[8,2,2]
binSize = np.prod(kernel)+1

normalize = False
pool_method = 'max'
unpool_method = 'mean'
nnsearch = 'sphere'
sample = 'FPS' #{'FPS','IDS','random'}

with_bn = True
with_bias = False