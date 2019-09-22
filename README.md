## *Spherical Kernel for Efficient Graph Convolution on 3D Point Clouds*
Created by Huan Lei, Naveed Akhtar and Ajmal Mian

![alt text](https://github.com/hlei-ziyan/SPH3D-GCN/blob/master/image/intro_arch.png)

### Introduction
This work is based on our [Arxiv tech report](https://arxiv.org/submit/2851732), which is a **significant** extension of [our CVPR2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Lei_Octree_Guided_CNN_With_Spherical_Kernels_for_3D_Point_Clouds_CVPR_2019_paper.html).

We propose a spherical kernel for efficient graph convolution of 3D point clouds. 
Our metric-based kernels systematically quantize the local 3D space 
to identify distinctive geometric relationships in the data. Similar to the regular grid CNN kernels, the spherical kernel maintains translation-invariance and asymmetry properties, where the former guarantees weight sharing among similar local structures in the  data and the latter facilitates fine geometric learning. 
The proposed kernel is applied to graph neural networks without edge-dependent filter generation, making it computationally attractive for large point clouds. 
In our graph networks, each vertex is associated with a single point location and edges connect the neighborhood points within a defined range. The graph gets coarsened in the network with farthest point sampling. 
Analogous to the standard CNNs, we define pooling and unpooling operations for our network. 
We demonstrate the effectiveness of the proposed spherical kernel with graph neural networks for point cloud classification and semantic segmentation  using ModelNet, ShapeNet, RueMonge2014, ScanNet and S3DIS datasets.

In this repository, we release code and trained models for classification and segmentation.

### Citation
If you find our work useful in your research, please consider citing:

```
@article{lei2019octree,  
title={Octree guided CNN with Spherical Kernels for 3D Point Clouds},  
author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},  
journal={IEEE Conference on Computer Vision and Pattern Recognition},  
year={2019}  
}  
```
```
@article{lei2019spherical,  
title={Spherical Kernel for Efficient Graph Convolution on 3D Point Clouds},  
author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},  
journal={arXiv preprint arXiv:1920.xxxxx},  
year={2019}  
}
```

### License
Our code is released under MIT License (see LICENSE file for details).

### Installation


### Usage

- ModelNet
`cd io`  
`python make_tfrecord_modelnet.py`  
`cd modelnet40_cls`  
`python train_modelnet.py`  
`python evaluate_modelnet.py` --num_votes=12  

- ShapeNet
run `preprocessing/shapenet_removeSingularPoints.m` in matlab  
`cd io`  
`python make_tfrecord_shapenet.py`  
`cd shapenet_seg`  
`python train_shapenet.py --shape_name=Table`  
`python evaluate_modelnet.py`  

- RueMonge2014
run `preprocessing/ruemonge2014_prepare_data.m` in matlab  
`cd io`  
`python make_tfrecord_ruemonge2014.py`  
`cd ruemonge2014_seg`  
`python train_ruemonge2014.py`  
`python evaluate_ruemonge2014.py`  

- ScanNet V2
run `preprocessing/scannet_prepare_data.m` in matlab  
`cd io`  
`python make_tfrecord_scannet.py`  
`cd scannet_seg`  
`python train_scannet.py`  
`python evaluate_scannet_with_overlap.py`  
`python scannet_block2index_with_overlap.py`  
run `post-merging/scannet_merge.m` in matlab  

- S3DIS
run `preprocessing/s3dis_prepare_data.m` in matlab  
`cd io`  
`python make_tfrecord_s3dis.py`  
`python make_tfrecord_s3dis_no_split.py`  
`cd s3dis_seg`  
`python train_s3dis.py`  
`python evaluate_s3dis_with_overlap.py --model_name=xxxx`  
`python s3dis_block2index_with_overlap.py`  
run `post-merging/s3dis_merge.m` in matlab  

# ...... incomplete Code, ReadMe in construction ......
