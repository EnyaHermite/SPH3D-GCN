## *Spherical Kernel for Efficient Graph Convolution on 3D Point Clouds*
Created by Huan Lei, Naveed Akhtar and Ajmal Mian

![alt text](https://github.com/hlei-ziyan/SPH3D-GCN/blob/master/image/intro_arch.png)

### Introduction
This work is based on our [Arxiv tech report](https://arxiv.org/submit/2851732), which is a significant extension of the CVPR2019 paper ``Octree guided CNN with Spherical Kernels for 3D Point Clouds``.

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

@article{lei2019octree,
title={Octree guided CNN with Spherical Kernels for 3D Point Clouds},
author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
journal={IEEE Conference on Computer Vision and Pattern Recognition},
year={2019}
}
@article{lei2019spherical,
title={Spherical Kernel for Efficient Graph Convolution on 3D Point Clouds},
author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
journal={arXiv preprint arXiv:1920.07872},
year={2019}
}

### License
Our code is released under MIT License (see LICENSE file for details).

### Installation


### Usage

- ModelNet

- ShapeNet

- RueMonge2014

- ScanNet V2

- S3DIS


# ...... incomplete Code, ReadMe in construction ......
