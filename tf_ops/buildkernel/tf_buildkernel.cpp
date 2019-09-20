#include <cmath> // sqrtf
#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;


REGISTER_OP("SphericalKernel")
    .Attr("radius: float")          // range search radius
    .Attr("n_azim: int")            // division along azimuth direction
    .Attr("p_elev: int")            // division along elevation direction
    .Attr("q_radi: int")            // division along radius direction
    .Input("database: float32")     // database points: batch * npoint * 3 (x,y,z)
    .Input("query: float32")        // query points: batch * mpoint * 3
    .Input("nn_index: int32")      // neighbor and kernel bin indices: batch * mpoint * nn_sample
    .Input("nn_count: int32")      // number of neighbors: batch * mpoint
    .Input("nn_dist: float32")     // distance to the neighbors: batch * mpoint * nn_sample
    .Output("filt_index: int32")   // kernel bin indices: batch * mpoint * nn_sample
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle nn_index; // batch * mpoint * nn_sample
        c->WithRank(c->input(2), 3, &nn_index);

        c->set_output(0, nn_index);
        return Status::OK();
    });


void sphericalKernelLauncher(int B, int N, int M, int K, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const int* nnCount, const float* nnDist, int* filtIndex);
class SphericalKernelGpuOp : public OpKernel {
    public:
        explicit SphericalKernelGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Range search requires radius>0, got ", radius_));

            OP_REQUIRES_OK(context, context->GetAttr("n_azim", &n_));
            OP_REQUIRES(context, n_>2 && n_%2==0, errors::InvalidArgument("Need n_>2 and n_%2==0, got ", n_));

            OP_REQUIRES_OK(context, context->GetAttr("p_elev", &p_));
            OP_REQUIRES(context, p_>0 && p_%2==0, errors::InvalidArgument("Need p_>0 and p_%2==0, got ", p_));

            OP_REQUIRES_OK(context, context->GetAttr("q_radi", &q_));
            OP_REQUIRES(context, q_>0, errors::InvalidArgument("Need q_>0, got ", q_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& database_tensor = context->input(0);
            const Tensor& query_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);
            const Tensor& nn_dist_tensor = context->input(4);

            // get the dims required by computations
            int B = database_tensor.shape().dim_size(0); // batch size
            int N = database_tensor.shape().dim_size(1); // number of database points
            int M = query_tensor.shape().dim_size(1);    // number of query points
            int K = nn_index_tensor.shape().dim_size(2);

            OP_REQUIRES(context, database_tensor.dims()==3 && database_tensor.shape().dim_size(2)==3, errors::InvalidArgument("Shape of database points requires to be (batch, npoint, 3)"));
            OP_REQUIRES(context, query_tensor.dims()==3 && query_tensor.shape().dim_size(2)==3, errors::InvalidArgument("Shape of query points requires to be (batch, mpoint, 3)"));
            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("Shape of nn_index requires to be of rank 3"));

            // flatten the input tensors
            auto database_flat = database_tensor.flat<float>();
            auto query_flat = query_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_dist_flat = nn_dist_tensor.flat<float>();

            // Create an output tensor
            Tensor* filt_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,K}, &filt_index_tensor));
            auto filt_index_flat = filt_index_tensor->flat<int32>();

            const float* database = &(database_flat(0));
            const float* query = &(query_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const float* nnDist = &(nn_dist_flat(0));

            int* filtIndex = &(filt_index_flat(0));
            cudaMemset(filtIndex, 0, sizeof(int)*B*M*K);

            sphericalKernelLauncher(B, N, M, K, n_, p_, q_, radius_,
                                        database, query, nnIndex, nnCount, nnDist,
                                        filtIndex);
        }
    private:
        float radius_;
        int n_, p_, q_;
};
REGISTER_KERNEL_BUILDER(Name("SphericalKernel").Device(DEVICE_GPU), SphericalKernelGpuOp);












