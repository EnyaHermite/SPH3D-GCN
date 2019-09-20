#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// for the unpooling modules, we have in_mpoint<out_npoint
REGISTER_OP("MeanInterpolate")
    .Input("input: float32") // batch * in_mpoint * in_channels
    .Input("nn_index: int32") // neighbor indices: batch * out_npoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_npoint
    .Output("output: float32") // batch * out_npoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        c->WithRank(c->input(0), 3, &input);
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        c->WithRank(c->input(2), 2, &nn_count);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(input, 0), c->Dim(nn_count, 1), c->Dim(input, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("MeanInterpolateGrad")
    .Input("input: float32")  // batch * in_mpoint * in_channels
    .Input("grad_output: float32") // batch * out_npoint * in_channels
    .Input("nn_index: int32") // neighbor indices: batch * out_npoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_npoint
    .Output("grad_input: float32") // batch * in_mpoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
REGISTER_OP("WeightedInterpolate")
    .Input("input: float32") // batch * in_mpoint * in_channels
    .Input("weight: float32") // weights: batch * out_npoint * nn_sample
    .Input("nn_index: int32") // neighbor indices: batch * out_npoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_npoint
    .Output("output: float32") // batch * out_npoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        c->WithRank(c->input(0), 3, &input);
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        c->WithRank(c->input(3), 2, &nn_count);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(input, 0), c->Dim(nn_count, 1), c->Dim(input, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("WeightedInterpolateGrad")
    .Input("input: float32")  // batch * in_mpoint * in_channels
    .Input("grad_output: float32") // batch * out_npoint * in_channels
    .Input("weight: float32") // weights: batch * out_npoint * nn_sample
    .Input("nn_index: int32") // neighbor and kernel bin indices: batch * out_npoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_npoint
    .Output("grad_input: float32") // batch * in_mpoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void meanInterpolateLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                             const float* input, float* output);
class MeanInterpolateGpuOp : public OpKernel {
    public:
        explicit MeanInterpolateGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& nn_index_tensor = context->input(1);
            const Tensor& nn_count_tensor = context->input(2);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int M = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int N = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,N,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* out = &(output_flat(0));
            cudaMemset(out, 0, sizeof(float)*B*N*C);
            meanInterpolateLauncher(B, N, M, C, K, nnIndex, nnCount, in, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeanInterpolate").Device(DEVICE_GPU), MeanInterpolateGpuOp);


void meanInterpolateGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                 const float* gradOutput, float* gradInput);
class MeanInterpolateGradGpuOp : public OpKernel {
    public:
        explicit MeanInterpolateGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int M = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int N = grad_output_tensor.shape().dim_size(1);    // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*B*M*C);
            meanInterpolateGradLauncher(B, N, M, C, K, nnIndex, nnCount, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeanInterpolateGrad").Device(DEVICE_GPU), MeanInterpolateGradGpuOp);


void weightedInterpolateLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                 const float* input, const float* weight, float* output);
class WeightedInterpolateGpuOp : public OpKernel {
    public:
        explicit WeightedInterpolateGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int M = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int N = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto weight_flat = weight_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,N,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* weight = &(weight_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* out = &(output_flat(0));
            cudaMemset(out, 0, sizeof(float)*B*N*C);
            weightedInterpolateLauncher(B, N, M, C, K, nnIndex, nnCount, in, weight, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("WeightedInterpolate").Device(DEVICE_GPU), WeightedInterpolateGpuOp);


void weightedInterpolateGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                     const float* gradOutput, const float* weight, float* gradInput);
class WeightedInterpolateGradGpuOp : public OpKernel {
    public:
        explicit WeightedInterpolateGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
           // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& weight_tensor = context->input(2);
            const Tensor& nn_index_tensor = context->input(3);
            const Tensor& nn_count_tensor = context->input(4);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int M = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int N = grad_output_tensor.shape().dim_size(1);    // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto weight_flat = weight_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const float* weight = &(weight_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*B*M*C);
            weightedInterpolateGradLauncher(B, N, M, C, K, nnIndex, nnCount, gradOut, weight, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("WeightedInterpolateGrad").Device(DEVICE_GPU), WeightedInterpolateGradGpuOp);