#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("MaxPool3d")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("nn_index: int32") // neighbor and kernel bin indices: batch * out_mpoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_mpoint
    .Output("output: float32") // batch * out_mpoint * in_channels
    .Output("max_index: int32") // batch * out_mpoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        c->WithRank(c->input(0), 3, &input);
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        c->WithRank(c->input(2), 2, &nn_count);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(input, 0), c->Dim(nn_count, 1), c->Dim(input, 2)});
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });
REGISTER_OP("MaxPool3dGrad")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("grad_output: float32") // batch * out_mpoint * in_channels
    .Input("max_index: int32") // the neighbor gives maximum response: batch * out_mpoint * nn_sample
    .Output("grad_input: float32") // batch * in_npoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
REGISTER_OP("AvgPool3d")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("nn_index: int32") // neighbor and kernel bin indices: batch * out_mpoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_mpoint
    .Output("output: float32") // batch * out_mpoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        c->WithRank(c->input(0), 3, &input);
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        c->WithRank(c->input(2), 2, &nn_count);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(input, 0), c->Dim(nn_count, 1), c->Dim(input, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("AvgPool3dGrad")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("grad_output: float32") // batch * out_mpoint * in_channels
    .Input("nn_index: int32") // neighbor indices: batch * out_mpoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_mpoint
    .Output("grad_input: float32") // batch * in_npoint * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void maxPool3dLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       const float* input, float* output, int* maxIndex);
class MaxPool3dGpuOp : public OpKernel {
    public:
        explicit MaxPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& nn_index_tensor = context->input(1);
            const Tensor& nn_count_tensor = context->input(2);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int M = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            Tensor* max_index_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,C}, &output_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B,M,C}, &max_index_tensor));
            auto output_flat = output_tensor->flat<float>();
            auto max_index_flat = max_index_tensor->flat<int32>();

            const float* in = &(input_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* out = &(output_flat(0));
            int* maxIndex = &(max_index_flat(0));
            cudaMemset(out, 0, sizeof(float)*B*M*C);
            cudaMemset(maxIndex, 0, sizeof(int)*B*M*C);
            maxPool3dLauncher(B, N, M, C, K, nnIndex, nnCount, in, out, maxIndex);
        }
};
REGISTER_KERNEL_BUILDER(Name("MaxPool3d").Device(DEVICE_GPU), MaxPool3dGpuOp);


void maxPool3dGradLauncher(int B, int N, int M, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput);
class MaxPool3dGradGpuOp : public OpKernel {
    public:
        explicit MaxPool3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& max_index_tensor = context->input(2);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int M = grad_output_tensor.shape().dim_size(1);    // number of output points

            OP_REQUIRES(context, max_index_tensor.dims()==3, errors::InvalidArgument("rank of max_index should be 3, i.e. (batch, mpoint, in_channels)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto max_index_flat = max_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,N,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* maxIndex = &(max_index_flat(0));
            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*B*N*C);
            maxPool3dGradLauncher(B, N, M, C, maxIndex, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MaxPool3dGrad").Device(DEVICE_GPU), MaxPool3dGradGpuOp);


void avgPool3dLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       const float* input, float* output);
class AvgPool3dGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& nn_index_tensor = context->input(1);
            const Tensor& nn_count_tensor = context->input(2);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int M = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* out = &(output_flat(0));
            cudaMemset(out, 0, sizeof(float)*B*M*C);
            avgPool3dLauncher(B, N, M, C, K, nnIndex, nnCount, in, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("AvgPool3d").Device(DEVICE_GPU), AvgPool3dGpuOp);


void avgPool3dGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, float* gradInput);
class AvgPool3dGradGpuOp : public OpKernel {
    public:
        explicit AvgPool3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int M = grad_output_tensor.shape().dim_size(1);    // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("rank of nn_index should be 3, i.e. (batch, mpoint, nnsample)"));
            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("rank of nn_count should be 2, i.e. (batch, mpoint)"));

            // flatten the input tensors
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,N,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* gradOut = &(grad_output_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*B*N*C);
            avgPool3dGradLauncher(B, N, M, C, K, nnIndex, nnCount, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("AvgPool3dGrad").Device(DEVICE_GPU), AvgPool3dGradGpuOp);