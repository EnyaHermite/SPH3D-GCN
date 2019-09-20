#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("DepthwiseConv3d")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("filter: float32") // convolution: filter_size * in_channels * channel_multiplier   .
    .Input("nn_index: int32") // neighbor indices: batch * out_mpoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_mpoint
    .Input("bin_index: int32") // kernel bin indices: batch * out_mpoint * nn_sample
    .Output("output: float32") // batch * out_mpoint * out_channels  (out_channels = in_channels * channel_multiplier)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle filter;
        c->WithRank(c->input(1), 3, &filter);
        ::tensorflow::shape_inference::ShapeHandle nn_count;
        c->WithRank(c->input(3), 2, &nn_count);
        ::tensorflow::shape_inference::DimensionHandle Cout;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(filter, 1), c->Dim(filter, 2), &Cout));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(nn_count, 0), c->Dim(nn_count, 1), Cout});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("DepthwiseConv3dGrad")
    .Input("input: float32") // batch * in_npoint * in_channels
    .Input("filter: float32") // convolution: filter_size * in_channels * channel_multiplier
    .Input("grad_output:float32") // batch * out_mpoint * out_channels
    .Input("nn_index: int32") // neighbor indices: batch * out_mpoint * nn_sample
    .Input("nn_count: int32") // number of neighbors: batch * out_mpoint
    .Input("bin_index: int32") // kernel bin indices: batch * out_mpoint * nn_sample
    .Output("grad_input: float32") // batch * in_npoint * in_channels
    .Output("grad_filter: float32") // filter_size * in_channels * channel_multiplier
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });


void depthwiseConv3dLauncher(int B, int N, int M, int C, int r, int K,
                             const int* nnIndex, const int* nnCount, const int* binIndex,
                             const float* input, const float* filter, float* output);
class DepthwiseConv3dGpuOp : public OpKernel {
    public:
        explicit DepthwiseConv3dGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& nn_index_tensor = context->input(2);
            const Tensor& nn_count_tensor = context->input(3);
            const Tensor& bin_index_tensor = context->input(4);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int r = filter_tensor.shape().dim_size(2);   // depthwise channel multiplier
            int M = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C, errors::InvalidArgument("Input Channel size error of the filter"));
            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("The rank of nn_index should be 3."));
            OP_REQUIRES(context, bin_index_tensor.dims()==3, errors::InvalidArgument("The rank of bin_index should be 3."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,C*r}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* binIndex = &(bin_index_flat(0));

            float* out = &(output_flat(0));
            cudaMemset(out,0,sizeof(float)*B*M*C*r);
            depthwiseConv3dLauncher(B, N, M, C, r, K, nnIndex, nnCount, binIndex, in, filt, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv3d").Device(DEVICE_GPU), DepthwiseConv3dGpuOp);


void depthwiseConv3dGradLauncher(int B, int N, int M, int F, int C, int r, int K,
                                 const int* nnIndex, const int* nnCount, const int* binIndex,
                                 const float* input, const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter);
class DepthwiseConv3dGradGpuOp : public OpKernel {
    public:
        explicit DepthwiseConv3dGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor = context->input(0);
            const Tensor& filter_tensor = context->input(1);
            const Tensor& grad_output_tensor = context->input(2);
            const Tensor& nn_index_tensor = context->input(3);
            const Tensor& nn_count_tensor = context->input(4);
            const Tensor& bin_index_tensor = context->input(5);

            // get the dims required by computations
            int B = input_tensor.shape().dim_size(0);    // batch size
            int N = input_tensor.shape().dim_size(1);    // number of input points
            int C = input_tensor.shape().dim_size(2);    // number of input channels
            int F = filter_tensor.shape().dim_size(0);   // filter bin size
            int r = filter_tensor.shape().dim_size(2);   // depthwise channel multiplier
            int M = nn_index_tensor.shape().dim_size(1); // number of output points
            int K = nn_index_tensor.shape().dim_size(2); // max number of neighbors sampled

            OP_REQUIRES(context, filter_tensor.shape().dim_size(1)==C, errors::InvalidArgument("Channel size error of the filter"));
            OP_REQUIRES(context, nn_index_tensor.dims()==3, errors::InvalidArgument("The rank of nn_index should be 3."));
            OP_REQUIRES(context, bin_index_tensor.dims()==3, errors::InvalidArgument("The rank of bin_index should be 3."));

            // flatten the input tensors
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto nn_index_flat = nn_index_tensor.flat<int32>();
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto bin_index_flat = bin_index_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            Tensor* grad_filter_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,N,C}, &grad_input_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{F,C,r}, &grad_filter_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();
            auto grad_filter_flat = grad_filter_tensor->flat<float>();

            const float* in = &(input_flat(0));
            const float* filt = &(filter_flat(0));
            const float* gradOut = &(grad_output_flat(0));
            const int* nnIndex = &(nn_index_flat(0));
            const int* nnCount = &(nn_count_flat(0));
            const int* binIndex = &(bin_index_flat(0));

            float* gradIn = &(grad_input_flat(0));
            float* gradFilt = &(grad_filter_flat(0));
            cudaMemset(gradIn,0,sizeof(float)*B*N*C);
            cudaMemset(gradFilt,0,sizeof(float)*F*C*r);
            depthwiseConv3dGradLauncher(B, N, M, F, C, r, K, nnIndex, nnCount, binIndex,
                                        in, filt, gradOut, gradIn, gradFilt);
        }
};
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv3dGrad").Device(DEVICE_GPU), DepthwiseConv3dGradGpuOp);





