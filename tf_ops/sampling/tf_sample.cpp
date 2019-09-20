/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });


void farthestPointSampleLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestPointSampleLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);


