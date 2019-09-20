// nnIndex: B*N*K;
// nnCount: B*N;
// input:   B*M*C;
// output:  B*N*C (N>M)
__global__ void mean_interpolate_forward(int B, int N, int M, int C, int K, const int* nnIndex,
                                         const int* nnCount, const float* input, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                output[i*N*C+j] += input[i*M*C+m*C+c]/nnSize;
            }
        }
    }
}


__global__ void mean_interpolate_backward(int B, int N, int M, int C, int K, const int* nnIndex,
                                          const int* nnCount, const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                atomicAdd(&gradInput[i*M*C+m*C+c],gradOutput[i*N*C+j]/nnSize);
            }
        }
    }
}


__global__ void weighted_interpolate_forward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                             const float* input, const float* weight, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                float w = weight[i*N*K+n*K+k];
                output[i*N*C+j] += input[i*M*C+m*C+c]*w;
            }
        }
    }
}


__global__ void weighted_interpolate_backward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                              const float* gradOutput, const float* weight, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                float w = weight[i*N*K+n*K+k];
                atomicAdd(&gradInput[i*M*C+m*C+c],gradOutput[i*N*C+j]*w);
            }
        }
    }
}


void meanInterpolateLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                             const float* input, float* output)
{
    mean_interpolate_forward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, input, output);
    cudaDeviceSynchronize();
}

void meanInterpolateGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, float* gradInput)
{
    mean_interpolate_backward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, gradOutput, gradInput);
    cudaDeviceSynchronize();
}

void weightedInterpolateLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                 const float* input, const float* weight, float* output)
{
    weighted_interpolate_forward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, input, weight, output);
    cudaDeviceSynchronize();
}

void weightedInterpolateGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                     const float* gradOutput, const float* weight, float* gradInput)
{
    weighted_interpolate_backward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, gradOutput, weight, gradInput);
    cudaDeviceSynchronize();
}