// nnIndex: B*M*K;
// nnCount: B*M;
// input:   B*N*C;
// output:  B*M*C (M<N)
__global__ void max_pool3d_forward(int B, int N, int M, int C, int K, const int* nnIndex,
                                   const int* nnCount, const float* input, float* output, int* maxIndex)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
        {
            int m = j/C;
            int c = j%C;
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];
                if (k==0)
                {
                    output[i*M*C+j] = input[i*N*C+n*C+c];
                    maxIndex[i*M*C+j] = n;
                    continue;
                }

                if (input[i*N*C+n*C+c]>output[i*M*C+j])
                {
                    output[i*M*C+j] = input[i*N*C+n*C+c];
                    maxIndex[i*M*C+j] = n;
                }
            }
        }
    }
}


// maxIndex: B*M*C, indices of the maximum feature point
__global__ void max_pool3d_backward(int B, int N, int M, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
        {
            int c = j%C;
            int n = maxIndex[i*M*C+j];
            atomicAdd(&gradInput[i*N*C+n*C+c],gradOutput[i*M*C+j]);
        }
    }
}


__global__ void avg_pool3d_forward(int B, int N, int M, int C, int K, const int* nnIndex,
                                   const int* nnCount, const float* input, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
        {
            int m = j/C;
            int c = j%C;
            int nnSize = nnCount[i*M+m];
            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];
                output[i*M*C+j] += input[i*N*C+n*C+c]/nnSize;
            }
        }
    }
}


__global__ void avg_pool3d_backward(int B, int N, int M, int C, int K, const int* nnIndex,
                                   const int* nnCount, const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
        {
            int m = j/C;
            int c = j%C;
            int nnSize = nnCount[i*M+m];
            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k]; // only neighbor, no bin indices, dimension=(B,M,K)
                atomicAdd(&gradInput[i*N*C+n*C+c],gradOutput[i*M*C+j]/nnSize);
            }
        }
    }
}


void maxPool3dLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       const float* input, float* output, int* maxIndex)
{
    max_pool3d_forward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, input, output, maxIndex);
    cudaDeviceSynchronize();
}

void maxPool3dGradLauncher(int B, int N, int M, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput)
{
    max_pool3d_backward<<<32,1024>>>(B, N, M, C, maxIndex, gradOutput, gradInput);
    cudaDeviceSynchronize();
}

void avgPool3dLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       const float* input, float* output)
{
    avg_pool3d_forward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, input, output);
    cudaDeviceSynchronize();
}

void avgPool3dGradLauncher(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, float* gradInput)
{
    avg_pool3d_backward<<<32,1024>>>(B, N, M, C, K, nnIndex, nnCount, gradOutput, gradInput);
    cudaDeviceSynchronize();
}