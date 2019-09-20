#ifndef M_PI
#define M_PI           3.141592653589793F  /* pi */
#endif

#ifndef M_EPS
#define M_EPS          1.01e-3F             /* epsilon */
#endif

struct point3d
{
    float x=0, y=0, z=0;
};

// database:  B*N*3, (x,y,z)
// query:     B*M*3, (x,y,z)
// nnIndex:   B*M*K
// nnCount:   B*M
// nnDist:    B*M*K
// filtIndex: B*M*K
__global__ void build_spherical_kernel(const int B, const int N, const int M, const int K,
                                             const int n, const int p, const int q, const float radius,
                                             const float* database, const float* query, const int* nnIndex,
                                             const int* nnCount, const float* nnDist, int* filtIndex)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int nnSize = nnCount[i*M+j];
            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[i*M*K+j*K+k];   // input point ID

                pt.x = database[i*N*3+ptID*3];
                pt.y = database[i*N*3+ptID*3+1];
                pt.z = database[i*N*3+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist = nnDist[i*M*K+j*K+k];
                float dist2D = delta.x*delta.x + delta.y*delta.y;
                dist2D = sqrtf(dist2D);

                filtIndex[i*M*K+j*K+k] = 0;
                if (dist>M_EPS && fabs(dist-M_EPS)>1e-6) // update the bin index
                {
                    float theta = atan2f(delta.y, delta.x);
                    float phi = atan2f(delta.z, dist2D);

                    theta = theta<M_PI?theta:(-M_PI);
                    theta = theta>(-M_PI)?theta:(-M_PI);
                    theta += M_PI;

                    phi = phi<(M_PI/2)?phi:(M_PI/2);
                    phi = phi>(-M_PI/2)?phi:(-M_PI/2);
                    phi += M_PI/2;

                    float alpha = theta*n/2/M_PI;
                    float beta = phi*p/M_PI;
                    float gamma = dist*q/(radius+1e-6F);

                    int nID = min(n-1, int(alpha));
                    int pID = min(p-1, int(beta));
                    int qID = min(q-1, int(gamma));

                    filtIndex[i*M*K+j*K+k] = qID*p*n + pID*n + nID + 1;
                }
            }
        }
    }
}



void sphericalKernelLauncher(int B, int N, int M, int K, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const int* nnCount, const float* nnDist, int* filtIndex)
{
    build_spherical_kernel<<<32,1024>>>(B, N, M, K, n, p, q, radius,
                                database, query, nnIndex, nnCount, nnDist, filtIndex);
}

