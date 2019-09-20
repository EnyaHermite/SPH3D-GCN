#ifndef M_PI
#define M_PI           3.14159265358979323846F  /* pi */
#endif

struct point3d
{
    float x=0, y=0, z=0;
};

// database:  B*N*3
// query:     B*M*3
// nnIndex:   B*M*nnSample
// nnCount:   B*M
// nnDist:    B*M*nnSample
__global__ void cal_nn_binidx(int B, int N, int M, int nnSample, float radius,
                              const float* database, const float* query,
                              int* nnIndex, int* nnCount, float* nnDist)
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

            int s=0; // to count the number of neighbors
            while(s==0) //require a minimum of 1 neighbor point
            {
                //re-initialziation
                s = 0;

                for(int k=0;k<N;k++)
                {
                    pt.x = database[i*N*3+k*3];
                    pt.y = database[i*N*3+k*3+1];
                    pt.z = database[i*N*3+k*3+2];

                    delta.x = pt.x - ptQuery.x;
                    delta.y = pt.y - ptQuery.y;
                    delta.z = pt.z - ptQuery.z;

                    float dist2D = delta.x*delta.x + delta.y*delta.y; // squared 2D
                    float dist3D = dist2D + delta.z*delta.z; // squared 3D
                    dist3D = sqrtf(dist3D); //sqrt

                    if (dist3D<radius && fabs(dist3D-radius)>1e-6) // find a neighbor in range
                    {
                        if (s<nnSample) // sample NO=nnSample neighbor points, requires shuffling of points order in every epoch
                        {
                            nnIndex[i*M*nnSample+j*nnSample+s] = k;
                            nnDist[i*M*nnSample+j*nnSample+s] = sqrt(dist3D); // sqrt, not the squared one
                        }
                        s++;
                    }
                }
                radius += 0.05;
            }

            nnCount[i*M+j] = s<nnSample?s:nnSample;
        }
    }
}


// database:  B*N*3
// query:    B*M*3
// nnIndex:  B*M*nnSample*2
// nnCount:  B*M
__global__ void cal_nn_binidx_cube(int B, int N, int M, int gridSize, int nnSample, float length,
                              const float* database, const float* query, int* nnIndex, int* nnCount)
{
    // get the neighbor indices, and compute their indices in the filter/kernel bins
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int s = 0; // to count the number of neighbors
            for(int k=0;k<N;k++)
            {
                pt.x = database[i*N*3+k*3];
                pt.y = database[i*N*3+k*3+1];
                pt.z = database[i*N*3+k*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                if (abs(delta.x)<length/2 && abs(delta.y)<length/2 && abs(delta.z)<length/2 && s<nnSample)
                {
                    // calculate bin index in the cubic filter/kernel
                    int xId = (delta.x + length/2)/(length/gridSize); //[0, gridSize)
                    int yId = (delta.y + length/2)/(length/gridSize); //[0, gridSize)
                    int zId = (delta.z + length/2)/(length/gridSize); //[0, gridSize)

                    int binID = xId*gridSize*gridSize + yId*gridSize + zId;

                    nnIndex[i*M*nnSample*2+j*nnSample*2+s*2] = k;
                    nnIndex[i*M*nnSample*2+j*nnSample*2+s*2+1] = binID;
                    s++;
                }
            }
            nnCount[i*M+j] = s;
        }
    }
}

void buildSphereNeighborLauncher(int B, int N, int M, int nnSample, float radius,
                                 const float* database, const float* query, int* nnIndex,
                                 int* nnCount, float* nnDist)
{
    cal_nn_binidx<<<32,1024>>>(B, N, M, nnSample, radius,
                               database, query, nnIndex, nnCount, nnDist);
}

void buildCubeNeighborLauncher(int B, int N, int M, int gridSize, int nnSample, float length,
                                 const float* database, const float* query, int* nnIndex, int* nnCount)
{
    cal_nn_binidx_cube<<<32,1024>>>(B, N, M, gridSize, nnSample, length, database, query, nnIndex, nnCount);
}