#include <iostream>
#include <cstdio>
using namespace std;
#include <cuda_runtime.h>
#define TIMES 24


#include<sm_35_intrinsics.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////HELP FUNCTIONS/////////////////////////////////////////////////
void RandomInit(float* data, int n)
{
    for (int i=0; i<n; i++)
	{
        data[i] = rand() / (float)RAND_MAX;
	}
}

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////_VECTOR_ADDITION_///////////////////////////////////////////////////////
__global__ void triad(const float* A, const float* B, float* C, int scale)

{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

        C[i] = A[i] + scale * B[i];

}


// Host code
void VectorAddition(int N, int threadsPerBlock, int compute, int scale)
{
	cout<<"Vector Addition for input size "<<N<<" :\n";
	// Variables
	float* h_A;
	float* h_B;
	float* h_C;
        float* h_D;
	float* d_A;
	float* d_B;
	float* d_C;
        float* d_D;
	float total_time=0;
    size_t size = N * sizeof(float) * scale;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_D = (float*)malloc(size);
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        cout<<"deviceIndex"<<deviceIndex<<", SM = "<<deviceProperties.major<<"."<<deviceProperties.minor<<"L1 cache: " <<deviceProperties.globalL1CacheSupported<<endl;
        if (deviceProperties.major >= compute
            && deviceProperties.minor >= 0)
        {
            cout<<"Set device to "<<deviceIndex<<endl;
            cudaSetDevice(deviceIndex);
        }
    }

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&d_A, size) );
    checkCudaErrors( cudaMalloc((void**)&d_B, size) );
    checkCudaErrors( cudaMalloc((void**)&d_C, size) );
    checkCudaErrors( cudaMalloc((void**)&d_D, size) );

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );
 
    checkCudaErrors(cudaDeviceSynchronize());
    // Invoke kernel
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	for (int i = 0; i < 1; i++) {

    triad<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, scale);

	checkCudaErrors(cudaDeviceSynchronize());
	}

	double dSeconds = total_time/((double)TIMES * 1000);
	double dNumOps = N;
	double gflops = 1.0e-9 * dNumOps/dSeconds;
	cout<<"Time = "<<dSeconds*1.0e3<< "msec"<<endl<<"gflops = "<<gflops<<endl;

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
    
    // Verify result
    int i;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }

        // Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    cudaDeviceReset();

	if(i == N)
		cout<<"SUCCSESS"<<endl;
	else 
		cout<<"FAILED"<<endl;   
}
//////////////////////////////////////////////////////
int main(int argc,char *argv[])
{ 
  if(argc < 4)
     printf("Unsuffcient number of arguments!\n");
else
	{
		VectorAddition(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
	}
}
