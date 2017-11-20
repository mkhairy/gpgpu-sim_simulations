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

void RandomInit_int(int* data, int n, int max)
{
	srand (time(NULL));
    for (int i=0; i<n; i++)
	{
        data[i] =  rand() % max;
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

__global__ void locality_sector_nocont_variable(const float* A, const int* B, float* C, int N)

{

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int sum=0;



	for(int j =i*32; j<(i*32)+B[i]; j++)

		sum += A[j];

		

	C[i] = sum;



}

// Host code
void VectorAddition(int N, int threadsPerBlock, int compute, int scale)
{
	cout<<"Vector Addition for input size "<<N<<" :\n";
	// Variables
	float* h_A;
	int* h_B;
	float* h_C;

	float* d_A;
	int* d_B;
	float* d_C;

	float total_time=0;
    size_t size = N * sizeof(float) * scale;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (float*)malloc(size);
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit_int(h_B, N, 128);

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

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );
 
    checkCudaErrors(cudaDeviceSynchronize());
    // Invoke kernel
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	for (int i = 0; i < 1; i++) {

	locality_sector_nocont_variable<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //getLastCudaError("kernel launch failure");
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
