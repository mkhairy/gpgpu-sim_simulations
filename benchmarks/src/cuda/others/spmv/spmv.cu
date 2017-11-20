#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>

using namespace std; 

static const int BLOCK_SIZE = 128; 
static const int WARP_SIZE = 32;
static const double MAX_RELATIVE_ERROR = .02;
static const int TEMP_BUFFER_SIZE = 1024;  

void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (int i = 0; i < dim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}
template <typename floatType>
void fill(floatType *A, const int n, const float maxi)
{
    for (int j = 0; j < n; j++) 
    {
        A[j] = ((floatType) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}


// Forward declarations for kernels
template <typename fpType>
__global__ void 
spmv_csr_scalar_kernel(const fpType *  val,
                       const int    *  cols,
                       const int    *  rowDelimiters,
                       const int dim, fpType * out,
const fpType *  vec);



// ****************************************************************************
// Function: spmvCpu
//
// Purpose: 
//   Runs sparse matrix vector multiplication on the CPU 
//
// Arguements: 
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A; 
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation 
// 
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************
template <typename floatType>
void spmvCpu(const floatType *val, const int *cols, const int *rowDelimiters, 
	     const floatType *vec, int dim, floatType *out) 
{
    for (int i=0; i<dim; i++) 
    {
        floatType t = 0; 
        for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        {
            int col = cols[j]; 
            t += val[j] * vec[col];
        }    
        out[i] = t; 
    }
}


template <typename floatType>
bool verifyResults(const floatType *cpuResults, const floatType *gpuResults,
                   const int size, const int pass = -1) 
{
    bool passed = true; 
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] 
            > MAX_RELATIVE_ERROR) 
        {
//            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
//                " dev: " << gpuResults[i] << endl;
            passed = false;
        }
    }
    if (pass != -1) 
    {
        cout << "Pass "<<pass<<": ";
    }
    if (passed) 
    {
        cout << "Passed" << endl;
    }
    else 
    {
        cout << "---FAILED---" << endl;
    }
    return passed;
}

template <typename floatType>
void csrTest(floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded)
{
      // Device data structures
      floatType *d_val, *d_vec, *d_out;
      int *d_cols, *d_rowDelimiters;

      // Allocate device memory
      cudaMalloc(&d_val,  numNonZeroes * sizeof(floatType));
      cudaMalloc(&d_cols, numNonZeroes * sizeof(int));
      cudaMalloc(&d_vec,  numRows * sizeof(floatType));
      cudaMalloc(&d_out,  numRows * sizeof(floatType));
      cudaMalloc(&d_rowDelimiters, (numRows+1) * sizeof(int));

      // Setup events for timing
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      // Transfer data to device
      cudaEventRecord(start, 0);
      cudaMemcpy(d_val, h_val,   numNonZeroes * sizeof(floatType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_cols, h_cols, numNonZeroes * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_rowDelimiters, h_rowDelimiters,(numRows+1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);

      float iTransferTime, oTransferTime;
      cudaEventElapsedTime(&iTransferTime, start, stop);
      iTransferTime *= 1.e-3;


      // Setup thread configuration
      int nBlocksScalar = (int) ceil((floatType) numRows / BLOCK_SIZE);
      int nBlocksVector = (int) ceil(numRows /
                  (floatType)(BLOCK_SIZE / WARP_SIZE));
      int passes = 1;
      int iters  = 1;

      // Results description info
      char atts[TEMP_BUFFER_SIZE];
     // sprintf(atts, "%d_elements_%d_rows", numNonZeroes, numRows);
      string prefix = "";
      prefix += (padded) ? "Padded_" : "";
      double gflop = 2 * (double) numNonZeroes / 1e9;
      cout << "CSR Scalar Kernel\n";
      for (int k=0; k<passes; k++)
      {
          // Run Scalar Kernel
          cudaEventRecord(start, 0);
          for (int j = 0; j < iters; j++)
          {
              spmv_csr_scalar_kernel<floatType>
              <<<nBlocksScalar, BLOCK_SIZE>>>
              (d_val, d_cols, d_rowDelimiters, numRows, d_out, d_vec);
          }
          cudaEventRecord(stop, 0);
          cudaEventSynchronize(stop);
          float scalarKernelTime;
          cudaEventElapsedTime(&scalarKernelTime, start, stop);
          // Transfer data back to host
          cudaEventRecord(start, 0);
          cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),cudaMemcpyDeviceToHost);
          cudaEventRecord(stop, 0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&oTransferTime, start, stop);
          oTransferTime *= 1.e-3;
          // Compare reference solution to GPU result
         // if (! verifyResults(refOut, h_out, numRows, k))
         // {
         //     return;  // If results don't match, don't report performance
         // }
          scalarKernelTime = (scalarKernelTime / (float)iters) * 1.e-3;
          string testName = prefix+"CSR-Scalar";
          double totalTransfer = iTransferTime + oTransferTime;
      }
      cudaThreadSynchronize();

     
      // Free device memory
      cudaFree(d_rowDelimiters);
      cudaFree(d_vec);
      cudaFree(d_out);
      cudaFree(d_val);
      cudaFree(d_cols);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
}


template <typename floatType>
void RunTest(int nRows=0) 
{
    // Host data structures
    // Array of values in the sparse matrix
    floatType *h_val;
    // Array of column indices for each value in h_val
    int *h_cols;
    // Array of indices to the start of each row in h_Val
    int *h_rowDelimiters;
    // Dense vector and space for dev/cpu reference solution
    floatType *h_vec, *h_out, *refOut;
    // nItems = number of non zero elems
    int nItems, numRows;

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix

        numRows = nRows;
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        cudaMallocHost(&h_val, nItems * sizeof(floatType)); 
        cudaMallocHost(&h_cols, nItems * sizeof(int)); 
        cudaMallocHost(&h_rowDelimiters, (numRows + 1) * sizeof(int)); 
        fill(h_val, nItems, 10); 
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);

    // Set up remaining host data
    cudaMallocHost(&h_vec, numRows * sizeof(floatType)); 
    refOut = new floatType[numRows];
    fill(h_vec, numRows, 10);

    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);
	
    // Test CSR kernels on normal data
    cout << "CSR Test\n";
    csrTest<floatType>(h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false);


    delete[] refOut; 
    cudaFreeHost(h_val); 
    cudaFreeHost(h_cols); 
    cudaFreeHost(h_rowDelimiters);
    cudaFreeHost(h_vec); 
    cudaFreeHost(h_out);   
}

int main()
{
   
    int probSizes[5] = {1024, 8192, 12288, 16384,32768};
    int sizeClass = 32768;

    cout <<"Single precision tests:\n";
    RunTest<float>(sizeClass);
    return 0;
    
}

template <typename fpType>
__global__ void 
spmv_csr_scalar_kernel(const fpType *  val,
                       const int    *  cols,
                       const int    *  rowDelimiters,
                       const int dim, fpType *  out,
 			const fpType  *  vec)
{
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;

    if (myRow < dim) 
    {
        fpType t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j]; 
	    t += val[j] * vec[col];
        }
        out[myRow] = t; 
    }
}


