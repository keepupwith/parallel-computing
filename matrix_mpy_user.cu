#include <iostream>
#include "ee155_utils.hxx"
#include "matrix.hxx"
using namespace std;
const int BS = 32;	// The blocks are BS x BS.


///////////////////////////////
// This is the CUDA kernel function for you to write.
//
__global__ void mat_mult (float *d_A, float *d_B, float *d_C, int N) {
    int rb = blockIdx.x;
    int cb = blockIdx.y;
    int ri = threadIdx.x;
    int ci = threadIdx.y;
    
    __shared__ float SA[BS][BS], SB[BS][BS];
    //printf("In thread with r=(%d,%d) c=(%d,%d)\n", rB,rI,cB,cI);
    // Copy the data to shared memory
    for (int kb = 0; kb < gridDim.x; kb++) {
        SA[ri][ci] = d_A[N*(rb*BS+ri)+kb*BS+ci];
        SB[ri][ci] = d_B[N*(kb*BS+ri)+cb*BS+ci];
        __syncthreads();

        // Do actual computations
        for (int ki = 0; ki < BS; ki++) {
            d_C[N*(rb*BS+ri)+cb*BS+ci] += SA[ri][ki] * SB[ki][ci];
        }
        __syncthreads();
    }
    

}



///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
void Matrix::mpy1 (const Matrix &A, const Matrix &B, int BS) {

    // Copy A from host memory to device memory.
    int numElem=N()*N(), sizeBytes = numElem*4;
    float *d_A = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix A");

    err = cudaMemcpy (d_A, A.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix A from host to device");

    // Allocate device memory for B.
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix B");

    err = cudaMemcpy (d_B, B.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix B from host to device");

    // Allocate device memory for C.
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix C");

    // Set C to all zeroes
    err = cudaMemset(d_C, 0, sizeBytes);
    ERR_CHK (err, "Failed to set matrix C to zero");
    int N=A.N();
    int Nb=A.N()/BS;

    dim3 gridSize(Nb, Nb);
    dim3 blockSize(BS, BS);
    // Launch the CUDA Kernel
    mat_mult <<< gridSize, blockSize >>> (d_A, d_B, d_C, N);
    // err = cudaGetLastError();
    // ERR_CHK (err, "Failed to launch matrix multiplication kernel");

    // Copy the result from device memory to host memory.
    err = cudaMemcpy (data.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);
    ERR_CHK (err, "Failed to copy matrix C from device to host");

    // Free device memory.
    err = cudaFree(d_A);
    ERR_CHK (err, "Failed to free CUDA matrix A");
}