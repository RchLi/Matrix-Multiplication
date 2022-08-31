// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

__global__ void matMul(int N, _DOUBLE_ * __restrict__ C, _DOUBLE_ * __restrict__ A, _DOUBLE_ *__restrict__ B)
{
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int I = by*TW + ty;   
    int J = bx*TW + tx;

    if(N != 256){
    __shared__ double As[TW][TW];
    double Cs1[TW], Cs2[TW];
    double b1, b2;
    int tid = ty * blockDim.y + tx;

    // initialize Cs
    #pragma unroll
    for(int k = 0; k < TW; k++){
        Cs1[k] = 0;
        Cs2[k] = 0;
    }

  
    for(int m = 0; m < N/TW; m++){
        // load 16*16 A into shared memory with 8*8 thread block
        for(int i = 0; i < TW; i += blockDim.y){
            for(int j = 0; j < TW; j += blockDim.x){
                As[ty + i][tx + j] = A[(I + i) * N +  m * TW + tx + j];
            }
        }
        // As[ty][tx] = A[(I) * N +  m * TW + tx];
        __syncthreads();
        
        // perform block matriplication: A(I, m) * B(m, J)
        #pragma unroll
        for(int kk = 0; kk < TW; kk++){
            // load b into thread register
            b1 = B[(m * TW + kk) * N + 2 * bx * TL  + tid]; 
            b2 = B[(m * TW + kk) * N + (2 * bx + 1) * TL  + tid]; 

            #pragma unroll
            for(int k = 0; k < TW; k++){
                Cs1[k] +=  As[k][kk] * b1;
                Cs2[k] +=  As[k][kk] * b2;
            }
           
        }
        __syncthreads();
    }
    
    #pragma unroll
    for(int k = 0; k < TW; k++){
        C[(by * TW + k) * N + 2 * bx * TL + tid] = Cs1[k];
        C[(by * TW + k) * N + (2 * bx + 1) * TL + tid] = Cs2[k];
    }
    }
    else{
    __shared__ double As[TW][TW], Bs1[TW][TW], Bs2[TW][TW], Bs3[TW][TW], Bs4[TW][TW];
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (int kk=0; kk<N/TW; kk++){
        As[ty][tx] = A[I*N + kk*TW+tx];
        Bs1[ty][tx] = B[(kk*TW+ty)*N + 4 * bx * TW + tx];
        Bs2[ty][tx] = B[(kk*TW+ty)*N + (4 * bx + 1) * TW + tx];
        Bs3[ty][tx] = B[(kk*TW+ty)*N + (4 * bx + 2) * TW + tx];
        Bs4[ty][tx] = B[(kk*TW+ty)*N + (4 * bx + 3) * TW + tx];
        __syncthreads();

        for (int k=0; k<TW; k++){
            c1 += As[ty][k] * Bs1[k][tx];
            c2 += As[ty][k] * Bs2[k][tx];
            c3 += As[ty][k] * Bs3[k][tx];
            c4 += As[ty][k] * Bs4[k][tx];
        }    
        __syncthreads();
    }
    C[I*N + 4 * bx * TW + tx] = c1;
    C[I*N + (4 * bx + 1) * TW + tx] = c2;
    C[I*N + (4 * bx + 2) * TW + tx] = c3;
    C[I*N + (4 * bx + 3) * TW + tx] = c4;
    }
}

   