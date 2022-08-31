#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <immintrin.h>
#include <avx2intrin.h>

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,  // KC
		        //    int    m, // MR
                //    int    n, // NR
                   const double * restrict a,
                   const double * restrict b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < DGEMM_NR; ++j )
        { 
            for ( i = 0; i < DGEMM_MR; ++i )
            { 
	      // ldc is used here because a[] and b[] are not packed by the
	      // starter code
	      // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	      /* A:  a b c d   ->  a c e g  
                 e f g h       b d f h
            B: a e  
               b f
               c g
               d h
          */
          // TODO: change matrix change
            
	      c( i, j, ldc ) += a( i, l, DGEMM_KC) * b( l, j, DGEMM_NR );   
            }
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
// #define debug


inline void simd_4x12_ukr2(
                int    kc,
                int    m,
                int    n,
                const double * restrict A,
                const double * restrict B,
                double * C,
                unsigned long long ldc,
                aux_t* data
){
    register __m256d c00, c04, c08,
                     c10, c14, c18,
                     c20, c24, c28,
                     c30, c34, c38;   // 12 c_reg
    register __m256d a;  // 1 a_reg, to load with broadcast
    register __m256d b0, b1, b2; // 3 b_reg

    // load, each c_reg holds 4 doubles
    c00 = _mm256_loadu_pd(C + 0 * ldc + 0);
    c04 = _mm256_loadu_pd(C + 0 * ldc + 4);
    c08 = _mm256_loadu_pd(C + 0 * ldc + 8);

    c10 = _mm256_loadu_pd(C + 1 * ldc + 0);
    c14 = _mm256_loadu_pd(C + 1 * ldc + 4);
    c18 = _mm256_loadu_pd(C + 1 * ldc + 8);

    c20 = _mm256_loadu_pd(C + 2 * ldc + 0);
    c24 = _mm256_loadu_pd(C + 2 * ldc + 4);
    c28 = _mm256_loadu_pd(C + 2 * ldc + 8);

    c30 = _mm256_loadu_pd(C + 3 * ldc + 0);
    c34 = _mm256_loadu_pd(C + 3 * ldc + 4);
    c38 = _mm256_loadu_pd(C + 3 * ldc + 8);
    
    // calculation with 4-step loop unrolling
    for(int i_kernel = 0; i_kernel < kc; i_kernel++){
        // loop 0
        // broadcast a_reg
        a = _mm256_broadcast_sd(A + i_kernel * DGEMM_MR + 0);
        // load b
        b0 = _mm256_loadu_pd(B + i_kernel * DGEMM_NR + 0);
        b1 = _mm256_loadu_pd(B + i_kernel * DGEMM_NR + 4);
        b2 = _mm256_loadu_pd(B + i_kernel * DGEMM_NR + 8);
        // fmadd
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c04 = _mm256_fmadd_pd(a, b1, c04);
        c08 = _mm256_fmadd_pd(a, b2, c08);

        // loop 1
        a = _mm256_broadcast_sd(A + i_kernel * DGEMM_MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c14 = _mm256_fmadd_pd(a, b1, c14);
        c18 = _mm256_fmadd_pd(a, b2, c18);

        // loop 2
        a = _mm256_broadcast_sd(A + i_kernel * DGEMM_MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c24 = _mm256_fmadd_pd(a, b1, c24);
        c28 = _mm256_fmadd_pd(a, b2, c28);

        // loop 3
        a = _mm256_broadcast_sd(A + i_kernel * DGEMM_MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c34 = _mm256_fmadd_pd(a, b1, c34);
        c38 = _mm256_fmadd_pd(a, b2, c38);
    }

    // store from register C to packC
    _mm256_storeu_pd(C + 0 * ldc + 0, c00);
    _mm256_storeu_pd(C + 0 * ldc + 4, c04);
    _mm256_storeu_pd(C + 0 * ldc + 8, c08);

    _mm256_storeu_pd(C + 1 * ldc + 0, c10);
    _mm256_storeu_pd(C + 1 * ldc + 4, c14);
    _mm256_storeu_pd(C + 1 * ldc + 8, c18);

    _mm256_storeu_pd(C + 2 * ldc + 0, c20);
    _mm256_storeu_pd(C + 2 * ldc + 4, c24);
    _mm256_storeu_pd(C + 2 * ldc + 8, c28);

    _mm256_storeu_pd(C + 3 * ldc + 0, c30);
    _mm256_storeu_pd(C + 3 * ldc + 4, c34);
    _mm256_storeu_pd(C + 3 * ldc + 8, c38);

    
}