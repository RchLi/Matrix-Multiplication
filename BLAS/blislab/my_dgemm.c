/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order  
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"
#include "bl_config.h"

#include "../debugMat.cpp"
const char *dgemm_desc = "my blislab ";

/* 
 * pack one subpanel of A
 *
 * pack like this 
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *     
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 */
static inline void packA_mcxkc_d(
    int m, // MR
    int k, // KC
    double *XA,
    int ldXA, // length of A
    double *packA)
{
  int i, j;
  for (i = 0; i < m; i += 1)
  {
    for (j = 0; j < k; j += 1)
    {
       // l = j + i * KC
      // packA[j, i] = XA[i, j]
      packA[j * DGEMM_MR + i] = XA[i * ldXA + j];
    }
  }
  // padding
  // for(i = 0; i < DGEMM_KC; i++){
  //   for(j = m; j < DGEMM_MR; j++){
  //     packA[i * DGEMM_MR + j] = 0;
  //   }
  // }
  // for(i = k; i < DGEMM_KC; i++){
  //   for(j = 0; j < DGEMM_MR; j++){
  //     packA[i * DGEMM_MR + j] = 0;
  //   }
  // }

    for(i = 0; i < k; i++){
      for(j = m; j < DGEMM_MR; j++){
        packA[i * DGEMM_MR + j] = 0;
      }
    }

  // // printMat(ldXA, ldXA, "print matrixA", XA);
  // printMat(DGEMM_MR, DGEMM_KC, "printPaddedA", packA);
}

/*
 * --------------------------------------------------------------------------
 */

/* 
 * pack one subpanel of B
 * 
 * pack like this 
 * if B is 
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 *
 * Then pack 
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */

static inline void packB_kcxnc_d(
    int n, // NR +4
    int k, // min(k - pc, KC)
    double *XB,
    int ldXB, // ldXB is the original k
    double *packB)
{
  int i, j;
  for (i = 0; i < k; i += 1)
  {
    for (j = 0; j < n; j += 1)
    {
      packB[i * DGEMM_NR  + j] = XB[i * ldXB + j];
    }
  }
  //padding
  // for(i = 0; i < DGEMM_KC; i++){
  //   for(j = n; j < DGEMM_NR; j++){
  //     packB[i * DGEMM_NR  + j] = 0;
  //   }
  // }
  // for(i = k; i < DGEMM_KC; i++){
  //   for(j = 0; j < DGEMM_NR; j++){
  //     packB[i * DGEMM_NR  + j] = 0;
  //   }
  // }

  for(i = 0; i < k; i++){
    for(j = n; j < DGEMM_NR; j++){
      packB[i * DGEMM_NR  + j] = 0;
    }
  }
}

static inline void packC_mcnc(int m, int n, int ldxc, int ldpackc, double *XC, double *packC){
  int i, j;

  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      packC[i * ldpackc + j] = XC[i * ldxc + j];
    }
  }

  //pad
  for(i = m; i < DGEMM_MR; i++){
    for(j = 0; j < DGEMM_NR; j++){
      packC[i * ldpackc + j] = 0;
    }
  }
  for(i = 0; i < DGEMM_MR; i++){
    for(j = n; j < DGEMM_NR; j++){
      packC[i * ldpackc + j] = 0;
    }
  }
}

// static inline void writeC(int m, int n, int ldxc, int ldpackc, double *XC, double *packC){
//   int i, j;

//   for(i = 0; i < m; i++){
//     for(j = 0; j < n; j++){
//       XC[i * ldxc + j] = packC[i * DGEMM_NR + j];
//     }
//   }
// }




/*
 * --------------------------------------------------------------------------
 */

static inline void bl_macro_kernel(
    int m,
    int n,
    int k, //min(k-pc, KC)
    const double *packA,
    const double *packB,
    double *C,
    int ldc)
{
  int i, j;
  aux_t aux;

  for (i = 0; i < m; i += DGEMM_MR)
  { // 2-th loop around micro-kernel
    for (j = 0; j < n; j += DGEMM_NR)
    { // 1-th loop around micro-kernel

      (*bl_micro_kernel)(
          k,
          min(m - i, DGEMM_MR),
          min(n - j, DGEMM_NR),
          // DGEMM_KC,
          // DGEMM_MR,
          // DGEMM_NR,
          &packA[ i * DGEMM_KC ], // assumes sq matrix, otherwise use lda
          &packB[ j * DGEMM_KC ],       //
          &C[i * ldc + j],
          (unsigned long long)ldc,
          &aux);
    } // 1-th loop around micro-kernel
  }   // 2-th loop around micro-kernel
}

// TODO: pading to the matrix in each block
void bl_dgemm(
    int m,
    int n,
    int k,
    double *XA, // square matrix
    int lda,    // length of XA
    double *XB,
    int ldb,
    double *C,
    int ldc)
{
  int ic, ib, jc, jb, pc, pb;
  double *packA, *packB, *packC;

  // Allocate packing buffers
  //
  // FIXME undef NOPACK when you implement packing
  //
#define NOPACK
#undef NOPACK

#ifndef NOPACK
  // allocation some space, init packA, packB
  // Mc = 8, Mr = 2, (DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR = (8 / 3)floor * 2 = 4
  packA = bl_malloc_aligned(DGEMM_KC, (DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR, sizeof(double));
  packB = bl_malloc_aligned(DGEMM_KC, (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR, sizeof(double));
  packC = bl_malloc_aligned((DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR, (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR, sizeof(double));
#endif
  // m = 7, DGEMM_MC = 2, ib = 5, 2; 3, 4; 1, 6
  // => 2, 3, 1ƒ
  // 7 * 7 => 8 * 8

  for (ic = 0; ic < m; ic += DGEMM_MC)
  {                             // 5-th loop around micro-kernel
    // printf("5-th loop\n");
    ib = min(m - ic, DGEMM_MC); // align
    for (pc = 0; pc < k; pc += DGEMM_KC)
    { // 4-th loop around micro-kernel
      // printf("4-th loop\n");
      pb = min(k - pc, DGEMM_KC);

#ifdef NOPACK
      packA = &XA[pc + ic * lda];
#else
      int i, j;
      for (i = 0; i < ib; i += DGEMM_MR)
      {
        packA_mcxkc_d(
            min(ib - i, DGEMM_MR),    /* m */
            pb, /* k */               // normally DGEMM_KC
            &XA[pc + lda * (ic + i)], /* XA - start of micropanel in A */
            k,                        /* ldXA */
            &packA[0 * DGEMM_MC * pb + i * DGEMM_KC] /* packA */);
      }
#endif
      for (jc = 0; jc < n; jc += DGEMM_NC)
      { // 3-rd loop around micro-kernel
      // printf("3-rd loop\n");
        jb = min(m - jc, DGEMM_NC);

#ifdef NOPACK
        packB = &XB[ldb * pc + jc];
#else
        for (j = 0; j < jb; j += DGEMM_NR)
        {
          packB_kcxnc_d(
              min(jb - j, DGEMM_NR) /* n */,
              pb /* k */,
              &XB[ldb * pc + jc + j] /* XB - starting row and column for this panel */,
              n,             // should be ldXB instead /* ldXB */
              &packB[j * DGEMM_KC] /* packB */
          );


        }
#endif
        //packC_mcnc(ib, jb, ldc, &C[ic * ldc + jc], packC);
        
        int m_packc = (DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR;
        int n_packc =  (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR;
        for (i = 0; i < ib; i++){
          for(j = 0; j < jb; j++){
            packC[i * n_packc + j] = C[(ic + i) * ldc + jc + j];
          }
        }
        for(i = ib; i < m_packc; i++){
          for(j = 0; j < n_packc; j++){
            packC[i * n_packc + j] = 0;
          }
        }
        for(i = 0; i < m_packc; i++){
          for(j = jb; j < n_packc; j++){
            packC[i * n_packc + j] = 0;
          }
        }
        

        bl_macro_kernel(
            ib,   // min(m - ic, DGEMM_MC)
            jb,  // min(m - jc, DGEMM_NC)
            pb,  // min(k-pc, KC)
            packA,
            packB,
            packC, //to padd: min(MC-ic, MC) * min(NC-jc, NC) -> MC * NC
            n_packc);
        // printf("kernel 1\n");
        
        // for (i = 0; i < ib; i += DGEMM_MR){
        //   for(j = 0; j < jb; j += DGEMM_NR){
        //     writeC(
        //       min(ib - i, DGEMM_MR), 
        //       min(jb - j, DGEMM_NR), 
        //       ldc, 
        //       (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR,
        //       &C[(ic + i) * ldc + jc + j], 
        //       &packC[i * j]);
        //   }
        // }
        for (i = 0; i < ib; i++){
          for(j = 0; j < jb; j++){
            C[(ic + i) * ldc + jc + j] = packC[i * n_packc + j];
          }
        }


      } // End 3.rd loop around micro-kernel
    }   // End 4.th loop around micro-kernel
  }     // End 5.th loop around micro-kernel

#ifndef NOPACK
  free(packA);
  free(packB);
  free(packC);
#endif
// printf("free finish\n");
}

void square_dgemm(int lda, double *A, double *B, double *C)
{
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C, lda);
  // printf("bl_dgemm finish\n");
}


//---

// -> packC MC*NC
