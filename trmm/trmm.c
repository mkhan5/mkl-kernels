/*******************************************************************************
*   Copyright(C) 2012-2014 Intel Corporation. All Rights Reserved.
*
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*
*   *Third Party trademarks are the property of their respective owners.
*
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************/

/*******************************************************************************
*   This example computes real matrix C=alpha*A*B+beta*C using Intel(R) MKL
*   function dgemm, where A, B, and C are matrices and alpha and beta are
*   scalars in double precision.
*
*   In this simple example, practices such as memory management, data alignment,
*   and I/O that are necessary for good programming style and high MKL
*   performance are omitted to improve readability.
********************************************************************************/

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>


int main(int argc, char **argv)
{
    double *A, *B, *C;
    int m, n, p, i, j, N;
    double alpha, beta;
    int mythreads;
    int debug = 1;
    double starttime, stoptime;
    int scale;

  /*  printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");
   */
    m = n = p = N = atoi(argv[2]);
    mythreads = atoi(argv[1]);
    if (argc > 3)
        scale = atoi(argv[3]);
  /*  printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, p, p, n);
   */
    alpha = 1.0; beta = 0.0;

    int n2 = N * N;
    m = n = p = N;
    float var1, var2;
    int scale_100,minus_scale_100;

    var1 = (scale*N)/100;
    var2 = (100-scale)*N/100;

    scale_100  = var1;
    minus_scale_100 = var2;
     printf("Mul1 = |%d| , Mul2 = |%d| is int\n",scale_100, minus_scale_100 );

   if (!(fmod(var1, 1.0) == 0.0 && fmod(var2, 1.0) == 0.0))
   {
       printf("Mul1 = |%f| , Mul2 = |%f| is NOT int\n",var1, var2 );
       printf("Exiting\n");
       exit(0);
   }

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");

   /*
    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n"); */
    A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    B = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    C = (double *)mkl_malloc( n*n*sizeof( double ), 64 );



    if (A == NULL || B == NULL || B == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }


   // float tmpa[3][3] = { 1, 4, 7, 2, 5, 8, 3, 6, 9};
   // float tmpx[3] = { 13, 31, 49};

//   float alf = 1;
//  float bt = 1;
//  float tmpa[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
//  float tmpb[3][3] = { 1, -4, 2, -1, 1, -1, 3, -6, 4};
//  float tmpx[3] = { -5, -3, -6};
//  //sol : (9,-4,7)
//


//  float tmpa[3][3] = { 1, 0, 0, 2, 5, 0, 3, 6, -3};
//  float tmpb[3][3] = { 1, 0, 1, 3, -1, -2, 0, 2, -1};
 float tmpa[4][4] = { 1, 0, 0, 0, -17, 1, 0, 0, -8, 13, 1, 0, 7, 11, 19, 1}; // unit, lower triangular
   float tmpb[4][4] = { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9};


  //sol: c = (7, 4, -6)
  //         (17, 7, -14)
  //         (21, -12, -6)

   printf (" Intializing matrix data \n\n");
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A[j+i*n] = tmpa[i][j];
    }
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        B[j+i*n] = tmpb[i][j];
    }

  //  A[3][3] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
  //  x[3] = { 2, 1, 3};



   // for (i = 0; i < (m*n); i++) {
   //     C[i] = 0.0;
   // }

   if(debug)
	{
	    printf (" Top left corner of matrix A: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", A[j+i*p]);
	      }
	      printf ("\n");
	    }


	     printf ("\n Top left corner of matrix B: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", B[j+i*p]);
	      }
	      printf ("\n");
	    }
	}

    mkl_set_dynamic(0);
    //printf("The number of threads before %d \n", mkl_get_max_threads());
    mkl_set_num_threads(mythreads);
    //printf("The number of threads after %d \n", mkl_get_max_threads());
    //printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    starttime = omp_get_wtime();
    //cblas_dsymm (CblasRowMajor, CblasLeft, CblasLower, m, n, alpha, A, m, B, m, beta, C, n);
    cblas_dtrmm (CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, m, n, alpha, A, n, B, n);

    stoptime = omp_get_wtime();
    printf("Time for matrix multiplication: %3.3f s, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), atoi(argv[2]));
    //printf ("\n Computations completed.\n\n");
	if(debug)
	{


	    printf ("\n Top left corner of matrix Res B: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", B[j+i*n]);
	      }
	      printf ("\n");
	    }
	}


   //printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    //mkl_free(z);

    //mkl_set_num_threads(1);

    //printf (" Example completed. \n\n");
    return 0;
}
