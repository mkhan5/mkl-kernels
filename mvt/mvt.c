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
*   in any way  without Intel'x1  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*
*   *Third Party trademarks are the property of their respective owners.
*
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel'x1
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
    double *A, *y1, *x2, *y2, *x1;
    int m, n, i, j, N;
    double alpha, beta;
    int mythreads;
    int debug = 1;
    double starttime, stoptime;
    int scale;

  /*  printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");
   */
    m = n = N = atoi(argv[2]);
    mythreads = atoi(argv[1]);
    if (argc > 3)
        scale = atoi(argv[3]);
  /*  printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, y1, y1, n);
   */
    alpha = 1.0; beta = 1.0;

    int n2 = N * N;
    m = n = y1 = N;
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
    A = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    x1 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    x2 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    y1 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    y2 = (double *)mkl_malloc( n*sizeof( double ), 64 );


   // float tmpa[3][3] = { 1, 4, 7, 2, 5, 8, 3, 6, 9};
   // float tmpx[3] = { 13, 31, 49};

  double ta[3][3] = { 31, -18, 40, 22, 6, 18, -17, -11, 26};
  double tx1[3] = { -8, -14, 5};
  double tx2[3] = { -4, 49, 25};
  double ty1[3] = { -5, -13, 2};
  double ty2[3] = { 3, -9, -1};



   printf (" Intializing matrix data \n\n");
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A[j+i*n] = ta[i][j];
    }
    for (i = 0; i < (n); i++) {

        x1[i] = tx1[i];
        x2[i] = tx2[i];
        y1[i] = ty1[i];
        y2[i] = ty2[i];
    }
  //  A[3][3] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
  //  x[3] = { 2, 1, 3};


   // for (i = 0; i < (m*n); i++) {
   //     C[i] = 0.0;
   // }

    mkl_set_dynamic(0);
    //printf("The number of threads before %d \n", mkl_get_max_threads());
    mkl_set_num_threads(mythreads);
    //printf("The number of threads after %d \n", mkl_get_max_threads());
    //printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    starttime = omp_get_wtime();
if(debug)
	{
	    printf (" Top left corner of matrix A: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", A[j+i*n]);
	      }
	      printf ("\n");
	    }

	    printf ("\n Top left corner of x1: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x1[j+i*n]);
	      }
	      printf ("\n");
	    }

	    printf ("\n Top left corner of x2: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x2[j+i*n]);
	      }
	      printf ("\n");
	    }

	     printf ("\n Top left corner of y1: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", y1[j+i*n]);
	      }
	      printf ("\n");
	    }

	    printf ("\n Top left corner of y2: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", y2[j+i*n]);
	      }
	      printf ("\n");
	    }
	}


    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, m, y1, 1, beta, x1, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alpha, A, m, y2, 1, beta, x2, 1);
   // cblas_dgemv(CblasRowMajor, CblasTrans  , m, n, alpha, A, m, z, 1, beta, y, 1);
  // { 1 2 3 4 5 }
  //  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, scale_100, n, y1, alpha, A, y1, B, n, beta, C, n);


    stoptime = omp_get_wtime();
    printf("Time for matrix multiplication: %3.3f x1, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), atoi(argv[2]));
    //printf ("\n Computations completed.\n\n");
	if(debug)
	{


	    printf ("\n Top left corner of Res x1: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x1[j+i*n]);
	      }
	      printf ("\n");
	    }

	    printf ("\n Top left corner of Res x2: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x2[j+i*n]);
	      }
	      printf ("\n");
	    }
	}


   //printf ("\n Deallocating memory \n\n");
    mkl_free(A);

    //mkl_free(z);

    //mkl_set_num_threads(1);

    //printf (" Example completed. \n\n");
    return 0;
}
