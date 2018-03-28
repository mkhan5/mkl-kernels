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
    double *A, *x, *b, *z;
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
    x = (double *)mkl_malloc( n*sizeof( double ), 64 );
    b = (double *)mkl_malloc( n*sizeof( double ), 64 );
    z = (double *)mkl_malloc( n*sizeof( double ), 64 );
    if (A == NULL || x == NULL || b == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(x);
      mkl_free(b);
      return 1;
    }


   // float tmpa[3][3] = { 1, 4, 7, 2, 5, 8, 3, 6, 9};
   // float tmpx[3] = { 13, 31, 49};

   // float tmpa[3][3] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
   // float tmpx[3] = { 2, 1, 3};


  //   float tmpa[3][3] =  { -1, 0, 0, 2, 3, 0, -1, 4, -5};
  //       float tmpb[3] = { -1, 8, -8};
     // sol: (1,2,3)

      float tmpa[3][3] =  { 1, -2, 3, 2, 1, 1, -3, 2, -2};
       float tmpb[3] = { 7, 4, -10};

//
//        float tmpa[3][3] =  { 1, -2, 3, 5, 8, -1, 2, 1, 1};
//       float tmpb[3] = { 9, -5, 3};
       //sol ( 2,0,-1)


   printf (" Intializing matrix data \n\n");
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A[j+i*n] = tmpa[i][j];
    }
    for (i = 0; i < (n); i++) {

        b[i] = tmpb[i];
    }
  //  A[3][3] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
  //  x[3] = { 2, 1, 3};


    for (i = 0; i < (n); i++) {
        //b[i] = 0.0;
        z[i] = 0.0;
    }

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


	    printf ("\n Top left corner of matrix b: \n");
	    for (i=0; i<min(m,6); i++) {

		printf ("%12.0f", b[i]);

	      printf ("\n");
	    }

    }

    mkl_set_dynamic(0);
    //printf("The number of threads before %d \n", mkl_get_max_threads());
    mkl_set_num_threads(mythreads);
    //printf("The number of threads after %d \n", mkl_get_max_threads());
    //printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    starttime = omp_get_wtime();
int info;
//info  = LAPACKE_dtrtrs (LAPACK_ROW_MAJOR, 'L', 'N', 'N', n , 1, A, n, b , n);
 //cblas_dtrsm (CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, alpha, A, n, b, n);

 //cblas_dtrsv (CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
 //LAPACKE_mkl_dgetrfnpi (LAPACK_ROW_MAJOR, m, n, n, A, n);
 long long int *mkl_ipiv;
 mkl_ipiv = (long long int *)mkl_malloc( n*sizeof(long long int ), 64 );
 LAPACKE_dgetrf (LAPACK_COL_MAJOR, n, n, A, n, mkl_ipiv );
 LAPACKE_dgetrs (LAPACK_COL_MAJOR, 'T' , n , 1, A, n, mkl_ipiv, b, n);

 cblas_dcopy (n, b, 1, x, 1);
//   info = LAPACKE_dgesv (LAPACK_ROW_MAJOR , n , 1 , A, n , mkl_ipiv , b , n);
  // { 1 2 3 4 5 }
  //  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, scale_100, n, p, alpha, A, p, B, n, beta, C, n);
if (info == 0)
    printf("\nSuccesful exec\n");


    stoptime = omp_get_wtime();
    printf("Time for matrix multiplication: %3.3f s, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), atoi(argv[2]));
    //printf ("\n Computations completed.\n\n");
	if(debug)
	{


	    printf ("\n MAIN RESULT Top left corner of matrix x: \n");
	    for (i=0; i<min(n*n,6*n); i++) {
	      //for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", b[i]);
	     // }
	      printf ("\n");
	    }
	}


   //printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(x);
    mkl_free(b);
    mkl_free(z);

    //mkl_set_num_threads(1);

    //printf (" Example completed. \n\n");
    return 0;
}
