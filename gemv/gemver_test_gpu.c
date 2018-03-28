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
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv)
{
    double *A, *x, *y , *u1, *u2, *v1, *v2, *z, *w;
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
    m = n = p = N = 3;
    mythreads = 1;
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
    double A1[3][3];
    double x1[3];
    double w1[3];
    x = (double *)mkl_malloc( n*sizeof( double ), 64 );
    y = (double *)mkl_malloc( n*sizeof( double ), 64 );
     z = (double *)mkl_malloc( n*sizeof( double ), 64 );

     u1 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    v1 = (double *)mkl_malloc( n*sizeof( double ), 64 );
     u2 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    v2 = (double *)mkl_malloc( n*sizeof( double ), 64 );
    w = (double *)mkl_malloc( n*sizeof( double ), 64 );

    if (A == NULL || x == NULL || y == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(x);
      mkl_free(y);
      return 1;
    }


   // float tmpa[3][3] = { 1, 4, 7, 2, 5, 8, 3, 6, 9};
   // float tmpx[3] = { 13, 31, 49};

    float tmpa[3][3] = { -12, 11, 1, -5, -3, 7, 17, 19, -1};
    float tmpx[3] = { -7, -2, 4};
    float tmpu1[3] = { -9, -5, -11};
    float tmpv1[3] = { -1, -2, 14};
    float tmpu2[3] = { -7, -11, -3};
    float tmpv2[3] = { -2, -7, -5};
    float tmpz[3] = { -8, -7, 11};
    float tmpy[3] = { -51, -11, -71};
    float tmpw[3] = { -17, -13, -11};


   printf (" Intializing matrix data \n\n");
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A[j+i*n] = tmpa[i][j];

    }

     for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A1[i][j] = tmpa[i][j];
    }

      printf("\nImhere");
    for (i = 0; i < (n); i++) {

        x[i] = tmpx[i];
    }

      printf("\nImhere2");

       for (i = 0; i < (n); i++) {

        x1[i] = tmpx[i];
    }

      printf("\nImhere3");
    for (i = 0; i < (n); i++) {

        u1[i] = tmpu1[i];

    }
      printf("\nImhere4");
    for (i = 0; i < (n); i++) {


        u2[i] = tmpu2[i];

    }

          printf("\nImhere5");
    for (i = 0; i < (n); i++) {


        v1[i] = tmpv1[i];

    }

          printf("\nImhere6");
    for (i = 0; i < (n); i++) {


        v2[i] = tmpv2[i];

    }
  printf("\nImhere7");

    for (i = 0; i < (n); i++) {

        y[i] = tmpy[i];
    }

     printf("\nImhere8");
    for (i = 0; i < (n); i++) {


        z[i] = tmpz[i];
    }

     for (i = 0; i < (n); i++) {


        w[i] = tmpw[i];
    }

    for (i = 0; i < (n); i++) {


        w1[i] = tmpw[i];
    }
     printf("\nImhere9");


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


    printf (" Input matrix A: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", A[j+i*p]);
	      }
	      printf ("\n");
	    }

    printf (" Input matrix x: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", x[j+i*p]);
	      }
	      printf ("\n");
	    }

	      printf (" Input matrix w: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", w[j+i*p]);
	      }
	      printf ("\n");
	    }

// --- Loop1----------------
 cublasStatus_t status;
    cublasHandle_t handle;
    double *d_A = 0;
    double *d_x = 0;
    double *d_y = 0;
    double *d_u1 = 0, *d_u2 = 0,*d_v1 = 0, *d_v2 = 0;
    double *d_w = 0, *d_z = 0;

    const double cublas_alpha = 1.0;
    const double cublas_beta = 1.0;
    const double cublas_beta2 = 0.0;
    cublasCreate(&handle);

    double *d_Atemp1,*d_Atemp2;

     cudaMalloc((void **)&d_u1,  n * sizeof(d_u1[0]));
    cudaMalloc((void **)&d_u2,  n * sizeof(d_u2[0]));
     cudaMalloc((void **)&d_v1,  n * sizeof(d_v1[0]));
    cudaMalloc((void **)&d_v2,  n * sizeof(d_v2[0]));
    cudaMalloc((void **)&d_Atemp1, n * n * sizeof(d_Atemp1[0]));
    cudaMalloc((void **)&d_Atemp2, n * n * sizeof(d_Atemp2[0]));
    //Atemp1 = (double *)malloc( n*n*sizeof( double ));
    //Atemp2 = (double *)malloc( n*n*sizeof( double ));

    cudaMemcpy(d_u1, u1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, v1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2, u2, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Atemp2, A,  n * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasDger(handle, n, n, &cublas_alpha, d_u1, 1, d_v1, 1, d_Atemp1, n)
    cublasDger(handle, n, n, &cublas_alpha, d_u2, 1, d_v2, 1, d_Atemp2, n)
    cudaFree(d_u1);
    cudaFree(d_u2);
    cudaFree(d_v1);
    cudaFree(d_v2);

    cudaMalloc((void **)&d_A, n * n * sizeof(d_A[0]));
    //Matrix addition
    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cublas_alpha, d_Atemp1, n, &cublas_beta, d_Atemp2, n, d_A, n);
    cudaFree(d_Atemp1);

    cudaMalloc((void **)&d_w,  n * sizeof(d_w[0]));
    cudaMalloc((void **)&d_x,  n * sizeof(d_x[0]));
    cudaMalloc((void **)&d_y,  n * sizeof(d_y[0]));
    cudaMalloc((void **)&d_z,  n * sizeof(d_z[0]));

    cudaMemcpy(d_w, w, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, n * sizeof(double), cudaMemcpyHostToDevice);
    cublasDgemv(handle, CUBLAS_OP_N, n, n, &cublas_alpha, d_A, n, d_y, 1, &cublas_beta, d_x, 1);
    cublasDaxpy(handle, n, &cublas_alpha, d_z, 1, d_x, 1);

    cublasDgemv(handle, CUBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_w, 1);
    cudaMemcpy( w, d_w, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_T, n, n, &cublas_alpha, d_A, n, &cublas_beta2, d_Atemp1, n, d_Atemp2, n);
    cudaMemcpy( A, d_Atemp2, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_Atemp2);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cublasDestroy(handle);

    //---------------
   // cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
   // cblas_dgemv(CblasRowMajor, CblasTrans  , m, n, alpha, A, m, z, 1, beta, y, 1);
  // { 1 2 3 4 5 }
  //  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, scale_100, n, p, alpha, A, p, B, n, beta, C, n);


    stoptime = omp_get_wtime();
    printf("Time for matrix multiplication: %3.3f s, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), atoi(argv[2]));
    //printf ("\n Computations completed.\n\n");
	if(debug)
	{
	    printf ("RES Top left corner of matrix A: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(p,6); j++) {
		printf ("%12.0f", A[j+i*p]);
	      }
	      printf ("\n");
	    }

	    printf ("\nRES Top left corner of matrix x: \n");
	    for (i=0; i<min(p,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x[j+i*n]);
	      }
	      printf ("\n");
	    }


	    printf ("\nRES Top left corner of matrix w: \n");
	    for (i=0; i<min(p,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", w[j+i*n]);
	      }
	      printf ("\n");
	    }

	    printf ("\n Top left corner of matrix C: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", y[j+i*n]);
	      }
	      printf ("\n");
	    }

	     printf ("\n Top left corner of matrix u1: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", u1[j+i*n]);
	      }
	      printf ("\n");
	    }
	     printf ("\n Top left corner of matrix v1: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", v1[j+i*n]);
	      }
	      printf ("\n");
	    }
	     printf ("\n Top left corner of matrix u2: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", u2[j+i*n]);
	      }
	      printf ("\n");
	    }
	     printf ("\n Top left corner of matrix v2: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", v2[j+i*n]);
	      }
	      printf ("\n");
	    }


	     printf ("\n MKL res x: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", x[j+i*n]);
	      }
	      printf ("\n");
	    }

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A1[i][j] = A1[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x1[i] = x1[i] + 1 * A1[j][i] * y[j];
  for (i = 0; i < n; i++)
    x1[i] = x1[i] + z[i];


  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w1[i] = w1[i] +  1 * A1[i][j] * x[j];


	     printf ("\n Poly res A: \n");
	    for (i=0; i<min(m,6); i++) {
	      for (j=0; j<min(n,6); j++) {
		printf ("%12.0f", A[j+i*n]);
	      }
	      printf ("\n");
	    }

         printf ("\n Poly res x1: \n");
	    for (i=0; i<min(m,6); i++) {

		printf ("%12.0f", x1[i]);

	      printf ("\n");
	    }

	    printf ("\n Poly res x1: \n");
	    for (i=0; i<min(m,6); i++) {

		printf ("%12.0f", w1[i]);

	      printf ("\n");
	    }

	}


   //printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(x);
    mkl_free(y);
    //mkl_free(z);

    //mkl_set_num_threads(1);

    //printf (" Example completed. \n\n");
    return 0;
}
