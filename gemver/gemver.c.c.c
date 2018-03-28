
/* Header below added by Tulsi for replaced CUBLAS code */
#include <cuda_runtime.h>
#include <cublas_v2.h>


/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1;
  *beta = 1;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

//#pragma scop

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





//#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}
