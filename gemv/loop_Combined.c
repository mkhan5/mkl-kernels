double *Atemp1,*Atemp2;
Atemp1 = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
Atemp2 = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
for (i = 0; i < n*n ; i++)
  {
    Atemp1[i] = 0.0;
  }
cblas_dcopy (n*n, A, 1, Atemp2, 1);
const double mkl_alpha = 1.0;
const double mkl_beta = 1.0;
const double mkl_beta2 = 1.0;
cblas_dger (CblasRowMajor, n, n, mkl_alpha, u1, 1, v1, 1, Atemp1, n);
cblas_dger (CblasRowMajor, n, n, mkl_alpha, u2, 1, v2, 1, Atemp2, n);
mkl_domatadd ('R', 'N', 'N', n, n, mkl_alpha, Atemp1, n, mkl_beta, Atemp2, n, A, n);

const double inp_beta = 1.0;
cblas_dgemv(CblasRowMajor, CblasTrans, n, n, inp_beta, A, n, y, 1, mkl_beta, x, 1);
cblas_daxpy (n, 1, z, 1, x, 1);

cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, mkl_alpha, A, n, x, 1, mkl_beta2, w, 1);
