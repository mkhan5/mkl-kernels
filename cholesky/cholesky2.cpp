/************************************************************************************/
//Example of Cholesky square root and subsequent inverse, along with the log
//determinant along the way. Also, the use of the random number generators in Rmath.
//Make with "make -f Makefile-example-1"
/************************************************************************************/
#include <iostream>
#include <omp.h>
using namespace std;

#include <mkl.h>

#define MATHLIB_STANDALONE
//#include <Rmath.h>

int main(){

  int i,j;
  int info;
  char *lower = "L";
  char *upper = "U";
  char *ntran = "N";
  char *ytran = "T";
  char *rside = "R";
  char *lside = "L";
  const double one = 1.0;
  const double negOne = -1.0;
  const double zero = 0.0;
  const int incOne = 1;

  //set seed
  set_seed(123,456);

  //set threads
  omp_set_num_threads(4);

  int n = 3;
  int nn = n*n;
  double logDet = 0;
  double traceSum = 0;
  double *A = new double[nn];
  double *B = new double[nn];
  double *C = new double[nn];
   float tmpa[3][3] =  { 2, -1, 0, -1, 2, -1, 0, -1, 2};
 // for(i = 0; i < n*n; i++) A[i] = rnorm(0,1);

  printf (" Intializing matrix data \n\n");
    for (i = 0; i < (n); i++) {
        for (j=0; j<(n); j++)
        A[j+i*n] = tmpa[i][j];
    }

  //make a pd matrix
  dgemm(ntran, ytran, &n, &n, &n, &one, A, &n, A, &n, &zero, B, &n);

  //make a copy
  dcopy(&nn, B, &incOne, C, &incOne);

  //take the Cholesky square root
  dpotrf(lower, &n, C, &n, &info); if(info != 0){cout << "c++ error: Cholesky failed" << endl;}

  //get the log determinant
  for(i = 0; i < n; i++) logDet += 2*(C[i*n+i]);

  //take the inverse
  dpotri(lower, &n, C, &n, &info); if(info != 0){cout << "c++ error: Cholesky inverse failed" << endl;}

  //check the inverse for fun
  dsymm(lside, lower, &n, &n, &one, C, &n, B, &n, &zero, A, &n);

  for(i = 0; i < n; i++) traceSum += A[i*n+i];

  cout << "Log determinant: " <<  logDet << endl;
  cout << "Should be " << n << ": " << traceSum << endl;

  return(0);
}

