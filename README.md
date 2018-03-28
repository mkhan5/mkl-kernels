18 MKL Kernels
=======

18 Kernels - Linear Algebra Programs written using Intel MKL and OpenMP.
Serial C programs taken from Polybench 4.2.1 Benchmark and modified to run on multiple cores of a CPUs using MKL and OpenMP.

* 2mm		2 Matrix Multiplications (alpha * A * B * C + beta * D)
* 3mm		3 Matrix Multiplications ((A*B)*(C*D))
* atax		Matrix Transpose and Vector Multiplication
* bicg		BiCG Sub Kernel of BiCGStab Linear Solver
* cholesky	Cholesky Decomposition
* gemm		Matrix-multiply C=alpha.A.B+beta.C
* gemv		Vector Multiplication
* gemver		Vector Multiplication and Matrix Addition
* gesummv		Scalar, Vector and Matrix Multiplication
* QR (gramschmidt)	Gram-Schmidt decomposition
* lu		LU decomposition
* ludcmp		LU decomposition followed by Forward Substitution
* mvt		Matrix Vector Product and Transpose
* symm		Symmetric matrix-multiply
* syr2k		Symmetric rank-2k update
* syrk		Symmetric rank-k update
* trisolv		Triangular solver
* trmm		Triangular matrix-multiply


Download
--------

You can obtain the latest release from the repository by typing:

```bash
git clone https://github.com/mkhan5/mkl-kernels.git
```

Installation
------------

Needs Intel Math Kernel Library (MKL) to be installed.

Development
-----------

If you want to contribute new features or improvements, you're welcome to fork on github and send us your pull requests!
