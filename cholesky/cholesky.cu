void func(void)
{
    double *d_A;
    cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double));
    cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice);

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUDA CHOLESKY initialization
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, &work_size);

    // --- CUDA POTRF execution
    double *work;
    cudaMalloc(&work, work_size * sizeof(double));
    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, work, work_size, devInfo);
    int devInfo_h = 0;
   cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0)
        printf("Unsuccessful potrf execution\n\n");

    // --- At this point, the upper triangular part of A contains the elements of L. Showing this.
    printf("\nFactorized matrix\n");
    cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost);

    cusolverDnDestroy(solver_handle);
}

