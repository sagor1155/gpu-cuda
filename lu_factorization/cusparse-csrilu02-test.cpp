#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>


int main(int argc, char*argv[])
{
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;

    const int m = 3;
    const int nnz = 4;
    int h_csrRowPtr[nnz] = {0, 1, 3, 4}; //{0, 2, 3, 4 };
    int h_csrColInd[nnz] = {0, 0, 2, 1}; //{0, 1, 1, 1 };
    double h_csrVal[nnz] = {1,  5,  2, -1}; //{1.0, 2.0, 5.0, 8.0 };
    
    int *d_csrRowPtr = NULL;
    int *d_csrColInd = NULL;
    double *d_csrVal = NULL;

    double *d_x = NULL;
    double *d_y = NULL;
    double *d_z = NULL;
    
    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_M  = 0;
    csrsv2Info_t  info_L  = 0;
    csrsv2Info_t  info_U  = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha = 1.;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

    /* step 0: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cudaMalloc(&d_csrRowPtr, sizeof(int)*nnz);  // bug is there, size of d_csrRowPtr not always nnz
    cudaMalloc(&d_csrColInd, sizeof(int)*nnz);
    cudaMalloc(&d_csrVal, sizeof(double)*nnz);

    cudaMalloc(&d_x, sizeof(double)*m);
    cudaMalloc(&d_y, sizeof(double)*m);
    cudaMalloc(&d_z, sizeof(double)*m);

    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, sizeof(int)*nnz ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, sizeof(int)*nnz ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    h_csrVal, sizeof(double)*nnz ,cudaMemcpyHostToDevice);


    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has unit diagonal
    // - matrix U is base-1
    // - matrix U is upper triangular
    // - matrix U has non-unit diagonal
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateCsrilu02Info(&info_M);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseDcsrilu02_bufferSize(handle, m, nnz,        descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz,        descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz,        descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);


    printf("pBufferSize_M: %d\n", pBufferSize_M);
    printf("pBufferSize_L: %d\n", pBufferSize_L);
    printf("pBufferSize_U: %d\n", pBufferSize_U);

    pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on U
    // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    cusparseDcsrilu02_analysis(handle, m, nnz, descr_M,
        d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
        policy_M, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);

    cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_L, policy_L, pBuffer);
    
    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);

    cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_U, policy_U, pBuffer);
    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);
	fflush(stdout);
    
    // step 5: M = L * U
    cusparseDcsrilu02(handle, m, nnz, descr_M,
        d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
	fflush(stdout);
}
    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);

    // step 6: solve L*z = x
    cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_L, d_x, d_z, policy_L, pBuffer);
    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);

    // step 7: solve U*y = z
    cusparseDcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_U, d_z, d_y, policy_U, pBuffer);

    printf ("This is line %d of file \"%s\".\n", __LINE__, __FILE__);

    // step 8: free resources
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyCsrilu02Info(info_M);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_U);
    cusparseDestroy(handle);    
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    return 0;
}
