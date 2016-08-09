#include <cublasXt.h>
#include <stdio.h>
#include <sys/time.h>

void cublas_error_message(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            {
                printf("the library was not initialized\n");
                break;
            }
        case CUBLAS_STATUS_INVALID_VALUE:
            {
                printf("the parameters m,n,k<0\n");
                break;
            }
        case CUBLAS_STATUS_ARCH_MISMATCH:
            {
                printf("the device does not support double-precision\n");
                break;
            }
        case CUBLAS_STATUS_EXECUTION_FAILED:
            {
                printf("the function failed to launch on the GPU\n");
                break;
            }
        default:
            {
                printf("cublas status unknown");
            }
    }
}


#define CALL_CUBLAS(func__, args__)                                                 \
{                                                                                   \
    cublasStatus_t status;                                                          \
    if ((status = func__ args__) != CUBLAS_STATUS_SUCCESS)                          \
    {                                                                               \
        cublas_error_message(status);                                               \
        printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        exit(-100);                                                                 \
    }                                                                               \
}

int main(int argn, char** argv)
{
    int N = 4000;
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];
    double alpha = 1;
    double beta = 0;

    int deviceId[] = {0};

    for (int i = 0; i < N * N; i++) A[i] = B[i] = 1.0;

    cublasXtHandle_t handle;

    CALL_CUBLAS(cublasXtCreate, (&handle));
    CALL_CUBLAS(cublasXtDeviceSelect, (handle, 1, deviceId));

    timeval t0, t1;
    gettimeofday(&t0, NULL);
    
    CALL_CUBLAS(cublasXtDgemm, (handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N));

    gettimeofday(&t1, NULL);

    double val = double(t1.tv_sec - t0.tv_sec) + double(t1.tv_usec - t0.tv_usec) / 1e6;

    printf("performance : %f GFlops\n", 2e-9 * N * N * N / val);

    for (int i = 0; i < N * N; i++) 
    {
        if (std::abs(C[i] - N) > 1e-10)
        {
            printf("Fail\n");
            exit(0);
        }
    }
    
    CALL_CUBLAS(cublasXtDestroy, (handle));
}
