#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <sys/time.h>

inline double current_time()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

int main(int argn, char** argv)
{
    cuDoubleComplex* h_A;
    cuDoubleComplex* h_C;

    cuDoubleComplex* d_A;
    cuDoubleComplex* d_C;

    int N = 200;
    int n = 200;
    
    cublasHandle_t handle;

    cudaMallocHost((void**)&h_A, N * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&h_C, N * N * sizeof(cuDoubleComplex));

    cudaMalloc((void**)&d_A, N * n * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_C, N * N * sizeof(cuDoubleComplex));

    cublasCreate(&handle);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);

    for (int i = 0; i < N * n; i++) h_A[i] = make_cuDoubleComplex(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
    for (int i = 0; i < N * N; i++) h_C[i] = make_cuDoubleComplex(0, 0);


    cudaMemcpy(d_A, h_A, n * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    

    cudaMemcpy(d_A, h_A, n * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cublasGetStream(handle, &stream);


    double t1 = -current_time();
    for (int ia = 0; ia < 20; ia++)
    {

        cublasStatus_t status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, n, &alpha, 
                                            d_A, N, d_A, N, &alpha, d_C, N);

        if (status != CUBLAS_STATUS_SUCCESS)
        {

            printf("failed to execute cublasZgemm\n");
            
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
                    printf("unknown error\n");
                    break;
                }
            }

            exit(-1);
        }
    }
    cudaStreamSynchronize(stream);

    t1 += current_time();

    printf("performance: %12.6f GFlops\n", 20 * 8e-9 * N * N * n / t1);




    cublasDestroy(handle);

    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_C);
    
}
