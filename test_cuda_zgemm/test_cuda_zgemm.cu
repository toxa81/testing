#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <sys/time.h>
#include "cmd_args.h"
#include "cuda.hpp"
#include "cublas.hpp"

inline double current_time()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--N=", "N");
    args.register_key("--K=", "K");
    args.register_key("--ns=", "number of CUDA streams");
    args.parse_args(argn, argv);

    int N = args.value<int>("N", 256);
    int K = args.value<int>("K", 4000);
    int ns = args.value<int>("ns", 1);

    acc::create_streams(ns);
    cublas::create_stream_handles();

    std::vector<cuDoubleComplex*> mA(ns);
    std::vector<cuDoubleComplex*> mC(ns);

    cuDoubleComplex* h_A;
    cuDoubleComplex* h_C;

    cudaMallocHost((void**)&h_A, N * K * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&h_C, N * N * sizeof(cuDoubleComplex));

    for (int i = 0; i < N * K; i++) {
        h_A[i] = make_cuDoubleComplex(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
    }
    for (int i = 0; i < N * N; i++) {
        h_C[i] = make_cuDoubleComplex(0, 0);
    }

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);

    for (int i = 0; i < ns; i++) {
        mA[i] = acc::allocate<cuDoubleComplex>(N * K);
        mC[i] = acc::allocate<cuDoubleComplex>(N * N);
        cudaMemcpy(mA[i], h_A, K * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(mC[i], h_C, N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }


    double t1 = -current_time();

    for (int ia = 0; ia < 10; ia++)
    {
        for (int j = 0; j < ns; j++) {  
            cublasZgemm(cublas::stream_handle(j), CUBLAS_OP_N, CUBLAS_OP_T, N, N, K, &alpha, 
                                            mA[j], N, mA[j], N, &alpha, mC[j], N);
        }


        //if (status != CUBLAS_STATUS_SUCCESS)
        //{

        //    printf("failed to execute cublasZgemm\n");
        //    
        //    switch (status)
        //    {
        //        case CUBLAS_STATUS_NOT_INITIALIZED:
        //        {
        //            printf("the library was not initialized\n");
        //            break;
        //        }
        //        case CUBLAS_STATUS_INVALID_VALUE:
        //        {
        //            printf("the parameters m,n,k<0\n");
        //            break;
        //        }
        //        case CUBLAS_STATUS_ARCH_MISMATCH:
        //        {
        //            printf("the device does not support double-precision\n");
        //            break;
        //        }
        //        case CUBLAS_STATUS_EXECUTION_FAILED:
        //        {
        //            printf("the function failed to launch on the GPU\n");
        //            break;
        //        }
        //        default:
        //        {
        //            printf("unknown error\n");
        //            break;
        //        }
        //    }

        //    exit(-1);
        //}
    }
    for (int j = 0; j < ns; j++) {
        acc::sync_stream(j);
    }

    t1 += current_time();

    printf("M,N,K: %i %i %i, performance: %12.6f GFlops\n", N, N, K, ns * 10 * 8e-9 * N * N * K / t1);

    cublas::destroy_stream_handles();
    acc::destroy_streams();

    for (int i = 0; i < ns; i++) {
        acc::deallocate(mA[i]);
        acc::deallocate(mC[i]);
    }

    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    
}
