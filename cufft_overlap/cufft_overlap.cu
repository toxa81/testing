#include <cuda.h>
#include <cublas_v2.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <cufft.h>
#include <stdio.h>
#include <sys/time.h>

#ifdef NDEBUG
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error = func__ args__;                                                                             \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
    }                                                                                                              \
}
#else
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error;                                                                                             \
    func__ args__;                                                                                                 \
    cudaDeviceSynchronize();                                                                                       \
    error = cudaGetLastError();                                                                                    \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
    }                                                                                                              \
}
#endif

void cufft_error_message(cufftResult result)
{
    switch (result)
    {
        case CUFFT_INVALID_PLAN:
        {
            printf("CUFFT_INVALID_PLAN\n");
            break;
        }
        case CUFFT_ALLOC_FAILED:
        {
            printf("CUFFT_ALLOC_FAILED\n");
            break;
        }
        case CUFFT_INVALID_VALUE:
        {
            printf("CUFFT_INVALID_VALUE\n");
            break;
        }
        case CUFFT_INTERNAL_ERROR:
        {
            printf("CUFFT_INTERNAL_ERROR\n");
            break;
        }
        case CUFFT_SETUP_FAILED:
        {
            printf("CUFFT_SETUP_FAILED\n");
            break;
        }
        case CUFFT_INVALID_SIZE:
        {
            printf("CUFFT_INVALID_SIZE\n");
            break;
        }
        default:
        {
            printf("unknown error code %i\n", result);
            break;
        }
    }
}

#define CALL_CUFFT(func__, args__)                                                  \
{                                                                                   \
    cufftResult result;                                                             \
    if ((result = func__ args__) != CUFFT_SUCCESS)                                  \
    {                                                                               \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s: ", #func__, __LINE__, __FILE__); \
        cufft_error_message(result);                                                \
        exit(-100);                                                                 \
    }                                                                               \
}

inline double current_time()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

int main(int argn, char** argv)
{
    cudaStream_t stream1;
    cudaStream_t stream2;

    CALL_CUDA(cudaStreamCreate, (&stream1));
    CALL_CUDA(cudaStreamCreate, (&stream2));

    cufftHandle plan1;
    cufftHandle plan2;

    CALL_CUFFT(cufftCreate, (&plan1));
    CALL_CUFFT(cufftCreate, (&plan2));

    CALL_CUFFT(cufftSetAutoAllocation, (plan1, true));
    CALL_CUFFT(cufftSetAutoAllocation, (plan2, true));

    int dims[] = {128, 128, 128};
    size_t work_size;
    CALL_CUFFT(cufftMakePlanMany, (plan1, 3, dims, dims, 1, 1, dims, 1, 1, CUFFT_Z2Z, 1, &work_size));
    CALL_CUFFT(cufftMakePlanMany, (plan2, 3, dims, dims, 1, 1, dims, 1, 1, CUFFT_Z2Z, 1, &work_size));

    CALL_CUFFT(cufftSetStream, (plan1, stream1));
    CALL_CUFFT(cufftSetStream, (plan2, stream2));
    
    size_t buf_size = dims[0] * dims[1] * dims[2] * sizeof(cuDoubleComplex);
    cuDoubleComplex* buf1;
    cuDoubleComplex* buf2;
    CALL_CUDA(cudaMalloc, (&buf1, buf_size));
    CALL_CUDA(cudaMalloc, (&buf2, buf_size));

    CALL_CUDA(cudaMemset, (buf1, 0, buf_size));
    CALL_CUDA(cudaMemset, (buf2, 0, buf_size));

    double t = -current_time();
    for (int i = 0; i < 200; i++)
    {
        CALL_CUFFT(cufftExecZ2Z, (plan1, buf1, buf1, CUFFT_FORWARD));
    }
    CALL_CUDA(cudaStreamSynchronize, (stream1));
    t += current_time();
    printf("Execution time: %.4f sec.\n", t);
    
    t = -current_time();
    for (int i = 0; i < 100; i++)
    {
        CALL_CUFFT(cufftExecZ2Z, (plan1, buf1, buf1, CUFFT_FORWARD));
        CALL_CUFFT(cufftExecZ2Z, (plan2, buf2, buf2, CUFFT_FORWARD));
    }
    //for (int i = 0; i < 100; i++)
    //{
    //    CALL_CUFFT(cufftExecZ2Z, (plan2, buf2, buf2, CUFFT_FORWARD));
    //}

    CALL_CUDA(cudaStreamSynchronize, (stream1));
    CALL_CUDA(cudaStreamSynchronize, (stream2));

    t += current_time();
    printf("Execution time: %.4f sec.\n", t);

    CALL_CUDA(cudaFree, (buf1));
    CALL_CUDA(cudaFree, (buf2));

    CALL_CUFFT(cufftDestroy, (plan1));
    CALL_CUFFT(cufftDestroy, (plan2));

    CALL_CUDA(cudaStreamDestroy, (stream1));
    CALL_CUDA(cudaStreamDestroy, (stream2));

}
