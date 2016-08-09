#include <cuda.h>
#include <cublas_v2.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <cufft.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>

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

inline __host__ __device__ int num_blocks(int length, int block_size)
{
    return (length / block_size) + min(length % block_size, 1);
}

inline __device__ uint32_t random(size_t seed)
{
    uint32_t h = 5381;

    return (h << (seed % 15)) + h;
}

__global__ void randomize_on_gpu_kernel
(
    double* ptr__,
    size_t size__
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
}

extern "C" void randomize_on_gpu(double* ptr, size_t size)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size, grid_t.x));

    randomize_on_gpu_kernel <<<grid_b, grid_t>>>
    (
        ptr,
        size
    );
}

int main(int argn, char** argv)
{
    //== cudaStream_t stream1;
    //== cudaStream_t stream2;

    //== CALL_CUDA(cudaStreamCreate, (&stream1));
    //== CALL_CUDA(cudaStreamCreate, (&stream2));

    //== cufftHandle plan1;
    //== cufftHandle plan2;

    //== CALL_CUFFT(cufftCreate, (&plan1));
    //== CALL_CUFFT(cufftCreate, (&plan2));

    //== CALL_CUFFT(cufftSetAutoAllocation, (plan1, true));
    //== CALL_CUFFT(cufftSetAutoAllocation, (plan2, true));

    //== int dims[] = {128, 128, 128};
    //== size_t work_size;
    //== CALL_CUFFT(cufftMakePlanMany, (plan1, 3, dims, dims, 1, 1, dims, 1, 1, CUFFT_Z2Z, 1, &work_size));
    //== CALL_CUFFT(cufftMakePlanMany, (plan2, 3, dims, dims, 1, 1, dims, 1, 1, CUFFT_Z2Z, 1, &work_size));

    //== CALL_CUFFT(cufftSetStream, (plan1, stream1));
    //== CALL_CUFFT(cufftSetStream, (plan2, stream2));
    //== 
    //== size_t buf_size = dims[0] * dims[1] * dims[2] * sizeof(cuDoubleComplex);
    //== cuDoubleComplex* buf1;
    //== cuDoubleComplex* buf2;
    //== CALL_CUDA(cudaMalloc, (&buf1, buf_size));
    //== CALL_CUDA(cudaMalloc, (&buf2, buf_size));

    //== CALL_CUDA(cudaMemset, (buf1, 0, buf_size));
    //== CALL_CUDA(cudaMemset, (buf2, 0, buf_size));

    //== double t = -current_time();
    //== for (int i = 0; i < 200; i++)
    //== {
    //==     CALL_CUFFT(cufftExecZ2Z, (plan1, buf1, buf1, CUFFT_FORWARD));
    //== }
    //== CALL_CUDA(cudaStreamSynchronize, (stream1));
    //== t += current_time();
    //== printf("Execution time: %.4f sec.\n", t);
    //== 
    //== t = -current_time();
    //== for (int i = 0; i < 100; i++)
    //== {
    //==     CALL_CUFFT(cufftExecZ2Z, (plan1, buf1, buf1, CUFFT_FORWARD));
    //==     CALL_CUFFT(cufftExecZ2Z, (plan2, buf2, buf2, CUFFT_FORWARD));
    //== }
    //== //for (int i = 0; i < 100; i++)
    //== //{
    //== //    CALL_CUFFT(cufftExecZ2Z, (plan2, buf2, buf2, CUFFT_FORWARD));
    //== //}

    //== CALL_CUDA(cudaStreamSynchronize, (stream1));
    //== CALL_CUDA(cudaStreamSynchronize, (stream2));

    //== t += current_time();
    //== printf("Execution time: %.4f sec.\n", t);

    //== CALL_CUDA(cudaFree, (buf1));
    //== CALL_CUDA(cudaFree, (buf2));

    //== CALL_CUFFT(cufftDestroy, (plan1));
    //== CALL_CUFFT(cufftDestroy, (plan2));

    //== CALL_CUDA(cudaStreamDestroy, (stream1));
    //== CALL_CUDA(cudaStreamDestroy, (stream2));


    int nx{100}, ny{100};
    int dim_xy[] = {nx, ny};

    int nfft{100};

    cufftHandle plan1;
    CALL_CUFFT(cufftCreate, (&plan1));
    CALL_CUFFT(cufftSetAutoAllocation, (plan1, true));
    size_t work_size;
    CALL_CUFFT(cufftMakePlanMany, (plan1, 2, dim_xy, dim_xy, 1, 1, dim_xy, 1, 1, CUFFT_Z2Z, nfft, &work_size));
    
    size_t buf_size = nx * ny * nfft * sizeof(cuDoubleComplex);
    cuDoubleComplex* buf1;
    CALL_CUDA(cudaMalloc, (&buf1, buf_size));
    CALL_CUDA(cudaMemset, (buf1, 0, buf_size));

    double t = -current_time();
    for (int i = 0; i < 200; i++) {
        //CALL_CUDA(cudaMemset, (buf1, 0, buf_size));
        randomize_on_gpu((double*)buf1, nx * ny * nfft * 2);
        CALL_CUFFT(cufftExecZ2Z, (plan1, buf1, buf1, CUFFT_FORWARD));
    }
    CALL_CUDA(cudaStreamSynchronize, (NULL));
    t += current_time();
    printf("execution time: %.4f sec.\n", t);
    printf("speed: %.4f 2D FFTs / sec.\n", 200 * nfft / t);

    CALL_CUDA(cudaFree, (buf1));
    CALL_CUFFT(cufftDestroy, (plan1));
}
