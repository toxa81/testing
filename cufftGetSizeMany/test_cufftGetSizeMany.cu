#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

void cufft_check_error(const char* file_name, const int line_number, cufftResult result)
{
    if (result == CUFFT_SUCCESS) return;
    
    printf("CUFFT error at line %i of file %s: ", line_number, file_name);
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

int main(int argn, char** argv)
{
    cuInit(0);

    cufftHandle plan;

    cufftResult result = cufftCreate(&plan);
    cufft_check_error(__FILE__, __LINE__, result);

    int nfft = 4;
    int n[] = {100, 100, 100};
    int fft_size = n[0] * n[1] * n[2];
    size_t work_size;
    //result = cufftGetSizeMany(plan, 3, n, n, 1, fft_size, n, 1, fft_size, CUFFT_Z2Z, nfft, &work_size);
    //result = cufftGetSizeMany(plan, 3, n, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, nfft, &work_size);
    result = cufftEstimateMany(3, n, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, nfft, &work_size);
    cufft_check_error(__FILE__, __LINE__, result);

    printf("FFT size: %i\n", fft_size);
    printf("number of FFTs: %i\n", nfft);
    printf("estimated work size: %li\n", work_size);

}
