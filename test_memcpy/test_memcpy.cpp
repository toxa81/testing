#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <vector>

extern "C" void memcpy_custom(double* out, double* in, int length);

int main(int argn, char** argv)
{
    int repeat = 200;
    int N = (1 << 20);
    std::vector<double> inp(N, 1);
    std::vector<double> out(N, 2);

    double t = -omp_get_wtime();
    if (argn == 1)
    {
        for (int i = 0; i < repeat; i++)
        {
            memcpy(&out[0], &inp[0], N * sizeof(double));
        }
    }
    else
    {
        for (int i = 0; i < repeat; i++)
        {
            memcpy_custom(&out[0], &inp[0], N);
        }
    }
    t += omp_get_wtime();

    printf("performance: %.4f MB/sec.\n", repeat * N * sizeof(double) / t / (1 << 20)); 
}
