#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <complex>
#include <vector>

typedef int32_t ftn_int;
typedef int32_t ftn_len;
typedef double ftn_double;
typedef std::complex<double> ftn_double_complex;
typedef char const* ftn_char;

inline std::complex<double> randz()
{
    return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
}

extern "C" {

void zgemm_(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, 
            ftn_double_complex* alpha, ftn_double_complex* A, ftn_int* lda, ftn_double_complex* B,
            ftn_int* ldb, ftn_double_complex* beta, ftn_double_complex* C, ftn_int* ldc, ftn_len transa_len,
            ftn_len transb_len);
}

double test_zgemm()
{
    int N = 1024;
    ftn_double_complex* A = new ftn_double_complex[N * N];
    ftn_double_complex* B = new ftn_double_complex[N * N];
    ftn_double_complex* C = new ftn_double_complex[N * N];

    for (int i = 0; i < N * N; i++)
    {
        A[i] = randz();
        B[i] = randz();
        C[i] = 0.0;
    }
    ftn_double_complex alpha(1, 0);
    ftn_double_complex beta(0, 0);
    double t = -omp_get_wtime();
    zgemm_("N", "N", &N, &N, &N, &alpha, A, &N, B, &N, &beta, C, &N, 1, 1);
    t += omp_get_wtime();
    double perf = 8 * 1e-9 * N * N * N / t;

    delete[] A;
    delete[] B;
    delete[] C;

    return perf;
}


int main(int argn, char** argv)
{
    int repeat = 10;
    
    int nthreads = omp_get_max_threads();
    printf("number of OMP threads: %i\n", nthreads);
    std::vector<double> perf(repeat);
    double avg = 0;
    for (int i = 0; i < repeat; i++)
    {
        perf[i] = test_zgemm();
        avg += perf[i];
    }
    avg /= repeat;
    double variance = 0;
    for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    variance /= repeat;
    double sigma = std::sqrt(variance);
    printf("average performance: %12.4f GFlops\n", avg);
    printf("sigma: %12.4f GFlops\n", sigma);

    omp_set_num_threads(std::max(nthreads / 2, 1));

    printf("number of OMP threads: %i\n", omp_get_max_threads());
    avg = 0;
    for (int i = 0; i < repeat; i++)
    {
        perf[i] = test_zgemm();
        avg += perf[i];
    }
    avg /= repeat;
    variance = 0;
    for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    variance /= repeat;
    sigma = std::sqrt(variance);
    printf("average performance: %12.4f GFlops\n", avg);
    printf("sigma: %12.4f GFlops\n", sigma);

    omp_set_num_threads(nthreads);

    printf("number of OMP threads: %i\n", omp_get_max_threads());
    avg = 0;
    for (int i = 0; i < repeat; i++)
    {
        perf[i] = test_zgemm();
        avg += perf[i];
    }
    avg /= repeat;
    variance = 0;
    for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    variance /= repeat;
    sigma = std::sqrt(variance);
    printf("average performance: %12.4f GFlops\n", avg);
    printf("sigma: %12.4f GFlops\n", sigma);

}
