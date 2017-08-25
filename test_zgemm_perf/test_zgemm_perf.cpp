#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <complex>
#include <vector>
#include <sys/time.h>

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

void dgemm_(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, 
            ftn_double* alpha, ftn_double* A, ftn_int* lda, ftn_double* B,
            ftn_int* ldb, ftn_double* beta, ftn_double* C, ftn_int* ldc, ftn_len transa_len,
            ftn_len transb_len);
}

//double test_zgemm()
//{
//    int N = 1024;
//    ftn_double_complex* A = new ftn_double_complex[N * N];
//    ftn_double_complex* B = new ftn_double_complex[N * N];
//    ftn_double_complex* C = new ftn_double_complex[N * N];
//
//    for (int i = 0; i < N * N; i++)
//    {
//        A[i] = randz();
//        B[i] = randz();
//        C[i] = 0.0;
//    }
//    ftn_double_complex alpha(1, 0);
//    ftn_double_complex beta(0, 0);
//    double t = -omp_get_wtime();
//    zgemm_("N", "N", &N, &N, &N, &alpha, A, &N, B, &N, &beta, C, &N, 1, 1);
//    t += omp_get_wtime();
//    double perf = 8 * 1e-9 * N * N * N / t;
//
//    delete[] A;
//    delete[] B;
//    delete[] C;
//
//    return perf;
//}

inline double wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

double test_dgemm()
{
    int M = 30000;
    int N = 171;
    int K = 63;
    std::vector<ftn_double> A(M * K);
    std::vector<ftn_double> B(K * N);
    std::vector<ftn_double> C(M * N);

    for (int i = 0; i < M * K; i++) {
        A[i] = randz().real();
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = randz().real();
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0;
    }

    ftn_double alpha{1};
    ftn_double beta{0};
    double t = -wtime();
    dgemm_("N", "T", &M, &N, &K, &alpha, A.data(), &M, B.data(), &N, &beta, C.data(), &M, 1, 1);
    t += wtime();
    double perf = 2 * 1e-9 * M * N * K / t;

    return perf;
}


int main(int argn, char** argv)
{
    int repeat = 10;

    test_dgemm();
    double avg = 0;
    std::vector<double> perf(repeat);
    for (int i = 0; i < repeat; i++) {
        perf[i] = test_dgemm();
        avg += perf[i];
    }
    avg /= repeat;

    printf("average performance: %12.4f GFlops\n", avg);

    
    //int nthreads = omp_get_max_threads();
    //printf("number of OMP threads: %i\n", nthreads);
    //std::vector<double> perf(repeat);
    //double avg = 0;
    //for (int i = 0; i < repeat; i++)
    //{
    //    perf[i] = test_zgemm();
    //    avg += perf[i];
    //}
    //avg /= repeat;
    //double variance = 0;
    //for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    //variance /= repeat;
    //double sigma = std::sqrt(variance);
    //printf("average performance: %12.4f GFlops\n", avg);
    //printf("sigma: %12.4f GFlops\n", sigma);

    //omp_set_num_threads(std::max(nthreads / 2, 1));

    //printf("number of OMP threads: %i\n", omp_get_max_threads());
    //avg = 0;
    //for (int i = 0; i < repeat; i++)
    //{
    //    perf[i] = test_zgemm();
    //    avg += perf[i];
    //}
    //avg /= repeat;
    //variance = 0;
    //for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    //variance /= repeat;
    //sigma = std::sqrt(variance);
    //printf("average performance: %12.4f GFlops\n", avg);
    //printf("sigma: %12.4f GFlops\n", sigma);

    //omp_set_num_threads(nthreads);

    //printf("number of OMP threads: %i\n", omp_get_max_threads());
    //avg = 0;
    //for (int i = 0; i < repeat; i++)
    //{
    //    perf[i] = test_zgemm();
    //    avg += perf[i];
    //}
    //avg /= repeat;
    //variance = 0;
    //for (int i = 0; i < repeat; i++) variance += std::pow(perf[i] - avg, 2);
    //variance /= repeat;
    //sigma = std::sqrt(variance);
    //printf("average performance: %12.4f GFlops\n", avg);
    //printf("sigma: %12.4f GFlops\n", sigma);

}
