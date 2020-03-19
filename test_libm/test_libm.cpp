#include <complex>
#include <iostream>
#include <omp.h>

void run_test(int n, std::complex<double>& z)
{
    double d = 1.0 / n;
    for (int j = 0; j < n; j++) {
        double p = d * j * 2 * 3.141592;
        z += std::exp(std::complex<double>(0.0, p));
    }
}

int main(int argn, char** argv)
{
    int n = 10000000;
    std::complex<double> z(0, 0);

    double t = -omp_get_wtime();
    for (int j = 0; j < 10; j++) {
        run_test(n, z);
    }
    t += omp_get_wtime();
    std::cout << "time= " << t << " sec." << std::endl
              << "z=" << z << std::endl;
    return 0;
}
