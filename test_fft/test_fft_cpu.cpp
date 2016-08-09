#include <fftw3.h>
#include <omp.h>
#include <vector>
#include <complex>
#include <cstdlib>
#include <sys/time.h>

typedef std::complex<double> double_complex;

inline double current_time()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

inline std::complex<double> rnd()
{
    return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
}

int main(int argn, char** argv)
{
    int nx{100}, ny{100};
    int nfft{100};

    std::vector<fftw_plan> plan_forward_xy_;
    plan_forward_xy_  = std::vector<fftw_plan>(omp_get_max_threads());
    
    std::vector<double_complex*> fftw_buffer_xy_;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        fftw_buffer_xy_.push_back((double_complex*)fftw_malloc(nx * ny * sizeof(double_complex)));
    }

    for (int i = 0; i < omp_get_max_threads(); i++) {
        plan_forward_xy_[i] = fftw_plan_dft_2d(ny, nx, (fftw_complex*)fftw_buffer_xy_[i], 
                                               (fftw_complex*)fftw_buffer_xy_[i], FFTW_FORWARD, FFTW_ESTIMATE);
    }

    double t = -current_time();
    for (int i = 0; i < 200; i++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int iz = 0; iz < nfft; iz++) {
                for (int k = 0; k < nx * ny; k++) {
                    fftw_buffer_xy_[tid][k] = rnd();
                }
                fftw_execute(plan_forward_xy_[tid]);
            }
        }
    }
    t += current_time();
    printf("execution time: %.4f sec.\n", t);
    printf("speed: %.4f 2D FFTs / sec.\n", 200 * nfft / t);

}
