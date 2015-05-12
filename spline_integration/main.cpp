#include <sys/time.h>
#include "platform.h"
#include "mdarray.h"

extern "C" void spline_inner_product_gpu_v3(int const* idx_ri__,
                                            int num_ri__,
                                            int num_points__,
                                            double const* x__,
                                            double const* dx__,
                                            double const* f__, 
                                            double const* g__,
                                            double* result__);
inline double wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

void test_spline_integration()
{
    int num_x_points = 1000;
    mdarray<double, 1> x(num_x_points);
    mdarray<double, 1> dx(num_x_points - 1);
   
    /* linear grid */
    for (int i = 0; i < num_x_points; i++) x(i) = 0.01 * i;
    for (int i = 0; i < num_x_points - 1; i++) dx(i) = 0.01;

    /* spline coefficients */
    int N = 100;
    mdarray<double, 3> f(num_x_points, 4, N);
    mdarray<double, 3> g(num_x_points, 4, N);

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < num_x_points; j++)
            {
                f(j, k, i) = double(rand()) / RAND_MAX;
                g(j, k, i) = double(rand()) / RAND_MAX;
            }
        }
    }

    /* pairs of splines */
    mdarray<int, 2> idx(2, N * N);
    int n = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            idx(0, n) = i;
            idx(1, n) = j;
            n++;
        }
    }

    mdarray<double, 1> result(N * N);

    x.allocate_on_device();
    x.copy_to_device();
    dx.allocate_on_device();
    dx.copy_to_device();
    f.allocate_on_device();
    f.copy_to_device();
    g.allocate_on_device();
    g.copy_to_device();
    idx.allocate_on_device();
    idx.copy_to_device();
    result.allocate_on_device();
    
    double tval = -wtime();
    spline_inner_product_gpu_v3(idx.at<GPU>(), N * N, num_x_points, x.at<GPU>(), dx.at<GPU>(),
                                f.at<GPU>(), g.at<GPU>(), result.at<GPU>());
    cuda_device_synchronize();
    tval += wtime();
    result.copy_to_host();
    
    printf("number of mesh points : %i\n", num_x_points);
    printf("number of integrals   : %i\n", N * N);
    printf("total time            : %12.6f sec.\n", tval);
    printf("performance           : %12.6f GFlops\n", 1e-9 * N * N * num_x_points * 85 / tval);

}

int main(int argn, char** argv)
{
    Platform::initialize();
    
    for (int i = 0; i < 4; i++) test_spline_integration();

    Platform::finalize();

}
