#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define MAX_ITER 2048

inline __host__ __device__ int num_blocks(int length, int block_size)
{
    return (length / block_size) + min(length % block_size, 1);
}


inline int __device__ __host__ point_depth(double x, double y, int max_iter)
{
    double t1, t2;
    double xn = x;
    double yn = y;
    for (int iter = 0; iter < max_iter; iter++)
    {
        t1 = xn * xn;
        t2 = yn * yn;
    
        yn = 2 * xn * yn + y;
        xn = t1 - t2 + x;

        if (t1 + t2 > 4) return iter;
    }
    return max_iter;
}

__global__ void mandelbrot_set(int* raw_data, double x0, double y0, double scale, int Nx, int Ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < Nx && j < Ny)
    {
        double x = x0 + scale * double(i - (Nx >> 1)) / Nx;
        double y = y0 + scale * double(j - (Ny >> 1)) / Nx;

        int k = 0;
        
        double xn = x;
        double yn = y;
        while (true)
        {   
            double t1 = xn * xn;
            double t2 = yn * yn;

            if (t1 + t2 > 4 || k == MAX_ITER)
            {
                raw_data[i + Nx * j] = k;
                return;
            }

            double t3 = 2 * xn;
            xn = t1 - t2 + x;
            yn = t3 * yn + y;
            k++;
        }
    }
}
__global__ void mandelbrot_set_v2(int* raw_data, double x0, double y0, double scale, int Nx, int Ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < Nx && j < Ny)
    {
        double x = x0 + scale * double(i - (Nx >> 1)) / Nx;
        double y = y0 + scale * double(j - (Ny >> 1)) / Nx;

        double xn = x;
        double yn = y;
        
        raw_data[i + Nx * j] = 0;

        for (int k = 0; k < 100; k++)
        {
            double t1, t2;
            for (int iter = 0; iter < 20; iter++)
            {
                t1 = xn * xn;
                t2 = yn * yn;

                yn = 2 * xn * yn + y;
                xn = t1 - t2 + x;
            }

            if (t1 + t2 < 4)
            {
                raw_data[i + Nx * j] = k;
            }
            else
            {
                return;
            }
        }
    }
}

__global__ void mandelbrot_set_v3(int* raw_data, double x0, double y0, double scale, int Nx, int Ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockIdx.y;

    if (ix < Nx)
    {
        double x = x0 + scale * double(ix - (Nx >> 1)) / Nx;
        double y = y0 + scale * double(iy - (Ny >> 1)) / Ny;

        raw_data[ix + Nx * iy] = point_depth(x, y, MAX_ITER);
    }
}

void test3(int Nx, int Ny, int* d_ptr, int* h_ptr)
{
    dim3 dim_t(32);
    dim3 dim_b(num_blocks(Nx, dim_t.x), Ny);
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    mandelbrot_set_v3 <<<dim_b, dim_t>>>(d_ptr, -0.6922576383364505, 0.3261539381815615, 0.02, Nx, Ny);
    cudaDeviceSynchronize();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tdiff = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);
    double val = tdiff.count();

    printf("time: %18.12f\n", val);
    cudaMemcpy(h_ptr, d_ptr, Nx * Ny * sizeof(int), cudaMemcpyDeviceToHost);

    size_t nop = 0;
    for (size_t i = 0; i < Nx * Ny; i++) nop += h_ptr[i];
    printf("performance: %12.6f GFlops\n", 8e-9 * nop / val);
}

int main(int argn, char** argv)
{
    int* d_raw_data;
    int* h_raw_data;

    int Nx = 2048;
    int Ny = 2048;
    
    h_raw_data = (int*)malloc(Nx * Ny * sizeof(int));
    cudaMalloc(&d_raw_data, Nx * Ny * sizeof(int));

    for (int i = 0; i < 10; i++)
    {
        test3(Nx, Ny, d_raw_data, h_raw_data);
    }
    
    //dim3 threadsPerBlock(8, 8);
    //dim3 numBlocks(Nx / 8, Ny / 8);

    //dim3 dim_t(32);
    //dim3 dim_b(num_blocks(Nx, dim_t.x), Ny);
    //
    //std::chrono::high_resolution_clock::time_point starting_time_ = std::chrono::high_resolution_clock::now();
    ////mandelbrot_set<<<numBlocks, threadsPerBlock>>>(d_raw_data, -0.7241128263797995, 0.28645642661272747, 0.00001, Nx, Ny);

    ////mandelbrot_set_v2 <<<numBlocks, threadsPerBlock>>>(d_raw_data, -0.6922576383364505, 0.3261539381815615, 0.02, Nx, Ny);
    //mandelbrot_set_v3 <<<dim_b, dim_t>>>(d_raw_data, -0.6922576383364505, 0.3261539381815615, 0.02, Nx, Ny);
    //cudaDeviceSynchronize();
    //
    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> tdiff = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - starting_time_);
    //double val = tdiff.count();

    //printf("time: %18.12f\n", val);

    //cudaMemcpy(h_raw_data, d_raw_data, Nx * Ny * sizeof(int), cudaMemcpyDeviceToHost);
    //size_t nop = 0;
    //for (size_t i = 0; i < Nx * Ny; i++) nop += h_raw_data[i];
    //printf("performance: %12.6f GFlops\n", 8e-9 * nop / val);



    FILE* fout = fopen("raw_data.txt", "w");
    fprintf(fout, "%i %i %i\n", Nx, Ny, MAX_ITER);
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++) fprintf(fout, "%i ", h_raw_data[j + i * Nx]);
        fprintf(fout,"\n");
    }
    fclose(fout);


    cudaFree(d_raw_data);
    free(h_raw_data);

    return 0;


}
