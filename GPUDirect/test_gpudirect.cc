#include <vector>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>

void test(int N, double f)
{
    int num_ranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> h_A(N);
    double* d_A;
    cudaMalloc((void**)&d_A, N * sizeof(double));

    for (int i = 0; i < N; i++) h_A[i] = 1 + f * i;
    
    cudaMemcpy(d_A, &h_A[0], N * sizeof(double), cudaMemcpyHostToDevice);
    
    MPI_Allreduce(MPI_IN_PLACE, d_A, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    cudaMemcpy(&h_A[0], d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

    int num_err = 0;
    int i0 = -1;
    int i1 = -1;
    for (int i = 0; i < N; i++) 
    {
        if (std::abs(h_A[i] - double((1 + f * i) * num_ranks)) > 1e-12)
        {
            i1 = i;
            if (i0 == -1) i0 = i;
            num_err++;
        }
    }
    
    printf("rank: %i, vector size: %i, scaling factor: %f, ", rank, N, f);
    if (num_err != 0)
    {
        printf("test failed, number of errors: %i, first error at %i, last at %i\n", num_err, i0, i1);
    }
    else
    {
        printf("test passed!\n");
    }

    cudaFree(d_A);
}

int main(int argn, char** argv)
{
    if (argn != 2)
    {
        printf("Usage: %s N\n", argv[0]);
        exit(0);
    }

    int N = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    test(N, 0);
    test(N, 1);

    MPI_Finalize();
}
