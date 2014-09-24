#include <array>
#include <cuda.h>
#include <mpi.h>

void test(int N, double f)
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::vector<double> h_A(N);
    double* d_A;
    cudaMalloc((void**)&d_A, N * sizeof(double));

    for (int i = 0; i < N; i++) h_A[i] = 1 + f * i;
    
    cudaMemcpy(d_A, &h_A[0], N * sizeof(double), cudaMemcpyHostToDevice);
    
    MPI_Allreduce(MPI_IN_PLACE, d_A, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    cudaMemcpy(&h_A[0], d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

    int num_err = 0;
    for (int i = 0; i < N; i++) 
    {
        if (std::abs(h_A[i] - (1 + f * i) * num_ranks) > 1e-12) num_err++;
    }
    
    printf("Scaling factor: %f\n", f);
    if (num_err != 0)
    {
        printf("test failed, number of errors: %i\n", num_err);
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
