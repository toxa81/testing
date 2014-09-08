#include <array>
#include <cuda.h>
#include <mpi.h>

int main(int argn, char** argv)
{
    MPI_Init(NULL, NULL);

    double* d_A;

    const int N = 1000;
    std::array<double, N> h_A;
    
    cudaMalloc((void**)&d_A, N * sizeof(double));

    for (int i = 0; i < N; i++) h_A[i] = 1.0;

    cudaMemcpy(d_A, &h_A[0], N * sizeof(double), cudaMemcpyHostToDevice);
    
    MPI_Allreduce(MPI_IN_PLACE, d_A, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    cudaMemcpy(&h_A[0], d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << h_A[0] << " " << h_A[N - 1] << std::endl;

    cudaFree(d_A);
    MPI_Finalize();
}
