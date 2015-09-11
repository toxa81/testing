#include <mpi.h>
#include <vector>

int main(int argn, char** argv)
{
    MPI_Init(&argn, &argv);

    int N = 2 * (1 << 20);
    std::vector<double> a(N, 1234);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    double tval = -MPI_Wtime();

    if (rank == 0)
    {
        MPI_Isend(&a[0], N, MPI_DOUBLE, 1, 13, MPI_COMM_WORLD, &request);
    }
    else
    {
        MPI_Recv(&a[0], N, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    tval += MPI_Wtime();

    double sz = N * sizeof(double) / double(1 << 20);

    printf("size: %.4f MB, time: %.4f (us), transfer speed: %.4f MB/sec\n", sz, tval * 1e6, sz / tval);

    MPI_Finalize();
}
