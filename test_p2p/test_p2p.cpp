#include <mpi.h>
#include <stdlib.h>
#include <vector>

int main(int argn, char** argv)
{
    if (argn == 1)
    {
        printf("Usage: %s N\n", argv[0]);
        printf("  N is the message size in Kb\n");
        exit(0);
    }
    MPI_Init(&argn, &argv);

    int N = atoi(argv[1]) * 1024;
    std::vector<char> a(N, 1234);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request request;

    if (rank == 0)
    {
        //MPI_Isend(&a[0], N, MPI_CHAR, 1, 13, MPI_COMM_WORLD, &request);
        MPI_Send(&a[0], N, MPI_CHAR, 1, 13, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&a[0], N, MPI_CHAR, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tval = -MPI_Wtime();

    for (int i = 0; i < 20; i++)
    {
        if (rank == 0)
        {
            //MPI_Isend(&a[0], N, MPI_CHAR, 1, 13, MPI_COMM_WORLD, &request);
            MPI_Send(&a[0], N, MPI_CHAR, 1, 13, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&a[0], N, MPI_CHAR, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tval += MPI_Wtime();

    double sz = N / double(1 << 20);

    printf("size: %.4f MB, time: %.4f (us), transfer speed: %.4f MB/sec\n", sz, tval * 1e6, 20 * sz / tval);

    MPI_Finalize();
}
