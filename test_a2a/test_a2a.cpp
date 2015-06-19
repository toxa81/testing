#include <mpi.h>
#include <vector>
#include <stdio.h>

int main(int arg, char** argv)
{
    MPI_Init(NULL, NULL);
    int rank, num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (num_ranks != 4)
    {
        printf("test must be executed on 4 ranks\n");
        exit(0);
    }

    /* each rank sends 1 element */
    std::vector<double> sbuf(1, rank + 100.0);

    /* first rank receives 4 elements */
    std::vector<double> rbuf;
    if (rank == 0) rbuf = std::vector<double>(1 * 4);
    std::vector<int> sendcounts(4);
    std::vector<int> sdispls(4);
    std::vector<int> recvcounts(4);
    std::vector<int> rdispls(4);

    if (rank == 0)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {1, 1, 1, 1};
        rdispls    = {0, 1, 2, 3};
    }

    if (rank == 1)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {0, 0, 0, 0};
        rdispls    = {-1, -1, -1, -1};
    }

    if (rank == 2)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {0, 0, 0, 0};
        rdispls    = {-1, -1, -1, -1};
    }

    if (rank == 3)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {0, 0, 0, 0};
        rdispls    = {-1, -1, -1, -1};
    }

    MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_DOUBLE,
                  &rbuf[0], &recvcounts[0], &rdispls[0], MPI_DOUBLE,
                  MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < 4; i++)
        {
            if (rbuf[i] != 100.0 + i)
            {
                printf("Fail!\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        printf("OK\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
