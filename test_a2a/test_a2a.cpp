#include <mpi.h>
#include <vector>
#include <complex>
#include <stdio.h>
#include <stdlib.h>

void test_4_to_1()
{
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

    /* all ranks send the same thing */
    sendcounts = {1, 0, 0, 0};
    sdispls    = {0, -1, -1, -1};

    /* first rank receives all */
    if (rank == 0)
    {
        recvcounts = {1, 1, 1, 1};
        rdispls    = {0, 1, 2, 3};
    }
    else /* other ranks don't receive */
    {
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
}

void test_4_to_2()
{
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

    /* first and second ranks receive 2 elements each */
    std::vector<double> rbuf;
    if (rank == 0 || rank == 1) rbuf = std::vector<double>(1 * 2);
    std::vector<int> sendcounts(4);
    std::vector<int> sdispls(4);
    std::vector<int> recvcounts(4);
    std::vector<int> rdispls(4);

    /* first rank receives all */
    if (rank == 0)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {1, 1, 0, 0};
        rdispls    = {0, 1, -1, -1};
    }
    if (rank == 1)
    {
        sendcounts = {1, 0, 0, 0};
        sdispls    = {0, -1, -1, -1};

        recvcounts = {0, 0, 1, 1};
        rdispls    = {-1, -1, 0, 1};
    }
    if (rank == 2)
    {
        sendcounts = {0, 1, 0, 0};
        sdispls    = {-1, 0, -1, -1};

        recvcounts = {0, 0, 0, 0};
        rdispls    = {-1, -1, -1, -1};
    }
    if (rank == 3)
    {
        sendcounts = {0, 1, 0, 0};
        sdispls    = {-1, 0, -1, -1};

        recvcounts = {0, 0, 0, 0};
        rdispls    = {-1, -1, -1, -1};
    }

    MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_DOUBLE,
                  &rbuf[0], &recvcounts[0], &rdispls[0], MPI_DOUBLE,
                  MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < 2; i++)
        {
            if (rbuf[i] != 100.0 + i)
            {
                printf("Fail!\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        printf("OK\n");
    }
    if (rank == 1)
    {
        for (int i = 0; i < 2; i++)
        {
            if (rbuf[i] != 100.0 + 2 + i)
            {
                printf("Fail!\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        printf("OK\n");
    }
}

void test_3_to_2()
{
    int rank, num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (num_ranks != 3)
    {
        printf("test must be executed on 3 ranks\n");
        exit(0);
    }

    std::vector< std::complex<double> > sbuf(400);

    for (int i = 0; i < 400; i++) sbuf[i] = (rank + 1) * 1000.0 + i;

    std::vector< std::complex<double> > rbuf;
    if (rank == 0 || rank == 2) rbuf = std::vector< std::complex<double> >(600);
    std::vector<int> sendcounts(3);
    std::vector<int> sdispls(3);
    std::vector<int> recvcounts(3);
    std::vector<int> rdispls(3);

    if (rank == 0)
    {
        sendcounts = {400, 0, 0};
        sdispls    = {0, -1, -1};

        recvcounts = {400, 200, 0};
        rdispls    = {0, 400, -1};
    }
    if (rank == 1)
    {
        sendcounts = {200, 0, 200};
        sdispls    = {0, -1, 200};

        recvcounts = {0, 0, 0};
        rdispls    = {-1, -1, -1};
    }
    if (rank == 2)
    {
        sendcounts = {0, 0, 400};
        sdispls    = {-1, -1, 0};

        recvcounts = {0, 200, 400};
        rdispls    = {-1, 0, 200};
    }

    MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_COMPLEX16,
                  &rbuf[0], &recvcounts[0], &rdispls[0], MPI_COMPLEX16,
                  MPI_COMM_WORLD);

    //if (rank == 0)
    //{
    //    for (int i = 0; i < 2; i++)
    //    {
    //        if (rbuf[i] != 100.0 + i)
    //        {
    //            printf("Fail!\n");
    //            MPI_Abort(MPI_COMM_WORLD, -1);
    //        }
    //    }
    //    printf("OK\n");
    //}
    //if (rank == 1)
    //{
    //    for (int i = 0; i < 2; i++)
    //    {
    //        if (rbuf[i] != 100.0 + 2 + i)
    //        {
    //            printf("Fail!\n");
    //            MPI_Abort(MPI_COMM_WORLD, -1);
    //        }
    //    }
    //    printf("OK\n");
    //}
}

void test_3_to_2_double()
{
    int rank, num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (num_ranks != 3)
    {
        printf("test must be executed on 3 ranks\n");
        exit(0);
    }

    std::vector<double> sbuf(400);

    for (int i = 0; i < 400; i++) sbuf[i] = (rank + 1) * 1000.0 + i;

    std::vector<double> rbuf;
    if (rank == 0 || rank == 2) rbuf = std::vector<double>(600);
    std::vector<int> sendcounts(3);
    std::vector<int> sdispls(3);
    std::vector<int> recvcounts(3);
    std::vector<int> rdispls(3);

    if (rank == 0)
    {
        sendcounts = {400, 0, 0};
        sdispls    = {0, -1, -1};

        recvcounts = {400, 200, 0};
        rdispls    = {0, 400, -1};
    }
    if (rank == 1)
    {
        sendcounts = {200, 0, 200};
        sdispls    = {0, -1, 200};

        recvcounts = {0, 0, 0};
        rdispls    = {-1, -1, -1};
    }
    if (rank == 2)
    {
        sendcounts = {0, 0, 400};
        sdispls    = {-1, -1, 0};

        recvcounts = {0, 200, 400};
        rdispls    = {-1, 0, 200};
    }

    MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_DOUBLE,
                  &rbuf[0], &recvcounts[0], &rdispls[0], MPI_DOUBLE,
                  MPI_COMM_WORLD);

    //if (rank == 0)
    //{
    //    for (int i = 0; i < 2; i++)
    //    {
    //        if (rbuf[i] != 100.0 + i)
    //        {
    //            printf("Fail!\n");
    //            MPI_Abort(MPI_COMM_WORLD, -1);
    //        }
    //    }
    //    printf("OK\n");
    //}
    //if (rank == 1)
    //{
    //    for (int i = 0; i < 2; i++)
    //    {
    //        if (rbuf[i] != 100.0 + 2 + i)
    //        {
    //            printf("Fail!\n");
    //            MPI_Abort(MPI_COMM_WORLD, -1);
    //        }
    //    }
    //    printf("OK\n");
    //}
}

int main(int arg, char** argv)
{
    MPI_Init(NULL, NULL);
    
    //test_4_to_1();
    //test_4_to_2();
    //test_3_to_2();
    test_3_to_2_double();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
