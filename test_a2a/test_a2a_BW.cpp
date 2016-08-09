#include <mpi.h>
#include <vector>
#include <chrono>
#include <complex>
#include <stdio.h>
#include <stdlib.h>

#define NOW std::chrono::high_resolution_clock::now()

void test_a2a_BW(int local_size)
{
    int rank, num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<char> sbuf(local_size * num_ranks, 255);
    std::vector<char> rbuf(local_size * num_ranks, 255);

    std::vector<int> sendcounts(num_ranks);
    std::vector<int> sdispls(num_ranks);
    std::vector<int> recvcounts(num_ranks);
    std::vector<int> rdispls(num_ranks);

    for (int rank = 0; rank < num_ranks; rank++)
    {
        sendcounts[rank] = local_size;
        sdispls[rank]    = local_size * rank;

        recvcounts[rank] = local_size;
        rdispls[rank]    = local_size * rank;
    }

    /* warmup */
    MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_CHAR,
                  &rbuf[0], &recvcounts[0], &rdispls[0], MPI_CHAR,
                  MPI_COMM_WORLD);
    
    int repeat = 40;
    std::vector<double> times(repeat);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = NOW;
    for (int i = 0; i < repeat; i++)
    {
        auto tt = NOW;
        MPI_Alltoallv(&sbuf[0], &sendcounts[0], &sdispls[0], MPI_CHAR,
                      &rbuf[0], &recvcounts[0], &rdispls[0], MPI_CHAR,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        times[i] = std::chrono::duration_cast< std::chrono::duration<double> >(NOW - tt).count();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t = std::chrono::duration_cast< std::chrono::duration<double> >(NOW - t0).count();

    for (int i = 0; i < num_ranks; i++)
    {
        if (rank == i)
        {
            printf("rank: %i, injection bandwidth: %.3f GB/sec.\n", i, double(repeat) * (num_ranks - 1) * local_size / t / (1 << 30));
            double avg = 0;
            for (int j = 0; j < repeat; j++) avg += times[i];
            avg /= repeat;
            double variance = 0;
            for (int i = 0; i < repeat; i++) variance += std::pow(times[i] - avg, 2);
            double sigma = std::sqrt(variance);
            printf("         average time: %.4f (us), sigma: %.4f (us), CV: %.2f%%\n", avg * 1e6, sigma * 1e6, 100 * sigma / avg);
        }
    }
}

int main(int argn, char** argv)
{
    if (argn == 1)
    {
        printf("Usage: %s N\n", argv[0]);
        printf("  N is the message size in Kb\n");
        exit(0);
    }

    int N = atoi(argv[1]) * 1024;

    MPI_Init(NULL, NULL);

    test_a2a_BW(N);

    MPI_Finalize();
}
