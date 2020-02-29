#include <mpi.h>

int main(int argn, char** argv)
{
    int provided;

    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::printf("Warning! Required level of thread support is not provided");
    }

    MPI_Finalize();

    return 0;
}
