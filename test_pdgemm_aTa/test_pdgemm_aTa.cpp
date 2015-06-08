#include <mpi.h>
#include <cstdint>
#include <complex>
#include <vector>
#include "mpi_grid.h"

extern "C" void pdgemm_(const char* transa, const char* transb, 
                        int32_t* m, int32_t* n, int32_t* k, 
                        double* aplha,
                        double* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                        double* b, int32_t* ib, int32_t* jb, int32_t* descb,
                        double* beta,
                        double* c, int32_t* ic, int32_t* jc, int32_t* descc,
                        int32_t transa_len, int32_t transb_len);

extern "C" int32_t numroc_(int32_t* n, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" void descinit_(int32_t* desc, int32_t* m, int32_t* n, int32_t* mb, int32_t* nb, int32_t* irsrc, int32_t* icsrc, 
                          int32_t* ictxt, int32_t* lld, int32_t* info);

extern "C" int Csys2blacs_handle(MPI_Comm SysCtxt);
extern "C" MPI_Comm Cblacs2sys_handle(int BlacsCtxt);
extern "C" void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);
extern "C" void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);
extern "C" void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);
extern "C" void Cfree_blacs_system_handle(int ISysCtxt);
extern "C" void Cblacs_barrier(int ConTxt, const char* scope);
extern "C" void Cblacs_gridexit(int ConTxt);

#ifdef __LIBSCI_ACC
extern "C" int libsci_acc_HostAlloc(void **ptr, size_t numBytes);
extern "C" int libsci_acc_HostFree(void *ptr);
extern "C" void libsci_acc_init();
extern "C" void libsci_acc_finalize();
extern "C" int libsci_acc_HostRegister(void *ptr, size_t size);
extern "C" int libsci_acc_HostUnregister(void *ptr);
#endif

void test_real(Communicator const& comm, int M, int K, int num_ranks_row, int num_ranks_col, int bs_row, int bs_col)
{
    int32_t blacs_handler = Csys2blacs_handle(comm.mpi_comm());
    int32_t context = blacs_handler;

    /* create a BLACS context */
    Cblacs_gridinit(&context, "C", num_ranks_row, num_ranks_col);

    /* get row and column ranks */
    int32_t rank_row, rank_col;
    Cblacs_gridinfo(context, &num_ranks_row, &num_ranks_col, &rank_row, &rank_col);
    
    /* get local number of rows and columns of a matrix */
    int32_t num_A_rows_local, num_A_cols_local;
    int32_t num_C_rows_local, num_C_cols_local;
    int32_t isrc = 0;
    /* A is a K x M matrix, K is huge */
    num_A_rows_local = numroc_(&K, &bs_row, &rank_row, &isrc, &num_ranks_row);
    num_A_cols_local = numroc_(&M, &bs_col, &rank_col, &isrc, &num_ranks_col);
    /* C is M x M matrix */
    /* block size of C is the same in both directions */
    num_C_rows_local = numroc_(&M, &bs_col, &rank_row, &isrc, &num_ranks_row);
    num_C_cols_local = numroc_(&M, &bs_col, &rank_col, &isrc, &num_ranks_col);

    int32_t info;
    int32_t desc_A[9];
    int32_t desc_C[9];
    int32_t ld_A = std::max(1, num_A_rows_local); 
    int32_t ld_C = std::max(1, num_C_rows_local); 
    descinit_(desc_A, &K, &M, &bs_row, &bs_col, &isrc, &isrc, &context, &ld_A, &info);
    descinit_(desc_C, &M, &M, &bs_col, &bs_col, &isrc, &isrc, &context, &ld_C, &info);

    std::vector<double> A(num_A_rows_local * num_A_cols_local);
    std::vector<double> C(num_C_rows_local * num_C_cols_local, 0.0);
    
    #ifdef __LIBSCI_ACC
    libsci_acc_HostRegister(&A[0], A.size() * sizeof(double));
    libsci_acc_HostRegister(&C[0], C.size() * sizeof(double));
    #endif

    for (size_t i = 0; i < A.size(); i++) A[i] = double(rand()) / RAND_MAX;

    double alpha = 1.0;
    double beta = 0.0;
    int32_t ione = 1;

    const char* transA = "T";
    const char* transB = "N";

    double time = -MPI_Wtime();
    pdgemm_(transA, transB, &M, &M, &K, &alpha, &A[0], &ione, &ione, desc_A, &A[0], &ione, &ione, desc_A, &beta, &C[0], &ione, &ione, desc_C, 1, 1);
    comm.barrier();
    time += MPI_Wtime();

    #ifdef __LIBSCI_ACC
    libsci_acc_HostUnregister(&A[0]);
    libsci_acc_HostUnregister(&C[0]);
    #endif

    if (rank_row == 0 && rank_col == 0)
    {
        printf("time: %f (sec.), pdgemm performance: %f (GFlops / rank)\n", time, 2e-9 * M * M * K / time / num_ranks_row / num_ranks_col);
    }

    Cblacs_gridexit(context);
    Cfree_blacs_system_handle(blacs_handler);
}

#if defined(__PILAENV_BLOCKSIZE)
extern "C" int pilaenv_(int* ctxt, char* prec) 
{
    return __PILAENV_BLOCKSIZE;
}
#endif


int main(int argn, char** argv)
{
    if (argn != 6)
    {
        printf("Usage: %s K(large) M(small) bs_row bs_col comm_size\n", argv[0]);
        exit(0);
    }

    int32_t M, K, bs_row, bs_col, comm_size;
    std::istringstream(argv[1]) >> K;
    std::istringstream(argv[2]) >> M;
    std::istringstream(argv[3]) >> bs_row;
    std::istringstream(argv[4]) >> bs_col;
    std::istringstream(argv[5]) >> comm_size;
    
    MPI_Init(NULL, NULL);

    {
        int32_t num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        std::vector<int> grid_dims = {comm_size, num_ranks / comm_size};
        Communicator comm_world(MPI_COMM_WORLD);
        MPI_grid mpi_grid(grid_dims, comm_world);

        //int32_t rank;
        //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int32_t num_ranks_row, num_ranks_col;
        num_ranks_row = comm_size;
        num_ranks_col = 1;

        //== if (rank == 0)
        //== {
        //==     printf("Running on %i rank(s) (%i x %i grid), ", num_ranks, num_ranks_row, num_ranks_col);
        //==     printf("global martix dimensions (M, K): %i, %i\n", M, K);
        //==     printf("bs_row: %i, bs_col: %i\n", bs_row, bs_col);
        //== }

        #ifdef __LIBSCI_ACC
        libsci_acc_init();
        #endif

        test_real(mpi_grid.communicator(1 << 0), M, K, num_ranks_row, num_ranks_col, bs_row, bs_col);
        comm_world.barrier();
        if (comm_world.rank() == 0) printf("===============\n");
        test_real(mpi_grid.communicator(1 << 0), M, K, num_ranks_row, num_ranks_col, bs_row, bs_col);
        
        #ifdef __LIBSCI_ACC
        libsci_acc_finalize();
        #endif
    }

    MPI_Finalize();
}
