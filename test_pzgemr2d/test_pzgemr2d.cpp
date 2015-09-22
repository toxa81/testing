#include <mpi.h>
#include <cstdint>
#include <complex>
#include <vector>

typedef std::complex<double> double_complex;

extern "C" void pzgemm_(const char* transa, const char* transb, 
                        int32_t* m, int32_t* n, int32_t* k, 
                        double_complex* aplha,
                        double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                        double_complex* b, int32_t* ib, int32_t* jb, int32_t* descb,
                        double_complex* beta,
                        double_complex* c, int32_t* ic, int32_t* jc, int32_t* descc,
                        int32_t transa_len, int32_t transb_len);

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

extern "C" void pzgemr2d_(int32_t* m,
                          int32_t* n,
                          double_complex* A,
                          int32_t* ia,
                          int32_t* ja,
                          int32_t const* desca,  
                          double_complex* B,
                          int32_t* ib,
                          int32_t* jb,
                          int32_t const* descb,
                          int32_t* gcontext);

void test_gemr2d(int M, int N)
{
    int repeat = 10;
    
    int32_t one = 1;
    int32_t isrc = 0;

    int32_t num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    int32_t rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int32_t bs_row_A = M / num_ranks + std::min(1, M % num_ranks);
    int32_t bs_col_A = 1;

    int32_t bs_row_B = 1;
    int32_t bs_col_B = 1;

    int32_t blacs_handler = Csys2blacs_handle(MPI_COMM_WORLD);
    int32_t context1 = blacs_handler;
    int32_t context2 = blacs_handler;

    /* create BLACS context */
    Cblacs_gridinit(&context1, "C", num_ranks, 1);
    /* get row and column ranks */
    int32_t rank_row1, rank_col1;
    Cblacs_gridinfo(context1, &num_ranks, &one, &rank_row1, &rank_col1);
    /* get local number of rows and columns of a matrix */
    int32_t num_rows_local1, num_cols_local1;
    num_rows_local1 = numroc_(&M, &bs_row_A, &rank_row1, &isrc, &num_ranks);
    num_cols_local1 = numroc_(&N, &bs_col_A, &rank_col1, &isrc, &one);


    Cblacs_gridinit(&context2, "C", 1, num_ranks);
    int32_t rank_row2, rank_col2;
    Cblacs_gridinfo(context2, &one, &num_ranks, &rank_row2, &rank_col2);
    int32_t num_rows_local2, num_cols_local2;
    num_rows_local2 = numroc_(&M, &bs_row_B, &rank_row2, &isrc, &one);
    num_cols_local2 = numroc_(&N, &bs_col_B, &rank_col2, &isrc, &num_ranks);

    if (rank == 0)
    {
        printf("local dimensions of A: %i x %i\n", num_rows_local1, num_cols_local1);
        printf("local dimensions of B: %i x %i\n", num_rows_local2, num_cols_local2);
    } 

    int32_t descA[9], descB[9], info;
    descinit_(descA, &M, &N, &bs_row_A, &bs_col_A, &isrc, &isrc, &context1, &num_rows_local1, &info);
    descinit_(descB, &M, &N, &bs_row_B, &bs_col_B, &isrc, &isrc, &context2, &num_rows_local2, &info);

    std::vector<double_complex> A(num_rows_local1 * num_cols_local1);
    std::vector<double_complex> B(num_rows_local2 * num_cols_local2, double_complex(0, 0));
    std::vector<double_complex> C(num_rows_local1 * num_cols_local1);

    for (int i = 0; i < num_rows_local1 * num_cols_local1; i++)
    {
        A[i] = double_complex(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        C[i] = A[i];
    }

    double time = -MPI_Wtime();
    for (int i = 0; i < repeat; i++)
    {
        pzgemr2d_(&M, &N, &A[0], &one, &one, descA, &B[0], &one, &one, descB, &context1);
    }
    time += MPI_Wtime();

    if (rank == 0)
    {
        printf("average time %.4f sec, swap speed: %.4f GB/sec\n", time / repeat,
               sizeof(double_complex) * repeat * M * N / double(1 << 30) / time);
    }

    /* check correctness */
    pzgemr2d_(&M, &N, &B[0], &one, &one, descB, &A[0], &one, &one, descA, &context1);
    for (int i = 0; i < num_rows_local1 * num_cols_local1; i++)
    {
        if (std::abs(A[i] - C[i]) > 1e-14)
        {
            printf("Fail.\n");
            exit(0);
        }
    }

    Cblacs_gridexit(context1);
    Cblacs_gridexit(context2);
    Cfree_blacs_system_handle(blacs_handler);
}

int main(int argn, char** argv)
{
    if (argn != 3)
    {
        printf("Usage: %s M N\n", argv[0]);
        exit(0);
    }
    
    MPI_Init(NULL, NULL);

    int32_t rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int32_t M, N;
    std::istringstream iss(argv[1]);
    iss >> M;
    iss.str(argv[2]);
    iss.clear();
    iss >> N;
    if (rank == 0)
    {
        printf("global matrix dimensions: %i %i\n", M, N);
    }
    test_gemr2d(M, N);

    MPI_Finalize();
}
