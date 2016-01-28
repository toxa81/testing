#include <mpi.h>
#include <cmath>
#include <sstream>
#include "communicator.h"
#include "blacs_grid.h"
#include "dmatrix.h"
#include "linalg.h"
#include "evp_solver.h"

void test_diag(int N, BLACS_grid& blacs_grid)
{
    int bs = 32;

    dmatrix<ftn_double_complex> a(N, N, blacs_grid, bs, bs);
    dmatrix<ftn_double_complex> b(N, N, blacs_grid, bs, bs);

    dmatrix<ftn_double_complex> a_orig(N, N, blacs_grid, bs, bs);

    auto rnd = []()
    {
        return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
    };

    for (int jloc = 0; jloc < a.num_cols_local(); jloc++)
    {
        for (int iloc = 0; iloc < a.num_rows_local(); iloc++) a(iloc, jloc) = rnd();
    }
    /* conjugate transpose */
    linalg<CPU>::tranc(N, N, a, 0, 0, b, 0, 0);

    /* make a hermitian matrix */
    for (int jloc = 0; jloc < a.num_cols_local(); jloc++)
    {
        for (int iloc = 0; iloc < a.num_rows_local(); iloc++) a(iloc, jloc) = a(iloc, jloc) + b(iloc, jloc);
    }

    runtime::Timer t1("pzgemm_time");
    /* make a positively defined matrix */
    linalg<CPU>::gemm(2, 0, N, N, N, ftn_double_complex(1, 0), a, a, ftn_double_complex(0, 0), b);
    double tval = t1.stop();
    if (blacs_grid.comm().rank() == 0)
    {
        printf("\n");
        printf("pzgemm time  : %f (sec.), pzgemm performance: %f (GFlops / rank)\n", tval, 8e-9 * N * N * N / tval / blacs_grid.comm().size());
    }
    /* increase diagnonal a little bit */
    for (int i = 0; i < N; i++) b.add(i, i, 2.0 / (1.0 + i));
    
    //generalized_evp_scalapack solver(blacs_grid, -1.0, bs, bs); 
    generalized_evp_elpa1 solver(blacs_grid, bs);
    /* take 10% of eigenvectors */
    int nev = 0.1 * N;
    //int nev = N;
    dmatrix<ftn_double_complex> z(N, nev, blacs_grid, bs, bs);
    z.zero();
    std::vector<double> eval(nev, 0);

    runtime::Timer t2("gen_evp_time");
    solver.solve(N, a.num_rows_local(), a.num_cols_local(), nev, a.at<CPU>(), a.ld(), b.at<CPU>(), b.ld(), &eval[0], z.at<CPU>(), z.ld());
    tval = t2.stop();
    if (blacs_grid.comm().rank() == 0)
    {
       printf("evp time : %f (sec.)\n", tval);
    }

}

int main(int argn, char** argv)
{
    if (argn != 2)
    {
        printf("Usage: %s n\n", argv[0]);
        exit(0);
    }

    Communicator::initialize();

    /* assume a square grid */
    int32_t num_ranks_row, num_ranks_col;
    num_ranks_row = num_ranks_col = static_cast<int>(std::sqrt(mpi_comm_world().size() + 0.1));

    if (num_ranks_row * num_ranks_col != mpi_comm_world().size())
    {
        printf("wrong number of MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, -13);
    }

    int32_t n;
    std::istringstream iss(argv[1]);
    iss >> n;

    int32_t N = n * num_ranks_row;
    if (mpi_comm_world().rank() == 0)
    {
        printf("\n");
        printf("Running on %i rank(s) (%i x %i grid), ", mpi_comm_world().size(), num_ranks_row, num_ranks_col);
        printf("global martix dimensions: %i x %i\n", N, N);
    }

    {
        BLACS_grid blacs_grid(mpi_comm_world(), num_ranks_row, num_ranks_col);

        for (int i = 0; i < 5; i++) test_diag(N, blacs_grid);
    }

    Communicator::finalize();
}
