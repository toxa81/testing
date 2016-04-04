#include <stdio.h>
#include <vector>
#include "runtime.h"
#include "communicator.h"

int main(int argn, char** argv)
{
    Communicator::initialize();

    std::stringstream s;
    s << "test." << mpi_comm_world().rank() << ".bin";

    std::vector<double> v(1 << 30, 0);
    
    runtime::Timer t("fwrite");
    FILE* fout = fopen(s.str().c_str(), "w");
    fwrite(&v[0], sizeof(double), v.size(), fout);
    fclose(fout);
    mpi_comm_world().barrier();
    double tval = t.stop();

    printf("rank: %i, io speed: %f (Gb/sec.)", mpi_comm_world().rank(), v.size() * sizeof(double) / double(1 << 30) / tval);

    Communicator::finalize();

}
