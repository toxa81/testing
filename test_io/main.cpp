#include <stdio.h>
#include <vector>
#include "runtime.h"
#include "communicator.h"

void write_in_blocks()
{
    std::stringstream s;
    s << "test." << mpi_comm_world().rank() << ".bin";

    std::vector<double> v(1 << 20, 0);

    int N = 500;
    
    runtime::Timer t("fwrite");
    for (int i = 0; i < mpi_comm_world().size(); i++)
    {
        if (mpi_comm_world().rank() == i)
        {
            
            for (int j = 0; j < N; j++)
            {
                runtime::Timer t1("fwrite.chunk");
                FILE* fout = fopen(s.str().c_str(), "a");
                fwrite(&v[0], sizeof(double), v.size(), fout);
                fclose(fout);
                double tval = t1.stop();
                printf("%i %f\n", j, v.size() * sizeof(double) / double(1 << 30) / tval);
            }
        }
        mpi_comm_world().barrier();
    }
    double tval = t.stop();
    
    if (mpi_comm_world().rank() == 0)
    {
        printf("sequential io speed: %f (Gb/sec.)\n", mpi_comm_world().size() * N * v.size() * sizeof(double) / double(1 << 30) / tval);
    }

}

void write_sequential()
{
    std::stringstream s;
    s << "test." << mpi_comm_world().rank() << ".bin";

    std::vector<double> v(1 << 30, 0);
    
    runtime::Timer t("fwrite");
    for (int i = 0; i < mpi_comm_world().size(); i++)
    {
        if (mpi_comm_world().rank() == i)
        {
            FILE* fout = fopen(s.str().c_str(), "w");
            fwrite(&v[0], sizeof(double), v.size(), fout);
            fclose(fout);
        }
        mpi_comm_world().barrier();
    }
    double tval = t.stop();
    
    if (mpi_comm_world().rank() == 0)
    {
        printf("sequential io speed: %f (Gb/sec.)\n", mpi_comm_world().size() * v.size() * sizeof(double) / double(1 << 30) / tval);
    }
}

int main(int argn, char** argv)
{
    Communicator::initialize();

    write_in_blocks();

    //write_sequential();

    //std::stringstream s;
    //s << "test." << mpi_comm_world().rank() << ".bin";

    //std::vector<double> v(1 << 30, 0);
    //
    //runtime::Timer t("fwrite");
    //FILE* fout = fopen(s.str().c_str(), "w");
    //fwrite(&v[0], sizeof(double), v.size(), fout);
    //fclose(fout);
    //mpi_comm_world().barrier();
    //double tval = t.stop();

    //printf("rank: %i, io speed: %f (Gb/sec.)i\n", mpi_comm_world().rank(), v.size() * sizeof(double) / double(1 << 30) / tval);

    Communicator::finalize();

}