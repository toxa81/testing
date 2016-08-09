#include <stdio.h>
#include <vector>
#include "runtime.h"
#include "communicator.h"

double write_buffer(size_t len)
{
    std::vector<char> buf(len, '0');

    std::stringstream s;
    s << "test." << mpi_comm_world().rank() << ".bin";

    runtime::Timer t("fwrite");

    double t1 = -omp_get_wtime();
    FILE* fout = fopen(s.str().c_str(), "w");
    t1 += omp_get_wtime();

    double t2 = -omp_get_wtime();
    fwrite(&buf[0], sizeof(char), buf.size(), fout);
    t2 += omp_get_wtime();

    double t3 = -omp_get_wtime();
    fclose(fout);
    t3 += omp_get_wtime();

    printf("%f %f %f\n", t1, t2, t3);

    double tval = t.stop();
    
    return (buf.size() * sizeof(char) / double(1 << 30) / tval);
}

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

    for (int i = 1; i < 100; i += 5)
    {
        size_t len = i * (1 << 26);
        double speed = write_buffer(len);
        printf("%f %f\n", len / double(1 << 30), speed);
    }

        

    //write_in_blocks();

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
