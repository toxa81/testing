all:
	mpic++ -O0 -g -fopenmp -std=c++11 -D__SCALAPACK -D__ELPA -c linalg.cpp -o linalg.o
	mpic++ -O0 -g -fopenmp -std=c++11 -D__SCALAPACK -D__ELPA main.cpp linalg.o ~/src/LAPACK/scalapack-2.0.2/libscalapack.a ~/src/LAPACK/lapack-3.4.2/liblapack.a /Users/antonk/src/LAPACK/lapack-3.4.2/librefblas.a ~/local/lib/libmpi_mpifh.a -L/opt/local/lib -lz -lgfortran ./elpa/libelpa.a