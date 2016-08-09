all:
	CC -O3 -openmp -std=c++11 -D__SCALAPACK -D__ELPA -c linalg.cpp -o linalg.o
	CC -O3 -openmp -std=c++11 -D__SCALAPACK -D__ELPA main.cpp linalg.o ./elpa/libelpa.a $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread
