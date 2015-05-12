CC = CC
NVCC = nvcc
all:
	$(NVCC) -arch=sm_35 -m64 -DNDEBUG -c cuda_interface.cu
	$(NVCC) -arch=sm_35 -m64 -DNDEBUG -c cuda_spline_kernels.cu
	$(CC) -std=c++11 -D_GPU_ main.cpp platform.cpp cuda_interface.o cuda_spline_kernels.o -L/Developer/NVIDIA/CUDA-7.0/lib -lcublas -lcudart -lcufft -Wl,-rpath,/Developer/NVIDIA/CUDA-7.0/lib -lstdc++
