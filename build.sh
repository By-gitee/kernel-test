source env.sh

ARMFLANG_LIB=/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/lib
ARMPL=/root/package/armpl-26.01.0

mkdir -p build

g++ -O3 -march=native armpl_spmm_benchmark.cpp  -I/root/package/armpl-26.01.0/include   -L/root/package/armpl-26.01.0/lib -L/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/lib   -Wl,-rpath,/root/package/armpl-26.01.0/lib -Wl,-rpath,/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/lib   -Wl,-rpath-link,/root/package/armpl-26.01.0/lib -Wl,-rpath-link,/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/lib -fopenmp  -larmpl_mp  -o build/armpl_spmm_benchmark

g++ -O3 -march=native graphblas_spmm_benchmark.cpp   -I/root/package/GraphBLAS/Include   -L/root/package/GraphBLAS/lib -lgraphblas   -Wl,-rpath,/root/package/GraphBLAS/lib   -o build/graphblas_spmm_benchmark

g++ -O3 -march=native onednn_spmm_benchmark.cpp -L/root/package/ComputeLibrary/build  -I/usr/local/include -L/usr/local/lib -ldnnl  -larm_compute_graph -larm_compute -ldl -pthread  -Wl,-rpath,/usr/local/lib   -o build/onednn_spmm_benchmark
