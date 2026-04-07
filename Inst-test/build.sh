g++ compact_test.cpp -O0 -march=armv8-a+sve2 -msve-vector-bits=128 -o compact_test

g++ ld_test.cpp -O0 -march=armv8-a+sve -msve-vector-bits=128 -o ld_test