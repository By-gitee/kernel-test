#include <iostream>
#include <chrono>
#include <cstdint>
#include <random>
#include <algorithm>

#pragma GCC optimize("O0")


#define ITER 1000000
#define WARMUP 100

constexpr int MASK_SIZE = ITER * 4;  // 100000*4
//---------------------------------------------
// 方法1：SVE compact（单次核心实现，循环在 C++ 中）
//---------------------------------------------
void sve_compact(float *src, float *dst, uint8_t *mask)
{
    asm volatile(
        "mov x0, %[src]\n"
        "mov x1, %[dst]\n"
        "mov x2, %[mask]\n"

        "ptrue p0.s\n"          // load predicate for memory ops

        // load data
        "ld1w {z0.s}, p0/z, [x0]\n"

        // load mask bytes
        "ld1b {z1.b}, p0/z, [x2]\n"

        // convert mask -> predicate
        "cmpne p1.b, p0/z, z1.b, #0\n"

        // compact active lanes
        "compact z2.s, p1, z0.s\n"

        // store result
        "st1w {z2.s}, p0, [x1]\n"

        :
        : [src]"r"(src),
          [dst]"r"(dst),
          [mask]"r"(mask)
        : "x0","x1","x2",
          "z0","z1","z2",
          "p0","p1",
          "memory");
}

//---------------------------------------------
// 方法2：scalar + mask（单次核心实现，循环在 C++ 中）
//---------------------------------------------
void scalar_mask(float *src, float *dst, uint8_t *mask)
{
    asm volatile(
        "mov x0, %[src]\n"
        "mov x1, %[dst]\n"
        "mov x2, %[mask]\n"

        // i=0
        "ldrb w4, [x2]\n"
        "cbz w4, 2f\n"
        "ldr s0, [x0]\n"
        "str s0, [x1]\n"
        "add x1, x1, #4\n"
        "2:\n"

        // i=1
        "ldrb w4, [x2,#1]\n"
        "cbz w4, 3f\n"
        "ldr s0, [x0,#4]\n"
        "str s0, [x1]\n"
        "add x1, x1, #4\n"
        "3:\n"

        // i=2
        "ldrb w4, [x2,#2]\n"
        "cbz w4, 4f\n"
        "ldr s0, [x0,#8]\n"
        "str s0, [x1]\n"
        "add x1, x1, #4\n"
        "4:\n"

        // i=3
        "ldrb w4, [x2,#3]\n"
        "cbz w4, 5f\n"
        "ldr s0, [x0,#12]\n"
        "str s0, [x1]\n"
        "add x1, x1, #4\n"
        "5:\n"

        :
        : [src]"r"(src),
          [dst]"r"(dst),
          [mask]"r"(mask)
        : "x0","x1","x2","x4","s0","memory");
}

void print_io(const char *label, float *src, uint8_t *mask, float *dst, int dst_len)
{
    std::cout << "--- " << label << " ---\n";
    std::cout << "  Input  (src):  " << src[0] << ", " << src[1] << ", " << src[2] << ", " << src[3] << "\n";
    std::cout << "  Mask:          " << (int)mask[0] << ", " << (int)mask[1] << ", " << (int)mask[2] << ", " << (int)mask[3] << "\n";
    std::cout << "  Output (dst):  ";
    for (int i = 0; i < dst_len; ++i) std::cout << dst[i] << (i < dst_len - 1 ? ", " : "");
    std::cout << "\n";
}

int main()
{
    alignas(64) float data[4] = {1, 2, 3, 4};
    alignas(64) float out1[4] = {0};
    alignas(64) float out2[4] = {0};

    // mask: 100000*4 的 uint8_t 数组，随机 0/1，其中 1 占 80%
    uint8_t mask[MASK_SIZE];
    {
        const int num_ones = static_cast<int>(MASK_SIZE * 1);  // 80% 为 1
        const int num_zeros = MASK_SIZE - num_ones;
        std::fill(mask, mask + num_ones, 1);
        std::fill(mask + num_ones, mask + MASK_SIZE, 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(mask, mask + MASK_SIZE, g);
    }

    uint8_t* iter_mask = mask;
    std::cout << "========== Input (shared by both methods) ==========\n";
    std::cout << "  src[4]:  " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << "\n";
    std::cout << "  mask:    " << (int)mask[0] << ", " << (int)mask[1] << ", " << (int)mask[2] << ", " << (int)mask[3] << " ... (size " << MASK_SIZE << ", 80% ones)\n";
    std::cout << "  (mask=1 表示该 lane 有效，压缩后连续存放)\n\n";

    for (int i = 0; i < WARMUP; ++i) sve_compact(data, out1, mask);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; ++i) {
      sve_compact(data, out1, iter_mask);
      iter_mask = iter_mask + 4;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    iter_mask = mask;
    for (int i = 0; i < WARMUP; ++i) scalar_mask(data, out2, mask);
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; ++i) {
      scalar_mask(data, out2, iter_mask);
      iter_mask = iter_mask + 4;
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    double sve_time = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double scalar_time = std::chrono::duration<double, std::micro>(t4 - t3).count();

    std::cout << "========== Timing (x " << ITER << " iterations) ==========\n";
    std::cout << "  SVE compact:   " << sve_time << " us\n";
    std::cout << "  Scalar+mask:  " << scalar_time << " us\n\n";

    std::cout << "========== Output ==========\n";
    print_io("Method 1 (SVE compact)", data, mask, out1, 2);
    print_io("Method 2 (scalar+mask)", data, mask, out2, 2);
}