#include <iostream>
#include <cstdint>
#include <iomanip>

#pragma GCC optimize("O0")

constexpr int ITERS = 100000000;
constexpr int NODES = 1024;

struct alignas(16) Node {
    uint64_t next;
    uint64_t pad;
};

alignas(64) Node chain[NODES];
alignas(16) uint64_t gather_idx[2] = {0, 1};   // 连续 gather，下标 0 和 1

volatile uint64_t sink = 0;

// 读取 CNTVCT_EL0；前后用 ISB 防止乱序/投机影响测量
static inline uint64_t rdcycle()
{
    uint64_t c;
    asm volatile("mrs %0, cntvct_el0" : "=r"(c));
    return c;
}

static inline uint64_t rdfrq()
{
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}

// ----------------------------------
// 空循环基线
// ----------------------------------
__attribute__((noinline))
uint64_t test_empty()
{
    volatile uint64_t x = reinterpret_cast<uint64_t>(&chain[0]);

    uint64_t start = rdcycle();
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile("" : "+r"(x) :: "memory");
    }
    uint64_t end = rdcycle();

    sink = x;
    return end - start;
}

// ----------------------------------
// 1. scalar load latency
// ldr x0, [x0]
// ----------------------------------
__attribute__((noinline))
uint64_t test_scalar()
{
    uint64_t ptr = reinterpret_cast<uint64_t>(&chain[0]);

    uint64_t start = rdcycle();
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "ldr x1, [%0]\n"
            "mov %0, x1\n"
            : "+r"(ptr)
            :
            : "x1","memory"
        );
    }
    uint64_t end = rdcycle();

    sink = ptr;
    return end - start;
}

// ----------------------------------
// 2. NEON load latency
// ld1 {v0.2d}, [x0]
// umov x0, v0.d[0]
// ----------------------------------
__attribute__((noinline))
uint64_t test_neon()
{
    uint64_t ptr = reinterpret_cast<uint64_t>(&chain[0]);

    uint64_t start = rdcycle();
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "ld1 {v0.2d}, [%0]\n"
            "umov %0, v0.d[0]\n"
            : "+r"(ptr)
            :
            : "v0", "memory");
    }
    uint64_t end = rdcycle();

    sink = ptr;
    return end - start;
}

// ----------------------------------
// 3. SVE contiguous load latency
// 预设 VL=128，则 .d lanes=2
// ld1d {z0.d}, p0/z, [x0]
// umov x0, v0.d[0]
// 这里利用 z0 的低 128bit 与 v0 重叠
// ----------------------------------
__attribute__((noinline))
uint64_t test_sve()
{
    uint64_t ptr = reinterpret_cast<uint64_t>(&chain[0]);

    // ptrue 放到循环外，避免污染每次迭代
    asm volatile("ptrue p0.d\n" ::: "p0");

    uint64_t start = rdcycle();
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "ld1d {z0.d}, p0/z, [%0]\n"
            "umov %0, v0.d[0]\n"
            : "+r"(ptr)
            :
            : "z0", "v0", "p0", "memory");
    }
    uint64_t end = rdcycle();

    sink = ptr;
    return end - start;
}

// ----------------------------------
// 4. SVE gather load latency
// 连续 gather：index = {0,1}
// ld1d {z1.d}, p0/z, [idx] 放循环外
// 每次只测 gather 本身
// ld1d {z0.d}, p0/z, [x0, z1.d, lsl #3]
// umov x0, v0.d[0]
// ----------------------------------
__attribute__((noinline))
uint64_t test_gather()
{
    uint64_t ptr = reinterpret_cast<uint64_t>(&chain[0]);

    asm volatile(
        "ptrue p0.d\n"
        "ld1d {z1.d}, p0/z, [%0]\n"
        :
        : "r"(gather_idx)
        : "z1", "p0", "memory");

    uint64_t start = rdcycle();
    for (int i = 0; i < ITERS; i++)
    {
        asm volatile(
            "ld1d {z0.d}, p0/z, [%0, z1.d, lsl #3]\n"
            "umov %0, v0.d[0]\n"
            : "+r"(ptr)
            :
            : "z0", "v0", "z1", "p0", "memory");
    }
    uint64_t end = rdcycle();

    sink = ptr;
    return end - start;
}

// ----------------------------------

int main()
{
    // 构造 16B 节点链表
    // next 指向下一个节点起始地址
    for (int i = 0; i < NODES - 1; i++) {
        chain[i].next = reinterpret_cast<uint64_t>(&chain[i + 1]);
        chain[i].pad  = 0xDEADBEEFDEADBEEFULL;
    }
    chain[NODES - 1].next = reinterpret_cast<uint64_t>(&chain[0]);
    chain[NODES - 1].pad  = 0xDEADBEEFDEADBEEFULL;

    uint64_t c_empty  = test_empty();
    uint64_t c_scalar = test_scalar();
    uint64_t c_neon   = test_neon();
    uint64_t c_sve    = test_sve();
    uint64_t c_gather = test_gather();

    uint64_t adj_scalar = c_scalar - c_empty;
    uint64_t adj_neon   = c_neon   - c_empty;
    uint64_t adj_sve    = c_sve    - c_empty;
    uint64_t adj_gather = c_gather - c_empty;
    uint64_t freq = rdfrq();
    std::cout << "CNTVCT frequency: " << freq << " Hz\n\n";

    auto to_ns = [&](uint64_t ticks)
    {
        return (double)ticks * 1e9 / freq;
    };

        // 读取CPU频率 (简单写死也行)
        double cpu_freq = 3e9;   // 2.5 GHz

        double ratio = cpu_freq / freq;
    
    std::cout << "Iterations: " << ITERS << "\n\n";
    

    std::cout << "empty cycles : " << c_empty  << "\n";
    std::cout << "scalar cycles: " << c_scalar << "\n";
    std::cout << "neon cycles  : " << c_neon   << "\n";
    std::cout << "sve cycles   : " << c_sve    << "\n";
    std::cout << "gather cycles: " << c_gather << "\n\n";

    std::cout << "Adjusted cycles (subtract empty loop):\n";
    std::cout << "scalar: " << adj_scalar << "\n";
    std::cout << "neon  : " << adj_neon   << "\n";
    std::cout << "sve   : " << adj_sve    << "\n";
    std::cout << "gather: " << adj_gather << "\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average latency (cycles):\n";
    std::cout << "scalar : " << to_ns(adj_scalar) / ITERS << " ns\n";
    std::cout << "neon   : " << to_ns(adj_neon)   / ITERS << " ns\n";
    std::cout << "sve    : " << to_ns(adj_sve)    / ITERS << " ns\n";
    std::cout << "gather : " << to_ns(adj_gather) / ITERS << " ns\n";

    return 0;
}