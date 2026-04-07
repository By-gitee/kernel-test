// ======================= C++ VERSION =======================
// Direct port of graphblas_spmm_benchmark.c
//  - GrB_NONBLOCKING init
//  - M K N + thresholds from CLI, CSV output (graphblas_spmm_results.csv)
// ================================================================

#include <GraphBLAS.h>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static int M = 128;
static int K = 8192;
static int N = 8192;

#define WARMUP_RUNS 10
#define TIMED_RUNS  100
#define MAX_THRESHOLDS 64

// ------------------------------------------------------------
// thresholds (optional CLI override) — same as armpl_spmm_benchmark
// ------------------------------------------------------------
static const float DEFAULT_THRESHOLDS[] = {
    0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
    0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 0.99f
};
static const int NUM_DEFAULT_THRESHOLDS =
    static_cast<int>(sizeof(DEFAULT_THRESHOLDS) / sizeof(DEFAULT_THRESHOLDS[0]));

static inline double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

// Parse command-line: M K N [threshold1 [threshold2 ...]] or "t1,t2,..."
static bool parse_args(int argc, char **argv,
                       int *out_M, int *out_K, int *out_N,
                       float *out_thresholds, int max_thr, int *out_num_thr) {
    if (argc < 4) {
        std::fprintf(stderr,
                "Usage: %s M K N [threshold1 [threshold2 ...]]\n"
                "  M K N = matrix dimensions (sparse MxK * dense KxN -> MxN)\n"
                "  thresholds = optional list of floats, or single \"t1,t2,...\"\n",
                argv[0] ? argv[0] : "graphblas_spmm_benchmark");
        return false;
    }
    *out_M = std::atoi(argv[1]);
    *out_K = std::atoi(argv[2]);
    *out_N = std::atoi(argv[3]);
    *out_num_thr = 0;
    for (int i = 4; i < argc && *out_num_thr < max_thr; i++) {
        const char *arg = argv[i];
        const char *p = arg;
        while (*p) {
            float v;
            if (std::sscanf(p, "%f", &v) == 1) {
                out_thresholds[(*out_num_thr)++] = v;
                if (*out_num_thr >= max_thr) break;
            }
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    if (*out_num_thr == 0) {
        int n = NUM_DEFAULT_THRESHOLDS;
        if (n > max_thr) n = max_thr;
        for (int i = 0; i < n; i++)
            out_thresholds[i] = DEFAULT_THRESHOLDS[i];
        *out_num_thr = n;
    }
    return true;
}

// ------------------------------------------------------------
// Data loading: activation_{M}_{K}.bin, weight_{K}_{N}.bin (float32 row-major)
// ------------------------------------------------------------
static int load_activation_from_file(float *activation) {
    char path[256];
    int n = std::snprintf(path, sizeof(path), "activation_%d_%d.bin", M, K);
    if (n < 0 || static_cast<size_t>(n) >= sizeof(path))
        return -1;
    std::FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "Cannot open activation file: %s (%s)\n", path, std::strerror(errno));
        return -1;
    }
    size_t want = static_cast<size_t>(M) * K * sizeof(float);
    size_t got = std::fread(activation, 1, want, fp);
    std::fclose(fp);
    if (got != want) {
        std::fprintf(stderr, "Activation file %s: expected %zu bytes, got %zu\n", path, want, got);
        return -1;
    }
    return 0;
}

static int load_weight_from_file(float *weight_rm) {
    char path[256];
    int n = std::snprintf(path, sizeof(path), "weight_%d_%d.bin", K, N);
    if (n < 0 || static_cast<size_t>(n) >= sizeof(path))
        return -1;
    std::FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "Cannot open weight file: %s (%s)\n", path, std::strerror(errno));
        return -1;
    }
    size_t want = static_cast<size_t>(K) * N * sizeof(float);
    size_t got = std::fread(weight_rm, 1, want, fp);
    std::fclose(fp);
    if (got != want) {
        std::fprintf(stderr, "Weight file %s: expected %zu bytes, got %zu\n", path, want, got);
        return -1;
    }
    return 0;
}

// Copy row-major B(k,n) -> column-major. Done once outside timing (same as armpl/onednn).
static void weight_rm_to_cm(const float *weight_rm, float *weight_cm) {
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            weight_cm[static_cast<size_t>(n) * K + k] = weight_rm[static_cast<size_t>(k) * N + n];
}

// ------------------------------------------------------------
// Reference and correctness: ref = masked(activation, th) * weight_rm; compare with atol/rtol
// ------------------------------------------------------------
static void compute_reference(const float *activation, const float *weight_rm,
                              float th, float *ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                float a = activation[static_cast<size_t>(i) * K + k];
                if (std::fabs(a) > th)
                    sum += a * weight_rm[static_cast<size_t>(k) * N + j];
            }
            ref[static_cast<size_t>(i) * N + j] = sum;
        }
    }
}

static int verify_correctness(const float *output, const float *ref, float atol, float rtol) {
    size_t first_fail = static_cast<size_t>(M) * N;
    double max_diff = 0.0;
    for (size_t idx = 0; idx < static_cast<size_t>(M) * N; idx++) {
        float a = output[idx];
        float b = ref[idx];
        float tol = atol + rtol * std::fabs(b);
        double diff = static_cast<double>(std::fabs(a - b));
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (first_fail == static_cast<size_t>(M) * N) {
                first_fail = idx;
                std::fprintf(stderr, "Correctness check failed: first at [%zu,%zu] (idx %zu) got %g expected %g (diff %g > tol %g)\n",
                        first_fail / N, first_fail % N, first_fail,
                        static_cast<double>(a), static_cast<double>(b), diff, static_cast<double>(tol));
            }
        }
    }
    if (first_fail < static_cast<size_t>(M) * N) {
        std::fprintf(stderr, "  max|output-ref| = %g (atol=%g rtol=%g). If diff is small, try relaxing tolerance.\n",
                max_diff, static_cast<double>(atol), static_cast<double>(rtol));
        return -1;
    }
    return 0;
}

static GrB_Index sparsify_to_coo(
    const float *a, float thr,
    GrB_Index *ri, GrB_Index *ci, float *v,
    GrB_Index max_nnz)
{
    GrB_Index nnz = 0;
    for (int i = 0; i < M; i++) {
        const float *row = a + static_cast<size_t>(i) * K;
        for (int j = 0; j < K; j++) {
            float x = row[j];
            if (std::fabs(x) > thr) {
                assert(nnz < max_nnz);
                ri[nnz] = i;
                ci[nnz] = j;
                v[nnz]  = x;
                nnz++;
            }
        }
    }
    return nnz;
}

static GrB_Info create_A_from_coo(
    const GrB_Index *ri, const GrB_Index *ci,
    const float *v, GrB_Index nnz, GrB_Matrix *A)
{
    GrB_Info info = GrB_Matrix_new(A, GrB_FP32, M, K);
    if (info != GrB_SUCCESS) return info;

    info = GrB_Matrix_build_FP32(*A, ri, ci, v, nnz, GrB_FIRST_FP32);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(A);
        return info;
    }
    return info;
}
// ------------------------------------------------------------
// Timing: sparse_create, sparse_compute, sparse_total (same semantics as armpl/onednn)
// ------------------------------------------------------------
static double time_sparse_create(
    int runs, const float *a, float thr,
    GrB_Index *ri, GrB_Index *ci, float *v,
    GrB_Index max_nnz)
{
    GrB_Matrix A;
    
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        GrB_Index nnz = sparsify_to_coo(a, thr, ri, ci, v, max_nnz);
        create_A_from_coo(ri, ci, v, nnz, &A);
    }
    double measure_t = (now_ms() - t0) / runs;
    GrB_Matrix_free(&A);
    return measure_t;
}

static double time_sparse_compute(
    int warmup, int runs, GrB_Matrix A, GrB_Matrix B, GrB_Matrix C)
{
    for (int i = 0; i < warmup; i++) {
        GrB_Matrix_clear(C);
        GrB_mxm(C, GrB_NULL, GrB_NULL,
                GrB_PLUS_TIMES_SEMIRING_FP32,
                A, B, GrB_NULL);
        GrB_Matrix_wait(C, GrB_MATERIALIZE);
    }
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        GrB_Matrix_clear(C);
        GrB_mxm(C, GrB_NULL, GrB_NULL,
                GrB_PLUS_TIMES_SEMIRING_FP32,
                A, B, GrB_NULL);
        GrB_Matrix_wait(C, GrB_MATERIALIZE);
    }
    return (now_ms() - t0) / runs;
}

static double time_sparse_total(
    int runs, const float *a, float thr,
    GrB_Index *ri, GrB_Index *ci, float *v,
    GrB_Index max_nnz, GrB_Matrix B, GrB_Matrix C)
{
    GrB_Matrix A;
    double t0 = now_ms();

    for (int r = 0; r < runs; r++) {
        GrB_Index nnz = sparsify_to_coo(a, thr, ri, ci, v, max_nnz);
        create_A_from_coo(ri, ci, v, nnz, &A);

        GrB_Matrix_clear(C);
        GrB_mxm(C, GrB_NULL, GrB_NULL,
                GrB_PLUS_TIMES_SEMIRING_FP32,
                A, B, GrB_NULL);
        GrB_Matrix_wait(C, GrB_MATERIALIZE);

    }
    
    double measure_t = (now_ms() - t0) / runs;
    GrB_Matrix_free(&A);
    
    return measure_t;
}

// ------------------------------------------------------------
// Dense GEMM baseline: mask activation (zero out |x|<=th), import as full matrix via GxB_Matrix_import_FullR, then GrB_mxm.
// No COO; dense buffer passed directly to GraphBLAS. Only total time is reported.
// Requires SuiteSparse:GraphBLAS (GxB_Matrix_import_FullR).
// ------------------------------------------------------------
static double time_dense_gemm_total(int runs,
                                    const float *a,
                                    GrB_Matrix B,
                                    GrB_Matrix C) {
    const uint64_t nrows = static_cast<uint64_t>(M);
    const uint64_t ncols = static_cast<uint64_t>(K);
    const uint64_t val_size = static_cast<uint64_t>(static_cast<size_t>(M) * K);
    const size_t nbytes = static_cast<size_t>(M) * K * sizeof(float);

    double total_ms = 0.0;
    for (int r = 0; r < runs; r++) {
        // SuiteSparse import takes ownership; need a fresh buffer per iteration.
        float *import_buf = static_cast<float *>(std::malloc(nbytes));
        assert(import_buf && "malloc import buffer");
        std::memcpy(import_buf, a, nbytes);

        // Time only: import + mxm + wait + free
        double t_start = now_ms();

        GrB_Matrix A;
        void *val_ptr = static_cast<void *>(import_buf);
        GrB_Info info = GxB_Matrix_import_FullR(
            &A,
            GrB_FP32,
            nrows,
            ncols,
            &val_ptr,
            val_size,
            false,
            GrB_NULL);
        assert(info == GrB_SUCCESS && "GxB_Matrix_import_FullR (SuiteSparse)");

        GrB_Matrix_clear(C);
        info = GrB_mxm(C, GrB_NULL, GrB_NULL,
                       GrB_PLUS_TIMES_SEMIRING_FP32,
                       A, B, GrB_NULL);
        assert(info == GrB_SUCCESS && "GrB_mxm");

        info = GrB_Matrix_wait(C, GrB_MATERIALIZE);
        assert(info == GrB_SUCCESS && "GrB_Matrix_wait");

        GrB_Matrix_free(&A);
        total_ms += (now_ms() - t_start);
    }

    return total_ms / runs;
}
// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char **argv) {
    int in_M, in_K, in_N, num_thresholds;
    float thresholds[MAX_THRESHOLDS];

    if (!parse_args(argc, argv, &in_M, &in_K, &in_N,
                    thresholds, MAX_THRESHOLDS, &num_thresholds)) {
        return 1;
    }
    M = in_M;
    K = in_K;
    N = in_N;

    GrB_init(GrB_NONBLOCKING);
    GxB_Global_Option_set(GxB_NTHREADS, 16);

    std::vector<float> activation(static_cast<size_t>(M) * K);
    std::vector<float> weight_rm(static_cast<size_t>(K) * N);
    std::vector<float> weight_cm(static_cast<size_t>(K) * N);
    GrB_Index max_nnz = static_cast<GrB_Index>(M) * K;

    std::vector<GrB_Index> ri(max_nnz);
    std::vector<GrB_Index> ci(max_nnz);
    std::vector<float> v(max_nnz);

    std::vector<float> ref(static_cast<size_t>(M) * N);
    std::vector<float> output_buf(static_cast<size_t>(M) * N, 0.f);

    if (load_activation_from_file(activation.data()) != 0) {
        GrB_finalize();
        return 1;
    }
    if (load_weight_from_file(weight_rm.data()) != 0) {
        GrB_finalize();
        return 1;
    }
    weight_rm_to_cm(weight_rm.data(), weight_cm.data());

    GrB_Matrix B, C;
    GrB_Matrix_new(&B, GrB_FP32, K, N);
    GrB_Matrix_new(&C, GrB_FP32, M, N);

    for (GrB_Index j = 0; j < static_cast<GrB_Index>(N); j++)
        for (GrB_Index k = 0; k < static_cast<GrB_Index>(K); k++)
            GrB_Matrix_setElement_FP32(B, weight_cm[static_cast<size_t>(j) * K + k], k, j);

    GrB_Matrix_wait(B, GrB_MATERIALIZE);
    GrB_Matrix_wait(C, GrB_MATERIALIZE);

    std::vector<float> masked_dense(static_cast<size_t>(M) * K);

    std::printf("GraphBLAS Sparse SpMM Benchmark (aligned to ArmPL/oneDNN)\n");
    std::printf("M=%d K=%d N=%d\n", M, K, N);
    std::printf("threshold,density,latency_sparse_create,latency_sparse_compute,latency_sparse_total,latency_dense\n");

    std::FILE *csv = std::fopen("graphblas_spmm_results.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "Cannot open graphblas_spmm_results.csv for writing\n");
        GrB_Matrix_free(&B);
        GrB_Matrix_free(&C);
        GrB_finalize();
        return 1;
    }
    std::fprintf(csv, "threshold,density,latency_sparse_create,latency_sparse_compute,latency_sparse_total,latency_dense\n");

    int verified = 0;
    for (int ti = 0; ti < num_thresholds; ti++) {
        float th = thresholds[ti];
        GrB_Index nnz = sparsify_to_coo(activation.data(), th, ri.data(), ci.data(), v.data(), max_nnz);
        if (nnz == 0) continue;

        double density = static_cast<double>(nnz) / (static_cast<double>(M) * static_cast<double>(K));

        GrB_Matrix A;
        create_A_from_coo(ri.data(), ci.data(), v.data(), nnz, &A);
        if (!verified) {
            GrB_Matrix_clear(C);
            GrB_mxm(C, GrB_NULL, GrB_NULL,
                    GrB_PLUS_TIMES_SEMIRING_FP32,
                    A, B, GrB_NULL);
            GrB_Matrix_wait(C, GrB_MATERIALIZE);
            std::fill(output_buf.begin(), output_buf.end(), 0.f);
            for (GrB_Index i = 0; i < static_cast<GrB_Index>(M); i++)
                for (GrB_Index j = 0; j < static_cast<GrB_Index>(N); j++) {
                    float val;
                    if (GrB_Matrix_extractElement_FP32(&val, C, i, j) == GrB_SUCCESS)
                        output_buf[static_cast<size_t>(i) * N + static_cast<size_t>(j)] = val;
                }
            compute_reference(activation.data(), weight_rm.data(), th, ref.data());
            if (verify_correctness(output_buf.data(), ref.data(), 1e-2f, 1e-1f) != 0) {
                GrB_Matrix_free(&A);
                GrB_Matrix_free(&B);
                GrB_Matrix_free(&C);
                std::fclose(csv);
                GrB_finalize();
                return 1;
            }
            verified = 1;
            std::printf("Correctness verification passed (threshold=%.4f).\n", static_cast<double>(th));
        }

        double t_create  = time_sparse_create(TIMED_RUNS, activation.data(), th, ri.data(), ci.data(), v.data(), max_nnz);
        double t_compute = time_sparse_compute(WARMUP_RUNS, TIMED_RUNS, A, B, C);
        double t_total   = time_sparse_total(TIMED_RUNS, activation.data(), th, ri.data(), ci.data(), v.data(), max_nnz, B, C);
        double t_dense = time_dense_gemm_total(TIMED_RUNS, activation.data(), th,
                masked_dense.data(), B, C);

        std::printf("%.4f,%.6f,%.4f,%.4f,%.4f,%.4f\n",
               static_cast<double>(th), density, t_create, t_compute, t_total, t_dense);
        std::fprintf(csv, "%.4f,%.6f,%.4f,%.4f,%.4f,%.4f\n",
                static_cast<double>(th), density, t_create, t_compute, t_total, t_dense);

        GrB_Matrix_free(&A);
    }

    std::fclose(csv);
    std::printf("Results written to graphblas_spmm_results.csv\n");

    GrB_Matrix_free(&B);
    GrB_Matrix_free(&C);
    GrB_finalize();
    return 0;
}
