#include <armpl.h>
#include <armpl_sparse.h>
#include <cblas.h>

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <sstream>

/*
  ArmPL Sparse SpMM Benchmark
  (aligned with onednn_spmm_benchmark semantics)

  A: sparse CSR (M x K)
  B: dense column-major (K x N)  <- ArmPL expects B at weight_cm[n*K+k]
  C: dense row-major    (M x N)

  Unified logical B(k,n): we generate row-major weight_rm[k*N+n], then copy
  once to weight_cm[n*K+k] outside timing; ArmPL uses weight_cm.

  Output CSV:
    threshold, density,
    latency_sparse_create,
    latency_sparse_compute,
    latency_sparse_total
*/

static int M = 128;
static int K = 8192;
static int N = 8192;

static constexpr int WARMUP_RUNS = 0;
static constexpr int TIMED_RUNS  = 100;

static constexpr int CSV_WARMUP_RUNS = 10;
static constexpr int CSV_TIMED_RUNS  = 100;


// ------------------------------------------------------------
// thresholds (optional CLI override)
// ------------------------------------------------------------
static std::vector<float> default_thresholds() {
    return {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
            0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 0.99f};
}

// Parse command-line: M K N [threshold1 [threshold2 ...]] or "t1,t2,..."
static bool parse_args(int argc, char* argv[],
                       int& out_M, int& out_K, int& out_N,
                       std::vector<float>& out_thresholds) {
    if (argc < 4) {
        std::cerr << "Usage: " << (argv[0] ? argv[0] : "armpl_spmm_benchmark")
                  << " M K N [threshold1 [threshold2 ...]]\n"
                  << "  M K N = matrix dimensions (sparse MxK * dense KxN -> MxN)\n"
                  << "  thresholds = optional list of floats, or single \"t1,t2,...\"\n";
        return false;
    }
    out_M = std::atoi(argv[1]);
    out_K = std::atoi(argv[2]);
    out_N = std::atoi(argv[3]);
    out_thresholds.clear();
    for (int i = 4; i < argc; ++i) {
        std::string arg(argv[i]);
        for (size_t j = 0; j < arg.size(); ) {
            size_t k = arg.find(',', j);
            if (k == std::string::npos) k = arg.size();
            std::string part = arg.substr(j, k - j);
            float v;
            if (std::istringstream(part) >> v)
                out_thresholds.push_back(v);
            j = k + (k < arg.size() ? 1 : 0);
        }
    }
    if (out_thresholds.empty())
        out_thresholds = default_thresholds();
    return true;
}

// ------------------------------------------------------------
// Data loading: activation_{M}_{K}.bin, weight_{K}_{N}.bin (float32 row-major)
// ------------------------------------------------------------
// Load activation from same-dir activation_{M}_{K}.bin (M*K float32 row-major).
static bool load_activation_from_file(std::vector<float>& activation) {
    std::ostringstream oss;
    oss << "activation_" << M << "_" << K << ".bin";
    std::string path = oss.str();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open activation file: " << path << std::endl;
        return false;
    }
    size_t want = static_cast<size_t>(M) * K * sizeof(float);
    activation.resize(static_cast<size_t>(M) * K);
    f.read(reinterpret_cast<char*>(activation.data()), static_cast<std::streamsize>(want));
    if (!f || static_cast<size_t>(f.gcount()) != want) {
        std::cerr << "Activation file " << path << ": expected " << want << " bytes, got " << f.gcount() << std::endl;
        return false;
    }
    return true;
}

// Load weight from same-dir weight_{K}_{N}.bin (K*N float32 row-major).
static bool load_weight_from_file(std::vector<float>& weight_rm) {
    std::ostringstream oss;
    oss << "weight_" << K << "_" << N << ".bin";
    std::string path = oss.str();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open weight file: " << path << std::endl;
        return false;
    }
    size_t want = static_cast<size_t>(K) * N * sizeof(float);
    f.read(reinterpret_cast<char*>(weight_rm.data()), static_cast<std::streamsize>(want));
    if (!f || static_cast<size_t>(f.gcount()) != want) {
        std::cerr << "Weight file " << path << ": expected " << want << " bytes, got " << f.gcount() << std::endl;
        return false;
    }
    return true;
}

// Copy row-major B(k,n) -> column-major. Done once outside timing (same as graphblas).
static void weight_rm_to_cm(const std::vector<float>& weight_rm,
                            std::vector<float>& weight_cm) {
    weight_cm.resize(static_cast<size_t>(K) * N);
#pragma omp parallel for
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight_cm[static_cast<size_t>(n) * K + k] =
                weight_rm[static_cast<size_t>(k) * N + n];
}

// ------------------------------------------------------------
// Reference and correctness: ref = masked(activation, th) * weight_rm; compare with atol/rtol
// ------------------------------------------------------------
// Reference: ref = masked(activation, th) * weight_rm (row-major). ref is M*N row-major.
static void compute_reference(const std::vector<float>& activation,
                              const std::vector<float>& weight_rm,
                              float threshold,
                              std::vector<float>& ref) {
    ref.resize(static_cast<size_t>(M) * N);
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float a = activation[static_cast<size_t>(i) * K + k];
                if (std::fabs(a) > threshold)
                    sum += a * weight_rm[static_cast<size_t>(k) * N + j];
            }
            ref[static_cast<size_t>(i) * N + j] = sum;
        }
    }
}

static bool verify_correctness(const float* output, const std::vector<float>& ref,
                               float atol = 1e-3f, float rtol = 1e-2f) {
    for (size_t i = 0; i < ref.size(); ++i) {
        float a = output[i];
        float b = ref[i];
        float tol = atol + rtol * std::fabs(b);
        if (std::fabs(a - b) > tol) {
            std::cerr << "Correctness check failed: at index " << i
                      << " got " << a << " expected " << b << " (diff " << std::fabs(a - b)
                      << " > " << tol << ")\n";
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------
// sparsify to CSR
// ------------------------------------------------------------
static void sparsify_to_csr(const std::vector<float>& A,
                            float threshold,
                            std::vector<armpl_int_t>& rowptr,
                            std::vector<armpl_int_t>& colind,
                            std::vector<float>& vals)
{
    rowptr.resize(M + 1);
    armpl_int_t nnz = 0;
    for (int i = 0; i < M; ++i) {
        rowptr[i] = nnz;
        for (int j = 0; j < K; ++j)
            if (std::fabs(A[(size_t)i * K + j]) > threshold) nnz++;
    }
    rowptr[M] = nnz;
    colind.resize(nnz);
    vals.resize(nnz);
    nnz = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float v = A[(size_t)i * K + j];
            if (std::fabs(v) > threshold) {
                colind[nnz] = j;
                vals[nnz]   = v;
                nnz++;
            }
        }
    }

}

// ------------------------------------------------------------
// timing helpers
// ------------------------------------------------------------
static double time_sparse_create(const std::vector<float>& activation,
                                 float threshold)
{
    std::vector<armpl_int_t> rowptr, colind;
    std::vector<float> vals;
    armpl_spmat_t A = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < TIMED_RUNS; ++r) {
        sparsify_to_csr(activation, threshold, rowptr, colind, vals);

        armpl_spmat_create_csr_s(
            &A, M, K,
            rowptr.data(),
            colind.data(),
            vals.data(),
            0
        );
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    armpl_spmat_destroy(A);
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
           / TIMED_RUNS;
}

static double time_sparse_compute(armpl_spmat_t A,
                                  armpl_spmat_t B,
                                  armpl_spmat_t C)
{
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        armpl_spmm_exec_s(
            ARMPL_SPARSE_OPERATION_NOTRANS,
            ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f, A, B, 0.0f, C
        );
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TIMED_RUNS; ++i) {
        armpl_spmm_exec_s(
            ARMPL_SPARSE_OPERATION_NOTRANS,
            ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f, A, B, 0.0f, C
        );
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count()
           / TIMED_RUNS;
}

// Dense GEMM baseline: mask activation (zero out |x|<=th), pass as dense to sgemm (no CSR).
// Only total time is reported.
static void mask_activation_to_dense(const std::vector<float>& activation,
                                     float threshold,
                                     std::vector<float>& out) {
    out.resize(activation.size());
    for (size_t i = 0; i < activation.size(); ++i)
        out[i] = (std::fabs(activation[i]) > threshold) ? activation[i] : 0.f;
}

static double time_dense_gemm_total(const std::vector<float>& activation,
                                    float threshold,
                                    const std::vector<float>& weight_rm,
                                    std::vector<float>& output,
                                    int runs) {
    for (int w = 0; w < WARMUP_RUNS; ++w) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, activation.data(), K,
                    weight_rm.data(), N, 0.0f, output.data(), N);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, activation.data(), K,
                    weight_rm.data(), N, 0.0f, output.data(), N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
}

// weight must be column-major weight_cm[n*K+k] (from weight_rm_to_cm).
static double time_sparse_total(const std::vector<float>& activation,
                                float threshold,
                                const std::vector<float>& weight_cm,
                                std::vector<float>& output)
                                {
    std::vector<armpl_int_t> rowptr, colind;
    std::vector<float> vals;
    
    // B will be reused across all runs
    armpl_spmat_t  B = nullptr, C = nullptr;
    armpl_spmat_create_dense_s(
        &B, ARMPL_COL_MAJOR,
        K, N, K,
        const_cast<float*>(weight_cm.data()), 0
    );
    armpl_spmat_create_dense_s(
        &C, ARMPL_ROW_MAJOR,
        M, N, N,
        output.data(), ARMPL_SPARSE_CREATE_NOCOPY
    );
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; ++w) {
        sparsify_to_csr(activation, threshold, rowptr, colind, vals);
        armpl_spmat_t A = nullptr;
        armpl_spmat_create_csr_s(
            &A, M, K,
            rowptr.data(),
            colind.data(),
            vals.data(),
            0
        );
        armpl_spmm_exec_s(
            ARMPL_SPARSE_OPERATION_NOTRANS,
            ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f, A, B, 0.0f, C
        );
        armpl_spmat_destroy(A);
    }

    armpl_spmat_t A = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < TIMED_RUNS; ++r) {
        sparsify_to_csr(activation, threshold, rowptr, colind, vals);
        
        
        armpl_spmat_create_csr_s(
            &A, M, K,
            rowptr.data(),
            colind.data(),
            vals.data(),
            0
        );


        armpl_status_t st = armpl_spmm_exec_s(
            ARMPL_SPARSE_OPERATION_NOTRANS,
            ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f, A, B, 0.0f, C
        );
    
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    armpl_spmat_destroy(A);
    armpl_spmat_destroy(C);
    armpl_spmat_destroy(B);
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
    / TIMED_RUNS;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    int in_M, in_K, in_N;
    std::vector<float> thresholds;
    if (!parse_args(argc, argv, in_M, in_K, in_N, thresholds)) {
        return 1;
    }
    M = in_M;
    K = in_K;
    N = in_N;

    std::cout << "ArmPL Sparse SpMM Benchmark (aligned to oneDNN)\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "threshold,density,"
              << "latency_sparse_create,"
              << "latency_sparse_compute,"
              << "latency_sparse_total,"
              << "latency_dense\n";

    std::ofstream csv("armpl_spmm_results.csv");
    csv << "threshold,density,latency_sparse_create,"
           "latency_sparse_compute,latency_sparse_total,latency_dense\n";

    std::vector<float> activation(static_cast<size_t>(M) * K);
    std::vector<float> weight_rm(static_cast<size_t>(K) * N);
    std::vector<float> weight_cm;
    std::vector<float> output(static_cast<size_t>(M) * N, 0.f);

    if (!load_activation_from_file(activation)) return 1;
    if (!load_weight_from_file(weight_rm)) return 1;
    weight_rm_to_cm(weight_rm, weight_cm);  // once outside timing (ArmPL needs B column-major)

    // reusable dense wrappers (B uses column-major)
    armpl_spmat_t B = nullptr, C = nullptr;
    armpl_spmat_create_dense_s(
        &B, ARMPL_COL_MAJOR, K, N, K, weight_cm.data(), 0);
    armpl_spmat_create_dense_s(
        &C, ARMPL_ROW_MAJOR, M, N, N, output.data(), 0);

    bool verified = false;
    std::vector<float> ref;

    for (float th : thresholds) {
        std::vector<armpl_int_t> rowptr, colind;
        std::vector<float> vals;
        sparsify_to_csr(activation, th, rowptr, colind, vals);

        if (vals.empty()) continue;

        double density =
            double(vals.size()) / (double(M) * K);

        armpl_spmat_t A = nullptr;
        armpl_spmat_create_csr_s(
            &A, M, K,
            rowptr.data(),
            colind.data(),
            vals.data(),
            0
        );

        double t_create  = time_sparse_create(activation, th);
        double t_compute = time_sparse_compute(A, B, C);


        double t_total   = time_sparse_total(activation, th, weight_cm, output);
        double t_dense = time_dense_gemm_total(activation, th, weight_rm, output, CSV_TIMED_RUNS);

        if (!verified) {
            compute_reference(activation, weight_rm, th, ref);
            if (!verify_correctness(output.data(), ref)) {
                armpl_spmat_destroy(A);
                armpl_spmat_destroy(B);
                armpl_spmat_destroy(C);
                return 1;
            }
            verified = true;
            std::cout << "Correctness verification passed (threshold=" << th << ").\n";
        }

        std::cout << th << "," << density << ","
                  << t_create << ","
                  << t_compute << ","
                  << t_total << ","
                  << t_dense << "\n";

        csv << th << "," << density << ","
            << t_create << ","
            << t_compute << ","
            << t_total << ","
            << t_dense << "\n";

        armpl_spmat_destroy(A);
    }

    armpl_spmat_destroy(B);
    armpl_spmat_destroy(C);
    csv.close();

    std::cout << "Results written to armpl_spmm_results.csv\n";
    return 0;
}
