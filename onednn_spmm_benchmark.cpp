/**
 * oneDNN Sparse SpMM Benchmark (fair vs ArmPL-style semantics)
 *
 * activation [M, K] float32 row-major
 * weight     [K, N] float32 row-major
 * threshold controls sparsity: keep abs(x) > threshold
 *
 * Output CSV:
 *   threshold, density,
 *   latency_sparse_create, latency_sparse_compute, latency_sparse_total,
 *   latency_dense
 *
 * Timing semantics (ms, per-run average):
 *   sparse_create = sparsify_to_csr + create src_csr_md + primitive_desc + primitive
 *                  (NO execute)
 *   sparse_compute= execute only (primitive/mem pre-built), synchronized by stream.wait()
 *   sparse_total  = (sparsify + pd/prim + src_mem) + one execute, with weight/dst memory reused
 *   dense         = oneDNN dense matmul execute only (primitive/mem pre-built)
 */

 #include <oneapi/dnnl/dnnl.hpp>
 #include <cmath>
 #include <chrono>
 #include <cstdlib>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <random>
 #include <sstream>
 #include <string>
 #include <vector>

 using namespace dnnl;
 
 // ------------------------------------------------------------
 // thresholds (optional CLI override)
 // ------------------------------------------------------------
 static int M = 128;
 static int K = 8192;
 static int N = 8192;

 static constexpr int WARMUP_RUNS = 0;
 static constexpr int TIMED_RUNS  = 100;

 static constexpr int CSV_WARMUP_RUNS = 10;
 static constexpr int CSV_TIMED_RUNS  = 100;

 static std::vector<float> default_thresholds() {
     return {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
            0.6f, 0.7f, 0.8f, 0.9f, 0.95f, 0.99f};
 }

 // Parse command-line: M K N [threshold1 [threshold2 ...]] or "t1,t2,..."
 static bool parse_args(int argc, char* argv[],
                        int& out_M, int& out_K, int& out_N,
                        std::vector<float>& out_thresholds) {
     if (argc < 4) {
         std::cerr << "Usage: " << (argv[0] ? argv[0] : "onednn_spmm_benchmark")
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

// oneDNN uses weight row-major (ab); no weight_rm_to_cm needed.

// ------------------------------------------------------------
// Reference and correctness: ref = masked(activation, th) * weight_rm; compare with atol/rtol
// ------------------------------------------------------------
// Reference: ref = masked(activation, th) * weight_rm (row-major). ref is M*N row-major.
static void compute_reference(const std::vector<float>& activation,
                              const std::vector<float>& weight_rm,
                              float threshold,
                              std::vector<float>& ref) {
    ref.resize(static_cast<size_t>(M) * N);
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

static bool verify_correctness(const std::vector<float>& result, const std::vector<float>& ref,
                               float atol = 1e-3f, float rtol = 1e-2f) {
    for (size_t i = 0; i < ref.size(); ++i) {
        float a = result[i];
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
// Sparsify: activation [M,K] + threshold -> CSR; keep abs(x) > threshold
// CSR: values (nnz), col_ind (nnz), row_ptr (M+1), row_ptr[i]=start of row i, row_ptr[M]=nnz
// ------------------------------------------------------------
 static int64_t sparsify_to_csr(const std::vector<float>& activation, float threshold,
                             std::vector<float>& values,
                             std::vector<int32_t>& col_ind,
                             std::vector<int32_t>& row_ptr) {
     values.clear();
     col_ind.clear();
     row_ptr.clear();
     row_ptr.reserve(static_cast<size_t>(M) + 1);
     values.reserve(static_cast<size_t>(M) * K);
     col_ind.reserve(static_cast<size_t>(M) * K);

     for (int i = 0; i < M; ++i) {
         row_ptr.push_back(static_cast<int32_t>(values.size()));
         const float* row = activation.data() + static_cast<size_t>(i) * K;
         for (int j = 0; j < K; ++j) {
             float v = row[j];
             if (std::fabs(v) > threshold) {
                 values.push_back(v);
                 col_ind.push_back(j);
             }
         }
     }
     row_ptr.push_back(static_cast<int32_t>(values.size()));
     return (int64_t)values.size();
 }
 
 // ------------------------------------------------------------
 // Timing: sparse_create, sparse_compute, sparse_total (same semantics as armpl)
 // ------------------------------------------------------------
 // sparse_create = sparsify + md(csr) + primitive_desc + primitive (NO execute)
 static double time_sparse_create(int timed_runs,
     const engine& eng,
     const std::vector<float>& activation,
     float threshold,
     const memory::desc& wei_md,
     const memory::desc& dst_md,
     std::vector<float>& csr_values,
     std::vector<int32_t>& csr_col_ind,
     std::vector<int32_t>& csr_row_ptr) {

     const memory::dims src_dims = {M, K};

     auto t0 = std::chrono::high_resolution_clock::now();
     for (int r = 0; r < timed_runs; ++r) {
        int64_t nnz = sparsify_to_csr(activation, threshold, csr_values, csr_col_ind, csr_row_ptr);
         auto src_csr_md = memory::desc::csr(src_dims, memory::data_type::f32,
                                             nnz, memory::data_type::s32, memory::data_type::s32);
         auto pd = matmul::primitive_desc(eng, src_csr_md, wei_md, dst_md);
         auto prim = matmul(pd);
         (void)prim;
     }
     auto t1 = std::chrono::high_resolution_clock::now();
     return std::chrono::duration<double, std::milli>(t1 - t0).count() / timed_runs;
 }
 
 // sparse_compute = execute only (prim/mem already built), synchronized
 static double time_sparse_compute(int timed_runs,
     stream& s,
     const matmul& prim,
     const memory& src_csr_mem,
     const memory& wei_mem,
     const memory& dst_mem) {

     // Warmup
     for (int r = 0; r < WARMUP_RUNS; ++r) {
         prim.execute(s, {{DNNL_ARG_SRC, src_csr_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});
     }
     s.wait();

     auto t0 = std::chrono::high_resolution_clock::now();
     for (int r = 0; r < timed_runs; ++r) {
         prim.execute(s, {{DNNL_ARG_SRC, src_csr_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});
     }
     s.wait();
     auto t1 = std::chrono::high_resolution_clock::now();
     return std::chrono::duration<double, std::milli>(t1 - t0).count() / timed_runs;
 }


 // sparse_total = (sparsify + pd/prim + src_mem) + one execute; weight/dst reused
 static double time_sparse_total(int timed_runs,
     const engine& eng,
     stream& s,
     const std::vector<float>& activation,
     float threshold,
     const memory::desc& wei_md,
     const memory::desc& dst_md,
     const memory& wei_mem,
     const memory& dst_mem,
     std::vector<float>& csr_values,
     std::vector<int32_t>& csr_col_ind,
     std::vector<int32_t>& csr_row_ptr) {

     const memory::dims src_dims = {M, K};

     auto t0 = std::chrono::high_resolution_clock::now();
     for (int r = 0; r < timed_runs; ++r) {
         int64_t nnz = sparsify_to_csr(activation, threshold, csr_values, csr_col_ind, csr_row_ptr);

         auto src_csr_md = memory::desc::csr(src_dims, memory::data_type::f32,
                                             nnz, memory::data_type::s32, memory::data_type::s32);

         auto pd = matmul::primitive_desc(eng, src_csr_md, wei_md, dst_md);
         auto prim = matmul(pd);

         memory src_mem(src_csr_md, eng,
                        {csr_values.data(), csr_col_ind.data(), csr_row_ptr.data()});

         prim.execute(s, {{DNNL_ARG_SRC, src_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});
     }
     s.wait();
     auto t1 = std::chrono::high_resolution_clock::now();
     return std::chrono::duration<double, std::milli>(t1 - t0).count() / timed_runs;
 }
 
 // Dense baseline: execute only (primitive/mem pre-built)
 static double time_dense_compute(int timed_runs,
     stream& s,
     const matmul& prim,
     const memory& src_mem,
     const memory& wei_mem,
     const memory& dst_mem) {

     // Warmup
     for (int r = 0; r < WARMUP_RUNS; ++r) {
         prim.execute(s, {{DNNL_ARG_SRC, src_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});
     }
     s.wait();

     auto t0 = std::chrono::high_resolution_clock::now();
     for (int r = 0; r < timed_runs; ++r) {
         prim.execute(s, {{DNNL_ARG_SRC, src_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});
     }
     s.wait();
     auto t1 = std::chrono::high_resolution_clock::now();
     return std::chrono::duration<double, std::milli>(t1 - t0).count() / timed_runs;
 }
 
 int main(int argc, char* argv[]) {
    int in_M, in_K, in_N;
    std::vector<float> thresholds;
    if (!parse_args(argc, argv, in_M, in_K, in_N, thresholds)) {
        return 1;
    }
    M = in_M;
    K = in_K;
    N = in_N;

    std::cout << "oneDNN Sparse SpMM Benchmark (fair vs ArmPL semantics) (M="
              << M << ", K=" << K << ", N=" << N << ")\n";
    std::cout << "threshold,density,latency_sparse_create,latency_sparse_compute,"
                 "latency_sparse_total,latency_dense\n";

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    std::vector<float> activation(static_cast<size_t>(M) * K);
    std::vector<float> weight(static_cast<size_t>(K) * N);
    std::vector<float> result(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> ref_result(static_cast<size_t>(M) * N, 0.0f);

    if (!load_activation_from_file(activation)) return 1;
    if (!load_weight_from_file(weight)) return 1;

    // oneDNN dense descriptors (row-major: ab)
    const memory::dims src_dims = {M, K};
    const memory::dims wei_dims = {K, N};
    const memory::dims dst_dims = {M, N};

    auto dense_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ab);
    auto wei_md       = memory::desc(wei_dims, memory::data_type::f32, memory::format_tag::ab);
    auto dst_md       = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ab);

    // Reused dense memories (align ArmPL B/C wrappers reused)
    memory dense_src_mem(dense_src_md, eng, activation.data());
    memory wei_mem(wei_md, eng, weight.data());
    memory dst_mem(dst_md, eng, result.data());

    // Dense primitive (baseline) - uses row-major weights directly
    matmul dense_prim(matmul::primitive_desc(eng, dense_src_md, wei_md, dst_md));

    std::ofstream csv("onednn_spmm_results.csv");
    csv << "threshold,density,latency_sparse_create,latency_sparse_compute,latency_sparse_total,latency_dense\n";

    // Reused CSR buffers (avoid reallocation noise)
    std::vector<float> csr_values;
    std::vector<int32_t> csr_col_ind, csr_row_ptr;
    bool verified = false;

    for (float threshold : thresholds) {
        // Build CSR once for density + warmup/compute
        int64_t nnz = sparsify_to_csr(activation, threshold, csr_values, csr_col_ind, csr_row_ptr);
        const double density = static_cast<double>(nnz) / (static_cast<double>(M) * K);

        if (nnz == 0) {
            std::cerr << "Warning: threshold " << threshold << " produced 0 nnz, skipping\n";
            continue;
        }

        // Sparse md depends on nnz (CSR: values, col_ind, row_ptr)
        auto src_csr_md = memory::desc::csr(
            src_dims,
            memory::data_type::f32,
            nnz,
            memory::data_type::s32,
            memory::data_type::s32
        );

        // Build sparse primitive desc with weights=any (let oneDNN choose optimal)
        auto wei_any_md = memory::desc(wei_dims, memory::data_type::f32, memory::format_tag::any);
        matmul::primitive_desc sp_pd(eng, src_csr_md, wei_any_md, dst_md);
        matmul sp_prim(sp_pd);

        // CSR memory (data handles: values, col_ind, row_ptr)
        memory src_csr_mem(src_csr_md, eng, {csr_values.data(), csr_col_ind.data(), csr_row_ptr.data()});

        // Prepare weights memory for THIS sparse primitive (reorder if needed)
        memory sp_wei_opt(sp_pd.weights_desc(), eng);
        memory *wei_exec_mem = &sp_wei_opt;
        if (sp_pd.weights_desc() != wei_mem.get_desc()) {
            reorder(wei_mem, sp_wei_opt).execute(s, wei_mem, sp_wei_opt);
            s.wait();
        } else {
            // No reorder needed; use user weight memory directly
            wei_exec_mem = &wei_mem;
        }

        if (!verified) {
            compute_reference(activation, weight, threshold, ref_result);
            sp_prim.execute(s, {
                {DNNL_ARG_SRC, src_csr_mem},
                {DNNL_ARG_WEIGHTS, *wei_exec_mem},
                {DNNL_ARG_DST, dst_mem}
            });
            s.wait();

            if (!verify_correctness(result, ref_result)) {
                std::cerr << "Correctness verification failed (threshold=" << threshold << ").\n";
                return 1;
            }
            verified = true;
            std::cout << "Correctness verification passed (threshold=" << threshold << ").\n";
        }

        // Timings (warmup is done inside each time_* function)
        // NOTE: For dense baseline, always pass wei_mem (row-major) to avoid desc mismatch.
        double latency_create = time_sparse_create(
            TIMED_RUNS, eng, activation, threshold,
            wei_any_md, dst_md, csr_values, csr_col_ind, csr_row_ptr
        );

        double latency_compute = time_sparse_compute(
            TIMED_RUNS, s, sp_prim, src_csr_mem, *wei_exec_mem, dst_mem
        );

        double latency_total = time_sparse_total(
            TIMED_RUNS, eng, s, activation, threshold,
            wei_any_md, dst_md, *wei_exec_mem, dst_mem,
            csr_values, csr_col_ind, csr_row_ptr
        );

        double latency_dense = time_dense_compute(
            TIMED_RUNS, s, dense_prim, dense_src_mem, wei_mem, dst_mem
        );

        std::cout << threshold << "," << density << ","
                  << latency_create << "," << latency_compute << ","
                  << latency_total << "," << latency_dense << "\n";

        csv << threshold << "," << density << ","
            << latency_create << "," << latency_compute << ","
            << latency_total << "," << latency_dense << "\n";
    }

    csv.close();
    std::cout << "Results written to onednn_spmm_results.csv\n";
    return 0;
}
