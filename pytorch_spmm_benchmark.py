"""
PyTorch Sparse SpMM Benchmark (CSR) (fair vs ArmPL-style scan)

activation [M, K] float32, weight [K, N] float32, threshold controls sparsity
Pipeline:
  1) threshold sparsify -> CSR (no clone+zero-fill; build from nnz)
  2) sparse @ dense -> [M,N] via torch.sparse.mm or csr.matmul

CSV:
  threshold,density,latency_sparse_create,latency_sparse_compute,latency_sparse_total,latency_dense

Timing semantics (ms, per-run avg):
  create = sparsify scan + build sparse CSR (includes COO->CSR conversion)
  compute= only sparse matmul (CSR fixed across runs)
  total  = create + one sparse matmul
  dense  = torch.mm baseline (full dense, no masking)
  dense_gemm_total = mask activation (zero out |x|<=th) + torch.matmul(masked, weight); total time only (no CSR/COO).
"""

import argparse
import csv
import os
import struct
import time
import torch

WARMUP_RUNS = 0
TIMED_RUNS = 100
DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def parse_args():
    p = argparse.ArgumentParser(
        description="PyTorch Sparse SpMM Benchmark (CSR)",
        epilog="Example: python pytorch_spmm_benchmark.py 128 8192 8192 0.0 0.1 0.2 0.9",
    )
    p.add_argument("M", type=int, help="rows of sparse matrix (activation M x K)")
    p.add_argument("K", type=int, help="inner dimension (weight K x N)")
    p.add_argument("N", type=int, help="columns of dense matrix (result M x N)")
    p.add_argument(
        "thresholds",
        type=float,
        nargs="*",
        help="optional list of thresholds (default: built-in list)",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    p.add_argument("--no-verify", action="store_true", help="skip correctness verification")
    return p.parse_args()


def torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def load_activation_from_file(M, K, device, dtype, cwd=None):
    """Load activation from same-dir activation_{M}_{K}.bin (M*K float32 row-major). Same convention as weight."""
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__)) or "."
    path = os.path.join(cwd, f"activation_{M}_{K}.bin")
    with open(path, "rb") as f:
        raw = f.read()
    want = M * K * 4
    if len(raw) != want:
        raise FileNotFoundError(f"Activation file {path}: expected {want} bytes, got {len(raw)}")
    x = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().view(M, K)
    if dtype == torch.float16:
        x = x.half()
    return x.to(device=device, non_blocking=True)


def load_weight_from_file(K, N, device, dtype, cwd=None):
    """Load weight from same-dir weight_{K}_{N}.bin (K*N float32 row-major). PyTorch uses row-major, no transpose."""
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__)) or "."
    path = os.path.join(cwd, f"weight_{K}_{N}.bin")
    with open(path, "rb") as f:
        raw = f.read()
    want = K * N * 4
    if len(raw) != want:
        raise FileNotFoundError(f"Weight file {path}: expected {want} bytes, got {len(raw)}")
    w = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().view(K, N)
    if dtype == torch.float16:
        w = w.half()
    return w.to(device=device, non_blocking=True)


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def sparsify_to_csr(activation: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Fair CSR baseline:
    - No COO index materialization via nonzero()
    - No global sort (only per-row order as in K-scan)
    - Two-phase: count -> prefix-sum -> fill
    """
    x = torch.where(activation.abs() >= threshold, activation, torch.zeros_like(activation))
    return x.to_sparse_csr()

def sparse_mm_csr(csr: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Prefer torch.sparse.mm if supported; fallback to csr.matmul.
    """
    try:
        return torch.sparse.mm(csr, weight)
    except Exception:
        return csr.matmul(weight)


def dense_spmm(activation, weight):
    return torch.mm(activation, weight)


def verify_correctness(activation: torch.Tensor, weight: torch.Tensor, threshold: float,
                       atol: float = 1e-3, rtol: float = 1e-2) -> None:
    """Compare sparse SpMM result to reference (masked_activation @ weight). Raises on mismatch."""
    mask = (activation.abs() > threshold).to(activation.dtype)
    masked_activation = activation * mask
    ref = torch.mm(masked_activation, weight)
    csr = sparsify_to_csr(activation, threshold)
    sparse_result = sparse_mm_csr(csr, weight)
    if not torch.allclose(sparse_result, ref, atol=atol, rtol=rtol):
        diff = (sparse_result - ref).abs()
        max_diff = diff.max().item()
        raise AssertionError(
            f"Correctness check failed (threshold={threshold}): max|sparse-ref|={max_diff}, "
            f"atol={atol}, rtol={rtol}"
        )


def time_sparse_create(activation, threshold, timed_runs, device):
    for _ in range(WARMUP_RUNS):
        _ = sparsify_to_csr(activation, threshold)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        _ = sparsify_to_csr(activation, threshold)
    sync(device)
    return (time.perf_counter() - t0) * 1000.0 / timed_runs


def time_sparse_compute(sparse_activation_csr, weight, timed_runs, device):
    for _ in range(WARMUP_RUNS):
        _ = sparse_mm_csr(sparse_activation_csr, weight)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        _ = sparse_mm_csr(sparse_activation_csr, weight)
    sync(device)
    return (time.perf_counter() - t0) * 1000.0 / timed_runs


def time_sparse_total(activation, threshold, weight, timed_runs, device):
    for _ in range(WARMUP_RUNS):
        csr = sparsify_to_csr(activation, threshold)
        _ = sparse_mm_csr(csr, weight)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        csr = sparsify_to_csr(activation, threshold)
        _ = sparse_mm_csr(csr, weight)
    sync(device)
    return (time.perf_counter() - t0) * 1000.0 / timed_runs

def time_dense_gemm_total(activation, weight, threshold, timed_runs, device):
    """Mask activation (zero out |x|<=threshold), then torch.matmul(masked, weight). Total time only."""
    for _ in range(WARMUP_RUNS):
        _ = torch.matmul(activation, weight)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        _ = torch.mm(activation, weight)
    sync(device)
    return (time.perf_counter() - t0) * 1000.0 / timed_runs


def main():
    args = parse_args()
    M, K, N = args.M, args.K, args.N
    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    device = args.device
    dtype = torch_dtype(args.dtype)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"PyTorch Sparse SpMM Benchmark (CSR, fair) (M={M}, K={K}, N={N}, device={device}, dtype={dtype})")
    print("threshold,density,latency_sparse_create,latency_sparse_compute,latency_sparse_total,latency_dense")

    # activation/weight from run_all_benchmarks.py generated bin files (same-dir activation_{M}_{K}.bin, weight_{K}_{N}.bin)
    activation = load_activation_from_file(M, K, device=device, dtype=dtype)
    weight = load_weight_from_file(K, N, device, dtype)

    if not args.no_verify:
        verify_correctness(activation, weight, thresholds[0])
        print("Correctness verification passed (threshold=%.4f)." % thresholds[0])

    with open("pytorch_sparse_spmm_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "density", "latency_sparse_create",
                         "latency_sparse_compute", "latency_sparse_total", "latency_dense"])

        for threshold in thresholds:
            sparse_activation = sparsify_to_csr(activation, threshold)

            # nnz for CSR: values length
            nnz = int(sparse_activation.values().numel())
            density = nnz / (M * K)

            if nnz == 0:
                print(f"Warning: threshold {threshold} produced 0 nnz, skipping")
                continue

            latency_create = time_sparse_create(activation, threshold, TIMED_RUNS, device)
            latency_compute = time_sparse_compute(sparse_activation, weight, TIMED_RUNS, device)
            latency_total = time_sparse_total(activation, threshold, weight, TIMED_RUNS, device)
            latency_dense = time_dense_gemm_total(activation, weight, threshold, TIMED_RUNS, device)

            print(f"{threshold},{density},{latency_create},{latency_compute},{latency_total},{latency_dense}")
            writer.writerow([threshold, density, latency_create,
                             latency_compute, latency_total, latency_dense])

    print("Results written to pytorch_sparse_spmm_results.csv")


if __name__ == "__main__":
    main()
