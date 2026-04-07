#!/usr/bin/env python3
"""
运行 ArmPL / GraphBLAS / oneDNN / PyTorch 四个 SpMM benchmark，并生成性能对比结果。

用法:
  python run_all_benchmarks.py [选项]
  python run_all_benchmarks.py --M 128 --K 4096 --N 4096
  python run_all_benchmarks.py --thresholds 0.0,0.3,0.5,0.9 --no-build

环境:
  - 需先编译 C/C++ benchmark: ./build.sh（或 --build 时本脚本会尝试执行）
  - ArmPL/GraphBLAS/oneDNN 需已配置好库路径（见 env.sh / build.sh）
"""

import argparse
import csv
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 各 benchmark 可执行路径与输出 CSV（列名不含 threshold,density，按 CSV 第 3 列起顺序）
BENCHMARKS = {
    "armpl": {
        "cmd": ["./build/armpl_spmm_benchmark"],
        "csv": "armpl_spmm_results.csv",
        "cols": ["create", "compute", "total", "dense"],
    },
    "graphblas": {
        "cmd": ["./build/graphblas_spmm_benchmark"],
        "csv": "graphblas_spmm_results.csv",
        "cols": ["create", "compute", "total", "dense"],
    },
    "onednn": {
        "cmd": ["./build/onednn_spmm_benchmark"],
        "csv": "onednn_spmm_results.csv",
        "cols": ["create", "compute", "total", "dense"],
    },
    "pytorch": {
        "cmd": [sys.executable, "pytorch_spmm_benchmark.py"],
        "csv": "pytorch_sparse_spmm_results.csv",
        "cols": ["create", "compute", "total", "dense"],
    },
}

# 默认规模（与各 benchmark 内置默认一致）
DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 8192, 8192

# Weight 文件名约定：同目录下 weight_{K}_{N}.bin，K×N 个 float32 row-major
def weight_path(cwd: Path, K: int, N: int) -> Path:
    return cwd / f"weight_{K}_{N}.bin"


def ensure_weight_file(cwd: Path, K: int, N: int, seed: int = 123) -> Path:
    """生成或复用同目录下的 weight_{K}_{N}.bin（K×N row-major float32）。"""
    path = weight_path(cwd, K, N)
    if path.exists():
        expected = K * N * 4
        if path.stat().st_size == expected:
            return path
    import random
    random.seed(seed)
    with open(path, "wb") as f:
        for _ in range(K * N):
            # [-0.1, 0.1) 与各 benchmark 原 gen_weight 一致
            v = 0.2 * random.random() - 0.1
            f.write(struct.pack("<f", v))
    return path


# Activation 文件名约定：同目录下 activation_{M}_{K}.bin，M×K 个 float32 row-major（与 weight 同一口径）
def activation_path(cwd: Path, M: int, K: int) -> Path:
    return cwd / f"activation_{M}_{K}.bin"


def ensure_activation_file(cwd: Path, M: int, K: int, seed: int = 42) -> Path:
    """生成或复用同目录下的 activation_{M}_{K}.bin（M×K row-major float32）。"""
    path = activation_path(cwd, M, K)
    if path.exists():
        expected = M * K * 4
        if path.stat().st_size == expected:
            return path
    import random
    random.seed(seed)
    with open(path, "wb") as f:
        for _ in range(M * K):
            # [-1, 1) 与各 benchmark 原 activation 一致
            v = 2.0 * random.random() - 1.0
            f.write(struct.pack("<f", v))
    return path


def parse_args():
    p = argparse.ArgumentParser(description="运行四个 SpMM benchmark 并对比性能")
    p.add_argument("--M", type=int, default=None, help="矩阵维度 M（默认 %d）" % DEFAULT_M)
    p.add_argument("--K", type=int, default=None, help="矩阵维度 K（默认 %d）" % DEFAULT_K)
    p.add_argument("--N", type=int, default=None, help="矩阵维度 N（默认 %d）" % DEFAULT_N)
    p.add_argument("--thresholds", type=str, default=None, help="逗号分隔阈值，如 0.0,0.3,0.5,0.9；不传则用各 benchmark 内置默认")
    p.add_argument("--build", action="store_true", help="先执行 build.sh 再跑")
    p.add_argument("--no-build", action="store_true", help="不检查/不构建，直接跑（默认会检查可执行文件是否存在）")
    p.add_argument("--skip", type=str, default="", help="逗号分隔要跳过的后端，如 armpl,graphblas")
    p.add_argument("--out", type=str, default=None, help="合并对比输出 CSV 路径（默认: benchmark_YYYYMMDD_HHMMSS.csv）")
    p.add_argument("--timeout", type=int, default=0, help="每个 benchmark 最长运行秒数，0 表示不限制（默认 1800）")
    p.add_argument("--list-only", action="store_true", help="只列出将要运行的后端并退出")
    # OpenMP 环境（注入到子进程，供 ArmPL/oneDNN/GraphBLAS 等使用）
    p.add_argument("--omp-num-threads", type=int, default=16, help="设置 OMP_NUM_THREADS")
    p.add_argument("--omp-proc-bind", type=str, default=None, help="设置 OMP_PROC_BIND（如 close, spread）")
    p.add_argument("--omp-places", type=str, default="cores", help="设置 OMP_PLACES（如 cores, threads）")
    p.add_argument("--omp-dynamic", type=str, default="false", help="设置 OMP_DYNAMIC（true/false）")
    return p.parse_args()


def get_env(args):
    env = os.environ.copy()
    if args.M is not None:
        env["M"] = str(args.M)
    if args.K is not None:
        env["K"] = str(args.K)
    if args.N is not None:
        env["N"] = str(args.N)
    if args.thresholds is not None:
        env["THRESHOLDS"] = args.thresholds
    if getattr(args, "omp_num_threads", None) is not None:
        env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    if getattr(args, "omp_proc_bind", None) is not None:
        env["OMP_PROC_BIND"] = args.omp_proc_bind
    if getattr(args, "omp_places", None) is not None:
        env["OMP_PLACES"] = args.omp_places
    if getattr(args, "omp_dynamic", None) is not None:
        env["OMP_DYNAMIC"] = args.omp_dynamic
    return env


def run_build():
    if not Path("build.sh").exists():
        print("未找到 build.sh，跳过构建")
        return False
    print("执行 build.sh ...")
    ret = subprocess.run(["bash", "build.sh"], cwd=Path(__file__).resolve().parent)
    return ret.returncode == 0


def run_one(name, cfg, env, cwd, args):
    M = args.M if args.M is not None else DEFAULT_M
    K = args.K if args.K is not None else DEFAULT_K
    N = args.N if args.N is not None else DEFAULT_N
    th_str = args.thresholds
    thresholds = th_str.split(",") if th_str else []  # 作为 CLI 的多个参数

    cmd = list(cfg["cmd"])
    if name == "pytorch":
        cmd.extend([str(M), str(K), str(N)])
        if thresholds:
            cmd.extend([t.strip() for t in thresholds if t.strip()])
    else:
        # armpl / graphblas / onednn: 位置参数 M K N [threshold1 threshold2 ...]
        cmd.extend([str(M), str(K), str(N)])
        if thresholds:
            cmd.extend([t.strip() for t in thresholds if t.strip()])
        exe = (cwd / cmd[0]).resolve() if not os.path.isabs(cmd[0]) else Path(cmd[0])
        if not exe.exists():
            print(f"[{name}] 可执行文件不存在: {exe}，跳过")
            return False
    print(f"[{name}] 运行: {' '.join(cmd)}")
    kwargs = {"env": env, "cwd": cwd}
    if getattr(args, "timeout", 1800) > 0:
        kwargs["timeout"] = args.timeout
    ret = subprocess.run(cmd, **kwargs)
    if ret.returncode != 0:
        print(f"[{name}] 退出码 {ret.returncode}")
        return False
    return cwd / cfg["csv"]


def load_csv(path):
    if not path or not Path(path).exists():
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)  # skip header
        for row in r:
            if len(row) < 2:
                continue
            rows.append(row)
    return rows


def collect_by_threshold(results):
    """results: dict backend_name -> list of rows. 合并为按 threshold 的对比表。
    用 (round(th), round(den,4)) 做 key，避免各后端 density 浮点误差导致同一 threshold 拆成多行。
    """
    keys = set()
    by_key = {}
    for name, rows in results.items():
        if rows is None:
            continue
        for row in rows:
            if len(row) < 2:
                continue
            try:
                th = float(row[0])
                den = float(row[1])
            except (ValueError, IndexError):
                continue
            key = (round(th, 6), round(den, 4))
            keys.add(key)
            if key not in by_key:
                by_key[key] = {}
            by_key[key][name] = row[2:]
    return sorted(keys), by_key


def merged_header(backends):
    h = ["threshold", "density"]
    for b in backends:
        for c in BENCHMARKS[b]["cols"]:
            h.append(f"{b}_{c}")
    return h


def merged_row(key, by_key, backends):
    th, den = key
    row = [th, den]
    for b in backends:
        vals = by_key.get(key, {}).get(b, [])
        cols = BENCHMARKS[b]["cols"]
        for i in range(len(cols)):
            row.append(vals[i] if i < len(vals) else "")
    return row


def main():
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    os.chdir(cwd)
    skip = set(s.strip().lower() for s in (args.skip or "").split(",") if s.strip())

    if args.build:
        run_build()
    elif not args.no_build:
        for name, cfg in BENCHMARKS.items():
            if name in skip:
                continue
            if name == "pytorch":
                continue
            exe = (cwd / cfg["cmd"][0]).resolve()
            if not exe.exists():
                print(f"未找到 {exe}，请先执行 ./build.sh 或使用 --no-build 仅跑已有可执行文件")
                # 不退出，继续跑能跑的

    if args.list_only:
        for name in BENCHMARKS:
            if name in skip:
                continue
            print(name)
        return 0

    M = args.M if args.M is not None else DEFAULT_M
    K = args.K if args.K is not None else DEFAULT_K
    N = args.N if args.N is not None else DEFAULT_N
    ensure_weight_file(cwd, K, N)
    ensure_activation_file(cwd, M, K)
    print(f"Weight file: {weight_path(cwd, K, N)} (K×N row-major)")
    print(f"Activation file: {activation_path(cwd, M, K)} (M×K row-major)")

    env = get_env(args)
    results = {}
    for name in ["armpl", "graphblas", "onednn", "pytorch"]:
        if name in skip:
            continue
        cfg = BENCHMARKS[name]
        path = run_one(name, cfg, env, cwd, args)
        if path and Path(path).exists():
            results[name] = load_csv(path)
        else:
            results[name] = None

    # 合并对比
    keys, by_key = collect_by_threshold(results)
    if not keys:
        print("无有效数据可对比")
        return 0

    backends = [b for b in BENCHMARKS if b not in skip and results.get(b)]
    out_header = merged_header(backends)
    out_rows = [merged_row(k, by_key, backends) for k in keys]

    out_name = args.out
    if out_name is None:
        out_name = f"benchmark_"+str(M)+"_"+str(K)+"_"+str(N)+".csv"
    out_path = cwd / out_name
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        w.writerows(out_rows)
    print(f"\n对比结果已写入: {out_path}")

    # 控制台：展示所有列（create / compute / total / dense），与合并 CSV 一致
    print("\n--- 性能对比 (ms)，全部列 ---")
    col_show = ["create", "compute", "total", "dense"]
    head = ["threshold", "density"]
    for b in backends:
        for c in col_show:
            if c in BENCHMARKS[b]["cols"]:
                head.append(f"{b}_{c}")
    print("  ".join(head))
    for key in keys:
        data = by_key.get(key, {})
        parts = [f"{key[0]:.2f}", f"{key[1]:.4f}"]
        for b in backends:
            vals = data.get(b, [])
            cols = BENCHMARKS[b]["cols"]
            for c in col_show:
                if c in cols:
                    i = cols.index(c)
                    parts.append(vals[i] if i < len(vals) else "-")
        print("  ".join(parts))
    print(f"共 {len(keys)} 行，完整数据见 {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
