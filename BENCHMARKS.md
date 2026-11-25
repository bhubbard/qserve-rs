# Benchmarks & Coverage

Measured on **Apple M3 Pro** (`arm64`, 11 cores, 18 GiB RAM, macOS) with `rustc 1.100.0` (`--release`) vs. `Python 3.15.2` (same host) and upstream `mit-han-lab/qserve`.

## 1. Code Coverage (`cargo llvm-cov --workspace --summary-only`)

| File | Region Coverage | Function Coverage | Line Coverage |
| :--- | ---: | ---: | ---: |
| `qserve-core/src/qoq.rs` | 96.10% | 100.00% | 90.59% |
| `qserve-core/src/scheduler.rs` | 86.25% | 100.00% | 89.04% |
| `qserve-core/src/paged_kv.rs` | 85.83% | 87.50% | 87.72% |
| **Workspace Total** *(incl. CLI)* | **80.24%** | **84.09%** | **79.35%** |

## 2. Footprint, Startup & Microbenchmarks

| Metric / Operation | `qserve-rs` (Rust) | Upstream `mit-han-lab/qserve` (Python / CUDA) | Comparison |
| :--- | ---: | ---: | :--- |
| **Standalone Binary Size** | **1.01 MiB** | PyTorch + `omniserve_backend` CUDA | Portable CPU/Edge binary |
| **Cold Startup (5-run median)** | **5.27 ms** | **23.61 ms** (`python3 -c ""` bare stdlib) | **4.48× faster** cold start |
| `W4A8Linear::forward(256x256)` | **0.0283 ms** | Requires CUDA `qgemm_w4a8` | Pure-Rust `i4 * i8 -> i32` |
| `Kv4CacheBlock` quant+dequant (1000x, d=128) | **0.4892 ms** | Requires CUDA KV4 kernel | **4×** KV memory reduction |
| `Scheduler(16 seqs x 32 steps)` | **0.0101 ms** | Python `BlockSpaceManager` loop | Sub-microsecond per step |

Reproduce locally:
```bash
cargo run --release -p qserve-core --example bench
cargo llvm-cov --workspace --summary-only
```
