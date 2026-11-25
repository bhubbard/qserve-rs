# qserve-rs

Pure-Rust port of [mit-han-lab/qserve](https://github.com/mit-han-lab/qserve) — W4A8KV4 (QoQ) quantization, 4-bit paged KV-cache management (with LServe sink+local streaming eviction), and continuous-batching request scheduling.

Zero Python or CUDA runtime required.

## Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `qserve-core` | `W4A8Linear` progressive 4-bit weight + dynamic `i8` activation GEMM, `Kv4CacheBlock` asymmetric 4-bit KV quantization, `BlockSpaceManager`, and `Scheduler` |
| `qserve-cli` | `qserve` CLI (`quant-demo`, `kv4-demo`, `scheduler-sim`) |

## Quickstart

```bash
cargo test --workspace
cargo run -p qserve-cli -- quant-demo
cargo run -p qserve-cli -- kv4-demo
cargo run -p qserve-cli -- scheduler-sim
```
