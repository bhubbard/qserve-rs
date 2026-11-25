//! Release benchmark for `qserve-core` W4A8 forward, KV4 quantization, and paged scheduler.

use std::hint::black_box;
use std::time::Instant;

use qserve_core::{
    BlockSpaceManager, Kv4CacheBlock, Scheduler, SequenceGroup, SequenceStatus, W4A8Linear,
};

fn bench<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{name}: median={:.4} ms (min={:.4} ms)",
        times[times.len() / 2],
        times[0]
    );
}

fn main() {
    let weights: Vec<f32> = (0..(256 * 256))
        .map(|i| (i as f32 * 0.17).sin() * 0.4)
        .collect();
    let act: Vec<f32> = (0..256).map(|i| (i as f32 * 0.11).cos()).collect();
    let layer = W4A8Linear::from_float(&weights, 256, 256, 32, None).expect("w4a8");

    bench("W4A8Linear::forward(256x256)", 3, 15, || {
        let _ = black_box(layer.forward(black_box(&act)));
    });

    let kv: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
    bench(
        "Kv4CacheBlock::quantize+dequantize(128, 1000x)",
        3,
        15,
        || {
            for _ in 0..1000 {
                let q = Kv4CacheBlock::quantize(black_box(&kv)).expect("kv4");
                black_box(q.dequantize());
            }
        },
    );

    bench("Scheduler(16 seqs x 32 steps)", 3, 15, || {
        let bm = BlockSpaceManager::new(16, 128, Some((1, 8)));
        let mut sched = Scheduler::new(bm, 16);
        for id in 1..=16 {
            sched.add_request(SequenceGroup {
                request_id: id,
                prompt_len: 32,
                max_new_tokens: 32,
                generated_tokens: 0,
                status: SequenceStatus::Waiting,
            });
        }
        for _ in 0..32 {
            let _ = black_box(sched.step());
        }
    });
}
