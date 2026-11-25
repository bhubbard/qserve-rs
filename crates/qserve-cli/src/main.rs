//! CLI for QServe W4A8KV4 quantization and continuous-batching KV simulation.

use anyhow::Result;
use clap::{Parser, Subcommand};
use qserve_core::{
    BlockSpaceManager, Kv4CacheBlock, Scheduler, SequenceGroup, SequenceStatus, W4A8Linear,
};

#[derive(Parser)]
#[command(
    name = "qserve",
    about = "Pure-Rust QServe W4A8KV4 quantization & serving CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quantize a synthetic layer to W4A8 and compare against FP32.
    QuantDemo {
        #[arg(long, default_value_t = 64)]
        in_features: usize,
        #[arg(long, default_value_t = 32)]
        out_features: usize,
    },
    /// Quantize a KV vector to 4-bit (KV4) and report compression & error.
    Kv4Demo {
        #[arg(long, default_value_t = 128)]
        head_dim: usize,
    },
    /// Simulate continuous-batching paged KV scheduling.
    SchedulerSim,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::QuantDemo {
            in_features,
            out_features,
        } => {
            let weights: Vec<f32> = (0..(out_features * in_features))
                .map(|i| (i as f32 * 0.23).sin() * 0.4)
                .collect();
            let act: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.11).cos()).collect();
            let layer = W4A8Linear::from_float(&weights, out_features, in_features, 16, None)?;
            let out = layer.forward(&act)?;
            println!(
                "W4A8 forward complete: {} -> {} ({} packed bytes vs {} FP32 bytes)",
                in_features,
                out.len(),
                layer.qweight.len(),
                weights.len() * 4
            );
        }
        Commands::Kv4Demo { head_dim } => {
            let k: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.17).sin()).collect();
            let kv4 = Kv4CacheBlock::quantize(&k)?;
            let recon = kv4.dequantize();
            let max_err = k
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "KV4 packed {} bytes (vs {} FP32 bytes) | max_err={max_err:.5}",
                kv4.packed_u4.len(),
                head_dim * 4
            );
        }
        Commands::SchedulerSim => {
            let bm = BlockSpaceManager::new(16, 32, Some((1, 4)));
            let mut sched = Scheduler::new(bm, 4);
            for id in 1..=4 {
                sched.add_request(SequenceGroup {
                    request_id: id,
                    prompt_len: 24,
                    max_new_tokens: 4,
                    generated_tokens: 0,
                    status: SequenceStatus::Waiting,
                });
            }
            let mut steps = 0;
            let mut finished = 0;
            while finished < 4 && steps < 20 {
                finished += sched.step()?.len();
                steps += 1;
            }
            println!("Completed {finished} requests in {steps} continuous-batching steps");
        }
    }
    Ok(())
}
