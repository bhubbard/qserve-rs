//! Pure-Rust port of MIT Han Lab's QServe (`W4A8KV4` QoQ quantization and serving engine).
//!
//! Provides:
//! - [`qoq`]: Progressive W4A8 group quantization (`W4A8Linear`) and asymmetric
//!   4-bit KV cache quantization (`Kv4CacheBlock`).
//! - [`paged_kv`]: PagedAttention physical block allocator and LServe
//!   sink+local streaming `BlockSpaceManager`.
//! - [`scheduler`]: Continuous-batching FCFS request scheduler with preemption.

pub mod error;
pub mod paged_kv;
pub mod qoq;
pub mod scheduler;

pub use error::QServeError;
pub use paged_kv::{AllocStatus, BlockAllocator, BlockSpaceManager, PhysicalTokenBlock};
pub use qoq::{quantize_activations_i8, Kv4CacheBlock, W4A8Linear};
pub use scheduler::{Scheduler, SequenceGroup, SequenceStatus};
