//! Error types for `qserve-core`.

use thiserror::Error;

/// Errors returned by QServe quantization, KV cache allocation, and scheduling.
#[derive(Debug, Error)]
pub enum QServeError {
    /// Invalid tensor shape, group size, or bit-packing alignment.
    #[error("dimension or alignment error: {0}")]
    Dimension(String),

    /// KV block pool exhausted or invalid block lifecycle transition.
    #[error("block allocator error: {0}")]
    OutOfBlocks(String),
}
