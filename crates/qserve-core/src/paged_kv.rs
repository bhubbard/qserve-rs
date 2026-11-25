//! Paged KV-cache physical block allocator and LServe streaming block manager.

use std::collections::HashMap;

use crate::QServeError;

/// A physical KV-cache page holding `block_size` token slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalTokenBlock {
    /// Zero-based physical block identifier.
    pub block_number: usize,
    /// Number of token slots per block.
    pub block_size: usize,
    /// Active reference count (for copy-on-write / prefix sharing).
    pub ref_count: usize,
}

/// Reference-counted pool of physical KV-cache blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    block_size: usize,
    total_blocks: usize,
    free_blocks: Vec<PhysicalTokenBlock>,
}

impl BlockAllocator {
    /// Creates an allocator with `num_blocks` free blocks of size `block_size`.
    #[must_use]
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let free_blocks = (0..num_blocks)
            .rev()
            .map(|block_number| PhysicalTokenBlock {
                block_number,
                block_size,
                ref_count: 0,
            })
            .collect();
        Self {
            block_size,
            total_blocks: num_blocks,
            free_blocks,
        }
    }

    /// Allocates one physical block with `ref_count = 1`.
    ///
    /// # Errors
    /// Returns [`QServeError::OutOfBlocks`] if no free blocks remain.
    pub fn allocate(&mut self) -> Result<PhysicalTokenBlock, QServeError> {
        let mut block = self
            .free_blocks
            .pop()
            .ok_or_else(|| QServeError::OutOfBlocks("no free KV blocks available".into()))?;
        block.ref_count = 1;
        Ok(block)
    }

    /// Decrements a block's reference count and returns it to the pool at 0.
    ///
    /// # Errors
    /// Returns [`QServeError::OutOfBlocks`] on double-free.
    pub fn free(&mut self, mut block: PhysicalTokenBlock) -> Result<(), QServeError> {
        if block.ref_count == 0 {
            return Err(QServeError::OutOfBlocks(format!(
                "double free of block {}",
                block.block_number
            )));
        }
        block.ref_count -= 1;
        if block.ref_count == 0 {
            self.free_blocks.push(block);
        }
        Ok(())
    }

    /// Returns the number of currently free blocks.
    #[must_use]
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Returns the total block capacity.
    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Returns tokens per block.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Allocation feasibility status for an incoming sequence group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStatus {
    /// Enough free blocks are available right now.
    Ok,
    /// Not enough free blocks right now, but fits within total GPU capacity.
    Later,
    /// Sequence requires more blocks than total GPU capacity.
    Never,
}

/// Manages logical-to-physical KV block tables for active sequences,
/// including optional LServe streaming `(sink_blocks, local_blocks)` eviction.
#[derive(Debug)]
pub struct BlockSpaceManager {
    allocator: BlockAllocator,
    block_tables: HashMap<u64, Vec<PhysicalTokenBlock>>,
    streaming_window: Option<(usize, usize)>,
}

impl BlockSpaceManager {
    /// Creates a [`BlockSpaceManager`] with `num_gpu_blocks` pages.
    #[must_use]
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        streaming_window: Option<(usize, usize)>,
    ) -> Self {
        Self {
            allocator: BlockAllocator::new(block_size, num_gpu_blocks),
            block_tables: HashMap::new(),
            streaming_window,
        }
    }

    /// Checks whether a sequence of `num_tokens` can be allocated.
    #[must_use]
    pub fn can_allocate(&self, num_tokens: usize) -> AllocStatus {
        let needed = num_tokens.div_ceil(self.allocator.block_size()).max(1);
        let effective_needed = match self.streaming_window {
            Some((sink, local)) => needed.min(sink + local),
            None => needed,
        };
        if effective_needed > self.allocator.total_blocks() {
            AllocStatus::Never
        } else if effective_needed > self.allocator.num_free_blocks() {
            AllocStatus::Later
        } else {
            AllocStatus::Ok
        }
    }

    /// Allocates initial KV blocks for sequence `seq_id`.
    ///
    /// # Errors
    /// Returns [`QServeError::OutOfBlocks`] if insufficient blocks are free.
    pub fn allocate(&mut self, seq_id: u64, num_tokens: usize) -> Result<(), QServeError> {
        let needed = num_tokens.div_ceil(self.allocator.block_size()).max(1);
        let mut table = Vec::with_capacity(needed);
        for _ in 0..needed {
            table.push(self.allocator.allocate()?);
        }
        self.block_tables.insert(seq_id, table);
        Ok(())
    }

    /// Appends one new block if `total_tokens` crosses a block boundary, evicting
    /// middle blocks when LServe streaming `(sink, local)` is enabled.
    ///
    /// # Errors
    /// Returns [`QServeError::OutOfBlocks`] if allocation or eviction fails.
    pub fn append_slot(&mut self, seq_id: u64, total_tokens: usize) -> Result<(), QServeError> {
        let needed = total_tokens.div_ceil(self.allocator.block_size()).max(1);
        let table = self
            .block_tables
            .get_mut(&seq_id)
            .ok_or_else(|| QServeError::OutOfBlocks(format!("unknown seq_id {seq_id}")))?;

        if needed > table.len() {
            if let Some((sink, local)) = self.streaming_window {
                let max_kept = sink + local;
                if table.len() >= max_kept && table.len() > sink {
                    let evicted = table.remove(sink);
                    self.allocator.free(evicted)?;
                }
            }
            table.push(self.allocator.allocate()?);
        }
        Ok(())
    }

    /// Frees all physical blocks held by `seq_id`.
    ///
    /// # Errors
    /// Returns [`QServeError::OutOfBlocks`] on allocator error.
    pub fn free(&mut self, seq_id: u64) -> Result<(), QServeError> {
        if let Some(table) = self.block_tables.remove(&seq_id) {
            for block in table {
                self.allocator.free(block)?;
            }
        }
        Ok(())
    }

    /// Returns the number of currently free physical blocks.
    #[must_use]
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_window_evicts_middle_blocks_and_preserves_sink() {
        let mut mgr = BlockSpaceManager::new(16, 8, Some((1, 2)));
        mgr.allocate(1, 16).expect("1 block");
        mgr.append_slot(1, 17).expect("2 blocks");
        mgr.append_slot(1, 33).expect("3 blocks (at cap)");
        assert_eq!(mgr.num_free_blocks(), 5);

        // 4th block evicts index 1 (first non-sink block), keeping total at 3.
        mgr.append_slot(1, 49).expect("streaming eviction");
        assert_eq!(mgr.num_free_blocks(), 5);
        mgr.free(1).expect("free all");
        assert_eq!(mgr.num_free_blocks(), 8);
    }
}
