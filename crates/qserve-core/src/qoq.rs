//! QoQ (`W4A8KV4`) progressive quantization and integer-accumulated GEMM.

use crate::QServeError;

/// Packs two signed 4-bit integers (`-8..=7`) into a single byte.
#[must_use]
pub fn pack_i4_pair(lo: i8, hi: i8) -> u8 {
    let l = (lo as u8) & 0x0F;
    let h = (hi as u8) & 0x0F;
    (h << 4) | l
}

/// Unpacks a byte into two signed 4-bit integers (`-8..=7`).
#[must_use]
pub fn unpack_i4_pair(byte: u8) -> (i8, i8) {
    let lo = ((byte & 0x0F) as i8) << 4 >> 4;
    let hi = (((byte >> 4) & 0x0F) as i8) << 4 >> 4;
    (lo, hi)
}

/// Dynamically quantizes a float activation row to symmetric `i8` with per-token scale.
#[must_use]
pub fn quantize_activations_i8(row: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = (max_abs / 127.0).max(1e-8);
    let q = row
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Progressive W4A8 linear layer (`W` quantized to 4-bit with per-group `i8`
/// intermediate scales + per-channel `f32` scales).
#[derive(Debug, Clone)]
pub struct W4A8Linear {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Quantization group size along `in_features`.
    pub group_size: usize,
    /// Packed 4-bit weights (`out_features * (in_features / 2)`).
    pub qweight: Vec<u8>,
    /// Per-channel outer scale (`out_features`).
    pub s1_scales: Vec<f32>,
    /// Per-group inner integer scale (`out_features * (in_features / group_size)`).
    pub s2_scales: Vec<i8>,
    /// Optional bias (`out_features`).
    pub bias: Option<Vec<f32>>,
}

impl W4A8Linear {
    /// Quantizes an `out_features x in_features` row-major float weight matrix
    /// into QServe's progressive W4A8 format.
    ///
    /// # Errors
    /// Returns [`QServeError::Dimension`] if dimensions are incompatible with `group_size`.
    pub fn from_float(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, QServeError> {
        if weights.len() != out_features * in_features {
            return Err(QServeError::Dimension(
                "weight buffer length does not match out_features * in_features".into(),
            ));
        }
        if group_size == 0
            || !in_features.is_multiple_of(group_size)
            || !in_features.is_multiple_of(2)
        {
            return Err(QServeError::Dimension(
                "in_features must be even and divisible by group_size".into(),
            ));
        }
        if let Some(ref b) = bias {
            if b.len() != out_features {
                return Err(QServeError::Dimension("bias length mismatch".into()));
            }
        }

        let num_groups = in_features / group_size;
        let mut qweight = vec![0u8; out_features * (in_features / 2)];
        let mut s1_scales = vec![0.0f32; out_features];
        let mut s2_scales = vec![0i8; out_features * num_groups];

        for r in 0..out_features {
            let row = &weights[r * in_features..(r + 1) * in_features];
            let ch_max = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            // Progressive 2-level scaling: s1_scale * s2_scale(i8) * q4(-7..7)
            let s1 = (ch_max / (127.0 * 7.0)).max(1e-8);
            s1_scales[r] = s1;

            for g in 0..num_groups {
                let grp = &row[g * group_size..(g + 1) * group_size];
                let grp_max = grp.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
                let s2 = ((grp_max / (7.0 * s1)).round().clamp(1.0, 127.0)) as i8;
                s2_scales[r * num_groups + g] = s2;
                let eff_scale = s1 * f32::from(s2);

                for pair in 0..(group_size / 2) {
                    let c0 = g * group_size + pair * 2;
                    let q0 = (grp[pair * 2] / eff_scale).round().clamp(-7.0, 7.0) as i8;
                    let q1 = (grp[pair * 2 + 1] / eff_scale).round().clamp(-7.0, 7.0) as i8;
                    qweight[r * (in_features / 2) + c0 / 2] = pack_i4_pair(q0, q1);
                }
            }
        }

        Ok(Self {
            in_features,
            out_features,
            group_size,
            qweight,
            s1_scales,
            s2_scales,
            bias,
        })
    }

    /// Executes W4A8 forward pass over a batch of `f32` activation rows,
    /// dynamically quantizing each token to `i8` and accumulating in `i32`.
    ///
    /// # Errors
    /// Returns [`QServeError::Dimension`] if `activations.len() % in_features != 0`.
    pub fn forward(&self, activations: &[f32]) -> Result<Vec<f32>, QServeError> {
        if !activations.len().is_multiple_of(self.in_features) {
            return Err(QServeError::Dimension(
                "activations length must be a multiple of in_features".into(),
            ));
        }
        let batch = activations.len() / self.in_features;
        let num_groups = self.in_features / self.group_size;
        let mut out = vec![0.0f32; batch * self.out_features];

        for b in 0..batch {
            let act_row = &activations[b * self.in_features..(b + 1) * self.in_features];
            let (q_act, act_scale) = quantize_activations_i8(act_row);

            for r in 0..self.out_features {
                let mut row_sum = 0.0f32;
                for g in 0..num_groups {
                    let mut grp_acc: i32 = 0;
                    for pair in 0..(self.group_size / 2) {
                        let c0 = g * self.group_size + pair * 2;
                        let packed = self.qweight[r * (self.in_features / 2) + c0 / 2];
                        let (w0, w1) = unpack_i4_pair(packed);
                        grp_acc += i32::from(w0) * i32::from(q_act[c0])
                            + i32::from(w1) * i32::from(q_act[c0 + 1]);
                    }
                    let s2 = f32::from(self.s2_scales[r * num_groups + g]);
                    row_sum += (grp_acc as f32) * s2;
                }
                let mut val = row_sum * self.s1_scales[r] * act_scale;
                if let Some(ref bias) = self.bias {
                    val += bias[r];
                }
                out[b * self.out_features + r] = val;
            }
        }

        Ok(out)
    }
}

/// Asymmetric 4-bit quantized KV cache vector (KV4) for a single head/token.
#[derive(Debug, Clone)]
pub struct Kv4CacheBlock {
    /// Head dimension (must be even).
    pub head_dim: usize,
    /// Packed 4-bit unsigned values (`0..=15`, two per byte).
    pub packed_u4: Vec<u8>,
    /// Asymmetric quantization scale.
    pub scale: f32,
    /// Asymmetric quantization zero-point.
    pub zero_point: f32,
}

impl Kv4CacheBlock {
    /// Quantizes a float key/value head vector into 4-bit unsigned representation.
    ///
    /// # Errors
    /// Returns [`QServeError::Dimension`] if `values.is_empty()` or odd-length.
    pub fn quantize(values: &[f32]) -> Result<Self, QServeError> {
        if values.is_empty() || !values.len().is_multiple_of(2) {
            return Err(QServeError::Dimension(
                "KV4 vector must be non-empty and even-length".into(),
            ));
        }
        let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((max_v - min_v) / 15.0).max(1e-8);
        let zero_point = min_v;

        let mut packed_u4 = Vec::with_capacity(values.len() / 2);
        for pair in values.as_chunks::<2>().0 {
            let q0 = ((pair[0] - zero_point) / scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((pair[1] - zero_point) / scale).round().clamp(0.0, 15.0) as u8;
            packed_u4.push((q1 << 4) | q0);
        }

        Ok(Self {
            head_dim: values.len(),
            packed_u4,
            scale,
            zero_point,
        })
    }

    /// Dequantizes the 4-bit KV vector back to `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.head_dim);
        for &byte in &self.packed_u4 {
            let q0 = f32::from(byte & 0x0F);
            let q1 = f32::from((byte >> 4) & 0x0F);
            out.push(self.zero_point + q0 * self.scale);
            out.push(self.zero_point + q1 * self.scale);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w4a8_forward_matches_float_reference() {
        let in_f = 16;
        let out_f = 8;
        let w: Vec<f32> = (0..(out_f * in_f))
            .map(|i| (i as f32 * 0.29).sin() * 0.5)
            .collect();
        let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.17).cos()).collect();

        let layer = W4A8Linear::from_float(&w, out_f, in_f, 8, None).expect("valid W4A8");
        let y_q = layer.forward(&x).expect("forward succeeds");

        let mut y_ref = vec![0.0f32; out_f];
        for r in 0..out_f {
            for c in 0..in_f {
                y_ref[r] += w[r * in_f + c] * x[c];
            }
        }
        let max_diff = y_q
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.35, "W4A8 max diff too high: {max_diff}");
    }

    #[test]
    fn kv4_round_trip_compresses_4x() {
        let k: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let kv4 = Kv4CacheBlock::quantize(&k).expect("kv4 quantize");
        assert_eq!(kv4.packed_u4.len(), 32); // 4-bit = 0.5 bytes/elem vs 4 bytes/f32
        let recon = kv4.dequantize();
        let max_err = k
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "KV4 error too high: {max_err}");
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn verify_i4_pack_unpack_roundtrip() {
        let lo: i8 = kani::any();
        let hi: i8 = kani::any();
        kani::assume((-8..=7).contains(&lo));
        kani::assume((-8..=7).contains(&hi));
        let packed = pack_i4_pair(lo, hi);
        let (u_lo, u_hi) = unpack_i4_pair(packed);
        assert_eq!(u_lo, lo);
        assert_eq!(u_hi, hi);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_quantize_activations_i8_never_panics() {
        let vals: [f32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for &v in &vals {
            kani::assume(v.is_finite());
            kani::assume(v.abs() < 1e6);
        }
        let (q, scale) = quantize_activations_i8(&vals);
        assert_eq!(q.len(), 4);
        assert!(scale > 0.0 && scale.is_finite());
    }
}
