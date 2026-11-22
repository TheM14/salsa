//! JOC (EMDF ID14) payload parser.
//!
//! Output matrices are the dequantized `dq` values consumed by the DSP kernel.

use crate::bit::BitReader;
use crate::emdf::{self, EmdfContainer};
use crate::error::{Eac3JocError, Result};
use crate::huffman_tables::*;

pub const MAX_OBJECTS: usize = 15;
pub const MAX_DPOINTS: usize = 2;
pub const CORE_CHANNELS: usize = 5;
pub const MAX_CORE_CHANNELS: usize = 7;
pub const MAX_PARAMETER_BANDS: usize = 23;
pub const FRAME_SAMPLES: usize = 1536;
/// Timeslots an `offset_ts` may address.
pub const TIMESLOTS: usize = 24;

/// How completely an ID14 payload could be parsed.
///
/// Only a [`Complete`](Self::Complete) payload is rendered. The other states
/// describe a payload that is identifiable as this frame's JOC metadata but
/// cannot be decoded in full: the header fields (`dmx_config_idx`, clipgain,
/// `seq_count`) and the object parameters read before the payload ran out are
/// kept so the kernel can hold the previous matrices instead of dropping the
/// frame. Treating those payloads as fatal would make one damaged frame stop
/// the whole stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum JocPayloadStatus {
    /// Every declared object parameter was read.
    #[default]
    Complete,
    /// `joc_data` ended before the last object parameter.
    Truncated,
    /// `slope_idx == 1` with an `offset_ts` outside `0..TIMESLOTS` or not
    /// strictly increasing within one object.
    BadOffsetTs,
}

impl JocPayloadStatus {
    pub fn is_complete(self) -> bool {
        matches!(self, Self::Complete)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JocObject {
    pub present: bool,
    pub num_bands_idx: u8,
    pub n_bands: usize,
    pub sparse: bool,
    pub quant_idx: u8,
    pub slope_idx: u8,
    pub n_dpoints: usize,
    pub offset_ts: [u8; MAX_DPOINTS],
    /// `dq[data_point][core_channel][parameter_band]`.
    pub dq: [[[f64; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
}

impl Default for JocObject {
    fn default() -> Self {
        Self {
            present: false,
            num_bands_idx: 0,
            n_bands: 1,
            sparse: false,
            quant_idx: 0,
            slope_idx: 0,
            n_dpoints: 1,
            offset_ts: [0; MAX_DPOINTS],
            dq: [[[0.0; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JocParams {
    pub dmx_config_idx: u8,
    pub num_objects_bits: u8,
    pub ext_config_idx: u8,
    pub n_channels: u8,
    pub n_objects: u8,
    pub objects_clamped: bool,
    pub clipgain_x_bits: u8,
    pub clipgain_y_bits: u8,
    pub seq_count: u16,
    pub clipgain: f64,
    pub objects: [JocObject; MAX_OBJECTS],
    pub present_mask: u16,
    pub data_end_bits: u32,
    pub trailing_bits: u32,
    /// Whether this payload could be decoded in full.
    pub status: JocPayloadStatus,
}

impl JocParams {
    /// Phase-shift (sigma(k)) switch for `dmx_config_idx` 3 and 4.
    pub fn phase_shift(&self) -> bool {
        matches!(self.dmx_config_idx, 3 | 4)
    }
}

pub fn num_channels_for_config(config: u32) -> Option<u8> {
    match config {
        0 | 3 => Some(5),
        1 | 2 | 4 => Some(7),
        _ => None,
    }
}

pub fn num_bands_for_index(index: u32) -> Option<usize> {
    match index {
        0 => Some(1),
        1 => Some(3),
        2 => Some(5),
        3 => Some(7),
        4 => Some(9),
        5 => Some(12),
        6 => Some(15),
        7 => Some(23),
        _ => None,
    }
}

fn floored_mod(value: i64, modulus: i64) -> i64 {
    let remainder = value % modulus;
    if remainder < 0 {
        remainder + modulus
    } else {
        remainder
    }
}

#[derive(Clone, Copy)]
enum SymbolKind {
    Mtx,
    Idx,
    Vec,
}

fn select_tree(quant_idx: u8, kind: SymbolKind, n_channels: usize) -> &'static [[i16; 2]] {
    match kind {
        SymbolKind::Idx => {
            if n_channels == 5 {
                &JOC_HUFF_CODE_5CH_POS_INDEX_SPARSE
            } else {
                &JOC_HUFF_CODE_7CH_POS_INDEX_SPARSE
            }
        }
        SymbolKind::Vec => {
            if quant_idx == 0 {
                &JOC_HUFF_CODE_COARSE_COEFF_SPARSE
            } else {
                &JOC_HUFF_CODE_FINE_COEFF_SPARSE
            }
        }
        SymbolKind::Mtx => {
            if quant_idx == 0 {
                &JOC_HUFF_CODE_COARSE_GENERIC
            } else {
                &JOC_HUFF_CODE_FINE_GENERIC
            }
        }
    }
}

fn huff_decode(tree: &[[i16; 2]], reader: &mut BitReader<'_>) -> Result<i16> {
    let mut node = 0i32;
    let mut steps = 0usize;
    while node >= 0 {
        if node as usize >= tree.len() || steps > tree.len() {
            return Err(Eac3JocError::JocSyntax(
                "Huffman tree walk left the valid node range".into(),
            ));
        }
        steps += 1;
        let bit = reader.read(1) as usize;
        if reader.is_failed() {
            return Err(Eac3JocError::BitstreamTruncated);
        }
        node = i32::from(tree[node as usize][bit]);
    }
    Ok(-node as i16 - 1)
}

/// [`huff_decode`] that reports an exhausted payload as `None` instead of an
/// error, so a frame that ends inside `joc_data` keeps the parameters read so
/// far. A tree walk that leaves the valid node range stays an error.
fn huff_decode_optional(tree: &[[i16; 2]], reader: &mut BitReader<'_>) -> Result<Option<i16>> {
    match huff_decode(tree, reader) {
        Ok(value) => Ok(Some(value)),
        Err(Eac3JocError::BitstreamTruncated) => Ok(None),
        Err(error) => Err(error),
    }
}

/// `offset_ts` of one object addresses timeslots and must increase. The field
/// is coded in five bits, so the values above [`TIMESLOTS`] are invalid.
fn offsets_invalid(object: &JocObject) -> bool {
    let offsets = &object.offset_ts[..object.n_dpoints];
    if offsets
        .iter()
        .any(|offset| usize::from(*offset) >= TIMESLOTS)
    {
        return true;
    }
    offsets.windows(2).any(|pair| pair[1] <= pair[0])
}

/// `clip_gain = 2^(x-4) * (1 + y/32)`.
///
/// The range is 0.0625..15.75 and `x = 4` reduces the field to `1 + y/32`.
/// The exponent scales the whole mantissa; `powi` builds the power of two
/// exactly over the whole legal range.
fn clipgain(x_bits: u8, y_bits: u8) -> f64 {
    (2.0f64).powi(i32::from(x_bits) - 4) * (1.0 + f64::from(y_bits) / 32.0)
}

#[allow(clippy::needless_range_loop)]
fn reconstruct_dense(
    mtx: &[[[i16; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
    data_point: usize,
    n_bands: usize,
    n_channels: usize,
    nquant: i64,
    offset: i64,
    q: &mut [[[i64; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
) {
    for channel in 0..n_channels {
        q[data_point][channel][0] =
            floored_mod(offset + i64::from(mtx[data_point][channel][0]), nquant);
        for band in 1..n_bands {
            q[data_point][channel][band] = floored_mod(
                q[data_point][channel][band - 1] + i64::from(mtx[data_point][channel][band]),
                nquant,
            );
        }
    }
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn reconstruct_sparse(
    idx: &[[u8; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
    vec: &[[i16; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
    data_point: usize,
    n_bands: usize,
    n_channels: usize,
    nquant: i64,
    offset: i64,
    q: &mut [[[i64; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
) -> Result<()> {
    if n_channels != 5 && n_channels != 7 {
        return Err(Eac3JocError::UnsupportedVariant(format!(
            "sparse JOC requires 5 or 7 core channels, got {n_channels}"
        )));
    }
    let initial_channel = usize::from(idx[data_point][0]);
    if initial_channel >= n_channels {
        return Err(Eac3JocError::JocSyntax(format!(
            "sparse JOC initial channel {initial_channel} out of range for {n_channels} channels"
        )));
    }
    for channel in 0..n_channels {
        for band in 0..n_bands {
            q[data_point][channel][band] = nquant / 2;
        }
    }
    let mut active = initial_channel;
    let mut coefficient = offset;
    for band in 0..n_bands {
        if band != 0 {
            active = floored_mod(
                active as i64 + i64::from(idx[data_point][band]),
                n_channels as i64,
            ) as usize;
        }
        coefficient = floored_mod(coefficient + i64::from(vec[data_point][band]), nquant);
        q[data_point][active][band] = coefficient;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ObjectSymbols {
    n_bands: usize,
    n_dpoints: usize,
    sparse: bool,
    quant_idx: u8,
    mtx: [[[i16; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
    idx: [[u8; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
    vec: [[i16; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
}

impl Default for ObjectSymbols {
    fn default() -> Self {
        Self {
            n_bands: 1,
            n_dpoints: 1,
            sparse: false,
            quant_idx: 0,
            mtx: [[[0; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS],
            idx: [[0; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
            vec: [[0; MAX_PARAMETER_BANDS]; MAX_DPOINTS],
        }
    }
}

/// Parse an ID14 payload.
pub fn parse_id14(payload: &[u8]) -> Result<JocParams> {
    let mut reader = BitReader::new(payload);
    let dmx_config_idx = reader.read(3) as u8;
    let num_objects_bits = reader.read(6) as u8;
    let ext_config_idx = reader.read(3) as u8;
    // A payload that ends inside the fixed header cannot be reported as a
    // damaged frame: the reader substitutes zeros for the fields it could not
    // read, so the frame would carry a downmix configuration and an object count
    // the bitstream never declared.
    if reader.is_failed() {
        return Err(Eac3JocError::BitstreamTruncated);
    }

    let declared_objects = u32::from(num_objects_bits) + 1;
    let n_channels = num_channels_for_config(u32::from(dmx_config_idx)).ok_or_else(|| {
        Eac3JocError::JocSyntax(format!(
            "unknown JOC downmix configuration {dmx_config_idx}"
        ))
    })?;
    let objects_clamped = declared_objects > MAX_OBJECTS as u32;
    let n_objects = declared_objects.min(MAX_OBJECTS as u32) as u8;

    let clipgain_x_bits = reader.read(3) as u8;
    let clipgain_y_bits = reader.read(5) as u8;
    let seq_count = reader.read(10) as u16;
    if reader.is_failed() {
        return Err(Eac3JocError::BitstreamTruncated);
    }
    let clipgain = clipgain(clipgain_x_bits, clipgain_y_bits);

    // The header is intact from here on. An object parameter that cannot be
    // read turns the frame into a damaged one instead of failing the stream.
    let mut truncated = false;
    let mut status = JocPayloadStatus::Complete;
    let mut objects = [JocObject::default(); MAX_OBJECTS];
    'headers: for object in objects.iter_mut().take(n_objects as usize) {
        object.present = reader.read(1) != 0;
        if reader.is_failed() {
            truncated = true;
            break 'headers;
        }
        if !object.present {
            continue;
        }
        object.num_bands_idx = reader.read(3) as u8;
        if reader.is_failed() {
            truncated = true;
            break 'headers;
        }
        let Some(n_bands) = num_bands_for_index(u32::from(object.num_bands_idx)) else {
            // Every three-bit index has a table entry, so this is a guard rather
            // than a reachable state; the symbol count would be unknown.
            return Err(Eac3JocError::JocSyntax(format!(
                "unknown JOC num_bands index {}",
                object.num_bands_idx
            )));
        };
        object.n_bands = n_bands;
        object.sparse = reader.read(1) != 0;
        object.quant_idx = reader.read(1) as u8;
        object.slope_idx = reader.read(1) as u8;
        object.n_dpoints = reader.read(1) as usize + 1;
        if reader.is_failed() {
            truncated = true;
            break 'headers;
        }
        if object.slope_idx == 1 {
            for data_point in 0..object.n_dpoints {
                object.offset_ts[data_point] = reader.read(5) as u8;
            }
            if reader.is_failed() {
                truncated = true;
                break 'headers;
            }
            if status.is_complete() && offsets_invalid(object) {
                status = JocPayloadStatus::BadOffsetTs;
            }
        }
    }
    if truncated {
        status = JocPayloadStatus::Truncated;
    }
    // A truncated payload leaves `joc_data` unread, so the object matrices are
    // unknown even for the objects whose header was parsed.
    let mut data_unread = truncated;

    let mut symbols = [ObjectSymbols::default(); MAX_OBJECTS];
    if !data_unread {
        'symbols: for (object_index, object) in objects.iter().take(n_objects as usize).enumerate()
        {
            if !object.present {
                continue;
            }
            let symbol = &mut symbols[object_index];
            symbol.n_bands = object.n_bands;
            symbol.n_dpoints = object.n_dpoints;
            symbol.sparse = object.sparse;
            symbol.quant_idx = object.quant_idx;
            for data_point in 0..object.n_dpoints {
                if object.sparse {
                    let idx_tree =
                        select_tree(object.quant_idx, SymbolKind::Idx, usize::from(n_channels));
                    symbol.idx[data_point][0] = reader.read(3) as u8;
                    if reader.is_failed() {
                        data_unread = true;
                        break 'symbols;
                    }
                    for slot in symbol.idx[data_point]
                        .iter_mut()
                        .take(object.n_bands)
                        .skip(1)
                    {
                        let Some(value) = huff_decode_optional(idx_tree, &mut reader)? else {
                            data_unread = true;
                            break 'symbols;
                        };
                        *slot = value as u8;
                    }
                    let vec_tree =
                        select_tree(object.quant_idx, SymbolKind::Vec, usize::from(n_channels));
                    for slot in symbol.vec[data_point].iter_mut().take(object.n_bands) {
                        let Some(value) = huff_decode_optional(vec_tree, &mut reader)? else {
                            data_unread = true;
                            break 'symbols;
                        };
                        *slot = value;
                    }
                } else {
                    let mtx_tree =
                        select_tree(object.quant_idx, SymbolKind::Mtx, usize::from(n_channels));
                    for row in symbol.mtx[data_point]
                        .iter_mut()
                        .take(usize::from(n_channels))
                    {
                        for slot in row.iter_mut().take(object.n_bands) {
                            let Some(value) = huff_decode_optional(mtx_tree, &mut reader)? else {
                                data_unread = true;
                                break 'symbols;
                            };
                            *slot = value;
                        }
                    }
                }
            }
        }
    }
    if data_unread {
        status = JocPayloadStatus::Truncated;
    }

    let data_end_bits = reader.position() as u32;
    let trailing_bits = (payload.len() * 8).saturating_sub(reader.position()) as u32;

    let mut present_mask = 0u16;
    for (object_index, object) in objects.iter_mut().take(n_objects as usize).enumerate() {
        if !object.present {
            continue;
        }
        present_mask |= 1u16 << object_index;
        if data_unread {
            // The objects whose header was read are known but their matrices
            // are not; the kernel does not render this frame.
            continue;
        }
        let symbol = &symbols[object_index];
        let nquant: i64 = if object.quant_idx == 0 { 96 } else { 192 };
        let mut q = [[[0i64; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS];
        for data_point in 0..object.n_dpoints {
            if object.sparse {
                let offset: i64 = if object.quant_idx == 0 { 50 } else { 100 };
                reconstruct_sparse(
                    &symbol.idx,
                    &symbol.vec,
                    data_point,
                    object.n_bands,
                    usize::from(n_channels),
                    nquant,
                    offset,
                    &mut q,
                )?;
            } else {
                let offset: i64 = if object.quant_idx == 0 { 48 } else { 96 };
                reconstruct_dense(
                    &symbol.mtx,
                    data_point,
                    object.n_bands,
                    usize::from(n_channels),
                    nquant,
                    offset,
                    &mut q,
                );
            }
        }
        let nquant_half = nquant as f64 / 2.0;
        let denominator = 4096.0 * f64::from(1 + u16::from(object.quant_idx));
        #[allow(clippy::needless_range_loop)]
        for data_point in 0..object.n_dpoints {
            for channel in 0..usize::from(n_channels) {
                for band in 0..object.n_bands {
                    let value = q[data_point][channel][band] as f64 - nquant_half;
                    object.dq[data_point][channel][band] = value * 820.0 / denominator;
                }
            }
        }
    }

    Ok(JocParams {
        dmx_config_idx,
        num_objects_bits,
        ext_config_idx,
        n_channels,
        n_objects,
        objects_clamped,
        clipgain_x_bits,
        clipgain_y_bits,
        seq_count,
        clipgain,
        objects,
        present_mask,
        data_end_bits,
        trailing_bits,
        status,
    })
}

/// Score a container as this frame's JOC metadata: 2 when the ID11 and ID14
/// payloads share a group id and ID14 parses, 1 when ID14 parses, else 0.
///
/// A payload that stops at its object headers cannot be rendered, so it never
/// wins the container choice: another candidate may carry the frame's complete
/// metadata.
pub fn score_container_with_frame(frame: &[u8], container: &EmdfContainer) -> i32 {
    let Some(payload) = container.find(emdf::ID_JOC) else {
        return 0;
    };
    let Some(oamd) = container.find(emdf::ID_OAMD) else {
        return 0;
    };
    let parses = emdf::extract_payload_bytes(frame, payload)
        .ok()
        .and_then(|bytes| parse_id14(&bytes).ok())
        .is_some_and(|params| params.status != JocPayloadStatus::Truncated);
    let associated = oamd.has_group_id && payload.has_group_id && oamd.group_id == payload.group_id;
    if associated && parses {
        2
    } else if parses {
        1
    } else {
        0
    }
}

/// Parse the JOC payload from a syncframe, returning the params and container.
pub fn parse_syncframe(frame: &[u8]) -> Result<(JocParams, EmdfContainer)> {
    let container = emdf::find_joc_emdf(frame, None)?;
    let payload = container
        .find(emdf::ID_JOC)
        .ok_or_else(|| Eac3JocError::EmdfTransport("EMDF container has no ID14 payload".into()))?;
    let bytes = emdf::extract_payload_bytes(frame, payload)?;
    let params = parse_id14(&bytes)?;
    Ok((params, container))
}

/// Parse the JOC payload, or `None` when the syncframe carries no EMDF.
pub fn parse_syncframe_optional(frame: &[u8]) -> Result<Option<(JocParams, EmdfContainer)>> {
    let score = |container: &EmdfContainer| score_container_with_frame(frame, container);
    let container = emdf::find_joc_emdf_optional(frame, Some(&score))?;
    let Some(container) = container else {
        return Ok(None);
    };
    let payload = container
        .find(emdf::ID_JOC)
        .ok_or_else(|| Eac3JocError::EmdfTransport("EMDF container has no ID14 payload".into()))?;
    let bytes = emdf::extract_payload_bytes(frame, payload)?;
    Ok(Some((parse_id14(&bytes)?, container)))
}

/// Check that at most seven padding bits remain and all are zero.
///
/// A payload that stops before its object data has no `joc_data` end position,
/// so it keeps reporting the error the frame status now carries.
pub fn check_id14_padding(payload: &[u8]) -> Result<u32> {
    let params = parse_id14(payload)?;
    match params.status {
        JocPayloadStatus::Complete | JocPayloadStatus::BadOffsetTs => {}
        JocPayloadStatus::Truncated => {
            return Err(Eac3JocError::BitstreamTruncated);
        }
    }
    if params.trailing_bits > 7 {
        return Err(Eac3JocError::JocSyntax(format!(
            "more than 7 bits left after joc_data ({})",
            params.trailing_bits
        )));
    }
    for bit in params.data_end_bits as usize..payload.len() * 8 {
        if (payload[bit >> 3] >> (7 - (bit & 7))) & 1 != 0 {
            return Err(Eac3JocError::JocSyntax(format!(
                "non-zero trailing padding bit at {bit}"
            )));
        }
    }
    Ok(params.trailing_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `clip_gain = 2^(x-4) * (1 + y/32)`.
    fn clipgain_of(x_bits: u8, y_bits: u8) -> f64 {
        clipgain(x_bits, y_bits)
    }

    /// MSB-first bit assembler for the hand-built payloads.
    #[derive(Default)]
    struct Bits {
        data: Vec<u8>,
        bit: usize,
    }

    impl Bits {
        fn push(&mut self, count: u32, value: u32) {
            for index in (0..count).rev() {
                if self.bit.is_multiple_of(8) {
                    self.data.push(0);
                }
                let bit = (value >> index) & 1;
                let last = self.data.len() - 1;
                self.data[last] |= (bit as u8) << (7 - (self.bit % 8));
                self.bit += 1;
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.data
        }
    }

    /// A complete 12-bit JOC header plus the clipgain and the sequence counter.
    fn header(bits: &mut Bits, dmx_config_idx: u32, declared_objects: u32) {
        bits.push(3, dmx_config_idx);
        bits.push(6, declared_objects - 1);
        bits.push(3, 0); // ext_config_idx
        bits.push(3, 4); // clipgain x: unity exponent
        bits.push(5, 0); // clipgain y
        bits.push(10, 7); // seq_count
    }

    #[test]
    fn channel_counts_match_the_spec() {
        assert_eq!(num_channels_for_config(3), Some(5));
        assert_eq!(num_channels_for_config(1), Some(7));
        assert_eq!(num_channels_for_config(5), None);
    }

    #[test]
    fn band_counts_match_the_spec() {
        assert_eq!(num_bands_for_index(5), Some(12));
        assert_eq!(num_bands_for_index(8), None);
    }

    #[test]
    fn clipgain_is_a_power_of_two_times_the_mantissa() {
        // x = 4 reduces to 1 + y/32; other exponents scale the whole mantissa.
        assert_eq!(clipgain_of(4, 0), 1.0);
        assert_eq!(clipgain_of(4, 16), 1.5);
        assert_eq!(clipgain_of(0, 0), 0.0625);
        assert_eq!(clipgain_of(6, 16), 6.0);
        assert!((clipgain_of(7, 31) - 15.75).abs() < 1e-12);
        // The two candidate formulas only differ when x != 4.
        assert_ne!(clipgain_of(6, 16), 1.0 + 0.5 * 4.0);
    }

    #[test]
    fn dense_reconstruction_is_cumulative_per_data_point() {
        let mut mtx = [[[0i16; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS];
        mtx[0][0][0] = -48;
        mtx[0][0][1] = 1;
        mtx[1][0][0] = -96;
        mtx[1][0][1] = 2;
        let mut q = [[[0i64; MAX_PARAMETER_BANDS]; MAX_CORE_CHANNELS]; MAX_DPOINTS];
        reconstruct_dense(&mtx, 0, 2, 1, 96, 48, &mut q);
        reconstruct_dense(&mtx, 1, 2, 1, 96, 48, &mut q);
        assert_eq!(q[0][0][0], 0);
        assert_eq!(q[0][0][1], 1);
        assert_eq!(q[1][0][0], 48);
        assert_eq!(q[1][0][1], 50);
    }

    #[test]
    fn offsets_must_address_timeslots_and_increase() {
        let single = |offset| JocObject {
            n_dpoints: 1,
            offset_ts: [offset, 0],
            ..JocObject::default()
        };
        assert!(!offsets_invalid(&single(0)));
        assert!(!offsets_invalid(&single(23)));
        assert!(offsets_invalid(&single(24)));
        let pair = |first, second| JocObject {
            n_dpoints: 2,
            offset_ts: [first, second],
            ..JocObject::default()
        };
        assert!(offsets_invalid(&pair(5, 5)));
        assert!(offsets_invalid(&pair(6, 5)));
        assert!(!offsets_invalid(&pair(5, 6)));
    }

    #[test]
    fn offset_ts_keeps_the_raw_five_bit_slot() {
        let mut bits = Bits::default();
        header(&mut bits, 3, 1);
        bits.push(1, 1); // object present
        bits.push(3, 0); // one parameter band
        bits.push(1, 0); // dense
        bits.push(1, 0); // coarse quantisation
        bits.push(1, 1); // slope_idx: step mode
        bits.push(1, 0); // one data point
        bits.push(5, 7); // offset_ts = 7, as coded
        for _ in 0..5 {
            bits.push(1, 0); // the coarse matrix code for zero is a single 0 bit
        }
        let params = parse_id14(&bits.into_bytes()).unwrap();
        assert_eq!(params.status, JocPayloadStatus::Complete);
        assert_eq!(params.objects[0].offset_ts[0], 7);
        assert_eq!(params.objects[0].offset_ts[1], 0);
    }

    #[test]
    fn out_of_range_offset_ts_is_a_frame_status() {
        let mut bits = Bits::default();
        header(&mut bits, 3, 1);
        bits.push(1, 1);
        bits.push(3, 0);
        bits.push(1, 0);
        bits.push(1, 0);
        bits.push(1, 1);
        bits.push(1, 0);
        bits.push(5, 30); // beyond the 24 timeslots
        for _ in 0..5 {
            bits.push(1, 0);
        }
        let params = parse_id14(&bits.into_bytes()).unwrap();
        assert_eq!(params.status, JocPayloadStatus::BadOffsetTs);
        assert!(!params.status.is_complete());
    }

    #[test]
    fn decreasing_offset_ts_is_a_frame_status() {
        let mut bits = Bits::default();
        header(&mut bits, 3, 1);
        bits.push(1, 1);
        bits.push(3, 0);
        bits.push(1, 0);
        bits.push(1, 0);
        bits.push(1, 1);
        bits.push(1, 1); // two data points
        bits.push(5, 9);
        bits.push(5, 9); // not strictly increasing
        for _ in 0..10 {
            bits.push(1, 0); // five channels x two data points
        }
        let params = parse_id14(&bits.into_bytes()).unwrap();
        assert_eq!(params.status, JocPayloadStatus::BadOffsetTs);
    }

    #[test]
    fn a_payload_that_ends_in_the_object_headers_is_truncated() {
        let mut bits = Bits::default();
        header(&mut bits, 3, 2);
        bits.push(1, 1); // first object present
        let params = parse_id14(&bits.into_bytes()).unwrap();
        assert_eq!(params.status, JocPayloadStatus::Truncated);
        assert_eq!(params.present_mask, 0b1);
        assert!(params.objects[0].present);
        // The header fields survive so the kernel can hold the previous state.
        assert_eq!(params.n_channels, 5);
        assert_eq!(params.seq_count, 7);
        assert_eq!(params.clipgain, 1.0);
    }

    #[test]
    fn a_payload_that_ends_in_joc_data_keeps_its_header() {
        let mut bits = Bits::default();
        header(&mut bits, 3, 1);
        bits.push(1, 1);
        bits.push(3, 0);
        bits.push(1, 0); // dense
        bits.push(1, 0);
        bits.push(1, 0); // interp
        bits.push(1, 0); // one data point
        bits.push(2, 0b10); // a two-bit code with no leaf here: runs out
        let params = parse_id14(&bits.into_bytes()).unwrap();
        assert_eq!(params.status, JocPayloadStatus::Truncated);
    }

    #[test]
    fn a_payload_shorter_than_the_fixed_header_is_an_error() {
        // The fixed header is the 30-bit dmx configuration, object count,
        // extension index, clipgain and sequence counter. A payload that ends
        // inside it cannot be reported as a damaged frame: the reader would
        // substitute zeros, so the frame would carry a downmix configuration and
        // an object count the bitstream never declared.
        for length in 0..=3 {
            let payload = vec![0u8; length];
            let error = parse_id14(&payload).expect_err("a short payload must fail");
            assert!(
                matches!(error, Eac3JocError::BitstreamTruncated),
                "length {length}: {error:?}"
            );
        }
        // A one-byte payload whose low bits spell a reserved configuration is
        // still a truncation, not a syntax error.
        let error = parse_id14(&[0xE0]).expect_err("a short payload must fail");
        assert!(
            matches!(error, Eac3JocError::BitstreamTruncated),
            "{error:?}"
        );
    }

    #[test]
    fn an_unsupported_downmix_configuration_stays_an_error() {
        let mut bits = Bits::default();
        header(&mut bits, 5, 1);
        let error = parse_id14(&bits.into_bytes()).unwrap_err();
        assert!(matches!(error, Eac3JocError::JocSyntax(_)));
    }
}
