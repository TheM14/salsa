//! Audio-frame grouping and the codec-specific frame exposed by the parser.

use objectaudio_oamd::{OamdFrame, OamdLimits};

use crate::bed::{
    BedPlan, compute_bed_plan, compute_bed_plan_for_config, compute_bed_plan_from_headers,
    is_core_stream,
};
use crate::emdf;
use crate::error::{Eac3JocError, Result};
use crate::joc::{self, JocParams};
use crate::transport::{FrameReader, SyncFrame, SyncFrameHeader};

/// One E-AC-3 audio frame: every syncframe up to the next repeated
/// `(codec, strmtyp, substreamid)` combination.
#[derive(Debug, Clone)]
pub struct Eac3JocFrame {
    pub index: u64,
    pub sample_rate: u32,
    pub samples: usize,
    /// The independent substream header, when present.
    pub core_header: Option<SyncFrameHeader>,
    /// The first dependent substream header, when present.
    pub dependent_header: Option<SyncFrameHeader>,
    pub syncframes: Vec<SyncFrame>,
    pub joc: Option<Box<JocParams>>,
    pub oamd: Option<OamdFrame>,
    /// Raw ID11 payload, kept for the GStreamer meta.
    pub oamd_payload: Option<Vec<u8>>,
    /// EMDF payload sample offset that accompanies the OAMD payload.
    pub oamd_sample_offset: u16,
    pub bed_plan: Option<BedPlan>,
}

impl Eac3JocFrame {
    /// Concatenated syncframes, the packet shape an E-AC-3 decoder expects.
    pub fn core_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.syncframes.iter().map(|f| f.bytes.len()).sum());
        for frame in &self.syncframes {
            bytes.extend_from_slice(&frame.bytes);
        }
        bytes
    }

    pub fn has_joc(&self) -> bool {
        self.joc.is_some()
    }
}

/// Incremental parse -> `Eac3JocFrame`.
#[derive(Debug, Default)]
pub struct Eac3JocParser {
    reader: FrameReader,
    pending: Vec<SyncFrame>,
    pending_joc: Option<Box<JocParams>>,
    pending_oamd: Option<OamdFrame>,
    pending_oamd_payload: Option<Vec<u8>>,
    pending_oamd_offset: u16,
    last_bed_plan: Option<BedPlan>,
    last_sample_rate: u32,
    next_index: u64,
    finished: bool,
}

impl Eac3JocParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_bytes(&mut self, data: &[u8]) -> Result<Vec<Eac3JocFrame>> {
        self.reader.push(data);
        self.drain()
    }

    pub fn flush(&mut self) -> Result<Vec<Eac3JocFrame>> {
        self.reader.finish();
        self.finished = true;
        let mut frames = self.drain()?;
        if let Some(frame) = self.finish_pending()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn drain(&mut self) -> Result<Vec<Eac3JocFrame>> {
        let mut completed = Vec::new();
        while let Some(frame) = self.reader.next_frame()? {
            let repeats = self
                .pending
                .iter()
                .any(|prior| prior.header.same_stream(&frame.header));
            if repeats && let Some(group) = self.finish_pending()? {
                completed.push(group);
            }
            self.accept(frame)?;
        }
        Ok(completed)
    }

    fn accept(&mut self, frame: SyncFrame) -> Result<()> {
        if let Some(container) = emdf::find_joc_emdf_optional(
            &frame.bytes,
            Some(&|candidate| joc::score_container_with_frame(&frame.bytes, candidate)),
        )? {
            let joc_payload = container.find(emdf::ID_JOC).ok_or_else(|| {
                Eac3JocError::EmdfTransport("EMDF container has no ID14 payload".into())
            })?;
            let params = joc::parse_id14(&emdf::extract_payload_bytes(&frame.bytes, joc_payload)?)?;
            self.pending_joc = Some(Box::new(params));
            if let Some(oamd_payload) = container.find(emdf::ID_OAMD) {
                let bytes = emdf::extract_payload_bytes(&frame.bytes, oamd_payload)?;
                let oamd = objectaudio_oamd::parse_with_limits(&bytes, OamdLimits::default())?;
                self.pending_oamd = Some(oamd);
                self.pending_oamd_payload = Some(bytes);
                self.pending_oamd_offset = oamd_payload.sample_offset;
            }
        }
        self.pending.push(frame);
        Ok(())
    }

    fn finish_pending(&mut self) -> Result<Option<Eac3JocFrame>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let syncframes = std::mem::take(&mut self.pending);
        let joc = self.pending_joc.take();
        let oamd = self.pending_oamd.take();
        let oamd_payload = self.pending_oamd_payload.take();
        let oamd_sample_offset = std::mem::take(&mut self.pending_oamd_offset);

        let core_header = syncframes
            .iter()
            .map(|frame| frame.header)
            .find(is_core_stream);
        let dependent_header = syncframes
            .iter()
            .map(|frame| frame.header)
            .find(|header| !is_core_stream(header));
        let sample_rate = core_header
            .or(dependent_header)
            .map(|header| header.sample_rate)
            .unwrap_or(self.last_sample_rate);
        if sample_rate != 0 {
            self.last_sample_rate = sample_rate;
        }
        let samples = core_header
            .map(|header| header.samples())
            .unwrap_or(crate::joc::FRAME_SAMPLES);

        // Only a payload the kernel can render describes the bed layout. A
        // damaged one keeps the header it read but never decoded its object
        // data, so it must not replace the plan the substream headers or the
        // previous frames established.
        let usable_joc = joc.as_ref().filter(|params| params.status.is_complete());
        let bed_plan = if let Some(params) = usable_joc {
            let plan = compute_bed_plan(
                core_header.as_ref(),
                dependent_header.as_ref(),
                params.dmx_config_idx,
            )
            .or_else(|_| compute_bed_plan_for_config(params.dmx_config_idx))?;
            self.last_bed_plan = Some(plan);
            Some(plan)
        } else if let Some(plan) = self.last_bed_plan {
            Some(plan)
        } else if let Some(core) = core_header.as_ref() {
            // The first frame can precede the first JOC/OAMD payload.  The
            // substream headers still describe the bed layout.
            let plan = compute_bed_plan_from_headers(core, dependent_header.as_ref())?;
            self.last_bed_plan = Some(plan);
            Some(plan)
        } else {
            None
        };

        let frame = Eac3JocFrame {
            index: self.next_index,
            sample_rate,
            samples,
            core_header,
            dependent_header,
            syncframes,
            joc,
            oamd,
            oamd_payload,
            oamd_sample_offset,
            bed_plan,
        };
        self.next_index += 1;
        Ok(Some(frame))
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joc::{JocObject, JocPayloadStatus};
    use crate::transport::{Codec, SyncFrame};

    fn header(acmod: u8, lfeon: bool) -> SyncFrameHeader {
        SyncFrameHeader {
            codec: Codec::Eac3,
            stream_type: 0,
            substream_id: 0,
            acmod,
            lfeon,
            bsid: 16,
            sample_rate: 48_000,
            num_blocks_code: 3,
            chanmap: None,
            size: 0,
        }
    }

    /// A payload whose fixed header is intact but whose object data is not.
    fn damaged_payload(dmx_config_idx: u8) -> JocParams {
        JocParams {
            dmx_config_idx,
            num_objects_bits: 0,
            ext_config_idx: 0,
            n_channels: crate::joc::num_channels_for_config(u32::from(dmx_config_idx))
                .expect("supported configuration"),
            n_objects: 1,
            objects_clamped: false,
            clipgain_x_bits: 4,
            clipgain_y_bits: 0,
            seq_count: 1,
            clipgain: 1.0,
            objects: [JocObject::default(); crate::joc::MAX_OBJECTS],
            present_mask: 0,
            data_end_bits: 0,
            trailing_bits: 0,
            status: JocPayloadStatus::Truncated,
        }
    }

    /// A 5.1 core frame: six bed channels, so a five-lane JOC configuration.
    fn five_one_frame(parser: &mut Eac3JocParser, joc: Option<JocParams>) {
        parser.pending.push(SyncFrame {
            header: header(7, true),
            offset: 0,
            bytes: Vec::new(),
        });
        parser.pending_joc = joc.map(Box::new);
    }

    #[test]
    fn a_damaged_payload_does_not_replace_the_bed_plan() {
        // The payload claims a seven-core-channel configuration, which does not
        // match the six-channel 5.1 core. It cannot be rendered, so the layout
        // has to come from the substream headers instead of the payload.
        let mut parser = Eac3JocParser::new();
        five_one_frame(&mut parser, Some(damaged_payload(4)));
        let frame = parser.finish_pending().unwrap().expect("frame");
        let plan = frame.bed_plan.expect("a plan");
        assert_eq!(plan.channels, 6);
        assert_eq!(plan.lfe_channel, Some(3));
        // The trusted plan is what a later metadata-free frame reuses.
        assert_eq!(
            parser.last_bed_plan.map(|plan| plan.channels),
            Some(6),
            "the damaged payload must not be cached"
        );
    }

    #[test]
    fn a_complete_payload_is_still_trusted_for_the_bed_plan() {
        // A renderable payload keeps deciding the layout, including the
        // configuration-only plan a stream whose headers disagree with its
        // downmix configuration gets. Only unrenderable payloads changed.
        let mut parser = Eac3JocParser::new();
        let mut complete = damaged_payload(4);
        complete.status = JocPayloadStatus::Complete;
        // A 5.1 core cannot supply the extra pair configuration 4 needs.
        five_one_frame(&mut parser, Some(complete));
        let frame = parser.finish_pending().unwrap().expect("frame");
        assert_eq!(frame.bed_plan.expect("a plan").channels, 8);
    }
}
