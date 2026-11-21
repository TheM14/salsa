//! JOC reconstruction kernel: analysis QMF, object matrix, synthesis QMF, LFE.
//!
//! The kernel also owns the frame-level state machine. A JOC stream can carry a
//! damaged payload, a frame without metadata at all, or a programme restart; the
//! state machine decides whether a frame renders its own metadata, holds the
//! previous matrices, or hard-cuts.
//!
//! The transforms use a scalar `rustfft` path.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::manual_memcpy,
    clippy::too_many_arguments
)]

use std::f64::consts::PI;
use std::sync::Arc;

use eac3joc::{CORE_CHANNELS, FRAME_SAMPLES, JocParams, MAX_CORE_CHANNELS, MAX_OBJECTS, TIMESLOTS};
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::qmf_tables as tables;

/// Output channels: the LFE plus every JOC object.
pub const OUTPUT_CHANNELS: usize = MAX_OBJECTS + 1;
const LFE_DELAY: usize = 1217;
/// Samples the clipgain ramp covers at the start of a frame.
const RAMP_SAMPLES: usize = 256;
/// The structural 1/16 of the analysis scalar. The x16 at the output cancels it,
/// so the net effect of `ANALYSIS_UNIT * clipgain` is a multiply by `clipgain`.
const ANALYSIS_UNIT: f64 = 0.0625;
/// Sequence counter the JOC payload carries; it wraps from 1023 back to 1.
const MAX_SEQUENCE: u16 = 1023;

const DC_A: [f64; 21] = [
    -0.0006242550443857908,
    -0.0019234686624258757,
    -0.0042654648423194885,
    -0.008168308064341545,
    -0.014327201060950756,
    -0.023759860545396805,
    -0.03757232800126076,
    -0.05577569454908371,
    -0.07568276673555374,
    -0.09172472357749939,
    -0.5979374051094055,
    -0.09172472357749939,
    -0.07568276673555374,
    -0.05577569454908371,
    -0.03757232800126076,
    -0.023759860545396805,
    -0.014327201060950756,
    -0.008168308064341545,
    -0.0042654648423194885,
    -0.0019234686624258757,
    -0.0006242550443857908,
];

const DC_B: [f64; 21] = [
    0.0013996040215715766,
    0.003839150769636035,
    0.007512642536312342,
    0.012419373728334904,
    0.018367428332567215,
    0.0249701626598835,
    0.03167900815606117,
    0.03785000368952751,
    0.04283412545919418,
    0.04607561603188515,
    0.047200120985507965,
    0.04607561603188515,
    0.04283412545919418,
    0.03785000368952751,
    0.03167900815606117,
    0.0249701626598835,
    0.018367428332567215,
    0.012419373728334904,
    0.007512642536312342,
    0.003839150769636035,
    0.0013996040215715766,
];

fn clamp_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    InvalidCoreChannels(usize),
    InvalidBedLength { expected: usize, actual: usize },
    InvalidLfeLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    InvalidParameters(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCoreChannels(count) => {
                write!(f, "core channel count must be 5 or 7, got {count}")
            }
            Self::InvalidBedLength { expected, actual } => {
                write!(f, "bed PCM has {actual} samples, expected {expected}")
            }
            Self::InvalidLfeLength { expected, actual } => {
                write!(f, "LFE PCM has {actual} samples, expected {expected}")
            }
            Self::InvalidOutputLength { expected, actual } => {
                write!(f, "output has {actual} samples, expected {expected}")
            }
            Self::InvalidParameters(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// The JOC description of one audio frame.
#[derive(Debug, Clone, Copy)]
pub struct JocFrame<'a> {
    /// Core channel count this frame declares; 5 or 7.
    pub core_channels: usize,
    /// `sigma(k)` phase-shift switch for the surround lanes. `None` keeps the
    /// previous frame's setting, because a frame without JOC metadata carries no
    /// downmix configuration.
    pub phase_shift: Option<bool>,
    /// Parsed JOC payload, `None` when the frame carries no JOC metadata.
    pub params: Option<&'a JocParams>,
}

impl<'a> JocFrame<'a> {
    /// A frame that carries a JOC payload.
    pub fn payload(params: &'a JocParams) -> Self {
        Self {
            core_channels: usize::from(params.n_channels),
            phase_shift: Some(params.phase_shift()),
            params: Some(params),
        }
    }

    /// A frame that carries no JOC payload; the declared channel count and the
    /// phase-shift switch have to come from the stream.
    pub fn absent(core_channels: usize, phase_shift: Option<bool>) -> Self {
        Self {
            core_channels,
            phase_shift,
            params: None,
        }
    }
}

/// How one frame is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameMode {
    /// Render this frame's metadata, interpolating from the previous state.
    Normal,
    /// Reuse the previous matrices; a damaged or missing payload must not drop
    /// the object gains to zero.
    Hold,
    /// Publish silence for one frame and restart the programme.
    HardCut,
}

/// What the frame says about the JOC metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameStatus {
    /// The payload was decoded in full.
    Complete,
    /// The payload is identifiable but incomplete.
    Damaged,
    /// The frame carries no JOC payload.
    NoMetadata,
    /// The kernel's own tag: the declared core channel count changed.
    ChannelCountChange,
}

impl FrameStatus {
    fn usable(self) -> bool {
        matches!(self, Self::Complete | Self::ChannelCountChange)
    }
}

/// The sequence-counter state machine of one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceState {
    /// No usable frame has been rendered yet.
    Start,
    /// The sequence counter matched the expectation.
    Synchronised,
    /// Fewer than six rejected frames in a row.
    Invalid,
    /// Six or more rejected frames in a row.
    Halted,
    /// The previous frame was a hard cut.
    Restart,
}

/// The stateful JOC kernel.
pub struct JocKernel {
    fft: Arc<dyn Fft<f64>>,
    rotation_sin: [f64; 64],
    rotation_cos: [f64; 64],

    analysis_fifo: [[[f64; 64]; 9]; MAX_CORE_CHANNELS],
    analysis_delay: [[[f32; 64]; 10]; 5],
    analysis_phase: f32,
    surround_delay: [[[Complex64; 64]; 10]; 2],
    surround_history: [[Complex64; 20]; 2],
    phase_shift_enabled: bool,
    lfe_delay: [f64; LFE_DELAY],
    /// Clipgain applied to the previous frame's LFE, the ramp start value.
    lfe_clipgain: Option<f64>,
    /// Frame-end matrix of the previous frame, and whether it is defined.
    matrix_previous: [[[f64; 64]; MAX_CORE_CHANNELS]; MAX_OBJECTS],
    matrix_previous_present: [bool; MAX_OBJECTS],
    synthesis_state: [[f64; 640]; MAX_OBJECTS],

    sequence: SequenceState,
    expected_sequence: u16,
    invalid_count: u8,
    previous_core_channels: Option<usize>,

    x: [[[Complex64; 24]; 64]; MAX_CORE_CHANNELS],
    z: [[[Complex64; 24]; 64]; MAX_OBJECTS],
}

impl Default for JocKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl JocKernel {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(64);
        let mut rotation_sin = [0.0; 64];
        let mut rotation_cos = [0.0; 64];
        for index in 0..64 {
            let theta = PI * index as f64 / 128.0;
            rotation_sin[index] = 0.5 * theta.sin();
            rotation_cos[index] = 0.5 * theta.cos();
        }
        Self {
            fft,
            rotation_sin,
            rotation_cos,
            analysis_fifo: [[[0.0; 64]; 9]; MAX_CORE_CHANNELS],
            analysis_delay: [[[0.0; 64]; 10]; 5],
            analysis_phase: -0.5,
            surround_delay: [[[Complex64::new(0.0, 0.0); 64]; 10]; 2],
            surround_history: [[Complex64::new(0.0, 0.0); 20]; 2],
            phase_shift_enabled: true,
            lfe_delay: [0.0; LFE_DELAY],
            lfe_clipgain: None,
            matrix_previous: [[[0.0; 64]; MAX_CORE_CHANNELS]; MAX_OBJECTS],
            matrix_previous_present: [false; MAX_OBJECTS],
            synthesis_state: [[0.0; 640]; MAX_OBJECTS],
            sequence: SequenceState::Start,
            expected_sequence: 0,
            invalid_count: 0,
            previous_core_channels: None,
            x: [[[Complex64::new(0.0, 0.0); 24]; 64]; MAX_CORE_CHANNELS],
            z: [[[Complex64::new(0.0, 0.0); 24]; 64]; MAX_OBJECTS],
        }
    }

    /// Reset all streaming state. The first frame after a reset adopts the new
    /// scalar without an intra-frame transition and renders its metadata as a
    /// hard cut instead of fading in from silence.
    pub fn reset(&mut self) {
        self.analysis_fifo = [[[0.0; 64]; 9]; MAX_CORE_CHANNELS];
        self.analysis_delay = [[[0.0; 64]; 10]; 5];
        self.analysis_phase = -0.5;
        self.surround_delay = [[[Complex64::new(0.0, 0.0); 64]; 10]; 2];
        self.surround_history = [[Complex64::new(0.0, 0.0); 20]; 2];
        self.lfe_delay = [0.0; LFE_DELAY];
        self.lfe_clipgain = None;
        self.matrix_previous = [[[0.0; 64]; MAX_CORE_CHANNELS]; MAX_OBJECTS];
        self.matrix_previous_present = [false; MAX_OBJECTS];
        self.synthesis_state = [[0.0; 640]; MAX_OBJECTS];
        self.sequence = SequenceState::Start;
        self.expected_sequence = 0;
        self.invalid_count = 0;
        self.previous_core_channels = None;
        self.x = [[[Complex64::new(0.0, 0.0); 24]; 64]; MAX_CORE_CHANNELS];
        self.z = [[[Complex64::new(0.0, 0.0); 24]; 64]; MAX_OBJECTS];
    }

    /// Render one audio frame.
    ///
    /// `bed` is planar `core_channels * 1536`, `lfe` is 1536 samples (or `None`
    /// for silence) and `output` is planar `16 * 1536` with the LFE in channel 0
    /// and the JOC objects in channels 1..15.
    ///
    /// A frame whose payload is incomplete, or that carries no payload at all,
    /// reuses the previous frame's matrices instead of publishing silence: one
    /// damaged frame must not drop the object gains. Frames without any previous
    /// state are silent.
    pub fn process_frame(
        &mut self,
        bed: &[f32],
        lfe: Option<&[f32]>,
        frame: &JocFrame<'_>,
        output_scale: f32,
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        let core_channels = frame.core_channels;
        if core_channels != CORE_CHANNELS && core_channels != MAX_CORE_CHANNELS {
            return Err(KernelError::InvalidCoreChannels(core_channels));
        }
        let expected_bed = core_channels * FRAME_SAMPLES;
        if bed.len() != expected_bed {
            return Err(KernelError::InvalidBedLength {
                expected: expected_bed,
                actual: bed.len(),
            });
        }
        if let Some(lfe) = lfe
            && lfe.len() != FRAME_SAMPLES
        {
            return Err(KernelError::InvalidLfeLength {
                expected: FRAME_SAMPLES,
                actual: lfe.len(),
            });
        }
        if output.len() != OUTPUT_CHANNELS * FRAME_SAMPLES {
            return Err(KernelError::InvalidOutputLength {
                expected: OUTPUT_CHANNELS * FRAME_SAMPLES,
                actual: output.len(),
            });
        }
        if !output_scale.is_finite() {
            return Err(KernelError::InvalidParameters(
                "output scale must be finite".into(),
            ));
        }

        let payload = match frame.params {
            None => FrameStatus::NoMetadata,
            Some(params) if !params.status.is_complete() => FrameStatus::Damaged,
            Some(_) => FrameStatus::Complete,
        };
        // A damaged payload is never indexed, so its partially parsed object
        // fields must not reject the frame: the kernel holds instead.
        if payload == FrameStatus::Complete {
            validate_params(frame.params.expect("complete payload"))?;
        }
        // The declared channel count decides how the object syntax expands, so a
        // change hard-cuts the frame and clears the state of the unused lanes.
        let changed = self
            .previous_core_channels
            .is_some_and(|previous| previous != core_channels);
        if changed {
            self.clear_unused_channels(core_channels);
        }
        let status = if changed && payload.usable() {
            FrameStatus::ChannelCountChange
        } else {
            payload
        };
        let sequence = frame.params.map_or(0, |params| params.seq_count);
        self.advance_state(status, sequence);
        if status == FrameStatus::ChannelCountChange {
            // The delay ring holds PCM from the old layout; the clipgain ramp
            // state stays so the gain does not retune.
            self.lfe_delay = [0.0; LFE_DELAY];
        }
        let mode = self.select_mode(status);
        if let Some(phase_shift) = frame.phase_shift {
            self.phase_shift_enabled = phase_shift;
        }
        let (clipgain, analysis_scalar) = match frame.params {
            Some(params)
                if payload.usable() && matches!(mode, FrameMode::Normal | FrameMode::HardCut) =>
            {
                let clipgain = params.clipgain;
                (clipgain, (ANALYSIS_UNIT * clipgain) as f32)
            }
            // A held frame reuses the current scalar; it does not retune it.
            _ => (self.held_clipgain(), self.held_analysis_scalar()),
        };

        output.fill(0.0);
        self.analysis(bed, core_channels, analysis_scalar);
        self.render_objects(frame.params, core_channels, mode, output_scale, output);
        self.render_lfe(lfe, clipgain, output_scale, output);
        if mode == FrameMode::HardCut {
            // Silence this frame; the presence markers stay set so the next
            // usable frame fades in from zero.
            output.fill(0.0);
            for object in 0..MAX_OBJECTS {
                self.matrix_previous[object] = [[0.0; 64]; MAX_CORE_CHANNELS];
                self.matrix_previous_present[object] = true;
            }
        }
        self.previous_core_channels = Some(core_channels);
        Ok(())
    }

    /// Analysis scalar of the previous frame, or the structural unit when no
    /// frame has been rendered yet.
    fn held_analysis_scalar(&self) -> f32 {
        if self.analysis_phase < 0.0 {
            ANALYSIS_UNIT as f32
        } else {
            self.analysis_phase
        }
    }

    fn held_clipgain(&self) -> f64 {
        f64::from(self.held_analysis_scalar()) / ANALYSIS_UNIT
    }

    /// Drop the state of the core lanes a smaller layout does not use, so a
    /// later grow does not resume a stale filter or matrix.
    fn clear_unused_channels(&mut self, core_channels: usize) {
        if core_channels >= MAX_CORE_CHANNELS {
            return;
        }
        for channel in core_channels..MAX_CORE_CHANNELS {
            self.analysis_fifo[channel] = [[0.0; 64]; 9];
            for object in 0..MAX_OBJECTS {
                self.matrix_previous[object][channel] = [0.0; 64];
            }
        }
        if core_channels == CORE_CHANNELS {
            // A 5-channel frame does not drive the extra-pair delay lanes.
            self.analysis_delay[3] = [[0.0; 64]; 10];
            self.analysis_delay[4] = [[0.0; 64]; 10];
        }
    }

    /// The 10-bit JOC sequence counter wraps from 1023 back to 1.
    fn next_sequence(value: u16) -> u16 {
        if value == MAX_SEQUENCE { 1 } else { value + 1 }
    }

    /// Advance the sequence state machine for this frame.
    ///
    /// A frame without metadata neither advances the expected counter nor
    /// compares the sequence: mid-stream it holds, because a frame without
    /// metadata is a damaged frame rather than "no objects".
    fn advance_state(&mut self, status: FrameStatus, sequence: u16) {
        let usable = status.usable();
        if status == FrameStatus::NoMetadata && self.sequence != SequenceState::Start {
            self.sequence = SequenceState::Halted;
            return;
        }
        self.expected_sequence = Self::next_sequence(self.expected_sequence);
        if usable && sequence == 0 && self.sequence != SequenceState::Start {
            // The programme restarted: hard-cut and drop the object history.
            self.sequence = SequenceState::Restart;
            return;
        }
        self.sequence = match self.sequence {
            SequenceState::Start => {
                if usable {
                    self.expected_sequence = sequence;
                    SequenceState::Synchronised
                } else {
                    SequenceState::Start
                }
            }
            SequenceState::Synchronised => {
                if !usable {
                    self.invalid_count = 1;
                    SequenceState::Invalid
                } else if sequence != self.expected_sequence {
                    SequenceState::Restart
                } else {
                    SequenceState::Synchronised
                }
            }
            SequenceState::Invalid => {
                if usable {
                    self.expected_sequence = sequence;
                    SequenceState::Synchronised
                } else if self.invalid_count == 5 {
                    SequenceState::Halted
                } else {
                    self.invalid_count += 1;
                    SequenceState::Invalid
                }
            }
            SequenceState::Halted => {
                if usable {
                    self.expected_sequence = sequence;
                    SequenceState::Synchronised
                } else {
                    self.invalid_count = self.invalid_count.saturating_add(1);
                    SequenceState::Halted
                }
            }
            SequenceState::Restart => {
                if usable {
                    self.expected_sequence = sequence;
                    SequenceState::Synchronised
                } else {
                    self.invalid_count = 1;
                    SequenceState::Invalid
                }
            }
        };
        if status == FrameStatus::ChannelCountChange {
            self.sequence = SequenceState::Restart;
        }
    }

    fn select_mode(&self, status: FrameStatus) -> FrameMode {
        if status == FrameStatus::NoMetadata {
            // Before the first usable frame there is nothing to hold, so the
            // held state is all zeros; mid-stream it reuses the last matrices.
            return FrameMode::Hold;
        }
        match self.sequence {
            SequenceState::Restart => FrameMode::HardCut,
            SequenceState::Invalid | SequenceState::Halted => FrameMode::Hold,
            SequenceState::Synchronised if status == FrameStatus::Complete => FrameMode::Normal,
            SequenceState::Synchronised | SequenceState::Start => FrameMode::Hold,
        }
    }

    /// Frame the analysis QMF of the core PCM with the given analysis-side
    /// scalar (`ANALYSIS_UNIT * clipgain`). The scalar ramps over the frame's
    /// first 256 samples from the previous frame's value; the first frame after
    /// a reset takes it directly.
    fn analysis(&mut self, bed: &[f32], core_channels: usize, scalar: f32) {
        let mut phase_old = self.analysis_phase;
        if phase_old < 0.0 {
            phase_old = scalar;
        }
        for channel in 0..core_channels {
            self.analysis_channel(bed, channel, core_channels, phase_old, scalar);
        }
        self.analysis_phase = scalar;
        self.surround_post();
    }

    fn analysis_channel(
        &mut self,
        bed: &[f32],
        channel: usize,
        _core_channels: usize,
        phase_old: f32,
        phase_new: f32,
    ) {
        let ramp = phase_old != phase_new;
        let step = (phase_new - phase_old) / 256.0f32;
        let mut scaled = [0.0f32; 64];
        let mut delayed = [0.0f32; 64];
        for timeslot in 0..24 {
            for sample in 0..64 {
                let frame_sample = timeslot * 64 + sample;
                let mut gain = phase_new;
                if ramp && frame_sample < 256 {
                    gain = phase_old + frame_sample as f32 * step;
                }
                scaled[sample] = bed[channel * FRAME_SAMPLES + frame_sample] * gain;
            }
            let delay_lane = if channel < 3 { channel } else { channel - 2 };
            let ring = if channel < 3 || channel >= 5 {
                delayed.copy_from_slice(&self.analysis_delay[delay_lane][0]);
                let lanes = &mut self.analysis_delay[delay_lane];
                lanes.copy_within(1..10, 0);
                lanes[9] = scaled;
                &delayed
            } else {
                &scaled
            };
            let ring_copy = *ring;
            self.analysis_slot(channel, &ring_copy, timeslot);
        }
    }

    fn analysis_slot(&mut self, channel: usize, ring: &[f32; 64], timeslot: usize) {
        let mut v36 = [0.0f64; 64];
        let mut v40 = [0.0f64; 64];
        let fifo = &self.analysis_fifo[channel];
        for sample in 0..64 {
            v36[sample] = fifo[0][sample] * tables::KANALYSIS_WINDOW[8 * 64 + sample]
                + fifo[2][sample] * tables::KANALYSIS_WINDOW[6 * 64 + sample]
                + fifo[4][sample] * tables::KANALYSIS_WINDOW[4 * 64 + sample]
                + fifo[6][sample] * tables::KANALYSIS_WINDOW[2 * 64 + sample]
                + fifo[8][sample] * tables::KANALYSIS_WINDOW[sample];
            v40[sample] = fifo[1][sample] * tables::KANALYSIS_WINDOW[7 * 64 + sample]
                + fifo[3][sample] * tables::KANALYSIS_WINDOW[5 * 64 + sample]
                + fifo[5][sample] * tables::KANALYSIS_WINDOW[3 * 64 + sample]
                + fifo[7][sample] * tables::KANALYSIS_WINDOW[64 + sample]
                + f64::from(ring[sample]) * tables::KANALYSIS_WINDOW[9 * 64 + sample];
        }

        let mut frequency = [Complex64::new(0.0, 0.0); 64];
        for k in 0..64 {
            let source = 63 - k;
            let re = v40[source];
            let im = v36[source];
            let a = self.rotation_sin[k];
            let b = self.rotation_cos[k];
            frequency[k] = Complex64::new(im * a - re * b, im * b + re * a);
        }
        self.fft.process(&mut frequency);
        const SCALE: f64 = 1.0 / 64.0;
        for k in 0..32 {
            self.x[channel][2 * k][timeslot] =
                Complex64::new(frequency[k].re * SCALE, -frequency[k].im * SCALE);
            self.x[channel][2 * k + 1][timeslot] =
                Complex64::new(frequency[63 - k].re * SCALE, frequency[63 - k].im * SCALE);
        }

        let fifo = &mut self.analysis_fifo[channel];
        for history in (1..9).rev() {
            fifo[history] = fifo[history - 1];
        }
        for sample in 0..64 {
            fifo[0][sample] = f64::from(ring[sample]);
        }
    }

    fn surround_post(&mut self) {
        for surround in 0..2 {
            let channel = surround + 3;
            for group in (0..24).step_by(4) {
                let mut current = [[Complex64::new(0.0, 0.0); 64]; 4];
                let mut dc_buffer = [Complex64::new(0.0, 0.0); 24];
                for slot in 0..4 {
                    for band in 0..64 {
                        current[slot][band] = self.x[channel][band][group + slot];
                        let delayed = self.surround_delay[surround][slot][band];
                        self.x[channel][band][group + slot] = if self.phase_shift_enabled {
                            Complex64::new(delayed.im, -delayed.re)
                        } else {
                            delayed
                        };
                    }
                }
                dc_buffer[..20].copy_from_slice(&self.surround_history[surround]);
                for slot in 0..4 {
                    dc_buffer[20 + slot] = current[slot][0];
                }
                if self.phase_shift_enabled {
                    for slot in 0..4 {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for tap in 0..21 {
                            let sample = dc_buffer[slot + tap];
                            let cr = DC_B[tap];
                            let ci = DC_A[tap];
                            sum.re += sample.re * cr - sample.im * ci;
                            sum.im += sample.re * ci + sample.im * cr;
                        }
                        self.x[channel][0][group + slot] =
                            Complex64::new(2.0 * sum.re, 2.0 * sum.im);
                    }
                }
                self.surround_history[surround].copy_from_slice(&dc_buffer[4..24]);
                self.surround_delay[surround].copy_within(4..10, 0);
                for slot in 0..4 {
                    self.surround_delay[surround][6 + slot] = current[slot];
                }
            }
        }
    }

    /// Render this frame's metadata into `z`.
    ///
    /// A present object interpolates from its previous frame-end matrix, or
    /// hard-cuts to its first data point when the stream has no state for it yet
    /// (a new object must not fade in from silence). An object that disappears
    /// keeps publishing a zero matrix until it comes back.
    fn matrix_object(&mut self, object: usize, params: &JocParams) {
        self.z[object] = [[Complex64::new(0.0, 0.0); 24]; 64];
        let core_channels = usize::from(params.n_channels);
        if params.present_mask & (1 << object) == 0 {
            if self.matrix_previous_present[object] {
                self.matrix_previous[object][..core_channels].fill([0.0; 64]);
            }
            return;
        }
        let info = &params.objects[object];
        let bands = info.n_bands;
        let points = info.n_dpoints;
        let slope = info.slope_idx;
        let map = eac3joc::parameter_band_map(bands).expect("validated");
        let has_previous = self.matrix_previous_present[object];
        let half = TIMESLOTS / 2;
        for channel in 0..core_channels {
            for subband in 0..64 {
                let parameter_band = usize::from(map[subband]);
                let previous = if has_previous {
                    self.matrix_previous[object][channel][subband]
                } else {
                    0.0
                };
                let target0 = info.dq[0][channel][parameter_band];
                let target1 = if points == 2 {
                    info.dq[1][channel][parameter_band]
                } else {
                    target0
                };
                let mut last = target0;
                for timeslot in 0..TIMESLOTS {
                    let coefficient = if slope == 0 {
                        if !has_previous {
                            if points == 1 || timeslot < half {
                                target0
                            } else {
                                let alpha = (timeslot - (half - 1)) as f64 / half as f64;
                                target0 * (1.0 - alpha) + target1 * alpha
                            }
                        } else if points == 1 {
                            let alpha = (timeslot + 1) as f64 / TIMESLOTS as f64;
                            previous * (1.0 - alpha) + target0 * alpha
                        } else if timeslot < half {
                            let alpha = (timeslot + 1) as f64 / half as f64;
                            previous * (1.0 - alpha) + target0 * alpha
                        } else {
                            let alpha = (timeslot - (half - 1)) as f64 / half as f64;
                            target0 * (1.0 - alpha) + target1 * alpha
                        }
                    } else {
                        // Step mode switches targets at the raw offset slots.
                        let base = if has_previous { previous } else { target0 };
                        let mut value = if timeslot < usize::from(info.offset_ts[0]) {
                            base
                        } else {
                            target0
                        };
                        if points == 2 && timeslot >= usize::from(info.offset_ts[1]) {
                            value = target1;
                        }
                        value
                    };
                    let x = self.x[channel][subband][timeslot];
                    self.z[object][subband][timeslot].re += x.re * coefficient;
                    self.z[object][subband][timeslot].im += x.im * coefficient;
                    last = coefficient;
                }
                self.matrix_previous[object][channel][subband] = last;
            }
        }
        self.matrix_previous_present[object] = true;
    }

    /// Reuse the previous frame-end matrix for every timeslot.
    fn matrix_hold(&mut self, object: usize, core_channels: usize) {
        self.z[object] = [[Complex64::new(0.0, 0.0); 24]; 64];
        for channel in 0..core_channels {
            for subband in 0..64 {
                let previous = self.matrix_previous[object][channel][subband];
                if previous == 0.0 {
                    continue;
                }
                for timeslot in 0..TIMESLOTS {
                    let x = self.x[channel][subband][timeslot];
                    self.z[object][subband][timeslot].re += x.re * previous;
                    self.z[object][subband][timeslot].im += x.im * previous;
                }
            }
        }
    }

    fn synthesis_object(&mut self, object: usize, output_scale: f32, output: &mut [f32]) {
        let mut frequency = [Complex64::new(0.0, 0.0); 64];
        let mut rotated = [0.0f64; 128];
        let mut pcm64 = [0.0f64; 64];
        let destination = (object + 1) * FRAME_SAMPLES;
        for timeslot in 0..24 {
            for k in 0..32 {
                let even = self.z[object][2 * k][timeslot];
                let odd = self.z[object][2 * k + 1][timeslot];
                frequency[k] = Complex64::new(even.re, -even.im);
                frequency[63 - k] = odd;
            }
            self.fft.process(&mut frequency);
            for k in 0..64 {
                let re = frequency[k].re;
                let im = frequency[k].im;
                let sin = self.rotation_sin[k];
                let cos = self.rotation_cos[k];
                rotated[2 * k] = 2.0 * (re * cos + im * sin);
                rotated[2 * k + 1] = 2.0 * (im * cos - re * sin);
            }
            qmf5_step(&mut self.synthesis_state[object], &rotated, &mut pcm64);
            for sample in 0..64 {
                let clipped = clamp_unit(16.0 * pcm64[sample]);
                output[destination + timeslot * 64 + sample] = clipped as f32 * output_scale;
            }
        }
    }

    /// Render all object slots, including the ones this frame does not carry:
    /// the synthesis state has to advance with the matrix, or a returning object
    /// resumes from a stale filter.
    fn render_objects(
        &mut self,
        params: Option<&JocParams>,
        core_channels: usize,
        mode: FrameMode,
        output_scale: f32,
        output: &mut [f32],
    ) {
        for object in 0..MAX_OBJECTS {
            match mode {
                FrameMode::Normal => {
                    let params = params.expect("normal frames carry a complete payload");
                    self.matrix_object(object, params);
                }
                FrameMode::Hold => self.matrix_hold(object, core_channels),
                FrameMode::HardCut => self.z[object] = [[Complex64::new(0.0, 0.0); 24]; 64],
            }
            self.synthesis_object(object, output_scale, output);
        }
    }

    /// Core LFE: the full clipgain is applied as a sample-wise ramp before the
    /// 1217-sample delay, and the read-out is only clamped. The ramp is
    /// `old + n * (new - old) / 256` for the first 256 samples and `new`
    /// afterwards; a frame with the same gain is a plain multiply.
    fn render_lfe(
        &mut self,
        lfe: Option<&[f32]>,
        clipgain: f64,
        output_scale: f32,
        output: &mut [f32],
    ) {
        let previous = self.lfe_clipgain.unwrap_or(clipgain);
        let mut ramp = [0.0f64; RAMP_SAMPLES];
        for (index, value) in ramp.iter_mut().enumerate() {
            *value = if previous == clipgain {
                clipgain
            } else {
                previous + (index as f64) * (clipgain - previous) / RAMP_SAMPLES as f64
            };
        }
        self.lfe_clipgain = Some(clipgain);
        let sample_at = |index: usize| f64::from(lfe.map_or(0.0, |lfe| lfe[index]));
        for sample in 0..LFE_DELAY {
            output[sample] = clamp_unit(self.lfe_delay[sample]) as f32 * output_scale;
        }
        for sample in LFE_DELAY..FRAME_SAMPLES {
            let index = sample - LFE_DELAY;
            let gain = if index < RAMP_SAMPLES {
                ramp[index]
            } else {
                clipgain
            };
            output[sample] = clamp_unit(sample_at(index) * gain) as f32 * output_scale;
        }
        for sample in 0..LFE_DELAY {
            let index = sample + (FRAME_SAMPLES - LFE_DELAY);
            let gain = if index < RAMP_SAMPLES {
                ramp[index]
            } else {
                clipgain
            };
            self.lfe_delay[sample] = sample_at(index) * gain;
        }
    }
}

/// Reject a payload the kernel cannot index. Only a payload that will be
/// rendered reaches this check.
fn validate_params(params: &JocParams) -> Result<(), KernelError> {
    if !params.clipgain.is_finite() {
        return Err(KernelError::InvalidParameters(
            "clipgain must be finite".into(),
        ));
    }
    for object in 0..MAX_OBJECTS {
        if params.present_mask & (1 << object) == 0 {
            continue;
        }
        let info = &params.objects[object];
        if eac3joc::parameter_band_map(info.n_bands).is_none() {
            return Err(KernelError::InvalidParameters(format!(
                "unsupported parameter-band count {}",
                info.n_bands
            )));
        }
        if !(1..=2).contains(&info.n_dpoints) {
            return Err(KernelError::InvalidParameters(
                "n_dpoints must be 1 or 2".into(),
            ));
        }
        if info.slope_idx > 1 {
            return Err(KernelError::InvalidParameters(
                "slope_idx must be 0 or 1".into(),
            ));
        }
        if info.slope_idx == 1
            && info.offset_ts[..info.n_dpoints]
                .iter()
                .any(|offset| usize::from(*offset) >= TIMESLOTS)
        {
            return Err(KernelError::InvalidParameters(
                "offset_ts leaves the timeslot range".into(),
            ));
        }
    }
    Ok(())
}

fn qmf5_step(state: &mut [f64; 640], rotated: &[f64; 128], output: &mut [f64; 64]) {
    for block in 0..16 {
        for lane in 0..4 {
            let sample = block * 4 + lane;
            let state_base = block * 36 + lane;
            let even = rotated[block * 8 + lane * 2];
            let odd = rotated[block * 8 + lane * 2 + 1];
            output[sample] = 2.0 * (tables::KQMF5_WINDOW[sample] * even + state[state_base]);

            state[state_base] = tables::KQMF5_WINDOW[64 + sample] * odd + state[state_base + 4];
            for slot in 0..7 {
                let alternating = if slot & 1 == 0 { even } else { odd };
                state[state_base + (slot + 1) * 4] = tables::KQMF5_WINDOW[(slot + 2) * 64 + sample]
                    * alternating
                    + state[state_base + (slot + 2) * 4];
            }
            state[state_base + 8 * 4] = tables::KQMF5_WINDOW[9 * 64 + sample] * odd;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eac3joc::{JocObject, JocPayloadStatus};

    fn silent_object() -> JocObject {
        JocObject::default()
    }

    /// One present object whose matrix is `matrix` in every core channel and
    /// band, with a single data point.
    fn object_with(matrix: f64) -> JocObject {
        let mut object = JocObject {
            present: true,
            quant_idx: 1,
            n_bands: 1,
            n_dpoints: 1,
            ..silent_object()
        };
        for data_point in &mut object.dq {
            for channel in data_point {
                channel.fill(matrix);
            }
        }
        object
    }

    /// A complete payload for `core_channels` with the given objects.
    fn params(core_channels: u8, objects: &[(usize, f64)], seq_count: u16) -> JocParams {
        let mut params = JocParams {
            dmx_config_idx: if core_channels == 5 { 3 } else { 4 },
            num_objects_bits: 0,
            ext_config_idx: 0,
            n_channels: core_channels,
            n_objects: MAX_OBJECTS as u8,
            objects_clamped: false,
            clipgain_x_bits: 4,
            clipgain_y_bits: 0,
            seq_count,
            clipgain: 1.0,
            objects: [JocObject::default(); MAX_OBJECTS],
            present_mask: 0,
            data_end_bits: 0,
            trailing_bits: 0,
            status: JocPayloadStatus::Complete,
        };
        for (slot, matrix) in objects {
            params.objects[*slot] = object_with(*matrix);
            params.present_mask |= 1 << slot;
        }
        params
    }

    fn silent_bed(core_channels: usize) -> Vec<f32> {
        vec![0.0f32; core_channels * FRAME_SAMPLES]
    }

    /// A bed whose first core channel carries a constant level, so the analysis
    /// and synthesis stages have something to move through `z`.
    fn constant_bed(core_channels: usize, level: f32) -> Vec<f32> {
        let mut bed = silent_bed(core_channels);
        bed[..FRAME_SAMPLES].fill(level);
        bed
    }

    fn energy(output: &[f32], channel: usize) -> f64 {
        output[channel * FRAME_SAMPLES..(channel + 1) * FRAME_SAMPLES]
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum()
    }

    #[test]
    fn silence_produces_silence() {
        let mut kernel = JocKernel::new();
        let bed = silent_bed(CORE_CHANNELS);
        let lfe = vec![0.0f32; FRAME_SAMPLES];
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::absent(CORE_CHANNELS, None),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn rejects_a_bad_core_channel_count() {
        let mut kernel = JocKernel::new();
        let bed = vec![0.0f32; 6 * FRAME_SAMPLES];
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        assert_eq!(
            kernel.process_frame(&bed, None, &JocFrame::absent(6, None), 1.0, &mut output),
            Err(KernelError::InvalidCoreChannels(6))
        );
    }

    #[test]
    fn lfe_is_delayed_by_1217_samples() {
        let mut kernel = JocKernel::new();
        let bed = silent_bed(CORE_CHANNELS);
        let mut lfe = vec![0.0f32; FRAME_SAMPLES];
        lfe[0] = 0.5;
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::absent(CORE_CHANNELS, None),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!(output[..LFE_DELAY].iter().all(|value| *value == 0.0));
        assert!((output[LFE_DELAY] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_lfe_is_scaled_by_the_clipgain() {
        let mut kernel = JocKernel::new();
        let bed = silent_bed(CORE_CHANNELS);
        let mut lfe = vec![0.0f32; FRAME_SAMPLES];
        // Past the 256-sample ramp, so the gain is the frame's clipgain.
        lfe[300] = 0.25;
        let mut params = params(CORE_CHANNELS as u8, &[], 1);
        params.clipgain = 2.0;
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&params),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!((output[LFE_DELAY + 300] - 0.5).abs() < 1e-6);
        // The ramp start is the same gain on the first frame after a reset.
        assert!((output[LFE_DELAY + 1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn the_lfe_clipgain_ramps_over_256_samples() {
        let mut kernel = JocKernel::new();
        let bed = silent_bed(CORE_CHANNELS);
        let lfe = vec![1.0f32; FRAME_SAMPLES];
        let mut unity = params(CORE_CHANNELS as u8, &[], 1);
        unity.clipgain = 1.0;
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&unity),
                1.0,
                &mut output,
            )
            .unwrap();
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&unity),
                1.0,
                &mut output,
            )
            .unwrap();
        // A frame that halves the clipgain ramps the LFE down linearly.
        let mut quieter = params(CORE_CHANNELS as u8, &[], 2);
        quieter.clipgain = 0.5;
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&quieter),
                1.0,
                &mut output,
            )
            .unwrap();
        for index in [0usize, 64, 128, 255] {
            let expected = 1.0 - (index as f64) * 0.5 / RAMP_SAMPLES as f64;
            let actual = f64::from(output[LFE_DELAY + index]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "sample {index}: {actual} != {expected}"
            );
        }
        assert!((f64::from(output[LFE_DELAY + RAMP_SAMPLES]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_first_appearance_hard_cuts_to_the_target() {
        let mut kernel = JocKernel::new();
        kernel.x = [[[Complex64::new(0.0, 0.0); TIMESLOTS]; 64]; MAX_CORE_CHANNELS];
        for slot in &mut kernel.x[0] {
            slot.fill(Complex64::new(1.0, 0.0));
        }
        let params = params(CORE_CHANNELS as u8, &[(0, 1.0)], 1);
        kernel.matrix_object(0, &params);
        for timeslot in 0..TIMESLOTS {
            assert!((kernel.z[0][0][timeslot].re - 1.0).abs() < 1e-12);
        }
        assert!(kernel.matrix_previous_present[0]);
    }

    #[test]
    fn a_known_previous_state_interpolates_over_the_frame() {
        let mut kernel = JocKernel::new();
        kernel.x = [[[Complex64::new(0.0, 0.0); TIMESLOTS]; 64]; MAX_CORE_CHANNELS];
        for slot in &mut kernel.x[0] {
            slot.fill(Complex64::new(1.0, 0.0));
        }
        // Establish zero as the previous state.
        let zero = params(CORE_CHANNELS as u8, &[(0, 0.0)], 1);
        kernel.matrix_object(0, &zero);
        // Then fade to one over the frame.
        let one = params(CORE_CHANNELS as u8, &[(0, 1.0)], 2);
        kernel.matrix_object(0, &one);
        for timeslot in 0..TIMESLOTS {
            let expected = (timeslot + 1) as f64 / TIMESLOTS as f64;
            assert!(
                (kernel.z[0][0][timeslot].re - expected).abs() < 1e-12,
                "timeslot {timeslot}"
            );
        }
    }

    #[test]
    fn hold_reuses_the_previous_matrix() {
        let mut kernel = JocKernel::new();
        kernel.x = [[[Complex64::new(0.0, 0.0); TIMESLOTS]; 64]; MAX_CORE_CHANNELS];
        for slot in &mut kernel.x[0] {
            slot.fill(Complex64::new(1.0, 0.0));
        }
        let params = params(CORE_CHANNELS as u8, &[(0, 0.5)], 1);
        kernel.matrix_object(0, &params);
        kernel.matrix_hold(0, CORE_CHANNELS);
        for timeslot in 0..TIMESLOTS {
            assert!((kernel.z[0][0][timeslot].re - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn a_disappearing_object_publishes_a_zero_matrix() {
        let mut kernel = JocKernel::new();
        kernel.x = [[[Complex64::new(0.0, 0.0); TIMESLOTS]; 64]; MAX_CORE_CHANNELS];
        for slot in &mut kernel.x[0] {
            slot.fill(Complex64::new(1.0, 0.0));
        }
        let present = params(CORE_CHANNELS as u8, &[(0, 1.0)], 1);
        kernel.matrix_object(0, &present);
        let absent = params(CORE_CHANNELS as u8, &[], 2);
        kernel.matrix_object(0, &absent);
        assert_eq!(kernel.matrix_previous[0][0], [0.0; 64]);
        // Coming back now fades in from zero rather than jumping.
        kernel.matrix_object(0, &params(CORE_CHANNELS as u8, &[(0, 1.0)], 3));
        assert!((kernel.z[0][0][0].re - 1.0 / TIMESLOTS as f64).abs() < 1e-12);
    }

    /// Render three frames of a constant bed with one object at a unit matrix and
    /// return the last frame's object plane. The third frame carries `third`
    /// instead of a complete payload.
    fn render_third_frame(third: Option<&JocParams>) -> Vec<f32> {
        let mut kernel = JocKernel::new();
        let bed = constant_bed(CORE_CHANNELS, 0.5);
        let lfe = vec![0.0f32; FRAME_SAMPLES];
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        for seq_count in [1u16, 2] {
            let frame = params(CORE_CHANNELS as u8, &[(0, 1.0)], seq_count);
            kernel
                .process_frame(
                    &bed,
                    Some(&lfe),
                    &JocFrame::payload(&frame),
                    1.0,
                    &mut output,
                )
                .unwrap();
        }
        let frame = match third {
            Some(params) => JocFrame::payload(params),
            None => JocFrame::absent(CORE_CHANNELS, None),
        };
        kernel
            .process_frame(&bed, Some(&lfe), &frame, 1.0, &mut output)
            .unwrap();
        output[FRAME_SAMPLES..2 * FRAME_SAMPLES].to_vec()
    }

    /// Largest sample difference relative to the reference peak.
    fn mismatch(actual: &[f32], reference: &[f32]) -> f32 {
        let peak = reference
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        let worst = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst / peak.max(f32::MIN_POSITIVE)
    }

    #[test]
    fn a_frame_without_metadata_holds_the_previous_object() {
        let steady = render_third_frame(Some(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 3)));
        let held = render_third_frame(None);
        assert!(
            held.iter().any(|sample| *sample != 0.0),
            "a frame without metadata must not drop the object"
        );
        assert!(
            mismatch(&held, &steady) < 1e-6,
            "hold differs from the normal path: {}",
            mismatch(&held, &steady)
        );
    }

    #[test]
    fn a_damaged_payload_holds_the_previous_object() {
        let steady = render_third_frame(Some(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 3)));
        let mut damaged = params(CORE_CHANNELS as u8, &[(0, 1.0)], 3);
        damaged.status = JocPayloadStatus::Truncated;
        let held = render_third_frame(Some(&damaged));
        assert!(held.iter().any(|sample| *sample != 0.0));
        assert!(
            mismatch(&held, &steady) < 1e-6,
            "a damaged payload differs from the hold path: {}",
            mismatch(&held, &steady)
        );
    }

    #[test]
    fn a_bad_offset_ts_payload_holds_instead_of_failing() {
        let steady = render_third_frame(Some(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 3)));
        let mut damaged = params(CORE_CHANNELS as u8, &[(0, 1.0)], 3);
        damaged.status = JocPayloadStatus::BadOffsetTs;
        damaged.objects[0].slope_idx = 1;
        damaged.objects[0].offset_ts = [TIMESLOTS as u8, 0];
        let held = render_third_frame(Some(&damaged));
        assert!(mismatch(&held, &steady) < 1e-6);
    }

    #[test]
    fn a_frame_before_the_first_payload_is_silent() {
        let mut kernel = JocKernel::new();
        let bed = constant_bed(CORE_CHANNELS, 0.5);
        let lfe = vec![0.0f32; FRAME_SAMPLES];
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::absent(CORE_CHANNELS, None),
                1.0,
                &mut output,
            )
            .unwrap();
        for channel in 1..OUTPUT_CHANNELS {
            assert_eq!(energy(&output, channel), 0.0, "channel {channel}");
        }
    }

    #[test]
    fn a_sequence_jump_hard_cuts_and_the_next_frame_fades_in() {
        let mut kernel = JocKernel::new();
        let bed = constant_bed(CORE_CHANNELS, 0.5);
        let lfe = vec![0.0f32; FRAME_SAMPLES];
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 1)),
                1.0,
                &mut output,
            )
            .unwrap();
        // A gap in the counter restarts the programme: one silent frame.
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 9)),
                1.0,
                &mut output,
            )
            .unwrap();
        assert_eq!(energy(&output, 1), 0.0);
        assert!(kernel.matrix_previous_present[0]);
        assert_eq!(kernel.matrix_previous[0][0], [0.0; 64]);
    }

    #[test]
    fn a_core_channel_count_change_hard_cuts_and_clears_the_lfe_ring() {
        let mut kernel = JocKernel::new();
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        let five = silent_bed(CORE_CHANNELS);
        let mut lfe = vec![0.0f32; FRAME_SAMPLES];
        lfe[FRAME_SAMPLES - 1] = 0.75;
        kernel
            .process_frame(
                &five,
                Some(&lfe),
                &JocFrame::payload(&params(CORE_CHANNELS as u8, &[(0, 1.0)], 1)),
                1.0,
                &mut output,
            )
            .unwrap();
        // A seven-channel frame changes the layout: hard cut, ring cleared.
        let seven = silent_bed(MAX_CORE_CHANNELS);
        let silence = vec![0.0f32; FRAME_SAMPLES];
        kernel
            .process_frame(
                &seven,
                Some(&silence),
                &JocFrame::payload(&params(MAX_CORE_CHANNELS as u8, &[(0, 1.0)], 2)),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!(output.iter().all(|sample| *sample == 0.0));
        assert_eq!(kernel.lfe_delay, [0.0; LFE_DELAY]);
        assert_eq!(kernel.previous_core_channels, Some(MAX_CORE_CHANNELS));
    }

    #[test]
    fn a_held_frame_does_not_retune_the_clipgain() {
        let mut kernel = JocKernel::new();
        let bed = silent_bed(CORE_CHANNELS);
        let lfe = vec![0.0f32; FRAME_SAMPLES];
        let mut loud = params(CORE_CHANNELS as u8, &[], 1);
        loud.clipgain = 2.0;
        let mut output = vec![0.0f32; OUTPUT_CHANNELS * FRAME_SAMPLES];
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::payload(&loud),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!((kernel.held_clipgain() - 2.0).abs() < 1e-12);
        kernel
            .process_frame(
                &bed,
                Some(&lfe),
                &JocFrame::absent(CORE_CHANNELS, None),
                1.0,
                &mut output,
            )
            .unwrap();
        assert!((kernel.held_clipgain() - 2.0).abs() < 1e-12);
    }
}
