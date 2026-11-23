//! Deterministic terrain lifecycle correctness harness.
//!
//! This drives an existing typed `.mccap` lifecycle capture through the ordinary client and
//! render shell.  It deliberately does not select `diagnostic_simulation=frozen_steady_state`:
//! the wrapper supplies a fixed 20 Hz simulation clock while the client is out of its loading
//! phase, and freezes simulation while assets are loading.  Every checkpoint is an exact replay
//! virtual tick; readiness failure or a skipped checkpoint stops the run immediately.
//! Pass `--variant base` (the default) for the ten-point contract, or `--variant gap` for
//! the unloaded-Hx extension through ticks 1401 and 1521. Pass `--mesh async` to mesh on the live
//! client's worker path instead of the default synchronous replay path.
//!
//! This is a correctness diagnostic, not a benchmark and not a parity or image qualification.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mc_client::{Client, ClientConfig, Phase};
use mc_math::BlockPos;
use mc_render::frame::telemetry::FrameWorkStats;
use mc_render::shell::DiagnosticClock;
use mc_render::{
    AcquiredFrame, Camera, FrameCtx, GameSettings, Gpu, PhaseList, Shell, ShellApp, ShellConfig,
    TickCtx,
};
use serde_json::{Value, json};

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
const STEP_MICROS: u64 = 50_000;
const MAX_FRAMES: u64 = 4096;
const MAX_DURATION_SECS: u64 = 90;
const SECTION_Y: i32 = 4;
const GAP_CHECKPOINT_TICK: u64 = 1_401;
const GAP_FINAL_CHECKPOINT_TICK: u64 = 1_521;

const COLUMNS: [(i32, i32); 24] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (8, 0),
    (9, 0),
    (8, 1),
    (9, 1),
    (0, 8),
    (1, 8),
    (0, 9),
    (1, 9),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
];

const BASE_CHECKPOINTS: [CheckpointSpec; 10] = [
    CheckpointSpec::new(200, "initial_A_plus_x", 24, &[[8, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(320, "turn_A_plus_z", 24, &[[0, SECTION_Y, 8]], &[]),
    CheckpointSpec::new(440, "teleport_B_minus_x", 24, &[[0, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(
        560,
        "unload_B",
        20,
        &[[0, SECTION_Y, 0]],
        &[[8, SECTION_Y, 0]],
    ),
    CheckpointSpec::new(680, "reload_B", 24, &[[0, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(800, "block_edit_round_trip", 24, &[[0, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(920, "teleport_C_minus_z", 24, &[[0, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(1040, "unload_A", 20, &[], &[[0, SECTION_Y, 0]]),
    CheckpointSpec::new(1160, "reload_A", 24, &[[0, SECTION_Y, 0]], &[]),
    CheckpointSpec::new(1280, "final_C_minus_z", 24, &[[0, SECTION_Y, 0]], &[]),
];

const GAP_CHECKPOINTS: [CheckpointSpec; 12] = [
    BASE_CHECKPOINTS[0],
    BASE_CHECKPOINTS[1],
    BASE_CHECKPOINTS[2],
    BASE_CHECKPOINTS[3],
    BASE_CHECKPOINTS[4],
    BASE_CHECKPOINTS[5],
    BASE_CHECKPOINTS[6],
    BASE_CHECKPOINTS[7],
    BASE_CHECKPOINTS[8],
    BASE_CHECKPOINTS[9],
    CheckpointSpec::new(
        GAP_CHECKPOINT_TICK,
        "unload_Hx",
        23,
        &[[8, SECTION_Y, 0]],
        &[],
    ),
    CheckpointSpec::new(
        GAP_FINAL_CHECKPOINT_TICK,
        "reload_Hx",
        24,
        &[[8, SECTION_Y, 0]],
        &[],
    ),
];

#[test]
fn unload_checkpoints_remove_every_section_of_all_four_island_columns() {
    for spec in BASE_CHECKPOINTS {
        let resident = LifecycleDiagnostic::expected_resident_sections(spec);
        assert_eq!(resident.len(), spec.resident_columns * 16);
        for x in 8..=9 {
            for z in 0..=1 {
                for y in 0..16 {
                    assert_eq!(resident.contains(&(x, y, z)), spec.tick != 560);
                }
            }
        }
        for x in 0..=1 {
            for z in 0..=1 {
                for y in 0..16 {
                    assert_eq!(resident.contains(&(x, y, z)), spec.tick != 1040);
                }
            }
        }
    }
}

#[test]
fn gap_checkpoint_removes_exactly_the_hx_column() {
    let unload = GAP_CHECKPOINTS
        .iter()
        .find(|spec| spec.tick == GAP_CHECKPOINT_TICK)
        .copied()
        .expect("gap unload checkpoint");
    let resident = LifecycleDiagnostic::expected_resident_sections(unload);
    assert_eq!(resident.len(), 368);
    for y in 0..16 {
        assert!(!resident.contains(&(4, y, 0)));
    }
    let final_checkpoint = GAP_CHECKPOINTS
        .iter()
        .find(|spec| spec.tick == GAP_FINAL_CHECKPOINT_TICK)
        .copied()
        .expect("gap reload checkpoint");
    assert_eq!(
        LifecycleDiagnostic::expected_resident_sections(final_checkpoint).len(),
        384
    );
}

#[derive(Clone, Copy)]
struct CheckpointSpec {
    tick: u64,
    label: &'static str,
    resident_columns: usize,
    required_visible: &'static [[i32; 3]],
    forbidden_visible: &'static [[i32; 3]],
}

impl CheckpointSpec {
    const fn new(
        tick: u64,
        label: &'static str,
        resident_columns: usize,
        required_visible: &'static [[i32; 3]],
        forbidden_visible: &'static [[i32; 3]],
    ) -> Self {
        Self {
            tick,
            label,
            resident_columns,
            required_visible,
            forbidden_visible,
        }
    }
}

struct LifecycleDiagnostic {
    client: Client,
    output: PathBuf,
    checkpoints: &'static [CheckpointSpec],
    checkpoint_index: usize,
    pending_checkpoint: Option<CheckpointSpec>,
    captured_this_frame: bool,
    complete: bool,
    failure: Option<String>,
    last_virtual_micros: Option<u64>,
    camera: Option<Camera>,
    physical_viewport: Option<(u32, u32)>,
    viewport_locked: bool,
}

impl LifecycleDiagnostic {
    fn source_micros(&self) -> Option<u64> {
        self.client
            .session()
            .and_then(|session| session.replay_source())
            .map(|source| source.virtual_micros())
    }

    fn set_failure(&mut self, reason: impl Into<String>) {
        if self.failure.is_none() {
            self.failure = Some(reason.into());
        }
    }

    fn observe_source_clock(&mut self) {
        let Some(micros) = self.source_micros() else {
            return;
        };
        if micros % STEP_MICROS != 0 {
            self.set_failure(format!(
                "replay source virtual clock is not a 20 Hz boundary: {micros}"
            ));
            return;
        }
        if let Some(previous) = self.last_virtual_micros {
            if micros < previous {
                self.set_failure(format!(
                    "replay source virtual clock regressed: {previous} -> {micros}"
                ));
                return;
            }
            if micros > previous && micros != previous.saturating_add(STEP_MICROS) {
                self.set_failure(format!(
                    "replay source virtual clock skipped a step: {previous} -> {micros}"
                ));
                return;
            }
        }
        self.last_virtual_micros = Some(micros);

        let Some(spec) = self.checkpoints.get(self.checkpoint_index).copied() else {
            self.complete = true;
            return;
        };
        let target = spec.tick.saturating_mul(STEP_MICROS);
        if micros > target {
            self.set_failure(format!(
                "checkpoint {} was skipped: source reached tick {}",
                spec.label,
                micros / STEP_MICROS
            ));
        } else if micros == target {
            if self.pending_checkpoint.is_some() {
                self.set_failure("checkpoint remained pending across a rendered frame");
            } else {
                self.pending_checkpoint = Some(spec);
            }
        }
    }

    fn expected_resident_sections(spec: CheckpointSpec) -> BTreeSet<(i32, i32, i32)> {
        COLUMNS
            .iter()
            .copied()
            .filter(|&(x, z)| match spec.tick {
                560 => !((8..=9).contains(&x) && (0..=1).contains(&z)),
                1040 => !((0..=1).contains(&x) && (0..=1).contains(&z)),
                1401 => !(x == 4 && z == 0),
                _ => true,
            })
            .flat_map(|(x, z)| (0..16).map(move |y| (x, y, z)))
            .collect()
    }

    fn validate_checkpoint(&self, spec: CheckpointSpec) -> Result<Value> {
        let expected_sections = Self::expected_resident_sections(spec);
        let session = self
            .client
            .session()
            .context("checkpoint has no client session")?;
        if !matches!(session.phase(), Phase::Playing) {
            bail!("checkpoint {} reached outside Playing phase", spec.label);
        }
        let source = session
            .replay_source()
            .context("checkpoint has no replay source")?;
        if source.virtual_micros() != spec.tick * STEP_MICROS {
            bail!(
                "checkpoint {} is not an exact replay virtual tick",
                spec.label
            );
        }
        let predictor = session
            .predictor()
            .context("checkpoint has no local physics predictor")?;
        let (expected_x, expected_y, expected_z, expected_yaw) = expected_pose(spec.tick);
        let position = predictor.state.pos;
        if !position.is_finite()
            || !predictor.state.yaw.is_finite()
            || !predictor.state.pitch.is_finite()
            || (position.x - expected_x).abs() > 0.001
            || (position.y - expected_y).abs() > 0.001
            || (position.z - expected_z).abs() > 0.001
            || (predictor.state.yaw - expected_yaw).abs() > 0.01
            || predictor.state.pitch.abs() > 0.01
        {
            bail!(
                "checkpoint {} predictor pose mismatch: position={position:?} yaw={} pitch={}",
                spec.label,
                predictor.state.yaw,
                predictor.state.pitch
            );
        }

        let world = &session.state().world;
        let loaded_columns: BTreeSet<_> = world
            .loaded_chunks()
            .map(|(position, _)| (position.x, position.z))
            .collect();
        let expected_columns: BTreeSet<_> =
            expected_sections.iter().map(|&(x, _, z)| (x, z)).collect();
        if loaded_columns != expected_columns || loaded_columns.len() != spec.resident_columns {
            bail!(
                "checkpoint {} loaded-column identity mismatch: got {}, expected {}",
                spec.label,
                loaded_columns.len(),
                expected_columns.len()
            );
        }
        for (position, chunk) in world.loaded_chunks() {
            if chunk.queued_light_checks() != 4096 || chunk.is_gap_lighting_updated() {
                bail!(
                    "checkpoint {} has incomplete light state at {position:?}",
                    spec.label
                );
            }
        }

        let (resident, visible, ready_upload_count) = self
            .client
            .diagnostic_terrain_identity()
            .context("checkpoint has no terrain identity")?;
        let resident_set: BTreeSet<_> = resident.iter().copied().collect();
        let visible_set: BTreeSet<_> = visible.iter().copied().collect();
        if resident_set.len() != resident.len() {
            bail!(
                "checkpoint {} resident section identity contains duplicates",
                spec.label
            );
        }
        if visible_set.len() != visible.len() {
            bail!(
                "checkpoint {} visible section identity contains duplicates",
                spec.label
            );
        }
        if resident_set != expected_sections {
            bail!(
                "checkpoint {} resident section identity mismatch",
                spec.label
            );
        }
        if !visible_set.is_subset(&resident_set) {
            bail!(
                "checkpoint {} visible sections are not a resident subset",
                spec.label
            );
        }
        for sentinel in spec.required_visible {
            if !visible_set.contains(&(sentinel[0], sentinel[1], sentinel[2])) {
                bail!(
                    "checkpoint {} required remote sentinel is not visible: {sentinel:?}",
                    spec.label
                );
            }
        }
        for sentinel in spec.forbidden_visible {
            if visible_set.contains(&(sentinel[0], sentinel[1], sentinel[2])) {
                bail!(
                    "checkpoint {} forbidden sentinel remains visible: {sentinel:?}",
                    spec.label
                );
            }
        }
        if ready_upload_count != 0
            || self.client.mesh_queue_len() != 0
            || self.client.mesh_queue_deferred_len() != 0
            || self.client.mesh_queue_in_flight_len() != 0
        {
            bail!("checkpoint {} has pending terrain work", spec.label);
        }
        if self.client.uploaded_section_count() != expected_sections.len() {
            bail!(
                "checkpoint {} uploaded section identity mismatch",
                spec.label
            );
        }
        if matches!(spec.tick, 680 | 800) {
            let block_position = BlockPos::new(8 * 16, 64, 0);
            let block = world.block_at(block_position);
            if block != mc_data::BlockState::from_raw(1 << 4) {
                bail!(
                    "checkpoint {} block edit did not restore stone at {block_position:?}: got {block:?}",
                    spec.label,
                );
            }
        }
        if !predictor.state.flying {
            bail!(
                "checkpoint {} predictor lost the scripted flying ability",
                spec.label
            );
        }
        let work = self.client.diagnostic_work();
        let light = light_snapshot(world);
        let camera = self.camera.context("checkpoint has no finalized camera")?;
        Ok(json!({
            "schema": "mc-rs.optimization.lifecycle-checkpoint",
            "schema_version": 1,
            "status": "UNQUALIFIED_DIAGNOSTIC",
            "scoring_eligible": false,
            "phase": "diagnostic_readback",
            "readback_frame_eligible_for_timing": false,
            "tick": spec.tick,
            "label": spec.label,
            "camera": {
                "position": [camera.pos.x, camera.pos.y, camera.pos.z],
                "yaw": camera.yaw,
                "pitch": camera.pitch,
                "view": camera.view.to_cols_array(),
                "projection": camera.proj.to_cols_array(),
            },
            "physical_viewport": self.physical_viewport.map(|(width, height)| [width, height]),
            "fov": 70,
            "render_distance_chunks": 10,
            "replay_step_micros": STEP_MICROS,
            "work_at_readback": work_json(&work),
            "terrain_identity_at_readback": {
                "resident_sections": resident,
                "visible_sections": visible,
                "ready_upload_count": ready_upload_count,
            },
            "world_light_at_readback": light,
            "visibility_contract": {
                "required_remote_visible_sentinels": spec.required_visible,
                "forbidden_visible_sentinels": spec.forbidden_visible,
                "corridor_residency": "asserted by the exact resident section identity",
                "self_visibility_assertions": "none",
            },
        }))
    }

    fn failed_checkpoint_record(
        &self,
        spec: CheckpointSpec,
        reason: &str,
        image_captured_before_validation: bool,
    ) -> Value {
        let terrain = match self.client.diagnostic_terrain_identity() {
            Some((resident, visible, ready_upload_count)) => json!({
                "resident_sections": resident,
                "visible_sections": visible,
                "ready_upload_count": ready_upload_count,
                "mesh_pending": self.client.mesh_queue_len(),
                "mesh_deferred": self.client.mesh_queue_deferred_len(),
                "mesh_in_flight": self.client.mesh_queue_in_flight_len(),
            }),
            None => json!({"error": "terrain identity unavailable"}),
        };
        let (loaded_columns, light) = match self.client.session() {
            Some(session) => {
                let world = &session.state().world;
                let mut columns = world
                    .loaded_chunks()
                    .map(|(position, _)| [position.x, position.z])
                    .collect::<Vec<_>>();
                columns.sort_unstable();
                (json!(columns), light_snapshot(world))
            }
            None => (Value::Null, Value::Null),
        };
        let camera = self.camera.map(|camera| {
            json!({
                "position": [camera.pos.x, camera.pos.y, camera.pos.z],
                "yaw": camera.yaw,
                "pitch": camera.pitch,
                "view": camera.view.to_cols_array(),
                "projection": camera.proj.to_cols_array(),
            })
        });
        json!({
            "schema": "mc-rs.optimization.lifecycle-checkpoint",
            "schema_version": 1,
            "status": "FAILED_DIAGNOSTIC",
            "scoring_eligible": false,
            "phase": "diagnostic_failed",
            "readback_frame_eligible_for_timing": false,
            "tick": spec.tick,
            "label": spec.label,
            "validation_error": reason,
            "image": format!("checkpoints/{:04}-{}.png", spec.tick, spec.label),
            "image_captured_before_validation": image_captured_before_validation,
            "source_virtual_micros": self.source_micros(),
            "camera": camera,
            "physical_viewport": self.physical_viewport.map(|(width, height)| [width, height]),
            "loaded_columns_observed": loaded_columns,
            "terrain_identity_observed": terrain,
            "work_at_failure": work_json(&self.client.diagnostic_work()),
            "world_light_at_failure": light,
        })
    }

    fn record_failed_checkpoint(&self, spec: CheckpointSpec, error: anyhow::Error) -> Result<()> {
        self.record_failed_checkpoint_with_image(spec, error, false)
    }

    fn record_failed_checkpoint_with_image(
        &self,
        spec: CheckpointSpec,
        error: anyhow::Error,
        image_captured_before_validation: bool,
    ) -> Result<()> {
        let reason = error.to_string();
        let mut bytes = serde_json::to_vec_pretty(&self.failed_checkpoint_record(
            spec,
            &reason,
            image_captured_before_validation,
        ))
        .context("encode failed checkpoint JSON")?;
        bytes.push(b'\n');
        match write_exclusive(
            &self
                .output
                .join("checkpoints")
                .join(format!("{:04}-{}-FAILED.json", spec.tick, spec.label)),
            &bytes,
        ) {
            Ok(()) => Err(error),
            Err(report_error) => {
                Err(error.context(format!("failed to write failure evidence: {report_error}")))
            }
        }
    }

    fn capture_checkpoint(
        &mut self,
        gpu: &Gpu,
        frame: &AcquiredFrame,
        spec: CheckpointSpec,
    ) -> Result<()> {
        let pixels = match frame
            .read_pixels(gpu)
            .context("checkpoint pixel readback failed")
        {
            Ok(pixels) => pixels,
            Err(error) => return self.record_failed_checkpoint(spec, error),
        };
        let Some((width, height)) = self.physical_viewport else {
            return self.record_failed_checkpoint(
                spec,
                anyhow::anyhow!("checkpoint has no physical viewport"),
            );
        };
        if (width, height) != (WIDTH, HEIGHT) {
            return self.record_failed_checkpoint(
                spec,
                anyhow::anyhow!(
                    "checkpoint viewport is not the pinned {WIDTH}x{HEIGHT}: {width}x{height}"
                ),
            );
        }
        if pixels.len() != width as usize * height as usize * 4 {
            return self.record_failed_checkpoint(
                spec,
                anyhow::anyhow!("checkpoint pixel readback has unexpected byte length"),
            );
        }
        let image = match image::RgbaImage::from_raw(width, height, pixels) {
            Some(image) => image,
            None => {
                return self.record_failed_checkpoint(
                    spec,
                    anyhow::anyhow!("checkpoint pixels were not RGBA8"),
                );
            }
        };
        let mut png = Vec::new();
        if let Err(error) = image::DynamicImage::ImageRgba8(image)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .context("encode checkpoint PNG")
        {
            return self.record_failed_checkpoint(spec, error);
        }
        let stem = format!("{:04}-{}", spec.tick, spec.label);
        if let Err(error) = write_exclusive(
            &self.output.join("checkpoints").join(format!("{stem}.png")),
            &png,
        ) {
            return self.record_failed_checkpoint(spec, error);
        }
        let record = match self.validate_checkpoint(spec) {
            Ok(record) => record,
            Err(error) => return self.record_failed_checkpoint_with_image(spec, error, true),
        };
        let mut json_bytes =
            serde_json::to_vec_pretty(&record).context("encode checkpoint JSON")?;
        json_bytes.push(b'\n');
        write_exclusive(
            &self.output.join("checkpoints").join(format!("{stem}.json")),
            &json_bytes,
        )?;
        Ok(())
    }
}

impl ShellApp for LifecycleDiagnostic {
    fn init(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        self.physical_viewport = Some((gpu.config.width, gpu.config.height));
        self.client.init(gpu, phases);
    }

    fn before_ticks(&mut self, camera: &mut Camera) {
        self.client.before_ticks(camera);
    }

    fn tick(&mut self, ctx: &mut TickCtx<'_>) {
        ShellApp::tick(&mut self.client, ctx);
    }

    fn frame(&mut self, ctx: &mut FrameCtx<'_>) {
        self.captured_this_frame = false;
        ShellApp::frame(&mut self.client, ctx);
        self.observe_source_clock();
    }

    fn diagnostic_phase(&self) -> &'static str {
        if self.captured_this_frame {
            "diagnostic_readback"
        } else if self.failure.is_some() {
            "diagnostic_failed"
        } else if self.client.is_loading() {
            "loading"
        } else {
            "lifecycle"
        }
    }

    fn diagnostic_clock(&self) -> Option<DiagnosticClock> {
        if let Some(DiagnosticClock::Invalid(reason)) = ShellApp::diagnostic_clock(&self.client) {
            return Some(DiagnosticClock::Invalid(reason));
        }
        if self.client.is_loading() || self.client.session().is_none() {
            Some(DiagnosticClock::Frozen)
        } else {
            Some(DiagnosticClock::Step {
                micros: STEP_MICROS,
            })
        }
    }

    fn begin_diagnostic_frame(&mut self) {
        self.client.begin_diagnostic_frame();
    }

    fn diagnostic_work(&self) -> FrameWorkStats {
        self.client.diagnostic_work()
    }

    fn finalize_camera(&mut self, camera: &mut Camera) {
        self.client.finalize_camera(camera);
        self.camera = Some(*camera);
    }

    fn after_render(&mut self, gpu: &Gpu, frame: &AcquiredFrame) {
        self.client.after_render(gpu, frame);
        if let Some(spec) = self.pending_checkpoint.take() {
            match self.capture_checkpoint(gpu, frame, spec) {
                Ok(()) => {
                    self.checkpoint_index += 1;
                    self.captured_this_frame = true;
                    self.complete = self.checkpoint_index == self.checkpoints.len();
                    self.viewport_locked = true;
                }
                Err(error) => {
                    self.set_failure(format!("checkpoint {} failed: {error}", spec.label))
                }
            }
        }
    }

    fn on_key_event(&mut self, event: &winit::event::KeyEvent, sequence: u64) {
        self.client.on_key_event(event, sequence);
    }

    fn on_focus_changed(&mut self, focused: bool) {
        self.client.on_focus_changed(focused);
    }

    fn game_input_allowed(&self) -> bool {
        self.client.game_input_allowed()
    }

    fn effective_frame_cap(&self, settings: &GameSettings, focused: bool) -> Option<u32> {
        self.client.effective_frame_cap(settings, focused)
    }

    fn wants_fullscreen(&self) -> Option<bool> {
        self.client.wants_fullscreen()
    }

    fn resized(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.client.resized(gpu, width, height);
        if self.viewport_locked && self.physical_viewport != Some((width, height)) {
            self.set_failure(format!("lifecycle viewport changed to {width}x{height}"));
        }
        self.physical_viewport = Some((width, height));
    }

    fn keep_running(&self) -> bool {
        !self.complete && self.failure.is_none() && self.client.keep_running()
    }

    fn shutdown(&mut self, settings: &GameSettings) {
        ShellApp::shutdown(&mut self.client, settings);
    }
}

fn expected_pose(tick: u64) -> (f64, f64, f64, f32) {
    match tick {
        200 | 320 => (16.5, 65.0, 16.5, if tick == 200 { -90.0 } else { 0.0 }),
        440 | 560 | 680 | 800 => (144.5, 65.0, 16.5, 90.0),
        920 | 1040 | 1160 | 1280 => (16.5, 65.0, 144.5, 180.0),
        1401 | 1521 => (16.5, 65.0, 16.5, -90.0),
        _ => panic!("unexpected lifecycle checkpoint tick {tick}"),
    }
}

fn work_json(work: &FrameWorkStats) -> Value {
    json!({
        "mesh_pending": work.mesh_pending,
        "mesh_deferred": work.mesh_deferred,
        "mesh_in_flight": work.mesh_in_flight,
        "mesh_ready_uploads": work.mesh_ready_uploads,
        "mesh_enqueued": work.mesh_enqueued,
        "mesh_dispatched": work.mesh_dispatched,
        "mesh_synchronous_built": work.mesh_synchronous_built,
        "mesh_completed": work.mesh_completed,
        "mesh_cancelled": work.mesh_cancelled,
        "mesh_stale_result_rejected": work.mesh_stale_result_rejected,
        "terrain_loaded_sections": work.terrain_loaded_sections,
        "terrain_visible_sections": work.terrain_visible_sections,
        "terrain_quads": work.terrain_quads,
        "terrain_resident_bytes": work.terrain_resident_bytes,
        "entity_vertices": work.entity_vertices,
        "entity_batches": work.entity_batches,
        "entity_finish_calls": work.entity_finish_calls,
        "entity_upload_bytes": work.entity_upload_bytes,
    })
}

fn light_snapshot(world: &mc_world::World) -> Value {
    let mut chunks = world
        .loaded_chunks()
        .map(|(position, chunk)| {
            json!([
                position.x,
                position.z,
                chunk.queued_light_checks(),
                chunk.is_gap_lighting_updated()
            ])
        })
        .collect::<Vec<_>>();
    chunks.sort_by_key(|row| {
        (
            row[0].as_i64().unwrap_or_default(),
            row[1].as_i64().unwrap_or_default(),
        )
    });
    let mut sky = [0_u64; 16];
    let mut block = [0_u64; 16];
    for x in -20..=20 {
        for z in -20..=20 {
            let position = BlockPos::new(x, 64, z);
            sky[world.sky_light_at(position) as usize] += 1;
            block[world.block_light_at(position) as usize] += 1;
        }
    }
    json!({
        "chunks_x_z_queued_checks_gap_pending": chunks,
        "sample_grid": {"x": [-20, 20], "z": [-20, 20], "y": 64, "inclusive": true},
        "sky_light_histogram": sky,
        "block_light_histogram": block,
        "scope": "world light storage only; rendered vertex/GPU light data not qualified",
    })
}

fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create exclusive output {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write output {}", path.display()))?;
    file.flush()
        .with_context(|| format!("flush output {}", path.display()))?;
    Ok(())
}

fn parse_options() -> Result<(PathBuf, PathBuf, PathBuf, bool, bool)> {
    let mut args = std::env::args().skip(1);
    let mut capture = None;
    let mut output = None;
    let mut game_dir = None;
    let mut gap_variant = None;
    let mut async_mesh = None;
    while let Some(key) = args.next() {
        let value = args.next().context("each option requires a value")?;
        match key.as_str() {
            "--capture" => anyhow::ensure!(
                capture.replace(PathBuf::from(value)).is_none(),
                "duplicate --capture"
            ),
            "--output" => anyhow::ensure!(
                output.replace(PathBuf::from(value)).is_none(),
                "duplicate --output"
            ),
            "--game-dir" => anyhow::ensure!(
                game_dir.replace(PathBuf::from(value)).is_none(),
                "duplicate --game-dir"
            ),
            "--variant" => {
                let parsed = match value.as_str() {
                    "base" => false,
                    "gap" => true,
                    _ => bail!("--variant must be base or gap"),
                };
                anyhow::ensure!(gap_variant.replace(parsed).is_none(), "duplicate --variant");
            }
            "--mesh" => {
                let parsed = match value.as_str() {
                    "sync" => false,
                    "async" => true,
                    _ => bail!("--mesh must be sync or async"),
                };
                anyhow::ensure!(async_mesh.replace(parsed).is_none(), "duplicate --mesh");
            }
            _ => bail!("unknown option {key}"),
        }
    }
    Ok((
        capture.context("--capture is required")?.canonicalize()?,
        output.context("--output is required")?,
        game_dir.context("--game-dir is required")?,
        gap_variant.unwrap_or(false),
        async_mesh.unwrap_or(false),
    ))
}

fn main() -> Result<()> {
    mc_core::init_tracing();
    // The collector writes from a background thread; drain it on every return.
    let _flush_logs = mc_core::FlushLogsOnDrop;
    let (capture, output, game_dir, gap_variant, async_mesh) = parse_options()?;
    let checkpoints: &'static [CheckpointSpec] = if gap_variant {
        &GAP_CHECKPOINTS
    } else {
        &BASE_CHECKPOINTS
    };
    let variant = if gap_variant {
        "unloaded_corridor"
    } else {
        "base"
    };
    fs::create_dir(&output).context("output must be a new directory with an existing parent")?;
    fs::create_dir(&game_dir)
        .context("game-dir must be a new directory with an existing parent")?;
    fs::create_dir(output.join("checkpoints"))?;

    let frame_trace = std::env::var_os("MC_FRAME_TRACE")
        .filter(|path| !path.is_empty())
        .context("MC_FRAME_TRACE must point to the coordinator-owned frame trace path")?;
    let _frame_trace_max_samples = std::env::var_os("MC_FRAME_TRACE_MAX_SAMPLES")
        .filter(|value| !value.is_empty())
        .context("MC_FRAME_TRACE_MAX_SAMPLES must be set by the coordinator")?;
    let config_args = vec![
        "--offline".to_owned(),
        "BenchLocal".to_owned(),
        "--gameDir".to_owned(),
        game_dir.to_string_lossy().into_owned(),
        "--replay".to_owned(),
        capture.to_string_lossy().into_owned(),
        "--replayCamera".to_owned(),
        "fixed".to_owned(),
        "--replayStep".to_owned(),
        STEP_MICROS.to_string(),
        "--width".to_owned(),
        WIDTH.to_string(),
        "--height".to_owned(),
        HEIGHT.to_string(),
    ];
    let mut config = ClientConfig::parse_args(&config_args).map_err(anyhow::Error::msg)?;
    config.max_frames = None;
    let mut client = Client::new_deferred(config.with_discovered_assets())?;
    client.settings.fov = 70.0;
    client.settings.render_distance_chunks = 10;
    client.settings.enable_vsync = false;
    client.settings.limit_framerate = 120;
    client.set_replay_deterministic_mesh(!async_mesh);
    let settings = client.settings.clone();
    let mut app = LifecycleDiagnostic {
        client,
        output: output.clone(),
        checkpoints,
        checkpoint_index: 0,
        pending_checkpoint: None,
        captured_this_frame: false,
        complete: false,
        failure: None,
        last_virtual_micros: None,
        camera: None,
        physical_viewport: None,
        viewport_locked: false,
    };
    let shell_result = Shell::run(
        ShellConfig {
            title: "mc-rs lifecycle correctness diagnostic".into(),
            width: WIDTH,
            height: HEIGHT,
            max_frames: Some(MAX_FRAMES),
            max_duration: Some(Duration::from_secs(MAX_DURATION_SECS)),
            ..ShellConfig::default()
        },
        settings,
        &mut app,
    );
    if let Some(error) = app.failure.take() {
        bail!("lifecycle diagnostic failed: {error}");
    }
    shell_result.map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if app.checkpoint_index != app.checkpoints.len() {
        bail!(
            "lifecycle ended before all checkpoints: {}/{}",
            app.checkpoint_index,
            app.checkpoints.len()
        );
    }
    if let Some(error) = app.client.fatal_error() {
        bail!("client fatal error after lifecycle: {error}");
    }
    let mut receipt = json!({
        "schema": "mc-rs.optimization.lifecycle-run",
        "schema_version": 1,
        "status": "UNQUALIFIED_DIAGNOSTIC",
        "scoring_eligible": false,
        "simulation": "fixed_20hz_wrapper_over_ordinary_replay",
        "replay_step_micros": STEP_MICROS,
        "diagnostic_simulation": "not_selected",
        "camera_source": "typed replay PlayerPositionLook packets; no live camera seam",
        "settings": {"fov": 70, "render_distance_chunks": 10, "viewport": [WIDTH, HEIGHT]},
        "checkpoints": app.checkpoints.iter().map(|spec| json!({
            "tick": spec.tick,
            "label": spec.label,
            "json": format!("checkpoints/{:04}-{}.json", spec.tick, spec.label),
            "image": format!("checkpoints/{:04}-{}.png", spec.tick, spec.label),
        })).collect::<Vec<_>>(),
        "frame_trace": PathBuf::from(frame_trace).to_string_lossy().into_owned(),
        "no_fps_summary": true,
    });
    if gap_variant {
        receipt["fixture_variant"] = json!(variant);
    }
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    write_exclusive(&output.join("lifecycle.json"), &bytes)?;
    Ok(())
}
