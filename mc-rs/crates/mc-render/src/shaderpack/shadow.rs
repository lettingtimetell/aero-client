//! Sun and moon shadows: the shadow map, its matrices, and the
//! phase that renders it.
//!
//! # The map
//!
//! One orthographic depth map looking down the light direction, centred on the player. It is
//! **stable**: the centre is snapped in light space to a whole number of [`SNAP_TEXELS`] texels,
//! so standing still or moving never makes edges shimmer, and the light direction is quantised to
//! [`DIRECTION_STEP_DEGREES`] along the sun's path, so the sun's slow crawl moves shadows in steps
//! far below a texel's visible size instead of every frame.
//!
//! The quantised direction is a function of the key alone: the light is turned back onto
//! vanilla's untilted path, quantised there, and turned forward by the pack's sun-path tilt, so
//! every frame that keeps the cached map projects a point exactly where the map stored it.
//! (Quantising only the light's angle in the x-y plane let a tilted path's z component through:
//! receivers slid up to 3 texels off the cached map and snapped back at each redraw.) Under a held
//! clock (the Time Changer, or a server that stops the daylight cycle) the light does not move at
//! all ([`super::FrameInputs::clock_held`]), so the map is never redrawn for it.
//! `MC_W22_SHD_STABLE=0` brings back the old behaviour, for comparisons ([`steady_shading`]).
//!
//! # Rendering only when something changed
//!
//! The casters are drawn relative to a section-aligned origin near the eye
//! ([`ShadowFrame::caster_origin`]), not to the eye itself, so their matrix depends only on the
//! snapped centre, the quantised direction, the origin, the resolution and the distance
//! ([`ShadowKey`]). The terrain variant (`terrain/pack.rs`) re-renders the map only when that key
//! or the shadow-casting geometry changes; on every other frame the shadow phase records nothing.
//! Receivers use [`ShadowFrame::receiver_matrix`], rebuilt every frame from the exact eye in
//! `f64`, so they stay exact while the map is reused.
//!
//! # What a shadow darkens
//!
//! Only sky light. The terrain variant scales the sky-light part of vanilla's lightmap and
//! leaves the block-light part alone, so a torch or lava in a shaded spot lights it exactly as
//! before (see `terrain/pack.wgsl`).

use std::cell::RefCell;
use std::rc::Rc;

use glam::{DMat3, DVec3, Mat4, Vec3, Vec4};

use super::config::ShadowSettings;
use super::frame::PackFrame;
use super::shared::ShaderCore;
use crate::terrain::phase::TerrainRenderer;
use crate::{Camera, Gpu, RenderPhase};

/// The shadow map's depth format.
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The centre snaps to multiples of this many texels (light space).
pub const SNAP_TEXELS: f64 = 32.0;
/// The light direction is quantised to this angle.
pub const DIRECTION_STEP_DEGREES: f64 = 0.2;
/// Environment switch for the shading stability fixes, for comparisons: `MC_W22_SHD_STABLE=0`
/// quantises the shadow direction as before (the angle in the x-y plane only), lets a held
/// clock's sun follow the partial tick, and re-seeds the light rays' and the Underwater shafts'
/// dither every frame even without TAA, as before.
pub const STABLE_ENV: &str = "MC_W22_SHD_STABLE";

/// Whether the shading stability fixes are on (the default; see [`STABLE_ENV`]). Read once.
#[must_use]
pub fn steady_shading() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(STABLE_ENV).map_or(true, |v| v.trim() != "0"))
}

/// How far toward the light (from the centre) casters are kept, in blocks.
pub const CASTER_REACH: f64 = 320.0;
/// How far away from the light (below the centre) receivers are kept, beyond the distance.
pub const RECEIVER_MARGIN: f64 = 48.0;

/// What the casters' matrix depends on. A change re-renders the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ShadowKey {
    /// The quantised light angle, in steps.
    pub direction_step: i64,
    /// The sun-path tilt's bits (degrees, `f32`): the quantised direction is turned by it.
    pub tilt_bits: u32,
    /// The snapped centre, in snaps (light-space x and y).
    pub center: [i64; 2],
    /// The caster origin, in sections.
    pub origin: [i64; 3],
    /// Map edge in texels.
    pub resolution: u32,
    /// The distance's bits.
    pub distance_bits: u32,
}

/// This frame's shadow matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowFrame {
    /// Absolute, section-aligned point the casters' offsets are relative to.
    pub caster_origin: DVec3,
    /// Origin-relative world position to shadow clip space (x, y in -1..1, depth 0..1).
    pub caster_matrix: Mat4,
    /// Camera-relative world position to shadow clip space.
    pub receiver_matrix: Mat4,
    /// The quantised direction toward the light.
    pub light_dir: Vec3,
    /// What the caster matrix depends on.
    pub key: ShadowKey,
    /// World size of one texel, in blocks.
    pub texel_world: f32,
    /// The settings it was built for.
    pub settings: ShadowSettings,
    /// Where shadows start fading out and where they are gone, blocks from the eye.
    pub fade: [f32; 2],
}

impl ShadowFrame {
    /// The matrices for a light direction (toward the light) on vanilla's untilted sun path and an
    /// eye.
    #[must_use]
    pub fn compute(settings: ShadowSettings, light_dir: Vec3, eye: DVec3) -> ShadowFrame {
        ShadowFrame::compute_on_path(settings, light_dir, 0.0, eye)
    }

    /// The matrices for a light direction (toward the light) on the sun's path tilted `path_tilt`
    /// degrees toward the south ([`super::frame::sun_path_tilt`], the running pack's
    /// `CoreConfig::sun_tilt`) and an eye.
    ///
    /// The quantised direction depends on the key alone (the direction step and the tilt), never
    /// on the exact light, so every frame that keeps a cached map builds the matrices it was drawn
    /// with. A light off the tilted path (none in the game) is projected onto it.
    #[must_use]
    pub fn compute_on_path(
        settings: ShadowSettings,
        light_dir: Vec3,
        path_tilt: f32,
        eye: DVec3,
    ) -> ShadowFrame {
        let distance = f64::from(settings.distance);
        let resolution = f64::from(settings.resolution.max(1));
        let step = DIRECTION_STEP_DEGREES.to_radians();
        let l = light_dir.as_dvec3().normalize_or(DVec3::Y);
        let (direction_step, tilt_bits, lq) = if steady_shading() {
            // Back onto the untilted path (z = 0), where the light moves along one angle.
            let tilt = DMat3::from_rotation_x(f64::from(path_tilt).to_radians());
            let flat = tilt.transpose() * l;
            let direction_step = ((-flat.x).atan2(flat.y) / step).round() as i64;
            let tilt_bits = path_tilt.to_bits();
            let qa = direction_step as f64 * step;
            let lq = (tilt * DVec3::new(-qa.sin(), qa.cos(), 0.0)).normalize_or(DVec3::Y);
            (direction_step, tilt_bits, lq)
        } else {
            // Previously: the angle in the x-y plane only; a tilted path's z component (and the
            // planar length) stayed continuous.
            let angle = (-l.x).atan2(l.y);
            let direction_step = (angle / step).round() as i64;
            let qa = direction_step as f64 * step;
            let planar = (l.x * l.x + l.y * l.y).sqrt();
            let lq = DVec3::new(-qa.sin() * planar, qa.cos() * planar, l.z).normalize_or(DVec3::Y);
            (direction_step, 0, lq)
        };
        // Light-space basis: z toward the light, y along world Z (never parallel to the sun's
        // path), x completing a right-handed frame.
        let x_axis = DVec3::Z.cross(lq).normalize_or(DVec3::X);
        let y_axis = lq.cross(x_axis);
        let rot = DMat3::from_cols(x_axis, y_axis, lq).transpose();
        let texel = 2.0 * distance / resolution;
        let snap = texel * SNAP_TEXELS;
        let eye_ls = rot * eye;
        let cx = (eye_ls.x / snap).floor();
        let cy = (eye_ls.y / snap).floor();
        let origin = (eye / 16.0).floor() * 16.0;
        // The depth reference is the section-aligned origin (part of the key), not the eye: a jump
        // or a fall moves the eye along the light without changing the key, and a cached map's
        // depths must still match what receivers compare against (found). The origin is within a section's diagonal (< 28 blocks) of the eye, which the
        // 48-block RECEIVER_MARGIN covers.
        let center = DVec3::new(cx * snap, cy * snap, (rot * origin).z);
        let z_hi = CASTER_REACH;
        let z_lo = -(distance + RECEIVER_MARGIN);
        let ortho = |t: DVec3| -> Mat4 {
            // ls = rot * p + t; x' = ls.x / distance, y' = ls.y / distance,
            // depth = (z_hi - ls.z) / (z_hi - z_lo).
            let sx = 1.0 / distance;
            let sz = -1.0 / (z_hi - z_lo);
            let r = rot;
            let row = |axis: usize, scale: f64, add: f64, offset: f64| -> [f64; 4] {
                let m = r.row(axis);
                [
                    m.x * scale,
                    m.y * scale,
                    m.z * scale,
                    (t[axis] + offset) * scale + add,
                ]
            };
            let rx = row(0, sx, 0.0, 0.0);
            let ry = row(1, sx, 0.0, 0.0);
            let rz = row(2, sz, 0.0, -z_hi);
            let rows = [rx, ry, rz, [0.0, 0.0, 0.0, 1.0]];
            // glam is column-major: transpose the rows.
            Mat4::from_cols_array(&[
                rows[0][0] as f32,
                rows[1][0] as f32,
                rows[2][0] as f32,
                rows[3][0] as f32,
                rows[0][1] as f32,
                rows[1][1] as f32,
                rows[2][1] as f32,
                rows[3][1] as f32,
                rows[0][2] as f32,
                rows[1][2] as f32,
                rows[2][2] as f32,
                rows[3][2] as f32,
                rows[0][3] as f32,
                rows[1][3] as f32,
                rows[2][3] as f32,
                rows[3][3] as f32,
            ])
        };
        let caster_t = rot * origin - center;
        let receiver_t = eye_ls - center;
        let fade_end = (distance - 2.0 * snap).max(8.0) as f32;
        ShadowFrame {
            caster_origin: origin,
            caster_matrix: ortho(caster_t),
            receiver_matrix: ortho(receiver_t),
            light_dir: lq.as_vec3(),
            key: ShadowKey {
                direction_step,
                tilt_bits,
                center: [cx as i64, cy as i64],
                origin: [
                    (origin.x / 16.0) as i64,
                    (origin.y / 16.0) as i64,
                    (origin.z / 16.0) as i64,
                ],
                resolution: settings.resolution,
                distance_bits: settings.distance.to_bits(),
            },
            texel_world: texel as f32,
            settings,
            fade: [fade_end * 0.85, fade_end],
        }
    }

    /// The uniform the terrain variant and any pack reads (WGSL `ShadowUniform`).
    #[must_use]
    pub fn uniform(&self, light_strength: f32) -> ShadowUniform {
        let s = self.settings;
        ShadowUniform {
            receiver: self.receiver_matrix.to_cols_array_2d(),
            caster: self.caster_matrix.to_cols_array_2d(),
            params: [
                self.texel_world,
                s.softness,
                s.strength,
                1.0 / s.resolution.max(1) as f32,
            ],
            fade: [self.fade[0], self.fade[1], s.distance, light_strength],
            light: Vec4::from((self.light_dir, 0.0)).to_array(),
        }
    }
}

/// GPU mirror of the shadow state: WGSL `ShadowUniform` in `shaderpack/common.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    /// Camera-relative world position to shadow clip.
    pub receiver: [[f32; 4]; 4],
    /// Caster-origin-relative world position to shadow clip.
    pub caster: [[f32; 4]; 4],
    /// x texel size in blocks, y softness (texels), z strength, w 1 / resolution.
    pub params: [f32; 4],
    /// x fade start, y fade end (blocks from the eye), z distance, w light strength.
    pub fade: [f32; 4],
    /// xyz quantised direction toward the light.
    pub light: [f32; 4],
}

/// Read-only access to the shadow map for any pack ([`ShaderCore::shadow`]).
#[derive(Debug, Clone)]
pub struct ShadowView {
    /// The map receivers sample (`Depth32Float`; sample with a comparison sampler, or
    /// `textureLoad`): the terrain map, or with entity casters on, the terrain map plus this
    /// frame's entities.
    pub view: wgpu::TextureView,
    /// The cached terrain-only map (what the terrain caster pass renders into).
    pub terrain_view: wgpu::TextureView,
    /// The uniform buffer (`ShadowUniform`).
    pub uniform: wgpu::Buffer,
    /// This frame's matrices.
    pub frame: ShadowFrame,
}

/// Where this frame's entity casters draw: a copy of the terrain map they add their depth to.
pub struct EntityShadowTarget {
    /// The combined map's view (render attachment).
    pub view: wgpu::TextureView,
    /// The shadow uniform (`ShadowUniform`; entities are camera-relative, so they use
    /// `receiver`).
    pub uniform: wgpu::Buffer,
}

/// The shadow map's GPU half.
pub(crate) struct ShadowGpu {
    pub(crate) resolution: u32,
    texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) uniform: wgpu::Buffer,
    uploaded: Option<ShadowUniform>,
    /// The terrain map plus this frame's entities, while entity casters are on.
    combined: Option<(wgpu::Texture, wgpu::TextureView)>,
}

/// The shadow map, owned by [`ShaderCore`].
#[derive(Default)]
pub(crate) struct ShadowMap {
    pub(crate) gpu: Option<ShadowGpu>,
    pub(crate) frame: Option<ShadowFrame>,
    /// Bumped whenever the texture or the uniform buffer is (re)allocated.
    pub(crate) generation: u64,
}

impl ShadowMap {
    /// Build or resize the map and upload this frame's matrices. `None` releases it. `path_tilt` is
    /// the running pack's sun-path tilt in degrees (the tilt `frame.light_dir` was turned by).
    pub(crate) fn update(
        &mut self,
        gpu: &Gpu,
        settings: Option<ShadowSettings>,
        path_tilt: f32,
        frame: &PackFrame,
    ) {
        let Some(settings) = settings else {
            self.release();
            return;
        };
        let map = |label: &str, usage: wgpu::TextureUsages| {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: settings.resolution,
                    height: settings.resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        if self
            .gpu
            .as_ref()
            .is_none_or(|g| g.resolution != settings.resolution)
        {
            let (texture, view) = map("shaderpack shadow map", wgpu::TextureUsages::COPY_SRC);
            let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shaderpack shadow uniform"),
                size: std::mem::size_of::<ShadowUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.gpu = Some(ShadowGpu {
                resolution: settings.resolution,
                texture,
                view,
                uniform,
                uploaded: None,
                combined: None,
            });
            self.generation += 1;
        }
        let built = self.gpu.as_mut().expect("built above");
        if settings.entity_casters != built.combined.is_some() {
            built.combined = settings.entity_casters.then(|| {
                map(
                    "shaderpack shadow map (with entities)",
                    wgpu::TextureUsages::COPY_DST,
                )
            });
            self.generation += 1;
        }
        let shadow = ShadowFrame::compute_on_path(settings, frame.light_dir, path_tilt, frame.eye);
        let uniform = shadow.uniform(frame.light_strength);
        let built = self.gpu.as_mut().expect("built above");
        if built.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&built.uniform, 0, bytemuck::bytes_of(&uniform));
            built.uploaded = Some(uniform);
        }
        self.frame = Some(shadow);
    }

    pub(crate) fn release(&mut self) {
        if self.gpu.take().is_some() {
            self.generation += 1;
        }
        self.frame = None;
    }

    pub(crate) fn allocated(&self) -> bool {
        self.gpu.is_some()
    }

    pub(crate) fn view(&self) -> Option<ShadowView> {
        let gpu = self.gpu.as_ref()?;
        Some(ShadowView {
            view: gpu
                .combined
                .as_ref()
                .map_or_else(|| gpu.view.clone(), |(_, view)| view.clone()),
            terrain_view: gpu.view.clone(),
            uniform: gpu.uniform.clone(),
            frame: self.frame?,
        })
    }

    /// Copy the terrain map into the combined map and hand back where entities draw this frame.
    /// `None` while entity casters are off.
    fn begin_entity_casters(&self, enc: &mut wgpu::CommandEncoder) -> Option<EntityShadowTarget> {
        let gpu = self.gpu.as_ref()?;
        let (combined, view) = gpu.combined.as_ref()?;
        enc.copy_texture_to_texture(
            gpu.texture.as_image_copy(),
            combined.as_image_copy(),
            wgpu::Extent3d {
                width: gpu.resolution,
                height: gpu.resolution,
                depth_or_array_layers: 1,
            },
        );
        Some(EntityShadowTarget {
            view: view.clone(),
            uniform: gpu.uniform.clone(),
        })
    }
}

/// Draws this frame's entities into an [`EntityShadowTarget`] (the client wires it to its
/// entity phase; see [`ShadowPhase::set_entity_casters`]).
pub type EntityCasters = Box<dyn FnMut(&Gpu, &mut wgpu::CommandEncoder, &EntityShadowTarget)>;

/// The phase that renders the shadow map, before the sky and terrain
/// ([`super::order::SHADOW`]). Records nothing unless the running pack has shadows on, and
/// nothing on frames where the cached map is still right.
pub struct ShadowPhase {
    core: ShaderCore,
    terrain: Rc<RefCell<TerrainRenderer>>,
    entities: Option<EntityCasters>,
}

impl ShadowPhase {
    /// The phase's name in `PhaseList` and GPU timings.
    pub const NAME: &'static str = "shaderpack/shadow";

    /// A phase drawing `terrain`'s casters into `core`'s shadow map.
    #[must_use]
    pub fn new(core: ShaderCore, terrain: Rc<RefCell<TerrainRenderer>>) -> ShadowPhase {
        ShadowPhase {
            core,
            terrain,
            entities: None,
        }
    }

    /// How this frame's entities are drawn into the shadow map while the running pack has
    /// entity casters on. Each frame they draw into a copy of the cached terrain map.
    pub fn set_entity_casters(&mut self, casters: EntityCasters) {
        self.entities = Some(casters);
    }
}

impl RenderPhase for ShadowPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        // No direct light (the Nether, the End, the instant the sun hands over to the moon):
        // receivers skip the map too, so there is nothing to draw.
        self.core.any_running()
            && self.core.config().shadows.is_some()
            && self.core.frame().light_strength > 0.0
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.terrain
            .borrow_mut()
            .record_pack_shadows(gpu, enc, &self.core);
        // With entity casters on, receivers sample the combined map: refresh it from the terrain
        // map every frame (even with no entity hook installed, so it is never stale or empty),
        // then add this frame's entities.
        if let Some(target) = self.core.with_shadow(|map| map.begin_entity_casters(enc)) {
            if let Some(entities) = self.entities.as_mut() {
                entities(gpu, enc, &target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ShadowSettings {
        ShadowSettings::default()
    }

    fn project(m: Mat4, p: Vec3) -> Vec3 {
        let c = m * p.extend(1.0);
        c.truncate() / c.w
    }

    #[test]
    fn receivers_and_casters_agree_on_the_same_world_point() {
        let eye = DVec3::new(1234.567, 81.25, -9876.5);
        let frame = ShadowFrame::compute(settings(), Vec3::new(-0.5, 0.8, 0.0), eye);
        let world = DVec3::new(1240.0, 70.0, -9870.0);
        let as_receiver = project(frame.receiver_matrix, (world - eye).as_vec3());
        let as_caster = project(frame.caster_matrix, (world - frame.caster_origin).as_vec3());
        assert!(
            (as_receiver - as_caster).length() < 1e-4,
            "{as_receiver} vs {as_caster}"
        );
        assert!(as_receiver.x.abs() < 1.0 && as_receiver.y.abs() < 1.0);
        assert!((0.0..=1.0).contains(&as_receiver.z));
    }

    #[test]
    fn a_point_nearer_the_light_has_a_smaller_depth() {
        let eye = DVec3::new(0.5, 64.0, 0.5);
        let light = Vec3::new(-0.3, 0.95, 0.0).normalize();
        let frame = ShadowFrame::compute(settings(), light, eye);
        let low = project(frame.receiver_matrix, Vec3::new(0.0, -2.0, 0.0));
        let high = project(frame.receiver_matrix, Vec3::new(0.0, 10.0, 0.0));
        assert!(high.z < low.z);
    }

    /// A cached map must stay valid for as long as its
    /// key holds. A jump or a fall moves the eye along the light without changing the key, so the
    /// depths the map stored and the depths receivers compare against must not depend on the
    /// eye's exact light-space depth.
    #[test]
    fn a_cached_map_stays_valid_while_its_key_holds() {
        let light = Vec3::new(-0.4, 0.9, 0.0).normalize();
        let eye_a = DVec3::new(100.5, 70.0, 100.5);
        let cached = ShadowFrame::compute(settings(), light, eye_a);
        let world = DVec3::new(104.0, 66.0, 97.0);
        let stored = project(
            cached.caster_matrix,
            (world - cached.caster_origin).as_vec3(),
        );
        let mut checked = 0;
        // Jump heights in 0.1-block steps: every one that keeps the key must read the same depth.
        for tenth in 1..=12 {
            let eye_b = eye_a + DVec3::new(0.0, f64::from(tenth) * 0.1, 0.0);
            let now = ShadowFrame::compute(settings(), light, eye_b);
            if now.key != cached.key {
                continue;
            }
            checked += 1;
            let read = project(now.receiver_matrix, (world - eye_b).as_vec3());
            assert!(
                (read - stored).length() < 1e-5,
                "eye +{:.1}: receiver reads {read}, the cached map stored {stored}",
                f64::from(tenth) * 0.1
            );
        }
        assert!(
            checked > 0,
            "no jump height kept the key; move the test eye"
        );
    }

    #[test]
    fn walking_keeps_the_map_and_its_texel_grid() {
        let light = Vec3::new(-0.4, 0.9, 0.0).normalize();
        let world = DVec3::new(105.0, 66.0, 98.0);
        let res = settings().resolution as f32;
        let mut keys = std::collections::BTreeSet::new();
        let mut first_texel = None;
        // Walk 6 blocks in 0.05-block steps (120 frames of sprinting).
        for step in 0..120 {
            let eye = DVec3::new(100.2 + f64::from(step) * 0.05, 70.0, 100.2);
            let frame = ShadowFrame::compute(settings(), light, eye);
            keys.insert(frame.key);
            // The fixed world point's position inside its texel never changes: no shimmer.
            let p = project(frame.receiver_matrix, (world - eye).as_vec3());
            let texel = (p.truncate() * 0.5 * res).fract();
            let texel = texel.map(|v| if v < 0.0 { v + 1.0 } else { v });
            match first_texel {
                None => first_texel = Some(texel),
                Some(first) => {
                    let d = (texel - first).abs();
                    let d = d.map(|v| v.min(1.0 - v));
                    assert!(d.length() < 2e-2, "texel phase moved by {d} at step {step}");
                }
            }
        }
        // The cached map survives most of the walk: a new key every few blocks, not per frame.
        assert!(
            keys.len() <= 4,
            "{} distinct maps over 6 blocks",
            keys.len()
        );
    }

    #[test]
    fn the_sun_crawl_changes_the_map_in_steps() {
        let eye = DVec3::new(0.0, 70.0, 0.0);
        // 200 consecutive ticks (10 seconds) of sun movement: about 3 degrees.
        let mut steps = std::collections::BTreeSet::new();
        let mut changes = 0;
        let mut last = None;
        for tick in 0..200 {
            let dir = crate::shaderpack::frame::sun_direction(0.1 + tick as f32 / 24_000.0);
            let key = ShadowFrame::compute(settings(), dir, eye).key;
            steps.insert(key.direction_step);
            if last.is_some_and(|l| l != key) {
                changes += 1;
            }
            last = Some(key);
        }
        assert!(steps.len() > 5, "the direction does follow the sun");
        assert!(changes < 30, "re-rendered on {changes} of 200 ticks");
    }
}
