//! World-scoped state the server drives: time of day, weather, difficulty, and world border.
//!
//! java: `NetHandlerPlayClient.handleTimeUpdate` (`:952`) and `handleChangeGameState` (`:1358`),
//! `client/multiplayer/WorldClient.setRainStrength/setThunderStrength`,
//! `world/storage/WorldInfo.java`.

use std::time::{SystemTime, UNIX_EPOCH};

use mc_math::{Aabb, BlockPos};
use mc_protocol::play::s2c::{GameStateReason, ServerDifficultyLevel, WorldBorder};

/// Ticks in one Minecraft day. java: `World.getCelestialAngle` works modulo this.
pub const TICKS_PER_DAY: i64 = 24_000;

/// Java `WorldBorder`'s client-side default damage multiplier.
///
/// The 1.8.9 client does not receive damage amount/buffer mutations over S44: those setters are
/// server configuration.  The values nevertheless exist on every freshly-created client border
/// and are consumed by the local-player border-damage probe below.  Keep them named so the
/// arithmetic cannot silently drift back to an unexplained literal.
pub const DEFAULT_BORDER_DAMAGE_AMOUNT: f64 = 0.2;
/// Java `WorldBorder`'s client-side default damage buffer in blocks.
pub const DEFAULT_BORDER_DAMAGE_BUFFER: f64 = 5.0;

/// The world border, as the server describes it. java: `world/border/WorldBorder.java`
///
/// The client-side projection of `WorldBorder`. `S44PacketWorldBorder` supplies the endpoints and
/// duration; the receive instant is captured here because vanilla evaluates the transition from
/// `System.currentTimeMillis()` rather than from world ticks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderState {
    /// Border centre X.
    pub center_x: f64,
    /// Border centre Z.
    pub center_z: f64,
    /// The diameter the border is moving away from.
    pub old_diameter: f64,
    /// The diameter it is moving towards.
    pub new_diameter: f64,
    /// How long the packet's move takes, in milliseconds. This is retained as the wire-facing
    /// value; the Java-style accessors below use the captured start/end instants.
    pub speed_millis: i64,
    /// Wall-clock instant at which the current transition started, in Unix milliseconds.
    /// java: `WorldBorder.startTime`.
    pub start_time_millis: i64,
    /// Wall-clock instant at which the current transition ends, in Unix milliseconds.
    /// java: `WorldBorder.endTime`.
    pub end_time_millis: i64,
    /// The world's coordinate limit (`29999984` in vanilla).
    pub world_size: i32,
    /// How many blocks out the cyan world-border vignette tint starts.
    pub warning_blocks: i32,
    /// How many seconds of approach triggers the warning.
    pub warning_time: i32,
    /// Damage per block beyond the damage buffer. java: `WorldBorder.damageAmount`.
    ///
    /// S44 has no action for this field; it remains at the constructor default on the client.
    pub damage_amount: f64,
    /// Grace distance before border damage starts. java: `WorldBorder.damageBuffer`.
    ///
    /// S44 has no action for this field; it remains at the constructor default on the client.
    pub damage_buffer: f64,
}

impl Default for BorderState {
    /// java: `WorldBorder`'s field initialisers — a 60,000,000-block border centred on origin.
    fn default() -> Self {
        BorderState {
            center_x: 0.0,
            center_z: 0.0,
            old_diameter: 6.0e7,
            new_diameter: 6.0e7,
            speed_millis: 0,
            start_time_millis: 0,
            end_time_millis: 0,
            world_size: 29_999_984,
            warning_blocks: 5,
            warning_time: 15,
            damage_amount: DEFAULT_BORDER_DAMAGE_AMOUNT,
            damage_buffer: DEFAULT_BORDER_DAMAGE_BUFFER,
        }
    }
}

/// Time, weather, difficulty and border — everything about the world that is not blocks.
impl BorderState {
    /// Return the client-world view of this border for `dimension`.
    ///
    /// java: `WorldProviderHell.getWorldBorder` projects only the stored centre through the
    /// Nether movement factor. Packet state remains in server coordinates, and the diameter,
    /// transition clocks, warnings, and coordinate clamp are not scaled.
    #[must_use]
    pub fn dimension_view(mut self, dimension: i32) -> Self {
        if dimension == -1 {
            self.center_x /= 8.0;
            self.center_z /= 8.0;
        }
        self
    }

    /// The Java `EnumBorderStatus` id used by `RenderGlobal.renderWorldBorder`.
    ///
    /// Status is derived from the transition endpoints rather than the instantaneous diameter;
    /// that is what `WorldBorder.getStatus()` does while a resize is in flight.
    #[must_use]
    pub fn status_id(self) -> i32 {
        if self.new_diameter < self.old_diameter {
            16_724_016 // EnumBorderStatus.SHRINKING (0xFF3030)
        } else if self.new_diameter > self.old_diameter {
            4_259_712 // EnumBorderStatus.GROWING (0x40FF80)
        } else {
            2_138_367 // EnumBorderStatus.STATIONARY (0x20A0FF)
        }
    }

    /// The normalized RGB colour Java extracts from [`BorderState::status_id`].
    #[must_use]
    pub fn status_color(self) -> [f32; 4] {
        let id = self.status_id();
        [
            ((id >> 16) & 255) as f32 / 255.0,
            ((id >> 8) & 255) as f32 / 255.0,
            (id & 255) as f32 / 255.0,
            1.0,
        ]
    }

    /// The number of border blocks travelled per millisecond.
    ///
    /// java: `WorldBorder.getResizeSpeed()` (`:243-246`). Java explicitly returns `0.0` when
    /// `endTime == startTime`; it does not expose a `0/0` NaN for a stationary border.
    #[must_use]
    pub fn resize_speed(self) -> f64 {
        let duration = self.end_time_millis.wrapping_sub(self.start_time_millis);
        if duration == 0 {
            0.0
        } else {
            (self.new_diameter - self.old_diameter).abs() / duration as f64
        }
    }

    /// The minimum X edge for a border whose current diameter is `diameter`.
    ///
    /// java: `WorldBorder.minX()`, including its independent `worldSize` clamp.
    #[must_use]
    pub fn min_x(self, diameter: f64) -> f64 {
        let value = self.center_x - diameter / 2.0;
        if value < -f64::from(self.world_size) {
            -f64::from(self.world_size)
        } else {
            value
        }
    }

    /// The maximum X edge for a border whose current diameter is `diameter`.
    ///
    /// java: `WorldBorder.maxX()`.
    #[must_use]
    pub fn max_x(self, diameter: f64) -> f64 {
        let value = self.center_x + diameter / 2.0;
        if value > f64::from(self.world_size) {
            f64::from(self.world_size)
        } else {
            value
        }
    }

    /// The minimum Z edge for a border whose current diameter is `diameter`.
    ///
    /// java: `WorldBorder.minZ()`.
    #[must_use]
    pub fn min_z(self, diameter: f64) -> f64 {
        let value = self.center_z - diameter / 2.0;
        if value < -f64::from(self.world_size) {
            -f64::from(self.world_size)
        } else {
            value
        }
    }

    /// The maximum Z edge for a border whose current diameter is `diameter`.
    ///
    /// java: `WorldBorder.maxZ()`.
    #[must_use]
    pub fn max_z(self, diameter: f64) -> f64 {
        let value = self.center_z + diameter / 2.0;
        if value > f64::from(self.world_size) {
            f64::from(self.world_size)
        } else {
            value
        }
    }

    /// Distance to the nearest edge of a border whose current diameter is `diameter`.
    ///
    /// Negative values deliberately mean that the point is outside the border. java:
    /// `WorldBorder.getClosestDistance(Entity)` via its four `get*` bounds.
    #[must_use]
    pub fn closest_distance(self, x: f64, z: f64, diameter: f64) -> f64 {
        // `Math.min` propagates NaN, while Rust's f64::min intentionally chooses the numeric
        // operand. Keep the Java behaviour because the warning colour is a direct float port.
        let x_distance = java_min(x - self.min_x(diameter), self.max_x(diameter) - x);
        let xz_distance = java_min(x_distance, z - self.min_z(diameter));
        java_min(xz_distance, self.max_z(diameter) - z)
    }

    /// Whether any horizontal part of an entity box lies inside this border.
    ///
    /// java: `WorldBorder.contains(AxisAlignedBB)`.  The strict comparisons matter at the exact
    /// wall: a box that only touches an edge is outside for the purpose of the living-entity
    /// damage branch.  The Y coordinates are intentionally ignored, just as Java ignores them.
    #[must_use]
    pub fn contains_aabb(self, aabb: Aabb, diameter: f64) -> bool {
        aabb.max.x > self.min_x(diameter)
            && aabb.min.x < self.max_x(diameter)
            && aabb.max.z > self.min_z(diameter)
            && aabb.min.z < self.max_z(diameter)
    }

    /// Compute Java's one-tick world-border damage from a closest-edge distance.
    ///
    /// `WorldBorder` supplies `closestDistance(entity) + damageBuffer`; when the sum is negative,
    /// `EntityLivingBase.onEntityUpdate` calls `attackEntityFrom` with
    /// `max(1, floor(-distance * damageAmount))`.  Returning `None` at the non-negative boundary
    /// preserves Java's strict `< 0.0D` test.  This helper is deliberately pure so callers and
    /// tests can exercise the boundary, negative amount and custom amount/buffer controls without
    /// involving a session or a wall-clock border transition.
    #[must_use]
    pub fn damage_for_distance(
        closest_distance: f64,
        damage_amount: f64,
        damage_buffer: f64,
    ) -> Option<i32> {
        let distance = closest_distance + damage_buffer;
        if distance < 0.0 {
            Some(mc_math::floor_f64(-distance * damage_amount).max(1))
        } else {
            None
        }
    }

    /// Compute the damage amount for a point at the current border diameter.
    ///
    /// This is the geometry-facing convenience form of [`BorderState::damage_for_distance`].
    /// It intentionally does not apply the entity-AABB gate; callers that model
    /// `EntityLivingBase.onEntityUpdate` must first check [`BorderState::contains_aabb`].
    #[must_use]
    pub fn damage_for_point(self, x: f64, z: f64, diameter: f64) -> Option<i32> {
        Self::damage_for_distance(
            self.closest_distance(x, z, diameter),
            self.damage_amount,
            self.damage_buffer,
        )
    }

    /// Whether a block has any portion inside the border at `diameter`.
    ///
    /// java: `WorldBorder.contains(BlockPos)` (`WorldBorder.java:34`). The strict comparisons and
    /// the `+1` on the minimum side are observable at the exact edge.
    #[must_use]
    pub fn contains_block(self, pos: BlockPos, diameter: f64) -> bool {
        // Java performs `pos.getX() + 1` as an `int` before widening; wrapping_add keeps the
        // same result for the pathological i32::MAX coordinate too.
        f64::from(pos.x.wrapping_add(1)) > self.min_x(diameter)
            && f64::from(pos.x) < self.max_x(diameter)
            && f64::from(pos.z.wrapping_add(1)) > self.min_z(diameter)
            && f64::from(pos.z) < self.max_z(diameter)
    }

    /// Apply Java's `WorldBorder.setTransition(oldSize,newSize,time)` at a supplied clock value.
    fn set_transition_at(
        &mut self,
        old_diameter: f64,
        new_diameter: f64,
        time_millis: i64,
        now_millis: i64,
    ) {
        self.old_diameter = old_diameter;
        self.new_diameter = new_diameter;
        self.speed_millis = time_millis;
        self.start_time_millis = now_millis;
        self.end_time_millis = now_millis.wrapping_add(time_millis);
    }

    /// Evaluate `WorldBorder.getDiameter()` at an absolute Unix-millisecond instant.
    ///
    /// Vanilla casts both the elapsed time and duration to `float` before dividing, then widens
    /// the ratio back to `double` for the endpoint multiply. Once the ratio reaches `1.0` (or is
    /// NaN, as it is for a malformed zero-duration moving packet), vanilla calls
    /// `setTransition(endDiameter)`, collapsing the endpoints and timestamps. This method keeps
    /// that mutation visible to callers that hold the border mutably.
    // Keep the Java-shaped relational checks: `!=` would make NaN endpoints look like an active
    // transition, while Java's `<`/`>` pair treats them as stationary.
    #[allow(clippy::double_comparisons)]
    pub fn diameter_at(&mut self, now_millis: i64) -> f64 {
        if self.new_diameter < self.old_diameter || self.new_diameter > self.old_diameter {
            let elapsed = now_millis.wrapping_sub(self.start_time_millis);
            let duration = self.end_time_millis.wrapping_sub(self.start_time_millis);
            let progress = (elapsed as f32) / (duration as f32);
            if progress < 1.0 {
                return self.old_diameter
                    + (self.new_diameter - self.old_diameter) * f64::from(progress);
            }

            self.old_diameter = self.new_diameter;
            self.speed_millis = 0;
            self.start_time_millis = now_millis;
            self.end_time_millis = now_millis;
        }
        self.old_diameter
    }

    /// Immutable convenience form of [`BorderState::diameter_at`]. It computes the same Java
    /// value while leaving the caller's packet state untouched, which is useful to render passes
    /// that only have a shared `&BorderState`.
    #[must_use]
    pub fn current_diameter(self, now_millis: i64) -> f64 {
        let mut copy = self;
        copy.diameter_at(now_millis)
    }

    /// Projects the current border geometry for the border-aware physics collision helper.
    ///
    /// java: `World.getCollidingBoundingBoxes` reads the live `WorldBorder` every query and
    /// substitutes a stone state for positions outside it while the entity remains inside the
    /// border hysteresis region. Packet state stays owned here; a movement caller must still
    /// supply the returned projection and its entity-inside decision to
    /// [`mc_physics::collision::collision_boxes_in_with_border`].
    #[must_use]
    pub fn collision_border(self, now_millis: i64) -> mc_physics::collision::CollisionBorder {
        mc_physics::collision::CollisionBorder::new(
            self.center_x,
            self.center_z,
            self.current_diameter(now_millis),
            self.world_size,
        )
    }

    /// The exact `GuiIngame.renderVignette` world-border colour, if it overrides the normal
    /// grayscale vignette. Apply this only in vanilla's Fancy-graphics vignette pass; Fast
    /// graphics has no vignette or border warning.
    ///
    /// `current_diameter` is an input so callers can choose the same frame instant used by their
    /// render pass. [`BorderState::current_diameter`] provides the Java interpolation when only
    /// packet state is available.
    /// java: `GuiIngame.java:956-1001` (1.8.9).
    #[must_use]
    pub fn warning_vignette_tint(
        self,
        player_x: f64,
        player_z: f64,
        current_diameter: f64,
    ) -> Option<[f32; 4]> {
        let closest = self.closest_distance(player_x, player_z, current_diameter) as f32;
        let travel_in_warning_time = self.resize_speed() * f64::from(self.warning_time) * 1000.0;
        // Java's Math.min/max propagate NaN; keep that behaviour explicitly instead of using
        // Rust's f64::min/max, which choose the numeric operand.
        let remaining = (self.new_diameter - current_diameter).abs();
        let warning_travel = java_min(travel_in_warning_time, remaining);
        let warning_radius = java_max(f64::from(self.warning_blocks), warning_travel);
        let strength = if f64::from(closest) < warning_radius {
            1.0 - (f64::from(closest) / warning_radius) as f32
        } else {
            0.0
        };

        (strength > 0.0).then_some([0.0, strength, strength, 1.0])
    }
}

/// Java's `Math.min(double,double)`: NaN in either operand produces NaN.
#[inline]
fn java_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left < right || (left == 0.0 && left.is_sign_negative()) {
        left
    } else {
        right
    }
}

/// Java's `Math.max(double,double)`: NaN in either operand produces NaN.
#[inline]
fn java_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left > right || (left == 0.0 && left.is_sign_positive()) {
        left
    } else {
        right
    }
}

/// `System.currentTimeMillis()` represented as a signed Unix-millisecond value.
fn current_time_millis() -> i64 {
    let millis = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i128,
        Err(error) => -(error.duration().as_millis() as i128),
    };
    millis.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// World-scoped state synchronized from the server.
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    /// Ticks since the world was created. Never resets; drives moon phase and lightning.
    /// java: `WorldInfo.totalTime`
    pub total_world_time: i64,
    /// Signed `worldTime` from the last `S03PacketTimeUpdate`. A negative nonzero wire value
    /// signals `doDaylightCycle=false`; its Java-long absolute value is the client world's time,
    /// except `i64::MIN`, whose `Math.abs`/`wrapping_abs` result remains `i64::MIN`.
    ///
    /// Vanilla's `WorldClient.setWorldTime` consumes the sign into the gamerule and stores the
    /// absolute value; this model retains the signed wire value to derive frozen state directly.
    pub world_time: i64,
    /// `0.0`–`1.0`. java: `World.rainingStrength`
    pub rain_strength: f32,
    /// `0.0`–`1.0`. java: `World.thunderingStrength`
    pub thunder_strength: f32,
    /// java: `WorldInfo.raining`. See [`Environment::apply_game_state`] for why this is not
    /// simply `rain_strength > 0`.
    pub raining: bool,
    /// java: `WorldInfo.difficulty`, set by `S41PacketServerDifficulty`.
    pub difficulty: ServerDifficultyLevel,
    /// The world spawn. java: `WorldInfo.setSpawn` via `S05PacketSpawnPosition`.
    pub spawn_position: BlockPos,
    /// The world border.
    pub border: BorderState,
    /// `WorldClient.tick` calls this world has run. Unlike
    /// [`Self::total_world_time`], `S03PacketTimeUpdate` never touches it, so it is the steady
    /// clock [`Self::render_world_time`] measures the server's resync corrections against.
    client_ticks: i64,
    /// See [`Self::render_world_time`].
    render_clock: RenderWorldClock,
    /// While a replay plays: the recording's clock (Unix milliseconds) at
    /// the replay time being applied or drawn. The border stamps
    /// `System.currentTimeMillis()` when its packet is applied and reads it to place a moving
    /// border; on the wall clock a replay's shrinking border kept moving while paused and landed
    /// somewhere else after every seek and in every export. `None`, as in live play: the wall
    /// clock, as vanilla.
    replay_clock_millis: Option<i64>,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            total_world_time: 0,
            world_time: 0,
            rain_strength: 0.0,
            thunder_strength: 0.0,
            raining: false,
            difficulty: ServerDifficultyLevel::Normal,
            spawn_position: BlockPos::new(0, 0, 0),
            border: BorderState::default(),
            client_ticks: 0,
            render_clock: RenderWorldClock::default(),
            replay_clock_millis: None,
        }
    }
}

/// How fast [`RenderWorldClock`] absorbs a resync: ticks of correction per tick of play. A
/// one-tick `S03` correction is spread over ten ticks (half a second), during which the animation
/// runs at 90-110 % of its normal speed. It can never run backwards.
const RESYNC_SLEW_PER_TICK: f64 = 0.1;

/// A disagreement larger than this is not resync jitter but a real jump (the first `S03` after a
/// join, a long server stall), so the render clock jumps with it exactly as vanilla's does.
const RESYNC_SNAP_TICKS: f64 = 20.0;

/// The render-side copy of `getTotalWorldTime() + partialTicks`.
///
/// Vanilla animates the beacon beam from `(double) world.getTotalWorldTime() + partialTicks`
/// (`TileEntityBeaconRenderer:42`). `NetHandlerPlayClient.handleTimeUpdate` overwrites the total
/// time from every `S03` (once a second), and the client's own `WorldClient.tick` adds one per
/// tick. A server that runs a little below 20 TPS therefore pulls the client clock back a whole
/// tick every few seconds: the real 1.8.8 server capture in `crates/mc-parity/captures/` runs at
/// 19.91 TPS, so the beam turns back 2.15 degrees and its texture scrolls back a fifth of a
/// repeat about every 11 seconds. Network jitter near a tick boundary adds back-and-forth pairs.
///
/// This clock follows the server's time exactly while the two agree, so in steady state it is
/// bit-identical to vanilla, and it absorbs each small correction gradually instead of jumping.
/// Only render animations read it; gameplay state (`shouldBeamRender`'s counter, the 80-tick
/// segment refresh) stays on [`Environment::total_world_time`], exactly as vanilla.
#[derive(Clone, Debug, Default, PartialEq)]
struct RenderWorldClock {
    /// The smoothed `total_world_time - client_ticks`, `None` before the first sample.
    offset: Option<f64>,
    /// `client_ticks + partial_ticks` at the last sample.
    last_clock: f64,
}

impl RenderWorldClock {
    /// One frame's `(ticks, partial)`. Sampling again with the same clock changes nothing, so
    /// every renderer that needs the value in a frame can ask for it.
    fn sample(
        &mut self,
        client_ticks: i64,
        partial_ticks: f32,
        total_world_time: i64,
    ) -> (i64, f32) {
        let clock = client_ticks as f64 + f64::from(partial_ticks);
        let target = total_world_time.wrapping_sub(client_ticks) as f64;
        let offset = match self.offset {
            Some(previous)
                if clock >= self.last_clock && (target - previous).abs() <= RESYNC_SNAP_TICKS =>
            {
                let step = (clock - self.last_clock) * RESYNC_SLEW_PER_TICK;
                let gap = target - previous;
                // Land exactly on the target, so the early return below can fire again.
                if gap.abs() <= step {
                    target
                } else {
                    previous + step.copysign(gap)
                }
            }
            _ => target,
        };
        self.offset = Some(offset);
        self.last_clock = clock;
        if offset == target {
            // In step with the server: vanilla's own inputs, with no float round trip.
            return (total_world_time, partial_ticks);
        }
        let time = clock + offset;
        let ticks = time.floor();
        (ticks as i64, (time - ticks) as f32)
    }
}

/// What `World.getRainStrength`/`getThunderStrength` (`World.java:3517-3546`) read back after
/// `setRainStrength(value)`: `prev + (cur - prev) * delta` with `prev == cur == value`. That is
/// `value` itself for every finite input, but NaN for an infinite one (`inf - inf`).
// java: the self-subtraction is the point (`inf - inf` is NaN), not a typo.
#[allow(clippy::eq_op)]
fn java_read_of_equal_endpoints(value: f32) -> f32 {
    value + (value - value)
}

impl Environment {
    /// Fresh state for a newly-loaded world.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `S03PacketTimeUpdate`. java: `handleTimeUpdate` (`:952`)
    pub fn apply_time(&mut self, total_world_time: i64, world_time: i64) {
        self.total_world_time = total_world_time;
        self.world_time = world_time;
    }

    /// Advance the two client-side clocks by one world tick.
    ///
    /// java: `WorldClient.tick` always increments total time, and increments world time only when
    /// `doDaylightCycle` is enabled. `wrapping_add` matches Java `long` overflow.
    pub fn tick(&mut self) {
        self.total_world_time = self.total_world_time.wrapping_add(1);
        if !self.daylight_cycle_frozen() {
            self.world_time = self.world_time.wrapping_add(1);
        }
        self.client_ticks = self.client_ticks.wrapping_add(1);
    }

    /// The `(getTotalWorldTime(), partialTicks)` pair world-time-driven
    /// render animations (the beacon beam) use this frame: vanilla's values while the client and
    /// server clocks agree, and a smooth, never-backwards catch-up after an `S03` corrects the
    /// client by a few ticks. See [`RenderWorldClock`].
    pub fn render_world_time(&mut self, partial_ticks: f32) -> (i64, f32) {
        self.render_clock
            .sample(self.client_ticks, partial_ticks, self.total_world_time)
    }

    /// Whether the server has frozen the day/night cycle.
    ///
    /// The client treats a negative packet value as frozen and stores its negated Java `long`;
    /// this model keeps the signed wire value and checks negativity directly.
    #[must_use]
    pub fn daylight_cycle_frozen(&self) -> bool {
        self.world_time < 0
    }

    /// Time of day in `0..24000`, with the frozen-cycle sign removed.
    #[must_use]
    pub fn time_of_day(&self) -> i64 {
        self.world_time.wrapping_abs().rem_euclid(TICKS_PER_DAY)
    }

    /// Full positive world time, preserving the day count used for the moon phase.
    ///
    /// This is the value vanilla stores after `WorldClient.setWorldTime` consumes the packet's
    /// daylight-cycle sign. Consumers that perform their own day modulo must use this rather than
    /// [`Environment::time_of_day`].
    #[must_use]
    pub fn client_world_time(&self) -> i64 {
        self.world_time.wrapping_abs()
    }

    /// Apply the weather-relevant half of `S2BPacketChangeGameState`.
    ///
    /// Returns `true` if the reason was one this function handles, so the caller knows whether
    /// it still has to deal with the reason itself (game-mode change, demo prompt, credits).
    ///
    /// # A vanilla oddity, ported deliberately
    ///
    /// `handleChangeGameState` (`:1371-1380`) does this:
    ///
    /// ```java
    /// if (i == 1) { worldInfo.setRaining(true);  world.setRainStrength(0.0F); }
    /// else if (i == 2) { worldInfo.setRaining(false); world.setRainStrength(1.0F); }
    /// ```
    ///
    /// The `setRaining` flags are backwards relative to the reason names (1 is "end raining"),
    /// but the **strengths** are right, and `World.isRaining()` on the client is
    /// `getRainStrength(1.0F) > 0.2`, i.e. derived from the strength and not from the flag. The
    /// visible behaviour is therefore correct despite the flag being inverted. Both halves are
    /// reproduced here — the flag included — because a client that "fixed" the flag would
    /// diverge from vanilla for anything that reads it.
    pub fn apply_game_state(&mut self, reason: GameStateReason, value: f32) -> bool {
        match reason {
            GameStateReason::EndRaining => {
                self.raining = true;
                self.rain_strength = 0.0;
                true
            }
            GameStateReason::BeginRaining => {
                self.raining = false;
                self.rain_strength = 1.0;
                true
            }
            // java: `setRainStrength`/`setThunderStrength` store the packet's
            // float as both the previous and the current value, and every reader interpolates
            // `prev + (cur - prev) * delta` (`World.java:3517-3546`). That is the value
            // itself, except that an infinite one reads back as NaN (`inf - inf`). Storing the
            // infinity made `addRainParticles` loop `(int) inf` = 2^31 - 1 times: a Rust-only
            // freeze one S2B could trigger, where vanilla spawns `(int) NaN` = 0 particles.
            GameStateReason::RainLevelChange => {
                self.rain_strength = java_read_of_equal_endpoints(value);
                true
            }
            GameStateReason::ThunderLevelChange => {
                self.thunder_strength = java_read_of_equal_endpoints(value);
                true
            }
            _ => false,
        }
    }

    /// Whether it is visibly raining, by vanilla's client-side rule.
    /// java: `World.isRaining()` — `(double)getRainStrength(1.0F) > 0.2D`
    #[must_use]
    pub fn is_raining(&self) -> bool {
        f64::from(self.rain_strength) > 0.2
    }

    /// Thunder intensity after vanilla's rain weighting.
    ///
    /// `World.getThunderStrength(delta)` multiplies the interpolated thunder fade by
    /// `getRainStrength(delta)`. The change-game-state packet setters make the previous and
    /// current values equal, so the packet-driven state represented here has no remaining
    /// partial-tick interpolation; the exact client-visible value is this product.
    #[must_use]
    pub fn effective_thunder_strength(&self) -> f32 {
        self.thunder_strength * self.rain_strength
    }

    /// The current moon fullness factor used by `World.getCurrentMoonPhaseFactor()`.
    ///
    /// java: `WorldProvider.moonPhaseFactors` indexed by
    /// `(int)(worldTime / 24000L % 8L + 8L) % 8`. The signed packet convention used by this
    /// port is removed before the day calculation, matching `WorldClient.setWorldTime`'s
    /// `Math.abs` storage.
    #[must_use]
    pub fn current_moon_phase_factor(&self) -> f32 {
        const MOON_PHASE_FACTORS: [f32; 8] = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75];
        let phase = (self.client_world_time() / TICKS_PER_DAY % 8 + 8) % 8;
        MOON_PHASE_FACTORS[phase as usize]
    }

    /// The client-visible `DifficultyInstance.getAdditionalDifficulty()` value.
    ///
    /// java: `DifficultyInstance.calculateAdditionalDifficulty` (`DifficultyInstance.java:21-50`).
    /// A multiplayer client has no inhabited-time packet, so callers pass `0` for the chunk's
    /// inhabited time, exactly the value `World.getDifficultyForLocation` uses when its column
    /// is not loaded. The arithmetic deliberately stays in `f32`, including the intermediate
    /// clamps and the difficulty-id multiply.
    #[must_use]
    pub fn additional_difficulty(&self, chunk_inhabited_time: i64) -> f32 {
        let (difficulty_id, easy, hard) = match self.difficulty {
            ServerDifficultyLevel::Peaceful => return 0.0,
            ServerDifficultyLevel::Easy => (1.0_f32, true, false),
            ServerDifficultyLevel::Normal => (2.0_f32, false, false),
            ServerDifficultyLevel::Hard => (3.0_f32, false, true),
        };

        let mut base = 0.75_f32;
        let f1 =
            (((self.client_world_time() as f32) + -72_000.0) / 1_440_000.0).clamp(0.0, 1.0) * 0.25;
        base += f1;

        let inhabited_scale = if hard { 1.0 } else { 0.75 };
        let mut extra =
            ((chunk_inhabited_time as f32) / 3_600_000.0).clamp(0.0, 1.0) * inhabited_scale;
        extra += (self.current_moon_phase_factor() * 0.25).clamp(0.0, f1);
        if easy {
            extra *= 0.5;
        }

        difficulty_id * (base + extra)
    }

    /// Apply `S44PacketWorldBorder`. java: `handleWorldBorder` (`:1558`)
    pub fn apply_world_border(&mut self, packet: &WorldBorder) {
        self.apply_world_border_at(packet, self.border_now_millis());
    }

    /// The clock the world border is stamped and read with:
    /// `System.currentTimeMillis()`, or the recording's clock while a replay plays
    /// ([`Self::set_replay_clock`]).
    #[must_use]
    pub fn border_now_millis(&self) -> i64 {
        self.replay_clock_millis.unwrap_or_else(current_time_millis)
    }

    /// The replay clock, when a replay drives this world.
    #[must_use]
    pub fn replay_clock(&self) -> Option<i64> {
        self.replay_clock_millis
    }

    /// Run the border on a replay's clock (`Some`), or on the wall clock
    /// again (`None`). Only replays set it; a live session's environment never has it.
    pub fn set_replay_clock(&mut self, millis: Option<i64>) {
        self.replay_clock_millis = millis;
    }

    /// Deterministic form of [`Environment::apply_world_border`] for replay and parity tests.
    /// `now_millis` is the same Unix-millisecond clock used by `System.currentTimeMillis()`.
    pub fn apply_world_border_at(&mut self, packet: &WorldBorder, now_millis: i64) {
        match *packet {
            WorldBorder::SetSize { diameter } => {
                self.border
                    .set_transition_at(diameter, diameter, 0, now_millis);
            }
            WorldBorder::LerpSize {
                old_diameter,
                new_diameter,
                speed_millis,
            } => {
                self.border
                    .set_transition_at(old_diameter, new_diameter, speed_millis, now_millis);
            }
            WorldBorder::SetCenter { x, z } => {
                self.border.center_x = x;
                self.border.center_z = z;
            }
            WorldBorder::Initialize {
                x,
                z,
                old_diameter,
                new_diameter,
                speed_millis,
                world_size,
                warning_blocks,
                warning_time,
            } => {
                // java: `S44PacketWorldBorder.func_179788_a` sets the centre, then uses the
                // one-argument transition whenever `timeUntilTarget <= 0`, discarding the raw
                // diameter in that branch. The packet's remaining fields are then copied into
                // the existing border.
                self.border.center_x = x;
                self.border.center_z = z;
                if speed_millis > 0 {
                    self.border.set_transition_at(
                        old_diameter,
                        new_diameter,
                        speed_millis,
                        now_millis,
                    );
                } else {
                    self.border
                        .set_transition_at(new_diameter, new_diameter, 0, now_millis);
                }
                self.border.world_size = world_size;
                self.border.warning_blocks = warning_blocks;
                self.border.warning_time = warning_time;
            }
            WorldBorder::SetWarningTime { warning_time } => {
                self.border.warning_time = warning_time;
            }
            WorldBorder::SetWarningBlocks { warning_blocks } => {
                self.border.warning_blocks = warning_blocks;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nether_border_view_scales_only_the_center_without_mutating_packet_state() {
        let raw = BorderState {
            center_x: 80.0,
            center_z: -40.0,
            old_diameter: 120.0,
            new_diameter: 64.0,
            speed_millis: 9_000,
            start_time_millis: 1_000,
            end_time_millis: 10_000,
            world_size: 29_999_984,
            warning_blocks: 7,
            warning_time: 11,
            ..BorderState::default()
        };

        assert_eq!(raw.dimension_view(0), raw);
        assert_eq!(raw.dimension_view(1), raw);

        let nether = raw.dimension_view(-1);
        assert_eq!((nether.center_x, nether.center_z), (10.0, -5.0));
        let mut centers_restored = nether;
        centers_restored.center_x = raw.center_x;
        centers_restored.center_z = raw.center_z;
        assert_eq!(
            centers_restored, raw,
            "dimension projection must not scale diameter, transition, warnings, or world size"
        );
        assert_eq!(
            (raw.center_x, raw.center_z),
            (80.0, -40.0),
            "the stored packet border remains in server coordinates"
        );
    }

    #[test]
    fn a_negative_world_time_means_the_cycle_is_frozen() {
        let mut env = Environment::new();
        env.apply_time(1_000_000, 6000);
        assert!(!env.daylight_cycle_frozen());
        assert_eq!(env.time_of_day(), 6000);

        env.apply_time(1_000_001, -6000);
        assert!(env.daylight_cycle_frozen());
        assert_eq!(env.time_of_day(), 6000, "the magnitude is still the time");
        assert_eq!(env.total_world_time, 1_000_001);
    }

    #[test]
    fn time_of_day_wraps_at_one_day() {
        let mut env = Environment::new();
        env.apply_time(0, 24_000 * 7 + 13_000);
        assert_eq!(env.time_of_day(), 13_000);
        assert_eq!(
            env.client_world_time(),
            24_000 * 7 + 13_000,
            "the day count must remain available to the moon-phase calculation"
        );
    }

    #[test]
    fn minimum_packet_time_matches_java_long_negation_without_panicking() {
        let mut env = Environment::new();
        env.apply_time(0, i64::MIN);

        assert!(env.daylight_cycle_frozen());
        assert_eq!(
            env.client_world_time(),
            i64::MIN,
            "Java -Long.MIN_VALUE wraps back to Long.MIN_VALUE"
        );
        assert!((0..TICKS_PER_DAY).contains(&env.time_of_day()));
    }

    #[test]
    fn a_world_tick_advances_total_time_and_only_advances_an_active_day_cycle() {
        let mut env = Environment::new();
        env.apply_time(1_000, 6_000);
        env.tick();
        assert_eq!(env.total_world_time, 1_001);
        assert_eq!(env.client_world_time(), 6_001);

        env.apply_time(2_000, -6_000);
        env.tick();
        assert_eq!(env.total_world_time, 2_001);
        assert_eq!(
            env.world_time, -6_000,
            "the packet sign stands in for vanilla's disabled doDaylightCycle gamerule"
        );
    }

    /// The reason-1/reason-2 asymmetry described on [`Environment::apply_game_state`].
    #[test]
    fn rain_reasons_set_the_strengths_vanilla_sets() {
        let mut env = Environment::new();

        assert!(env.apply_game_state(GameStateReason::BeginRaining, 0.0));
        assert_eq!(env.rain_strength, 1.0);
        assert!(env.is_raining());
        assert!(!env.raining, "vanilla's flag really is inverted here");

        assert!(env.apply_game_state(GameStateReason::EndRaining, 0.0));
        assert_eq!(env.rain_strength, 0.0);
        assert!(!env.is_raining());
        assert!(env.raining, "and inverted the other way too");
    }

    #[test]
    fn rain_and_thunder_levels_are_set_directly() {
        let mut env = Environment::new();
        assert!(env.apply_game_state(GameStateReason::RainLevelChange, 0.25));
        assert!(env.apply_game_state(GameStateReason::ThunderLevelChange, 0.75));
        assert_eq!(env.rain_strength, 0.25);
        assert_eq!(env.thunder_strength, 0.75);
        assert!(env.is_raining(), "0.25 > 0.2");

        // `World.isRaining()` is `(double)getRainStrength(1.0F) > 0.2D`: an `f32` widened to
        // `f64`. `0.2f32` widens to 0.20000000298…, which **is** greater than the `f64` 0.2, so
        // vanilla counts a strength of exactly 0.2 as raining. Reproducing the widening rather
        // than comparing in `f32` is the difference.
        assert!(env.apply_game_state(GameStateReason::RainLevelChange, 0.2));
        assert!(env.is_raining(), "0.2f32 widens to just above 0.2f64");
        assert!(env.apply_game_state(GameStateReason::RainLevelChange, 0.19));
        assert!(!env.is_raining());
    }

    /// java: the setters store prev = current = the packet float and every
    /// reader returns `prev + (cur - prev) * delta`, so an infinite S2B 7/8 value reads back
    /// as NaN: not raining, and `addRainParticles` spawns `(int) NaN` = 0 particles. Keeping the
    /// infinity made that loop run `(int) inf` = 2^31 - 1 times (a client freeze the
    /// live-client packet fuzz found).
    #[test]
    fn an_infinite_rain_or_thunder_level_reads_back_as_javas_nan() {
        let mut env = Environment::new();
        for infinity in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(env.apply_game_state(GameStateReason::RainLevelChange, infinity));
            assert!(env.apply_game_state(GameStateReason::ThunderLevelChange, infinity));
            assert!(env.rain_strength.is_nan() && env.thunder_strength.is_nan());
            assert!(!env.is_raining());
            assert_eq!((100.0 * env.rain_strength * env.rain_strength) as i32, 0);
        }
        // Finite values (and NaN) are unchanged: `v + (v - v) * delta == v`.
        for value in [0.0, 0.25, 1.0, -3.0, f32::MAX, f32::MIN_POSITIVE] {
            assert!(env.apply_game_state(GameStateReason::RainLevelChange, value));
            assert_eq!(env.rain_strength, value);
        }
    }

    #[test]
    fn thunder_strength_is_weighted_by_rain_like_world_get_thunder_strength() {
        let mut env = Environment::new();
        assert!(env.apply_game_state(GameStateReason::ThunderLevelChange, 1.0));
        assert!(env.apply_game_state(GameStateReason::RainLevelChange, 0.5));

        assert_eq!(env.effective_thunder_strength(), 0.5);

        assert!(env.apply_game_state(GameStateReason::RainLevelChange, 0.0));
        assert_eq!(
            env.effective_thunder_strength(),
            0.0,
            "raw thunder remains one, but vanilla cannot thunder without rain"
        );
        assert_eq!(
            env.thunder_strength, 1.0,
            "store the packet fade, not the already-weighted product"
        );
    }

    #[test]
    fn non_weather_game_states_are_left_to_the_caller() {
        let mut env = Environment::new();
        assert!(!env.apply_game_state(GameStateReason::ChangeGameMode, 1.0));
        assert!(!env.apply_game_state(GameStateReason::Other(200), 0.0));
        assert_eq!(env, Environment::new());
    }

    #[test]
    fn world_border_actions_each_touch_only_their_own_fields() {
        let mut env = Environment::new();
        env.apply_world_border_at(&WorldBorder::SetCenter { x: 10.0, z: -20.0 }, 100);
        env.apply_world_border_at(&WorldBorder::SetSize { diameter: 128.0 }, 200);
        assert_eq!(env.border.center_x, 10.0);
        assert_eq!(env.border.new_diameter, 128.0);
        assert_eq!(env.border.old_diameter, 128.0);

        env.apply_world_border_at(
            &WorldBorder::LerpSize {
                old_diameter: 128.0,
                new_diameter: 64.0,
                speed_millis: 5_000,
            },
            300,
        );
        assert_eq!(env.border.speed_millis, 5_000);
        assert_eq!(env.border.center_z, -20.0, "the centre must not be touched");

        env.apply_world_border_at(&WorldBorder::SetWarningBlocks { warning_blocks: 9 }, 400);
        assert_eq!(env.border.warning_blocks, 9);
        assert_eq!(env.border.warning_time, 15);
    }

    #[test]
    fn world_border_action_table_preserves_every_packet_transition() {
        let mut env = Environment::new();
        env.apply_world_border_at(
            &WorldBorder::Initialize {
                x: 4.0,
                z: -6.0,
                old_diameter: 200.0,
                new_diameter: 100.0,
                speed_millis: 5_000,
                world_size: 300,
                warning_blocks: 7,
                warning_time: 11,
            },
            1_000,
        );

        let actions = [
            WorldBorder::SetCenter { x: 12.0, z: 13.0 },
            WorldBorder::SetSize { diameter: 80.0 },
            WorldBorder::LerpSize {
                old_diameter: 80.0,
                new_diameter: 40.0,
                speed_millis: 2_000,
            },
            WorldBorder::SetWarningTime { warning_time: 19 },
            WorldBorder::SetWarningBlocks { warning_blocks: 3 },
        ];

        env.apply_world_border_at(&actions[0], 2_000);
        assert_eq!((env.border.center_x, env.border.center_z), (12.0, 13.0));
        assert_eq!(env.border.old_diameter, 200.0);
        assert_eq!(env.border.new_diameter, 100.0);

        env.apply_world_border_at(&actions[1], 3_000);
        assert_eq!(env.border.old_diameter, 80.0);
        assert_eq!(env.border.new_diameter, 80.0);
        assert_eq!(env.border.speed_millis, 0);

        env.apply_world_border_at(&actions[2], 4_000);
        assert_eq!(env.border.old_diameter, 80.0);
        assert_eq!(env.border.new_diameter, 40.0);
        assert_eq!(env.border.speed_millis, 2_000);
        assert_eq!(env.border.start_time_millis, 4_000);
        assert_eq!(env.border.end_time_millis, 6_000);

        env.apply_world_border_at(&actions[3], 5_000);
        assert_eq!(env.border.warning_time, 19);
        assert_eq!(env.border.new_diameter, 40.0);
        env.apply_world_border_at(&actions[4], 6_000);
        assert_eq!(env.border.warning_blocks, 3);
        assert_eq!(env.border.new_diameter, 40.0);
    }

    #[test]
    fn world_border_status_ids_and_colours_match_java_enum() {
        let growing = BorderState {
            old_diameter: 10.0,
            new_diameter: 20.0,
            ..BorderState::default()
        };
        let shrinking = BorderState {
            old_diameter: 20.0,
            new_diameter: 10.0,
            ..BorderState::default()
        };
        let stationary = BorderState::default();

        assert_eq!(growing.status_id(), 4_259_712);
        assert_eq!(shrinking.status_id(), 16_724_016);
        assert_eq!(stationary.status_id(), 2_138_367);
        assert_eq!(
            growing.status_color(),
            [64.0 / 255.0, 1.0, 128.0 / 255.0, 1.0]
        );
        assert_eq!(
            shrinking.status_color(),
            [1.0, 48.0 / 255.0, 48.0 / 255.0, 1.0]
        );
        assert_eq!(
            stationary.status_color(),
            [32.0 / 255.0, 160.0 / 255.0, 1.0, 1.0]
        );

        // Negative control: status follows endpoints, not the midpoint diameter.
        let mut midpoint = BorderState {
            start_time_millis: 0,
            end_time_millis: 100,
            ..growing
        };
        assert_eq!(midpoint.diameter_at(50), 15.0);
        assert_eq!(midpoint.status_id(), growing.status_id());
    }

    #[test]
    fn world_border_lerp_uses_java_float_ratio_and_collapses_at_expiry() {
        let mut env = Environment::new();
        let packet = WorldBorder::LerpSize {
            old_diameter: 60_000_000.0,
            new_diameter: 0.0,
            speed_millis: 10_000,
        };
        env.apply_world_border_at(&packet, 1_000_000);

        let java_progress = f64::from(3_333_f32 / 10_000_f32);
        let double_progress = 3_333.0_f64 / 10_000.0;
        assert_ne!(
            java_progress, double_progress,
            "the negative control must distinguish f32"
        );
        let expected = 60_000_000.0 * (1.0 - java_progress);
        assert_eq!(env.border.diameter_at(1_003_333), expected);
        assert_eq!(env.border.old_diameter, 60_000_000.0);
        assert_eq!(env.border.new_diameter, 0.0);

        assert_eq!(env.border.diameter_at(1_010_000), 0.0);
        assert_eq!(env.border.old_diameter, 0.0);
        assert_eq!(env.border.new_diameter, 0.0);
        assert_eq!(env.border.speed_millis, 0);
        assert_eq!(env.border.start_time_millis, 1_010_000);
        assert_eq!(env.border.end_time_millis, 1_010_000);
        assert_eq!(env.border.diameter_at(1_010_001), 0.0);
    }

    #[test]
    fn physics_border_projection_tracks_static_and_moving_diameters() {
        let mut env = Environment::new();
        env.apply_world_border_at(
            &WorldBorder::Initialize {
                x: 0.0,
                z: 0.0,
                old_diameter: 8.0,
                new_diameter: 8.0,
                speed_millis: 0,
                world_size: 100,
                warning_blocks: 5,
                warning_time: 15,
            },
            1_000,
        );
        let static_border = env.border.collision_border(1_000);
        assert_eq!(static_border.diameter, 8.0);
        assert!(!static_border.contains_block(BlockPos::new(4, 64, 0)));

        env.apply_world_border_at(
            &WorldBorder::LerpSize {
                old_diameter: 8.0,
                new_diameter: 12.0,
                speed_millis: 1_000,
            },
            2_000,
        );
        assert_eq!(env.border.collision_border(2_000).diameter, 8.0);
        assert_eq!(env.border.collision_border(2_500).diameter, 10.0);
        assert_eq!(env.border.collision_border(3_000).diameter, 12.0);
        assert!(
            env.border
                .collision_border(2_500)
                .contains_block(BlockPos::new(4, 64, 0)),
            "the moving edge admits x=4 at the midpoint"
        );
    }

    #[test]
    fn world_border_initialize_without_time_uses_target_size_like_java() {
        let mut env = Environment::new();
        let packet = WorldBorder::Initialize {
            x: 12.0,
            z: -7.0,
            old_diameter: 200.0,
            new_diameter: 100.0,
            speed_millis: 0,
            world_size: 150,
            warning_blocks: 8,
            warning_time: 22,
        };
        assert_ne!(
            200.0, 100.0,
            "negative control: the wire diameter differs from the target"
        );
        env.apply_world_border_at(&packet, 42);
        assert_eq!(env.border.center_x, 12.0);
        assert_eq!(env.border.center_z, -7.0);
        assert_eq!(env.border.old_diameter, 100.0);
        assert_eq!(env.border.new_diameter, 100.0);
        assert_eq!(env.border.speed_millis, 0);
        assert_eq!(env.border.start_time_millis, 42);
        assert_eq!(env.border.end_time_millis, 42);
        assert_eq!(env.border.world_size, 150);
        assert_eq!(env.border.warning_blocks, 8);
        assert_eq!(env.border.warning_time, 22);
    }

    #[test]
    fn world_border_geometry_clamps_to_world_size_and_uses_strict_edges() {
        let border = BorderState {
            center_x: 100.0,
            center_z: -100.0,
            old_diameter: 400.0,
            new_diameter: 400.0,
            world_size: 150,
            ..BorderState::default()
        };
        assert_eq!(border.min_x(400.0), -100.0);
        assert_eq!(border.max_x(400.0), 150.0);
        assert_eq!(border.min_z(400.0), -150.0);
        assert_eq!(border.max_z(400.0), 100.0);
        assert!(border.contains_block(BlockPos::new(149, 64, 0), 400.0));
        assert!(!border.contains_block(BlockPos::new(150, 64, 0), 400.0));
        assert_eq!(border.closest_distance(149.5, 0.0, 400.0), 0.5);
        assert_eq!(border.closest_distance(151.0, 0.0, 400.0), -1.0);
    }

    #[test]
    fn world_border_damage_defaults_and_java_floor_controls() {
        let border = BorderState::default();
        assert_eq!(border.damage_amount, DEFAULT_BORDER_DAMAGE_AMOUNT);
        assert_eq!(border.damage_buffer, DEFAULT_BORDER_DAMAGE_BUFFER);

        // Java tests `d0 < 0.0D`, so exactly at the five-block grace boundary is harmless.
        assert_eq!(
            BorderState::damage_for_distance(-5.0, border.damage_amount, border.damage_buffer),
            None
        );
        // Any negative remainder still deals at least one point: floor(0.2 * epsilon) is zero,
        // then Java's Math.max(1, ...) supplies the minimum.
        assert_eq!(
            BorderState::damage_for_distance(
                -5.000_001,
                border.damage_amount,
                border.damage_buffer
            ),
            Some(1)
        );
        assert_eq!(
            BorderState::damage_for_distance(-10.0, border.damage_amount, border.damage_buffer),
            Some(1)
        );
        assert_eq!(
            BorderState::damage_for_distance(-10.5, 0.5, 5.0),
            Some(2),
            "floor is applied before the minimum-one clamp"
        );

        // Controls remain pure and explicit: a custom buffer changes only the activation edge,
        // while zero/negative amounts still preserve Java's minimum-one result.
        assert_eq!(BorderState::damage_for_distance(-1.0, 0.0, 0.0), Some(1));
        assert_eq!(BorderState::damage_for_distance(-1.0, -20.0, 0.0), Some(1));
        assert_eq!(border.damage_for_point(0.0, 0.0, 60_000_000.0), None);
    }

    #[test]
    fn world_border_aabb_damage_gate_uses_strict_horizontal_edges() {
        let border = BorderState {
            old_diameter: 10.0,
            new_diameter: 10.0,
            ..BorderState::default()
        };
        // West edge is x=-5.  A box ending exactly at the wall is outside; one epsilon inside is
        // contained, matching AxisAlignedBB's strict `maxX > minX`/`minX < maxX` tests.
        assert!(!border.contains_aabb(Aabb::from_bounds(-6.0, 64.0, -0.5, -5.0, 66.0, 0.5), 10.0));
        assert!(border.contains_aabb(
            Aabb::from_bounds(-6.0, 64.0, -0.5, -5.0 + 1.0e-12, 66.0, 0.5),
            10.0
        ));
        // Y is deliberately irrelevant to WorldBorder.contains(AxisAlignedBB): this box spans
        // the border in X/Z while living far below the normal world height.
        assert!(border.contains_aabb(
            Aabb::from_bounds(4.0, -1_000.0, -0.5, 6.0, -999.0, 0.5),
            10.0
        ));
    }

    #[test]
    fn java_min_max_keep_nan_and_signed_zero_controls() {
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn border_warning_vignette_is_cyan_at_the_exact_java_radius() {
        let border = BorderState {
            old_diameter: 200.0,
            new_diameter: 100.0,
            speed_millis: 10_000,
            start_time_millis: 0,
            end_time_millis: 10_000,
            warning_blocks: 5,
            warning_time: 15,
            ..BorderState::default()
        };
        // The current 150-wide border has an edge at 75.  Its 50-block warning radius comes
        // from min(0.01 blocks/ms * 15 sec * 1000, |100 - 150|), not the flat 5-block floor.
        assert_eq!(
            border.warning_vignette_tint(50.0, 0.0, 150.0),
            Some([0.0, 0.5, 0.5, 1.0])
        );
        assert_eq!(
            border.warning_vignette_tint(25.0, 0.0, 150.0),
            None,
            "the strict `f > 0` colour branch excludes the exact radius"
        );
    }

    #[test]
    fn border_warning_vignette_uses_negative_outside_distance_and_java_nan_rule() {
        let border = BorderState {
            old_diameter: 200.0,
            new_diameter: 100.0,
            speed_millis: 10_000,
            start_time_millis: 0,
            end_time_millis: 10_000,
            warning_blocks: 5,
            warning_time: 15,
            ..BorderState::default()
        };
        // Outside values are not clamped before the Java colour calculation.
        assert_eq!(
            border.warning_vignette_tint(80.0, 0.0, 150.0),
            Some([0.0, 1.1, 1.1, 1.0])
        );

        let stationary = BorderState {
            old_diameter: 100.0,
            new_diameter: 100.0,
            speed_millis: 0,
            start_time_millis: 0,
            end_time_millis: 0,
            ..BorderState::default()
        };
        assert_eq!(stationary.resize_speed(), 0.0);
        // Java narrows the quotient to float before subtracting from `1.0F`.  The literal
        // `0.4F` is one ULP above that evaluation (`0.39999998F`), so keep the independent
        // operation-order oracle instead of rounding the expected value for readability.
        let java_strength = 1.0_f32 - (3.0_f64 / 5.0_f64) as f32;
        assert_ne!(
            java_strength, 0.4_f32,
            "negative control must preserve float order"
        );
        assert_eq!(
            stationary.warning_vignette_tint(47.0, 0.0, 100.0),
            Some([0.0, java_strength, java_strength, 1.0]),
            "a stationary border still uses the flat warning distance"
        );
    }

    #[test]
    fn difficulty_instance_matches_vanilla_formula_and_peaceful_control() {
        let mut env = Environment::new();
        env.difficulty = ServerDifficultyLevel::Hard;
        env.apply_time(0, 1_512_000);
        let f1 = ((1_512_000_f32 - 72_000.0) / 1_440_000.0).clamp(0.0, 1.0) * 0.25;
        let moon = 0.25_f32; // day 63, phase 7 -> 0.75 would clamp to f1; use explicit control below
        let expected = 3.0_f32 * (0.75 + f1 + (moon * 0.25).clamp(0.0, f1));
        // The actual phase for day 63 is 7 (0.75); the expression is intentionally kept
        // separate above so a test mutant cannot hide the f1 clamp in the expected value.
        let expected_actual = 3.0_f32 * (0.75 + f1 + (0.75_f32 * 0.25).clamp(0.0, f1));
        assert_ne!(
            expected, expected_actual,
            "negative control must exercise moon factor"
        );
        assert_eq!(env.additional_difficulty(0), expected_actual);

        env.difficulty = ServerDifficultyLevel::Peaceful;
        assert_eq!(env.additional_difficulty(3_600_000), 0.0);
    }


    /// Frames per tick of a 360 Hz client at 20 TPS.
    const W7_FRAMES_PER_TICK: u32 = 18;

    /// One frame: `ticks` world ticks (`WorldClient.tick`), then both clocks as `f64` world time,
    /// `(vanilla's getTotalWorldTime() + partialTicks, the render clock)`.
    fn w7_frame(env: &mut Environment, ticks: u32, partial: f32) -> (f64, f64) {
        for _ in 0..ticks {
            env.tick();
        }
        let vanilla = env.total_world_time as f64 + f64::from(partial);
        let (ticks, partial) = env.render_world_time(partial);
        (vanilla, ticks as f64 + f64::from(partial))
    }

    /// The frame-`frame` tick count and partial of a client whose ticks fall every 18 frames.
    fn w7_step(frame: u32) -> (u32, f32) {
        let tick = u32::from(frame > 0 && frame.is_multiple_of(W7_FRAMES_PER_TICK));
        (
            tick,
            (frame % W7_FRAMES_PER_TICK) as f32 / W7_FRAMES_PER_TICK as f32,
        )
    }

    #[test]
    fn w7_render_world_time_is_vanillas_exact_pair_while_the_clocks_agree() {
        let mut env = Environment::new();
        // Past 2^24 ticks, where an `f32` clock could not even hold the partial tick.
        env.apply_time(50_000_000, 6_000);
        for frame in 0..W7_FRAMES_PER_TICK * 30 {
            let (ticks, partial) = w7_step(frame);
            for _ in 0..ticks {
                env.tick();
            }
            assert_eq!(
                env.render_world_time(partial),
                (env.total_world_time, partial),
                "frame {frame}"
            );
            // A once-a-second S03 that agrees with the client is no correction at all.
            if frame % (W7_FRAMES_PER_TICK * 20) == 7 {
                env.apply_time(env.total_world_time, env.world_time);
            }
        }
    }

    #[test]
    fn w7_an_s03_one_tick_back_never_turns_the_render_clock_back() {
        let mut env = Environment::new();
        env.apply_time(50_000_000, 6_000);
        let correction_frame = W7_FRAMES_PER_TICK * 10 + 5;
        let (mut last_vanilla, mut last_render) = (f64::MIN, f64::MIN);
        let mut vanilla_steps_back = 0;
        let mut rejoined_at = None;
        for frame in 0..W7_FRAMES_PER_TICK * 40 {
            // `before_ticks` applies the packet before the frame's ticks, as `runGameLoop` does.
            if frame == correction_frame {
                env.apply_time(env.total_world_time - 1, env.world_time - 1);
            }
            let (ticks, partial) = w7_step(frame);
            let (vanilla, render) = w7_frame(&mut env, ticks, partial);
            if vanilla < last_vanilla {
                vanilla_steps_back += 1;
                assert!((last_vanilla - vanilla - (1.0 - 1.0 / 18.0)).abs() < 1e-6);
            }
            assert!(
                render >= last_render,
                "frame {frame}: {render} < {last_render}"
            );
            if frame > correction_frame && rejoined_at.is_none() && render == vanilla {
                rejoined_at = Some(frame);
            }
            last_vanilla = vanilla;
            last_render = render;
        }
        assert_eq!(
            vanilla_steps_back, 1,
            "control: vanilla's clock does step back here"
        );
        let took = rejoined_at.expect("the render clock rejoins vanilla") - correction_frame;
        assert!(
            took <= W7_FRAMES_PER_TICK * 10 + 1,
            "one tick absorbed in ten, took {took}"
        );
    }

    #[test]
    fn w7_an_s03_two_ticks_ahead_speeds_the_render_clock_up_without_a_jump() {
        let mut env = Environment::new();
        env.apply_time(9_000, 9_000);
        let correction_frame = W7_FRAMES_PER_TICK * 3 + 2;
        let mut last_render = f64::MIN;
        for frame in 0..W7_FRAMES_PER_TICK * 40 {
            if frame == correction_frame {
                env.apply_time(env.total_world_time + 2, env.world_time + 2);
            }
            let (ticks, partial) = w7_step(frame);
            let (vanilla, render) = w7_frame(&mut env, ticks, partial);
            if frame > 0 {
                let step = render - last_render;
                let nominal = 1.0 / f64::from(W7_FRAMES_PER_TICK);
                assert!(step >= 0.0, "frame {frame}");
                // The f32 partial ticks carry ~6e-8 of rounding per step.
                assert!(
                    step <= nominal * 1.1 + 1e-6,
                    "frame {frame}: a {step} tick jump"
                );
            }
            if frame == W7_FRAMES_PER_TICK * 39 {
                assert_eq!(render, vanilla, "two ticks absorbed in twenty");
            }
            last_render = render;
        }
    }

    #[test]
    fn w7_a_big_correction_snaps_exactly_like_vanilla() {
        let mut env = Environment::new();
        let _ = w7_frame(&mut env, 3, 0.5);
        // The first S03 after a join: the client's clock is still near zero.
        env.apply_time(28_610, 28_610);
        let (vanilla, render) = w7_frame(&mut env, 0, 0.75);
        assert_eq!(render, vanilla);
        // A long server stall: twenty-one ticks at once is not resync jitter.
        env.apply_time(env.total_world_time + 21, env.world_time + 21);
        let (vanilla, render) = w7_frame(&mut env, 1, 0.25);
        assert_eq!(render, vanilla);
        // A fresh world starts a fresh clock (`ClientState` replaces the environment).
        assert_eq!(Environment::new().render_world_time(0.5), (0, 0.5));
    }

    #[test]
    fn w7_sampling_twice_in_one_frame_returns_the_same_pair() {
        let mut env = Environment::new();
        env.apply_time(1_000, 1_000);
        let _ = w7_frame(&mut env, 1, 0.5);
        env.apply_time(env.total_world_time - 1, env.world_time - 1);
        let _ = w7_frame(&mut env, 1, 0.25);
        let first = env.render_world_time(0.25);
        assert_ne!(first, (env.total_world_time, 0.25), "mid-correction");
        assert_eq!(env.render_world_time(0.25), first);
    }
}
