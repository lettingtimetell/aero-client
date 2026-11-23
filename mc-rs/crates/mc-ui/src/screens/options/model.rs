//! The option model: one enum over `options.txt`, plus vanilla's label and cycle rules.
//!
//! java: `client/settings/GameSettings.java` — the `Options` enum, `getKeyBinding`,
//! `setOptionValue`, `getOptionFloatValue`, `setOptionFloatValue`, `getOptionOrdinalValue`
//!
//! Every options screen in vanilla is the same loop over a `GameSettings.Options[]`, so this is
//! the one file that knows how an option is spelled, what it is worth, and what pressing it
//! does. The screens are then just layout.
//!
//! `mc-render`'s [`GameSettings`] is the single settings struct (§7 S10); nothing here keeps a
//! second copy.

use mc_assets::Language;
use mc_render::settings::{
    native_render_distance_max, GameSettings, FRAMERATE_UNLIMITED, MAX_FRAMERATE_LIMIT,
};

// Evenly spaced slider stops keep ordinary caps easy to select while extending to 2000.
// The final stop retains vanilla's saved Unlimited sentinel; 260 is never a finite option.
const FRAMERATE_CHOICES: &[i32] = &[
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    170,
    180,
    190,
    200,
    210,
    220,
    230,
    240,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
    1250,
    1500,
    1750,
    2000,
    FRAMERATE_UNLIMITED,
];

fn framerate_stop(value: f32) -> usize {
    let last = FRAMERATE_CHOICES.len() - 1;
    if !value.is_finite()
        || value == FRAMERATE_UNLIMITED as f32
        || value > MAX_FRAMERATE_LIMIT as f32
    {
        return last;
    }
    FRAMERATE_CHOICES[..last]
        .iter()
        .enumerate()
        .fold(0, |best, (i, &cap)| {
            if (cap as f32 - value).abs() <= (FRAMERATE_CHOICES[best] as f32 - value).abs() {
                i
            } else {
                best
            }
        })
}

/// Every option this port models.
///
/// The Twitch-stream options (`STREAM_*`) are omitted: the service is dead and §2 lists it as
/// permanently out of scope. `FBO_ENABLE`, `USE_VBO` and `SNOOPER_ENABLED` are kept because
/// `options.txt` round-trips them, even though nothing reads them under wgpu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptionId {
    /// `options.invertMouse`
    InvertMouse,
    /// `options.sensitivity`
    Sensitivity,
    /// `options.fov`, in degrees.
    Fov,
    /// `options.gamma`
    Gamma,
    /// `options.saturation`
    Saturation,
    /// `options.renderDistance`, in chunks.
    RenderDistance,
    /// `options.viewBobbing`
    ViewBobbing,
    /// `options.anaglyph`
    Anaglyph,
    /// `options.framerateLimit`
    FramerateLimit,
    /// `options.fboEnable`
    FboEnable,
    /// `options.renderClouds`
    RenderClouds,
    /// `options.graphics`
    Graphics,
    /// `options.ao`
    AmbientOcclusion,
    /// `options.guiScale`
    GuiScale,
    /// `options.particles`
    Particles,
    /// `options.chat.visibility`
    ChatVisibility,
    /// `options.chat.color`
    ChatColor,
    /// `options.chat.links`
    ChatLinks,
    /// `options.chat.opacity`
    ChatOpacity,
    /// `options.chat.links.prompt`
    ChatLinksPrompt,
    /// `options.snooper`
    SnooperEnabled,
    /// `options.fullscreen`
    Fullscreen,
    /// `options.vsync`
    Vsync,
    /// `options.vbo`
    UseVbo,
    /// `options.touchscreen`
    Touchscreen,
    /// `options.chat.scale`
    ChatScale,
    /// `options.chat.width`
    ChatWidth,
    /// `options.chat.height.focused`
    ChatHeightFocused,
    /// `options.chat.height.unfocused`
    ChatHeightUnfocused,
    /// `options.mipmapLevels`
    MipmapLevels,
    /// `options.forceUnicodeFont`
    ForceUnicodeFont,
    /// `options.blockAlternatives`
    BlockAlternatives,
    /// `options.reducedDebugInfo`
    ReducedDebugInfo,
    /// `options.entityShadows`
    EntityShadows,
    /// `options.realmsNotifications`
    RealmsNotifications,
    /// `options.modelPart.cape`
    ModelPartCape,
    /// `options.modelPart.jacket`
    ModelPartJacket,
    /// `options.modelPart.left_sleeve`
    ModelPartLeftSleeve,
    /// `options.modelPart.right_sleeve`
    ModelPartRightSleeve,
    /// `options.modelPart.left_pants_leg`
    ModelPartLeftPantsLeg,
    /// `options.modelPart.right_pants_leg`
    ModelPartRightPantsLeg,
    /// `options.modelPart.hat`
    ModelPartHat,
    /// `fullbright` — **not a vanilla option** (see `GameSettings::fullbright`). Not placed on any
    /// vanilla screen; a non-vanilla page lists it.
    Fullbright,
}

/// How an option is presented: a slider, an on/off toggle, or a cycle through named values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionKind {
    /// A slider between `min` and `max`, snapped to `step` (0 means continuous).
    Float {
        /// Lowest value.
        min: f32,
        /// Highest value.
        max: f32,
        /// Snap increment, or 0.
        step: f32,
    },
    /// An on/off button.
    Boolean,
    /// A button cycling through `count` named values.
    Cycle {
        /// How many values there are.
        count: i32,
    },
}

/// Vanilla's `GUISCALES`.
pub const GUI_SCALES: [&str; 4] = [
    "options.guiScale.auto",
    "options.guiScale.small",
    "options.guiScale.normal",
    "options.guiScale.large",
];
/// Vanilla's `PARTICLES`.
pub const PARTICLES: [&str; 3] = [
    "options.particles.all",
    "options.particles.decreased",
    "options.particles.minimal",
];
/// Vanilla's `AMBIENT_OCCLUSIONS`.
pub const AMBIENT_OCCLUSIONS: [&str; 3] = ["options.ao.off", "options.ao.min", "options.ao.max"];
/// Vanilla's `CLOUDS_TYPES`.
pub const CLOUDS_TYPES: [&str; 3] = [
    "options.off",
    "options.graphics.fast",
    "options.graphics.fancy",
];
/// Vanilla's chat-visibility keys, in `EnumChatVisibility` order.
pub const CHAT_VISIBILITIES: [&str; 3] = [
    "options.chat.visibility.full",
    "options.chat.visibility.system",
    "options.chat.visibility.hidden",
];

/// `getTranslation`: an out-of-range index falls back to element 0 rather than throwing.
// java: GameSettings.java:600
#[must_use]
pub fn translation_of<'a>(keys: &[&'a str], index: i32) -> &'a str {
    let i = if index < 0 || index as usize >= keys.len() {
        0
    } else {
        index as usize
    };
    keys[i]
}

impl OptionId {
    /// The translation key for the option's name.
    #[must_use]
    pub const fn translation_key(self) -> &'static str {
        match self {
            OptionId::InvertMouse => "options.invertMouse",
            OptionId::Sensitivity => "options.sensitivity",
            OptionId::Fov => "options.fov",
            OptionId::Gamma => "options.gamma",
            OptionId::Saturation => "options.saturation",
            OptionId::RenderDistance => "options.renderDistance",
            OptionId::ViewBobbing => "options.viewBobbing",
            OptionId::Anaglyph => "options.anaglyph",
            OptionId::FramerateLimit => "options.framerateLimit",
            OptionId::FboEnable => "options.fboEnable",
            OptionId::RenderClouds => "options.renderClouds",
            OptionId::Graphics => "options.graphics",
            OptionId::AmbientOcclusion => "options.ao",
            OptionId::GuiScale => "options.guiScale",
            OptionId::Particles => "options.particles",
            OptionId::ChatVisibility => "options.chat.visibility",
            OptionId::ChatColor => "options.chat.color",
            OptionId::ChatLinks => "options.chat.links",
            OptionId::ChatOpacity => "options.chat.opacity",
            OptionId::ChatLinksPrompt => "options.chat.links.prompt",
            OptionId::SnooperEnabled => "options.snooper",
            OptionId::Fullscreen => "options.fullscreen",
            OptionId::Vsync => "options.vsync",
            OptionId::UseVbo => "options.vbo",
            OptionId::Touchscreen => "options.touchscreen",
            OptionId::ChatScale => "options.chat.scale",
            OptionId::ChatWidth => "options.chat.width",
            OptionId::ChatHeightFocused => "options.chat.height.focused",
            OptionId::ChatHeightUnfocused => "options.chat.height.unfocused",
            OptionId::MipmapLevels => "options.mipmapLevels",
            OptionId::ForceUnicodeFont => "options.forceUnicodeFont",
            OptionId::BlockAlternatives => "options.blockAlternatives",
            OptionId::ReducedDebugInfo => "options.reducedDebugInfo",
            OptionId::EntityShadows => "options.entityShadows",
            OptionId::RealmsNotifications => "options.realmsNotifications",
            OptionId::ModelPartCape => "options.modelPart.cape",
            OptionId::ModelPartJacket => "options.modelPart.jacket",
            OptionId::ModelPartLeftSleeve => "options.modelPart.left_sleeve",
            OptionId::ModelPartRightSleeve => "options.modelPart.right_sleeve",
            OptionId::ModelPartLeftPantsLeg => "options.modelPart.left_pants_leg",
            OptionId::ModelPartRightPantsLeg => "options.modelPart.right_pants_leg",
            OptionId::ModelPartHat => "options.modelPart.hat",
            // No language file has a key for this non-vanilla option, and `Language::tr` returns
            // the key on a miss, so the key is the English label itself.
            OptionId::Fullbright => "Fullbright",
        }
    }

    /// Whether this option is a slider, a toggle, or a cycle.
    #[must_use]
    pub const fn kind(self) -> OptionKind {
        self.kind_for_render_distance_max(native_render_distance_max())
    }

    /// Build the option kind with an injected render-distance capability.
    ///
    /// Vanilla mutates the otherwise-global `RENDER_DISTANCE` enum value from 16 to 32 during
    /// `GameSettings` construction on a capable 64-bit runtime. Keeping the capability as an
    /// input makes both vanilla branches testable while [`Self::kind`] uses the native policy.
    const fn kind_for_render_distance_max(self, render_distance_max: i32) -> OptionKind {
        use OptionId as O;
        match self {
            O::Sensitivity
            | O::Gamma
            | O::Saturation
            | O::ChatOpacity
            | O::ChatScale
            | O::ChatWidth
            | O::ChatHeightFocused
            | O::ChatHeightUnfocused => OptionKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.0,
            },
            O::Fov => OptionKind::Float {
                min: 30.0,
                max: 110.0,
                step: 1.0,
            },
            O::RenderDistance => OptionKind::Float {
                min: 2.0,
                max: render_distance_max as f32,
                step: 1.0,
            },
            O::FramerateLimit => OptionKind::Float {
                min: 10.0,
                max: MAX_FRAMERATE_LIMIT as f32,
                step: 10.0,
            },
            O::MipmapLevels => OptionKind::Float {
                min: 0.0,
                max: 4.0,
                step: 1.0,
            },
            O::RenderClouds | O::AmbientOcclusion | O::Particles | O::ChatVisibility => {
                OptionKind::Cycle { count: 3 }
            }
            O::Graphics => OptionKind::Cycle { count: 2 },
            O::GuiScale => OptionKind::Cycle { count: 4 },
            _ => OptionKind::Boolean,
        }
    }

    /// The current value as a float, for the slider options.
    // java: getOptionFloatValue
    #[must_use]
    pub fn float_value(self, s: &GameSettings) -> f32 {
        use OptionId as O;
        match self {
            O::Sensitivity => s.mouse_sensitivity,
            O::Fov => s.fov,
            O::Gamma => s.gamma,
            O::Saturation => s.saturation,
            O::ChatOpacity => s.chat_opacity,
            O::ChatHeightFocused => s.chat_height_focused,
            O::ChatHeightUnfocused => s.chat_height_unfocused,
            O::ChatScale => s.chat_scale,
            O::ChatWidth => s.chat_width,
            O::FramerateLimit => s
                .framerate_limit()
                .map_or(FRAMERATE_UNLIMITED as f32, |fps| fps as f32),
            O::MipmapLevels => s.mipmap_levels as f32,
            O::RenderDistance => s.render_distance_chunks as f32,
            _ => 0.0,
        }
    }

    /// Write a slider option back.
    // java: setOptionFloatValue
    pub fn set_float_value(self, s: &mut GameSettings, value: f32) {
        use OptionId as O;
        match self {
            O::Sensitivity => s.mouse_sensitivity = value,
            O::Fov => s.fov = value,
            O::Gamma => s.gamma = value,
            O::Saturation => s.saturation = value,
            O::ChatOpacity => s.chat_opacity = value,
            O::ChatHeightFocused => s.chat_height_focused = value,
            O::ChatHeightUnfocused => s.chat_height_unfocused = value,
            O::ChatScale => s.chat_scale = value,
            O::ChatWidth => s.chat_width = value,
            O::FramerateLimit => s.limit_framerate = value as i32,
            O::MipmapLevels => s.mipmap_levels = value as i32,
            O::RenderDistance => s.render_distance_chunks = value as i32,
            _ => {}
        }
    }

    /// The current value of a boolean option.
    // java: getOptionOrdinalValue
    #[must_use]
    pub fn bool_value(self, s: &GameSettings) -> bool {
        use OptionId as O;
        match self {
            O::InvertMouse => s.invert_mouse,
            O::ViewBobbing => s.view_bobbing,
            O::Anaglyph => s.anaglyph,
            O::FboEnable => s.fbo_enable,
            O::ChatColor => s.chat_colors,
            O::ChatLinks => s.chat_links,
            O::ChatLinksPrompt => s.chat_links_prompt,
            O::SnooperEnabled => s.snooper_enabled,
            O::Fullscreen => s.fullscreen,
            O::Vsync => s.enable_vsync,
            O::UseVbo => s.use_vbo,
            O::Touchscreen => s.touchscreen,
            O::ForceUnicodeFont => s.force_unicode_font,
            O::BlockAlternatives => s.allow_block_alternatives,
            O::ReducedDebugInfo => s.reduced_debug_info,
            O::EntityShadows => s.entity_shadows,
            O::RealmsNotifications => s.realms_notifications,
            O::Graphics => s.fancy_graphics,
            O::ModelPartCape => s.model_parts.get("cape").copied().unwrap_or(true),
            O::ModelPartJacket => s.model_parts.get("jacket").copied().unwrap_or(true),
            O::ModelPartLeftSleeve => s.model_parts.get("left_sleeve").copied().unwrap_or(true),
            O::ModelPartRightSleeve => s.model_parts.get("right_sleeve").copied().unwrap_or(true),
            O::ModelPartLeftPantsLeg => {
                s.model_parts.get("left_pants_leg").copied().unwrap_or(true)
            }
            O::ModelPartRightPantsLeg => s
                .model_parts
                .get("right_pants_leg")
                .copied()
                .unwrap_or(true),
            O::ModelPartHat => s.model_parts.get("hat").copied().unwrap_or(true),
            O::Fullbright => s.fullbright,
            _ => false,
        }
    }

    /// The current value of a cycling option, as an index.
    #[must_use]
    pub fn cycle_value(self, s: &GameSettings) -> i32 {
        use OptionId as O;
        match self {
            O::RenderClouds => s.clouds,
            O::AmbientOcclusion => s.ambient_occlusion,
            O::GuiScale => s.gui_scale,
            O::Particles => s.particles,
            O::ChatVisibility => s.chat_visibility,
            O::Graphics => i32::from(s.fancy_graphics),
            _ => 0,
        }
    }

    /// Press the option's button: toggle a boolean, advance a cycle by `by`.
    ///
    /// Returns whether something changed that invalidates laid-out screens — a GUI scale or a
    /// font switch — so the caller knows to re-`init` every open screen.
    // java: setOptionValue
    pub fn cycle(self, s: &mut GameSettings, by: i32) -> bool {
        use OptionId as O;
        match self {
            // java: `this.guiScale = this.guiScale + value & 3` — a *mask*, not a modulo, so it
            // wraps through 0..3 and a negative step wraps the other way too.
            O::GuiScale => {
                s.gui_scale = (s.gui_scale + by) & 3;
                return true;
            }
            O::Particles => s.particles = (s.particles + by).rem_euclid(3),
            O::RenderClouds => s.clouds = (s.clouds + by).rem_euclid(3),
            O::AmbientOcclusion => s.ambient_occlusion = (s.ambient_occlusion + by).rem_euclid(3),
            O::ChatVisibility => s.chat_visibility = (s.chat_visibility + by).rem_euclid(3),
            O::Graphics => s.fancy_graphics = !s.fancy_graphics,
            O::InvertMouse => s.invert_mouse = !s.invert_mouse,
            O::ViewBobbing => s.view_bobbing = !s.view_bobbing,
            O::Anaglyph => s.anaglyph = !s.anaglyph,
            O::FboEnable => s.fbo_enable = !s.fbo_enable,
            O::ChatColor => s.chat_colors = !s.chat_colors,
            O::ChatLinks => s.chat_links = !s.chat_links,
            O::ChatLinksPrompt => s.chat_links_prompt = !s.chat_links_prompt,
            O::SnooperEnabled => s.snooper_enabled = !s.snooper_enabled,
            O::Fullscreen => s.fullscreen = !s.fullscreen,
            O::Vsync => s.enable_vsync = !s.enable_vsync,
            O::UseVbo => s.use_vbo = !s.use_vbo,
            O::Touchscreen => s.touchscreen = !s.touchscreen,
            O::BlockAlternatives => s.allow_block_alternatives = !s.allow_block_alternatives,
            O::ReducedDebugInfo => s.reduced_debug_info = !s.reduced_debug_info,
            O::EntityShadows => s.entity_shadows = !s.entity_shadows,
            O::RealmsNotifications => s.realms_notifications = !s.realms_notifications,
            O::ModelPartCape => toggle_model_part(s, "cape"),
            O::ModelPartJacket => toggle_model_part(s, "jacket"),
            O::ModelPartLeftSleeve => toggle_model_part(s, "left_sleeve"),
            O::ModelPartRightSleeve => toggle_model_part(s, "right_sleeve"),
            O::ModelPartLeftPantsLeg => toggle_model_part(s, "left_pants_leg"),
            O::ModelPartRightPantsLeg => toggle_model_part(s, "right_pants_leg"),
            O::ModelPartHat => toggle_model_part(s, "hat"),
            O::Fullbright => s.fullbright = !s.fullbright,
            O::ForceUnicodeFont => {
                s.force_unicode_font = !s.force_unicode_font;
                return true;
            }
            _ => {}
        }
        false
    }

    /// `snapToStepClamp`: round to the nearest step, then clamp into range.
    ///
    /// Java's `Math.round(float)` is `floor(x + 0.5)`, which differs from Rust's `f32::round`
    /// on exact `.5` negatives — reproduced here rather than substituted.
    // java: GameSettings.Options.snapToStepClamp
    #[must_use]
    pub fn snap_to_step_clamp(self, value: f32) -> f32 {
        if self == Self::FramerateLimit {
            return FRAMERATE_CHOICES[framerate_stop(value)] as f32;
        }
        let OptionKind::Float { min, max, step } = self.kind() else {
            return value;
        };
        let v = if step > 0.0 {
            step * (value / step + 0.5).floor()
        } else {
            value
        };
        v.clamp(min, max)
    }

    /// `normalizeValue`: the option's value as a 0..1 slider position.
    #[must_use]
    pub fn normalize(self, value: f32) -> f32 {
        if self == Self::FramerateLimit {
            return framerate_stop(value) as f32 / (FRAMERATE_CHOICES.len() - 1) as f32;
        }
        let OptionKind::Float { min, max, .. } = self.kind() else {
            return 0.0;
        };
        ((self.snap_to_step_clamp(value) - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// `denormalizeValue`: a 0..1 slider position back to the option's value.
    ///
    /// A continuous option (vanilla `step` 0) first
    /// snaps the position to the step its label shows ([`OptionId::display_steps`]), so the
    /// stored value is exactly the number on the button: "Sensitivity: 87%" stores 0.435, and
    /// every whole percentage has exactly one value.
    #[must_use]
    pub fn denormalize(self, position: f32) -> f32 {
        if self == Self::FramerateLimit {
            let index =
                (position.clamp(0.0, 1.0) * (FRAMERATE_CHOICES.len() - 1) as f32).round() as usize;
            return FRAMERATE_CHOICES[index] as f32;
        }
        let OptionKind::Float { min, max, step } = self.kind() else {
            return 0.0;
        };
        let mut position = position.clamp(0.0, 1.0);
        if step <= 0.0 {
            if let Some(steps) = self.display_steps() {
                // (As a float at or just above the step: a truncating label must read it.)
                position = crate::widget::snap_position(position, steps);
            }
        }
        self.snap_to_step_clamp(min + (max - min) * position)
    }

    /// How many steps a slider's label resolves across the option's range (item 33): the
    /// option's own step for a stepped option (FOV: 80), otherwise the label's resolution
    /// (sensitivity shows `n * 200` %, so 200). Continuous options snap to it
    /// ([`OptionId::denormalize`]) and the arrow keys move a slider one of these steps.
    #[must_use]
    pub fn display_steps(self) -> Option<f32> {
        if self == Self::FramerateLimit {
            return Some((FRAMERATE_CHOICES.len() - 1) as f32);
        }
        use OptionId as O;
        let OptionKind::Float { min, max, step } = self.kind() else {
            return None;
        };
        if step > 0.0 {
            return Some(((max - min) / step).round().max(1.0));
        }
        Some(match self {
            // "N%" of `n * 200`.
            O::Sensitivity => 200.0,
            // "N%" of `n * 400`.
            O::Saturation => 400.0,
            // "N%" of `n * 90 + 10`.
            O::ChatOpacity => 90.0,
            // Pixels: `n * 280 + 40` and `n * 160 + 20`.
            O::ChatWidth => 280.0,
            O::ChatHeightFocused | O::ChatHeightUnfocused => 160.0,
            // Gamma's "+N%" and chat scale's "N%" of `n * 100`.
            _ => 100.0,
        })
    }

    /// Vanilla's button label: `"<name>: <value>"`.
    ///
    /// This is `getKeyBinding`, the single 200-column ternary chain in `GameSettings.java`,
    /// unrolled. Each special case is one arm and keeps vanilla's exact rounding: `(int)` is a
    /// **truncation**, so 199.99% shows as 199%.
    // java: GameSettings.java:615
    #[must_use]
    pub fn display_string(self, s: &GameSettings, lang: &Language) -> String {
        use OptionId as O;
        let prefix = format!("{}: ", lang.tr(self.translation_key()));
        match self.kind() {
            OptionKind::Float { .. } => {
                let raw = self.float_value(s);
                let n = self.normalize(raw);
                let tail = match self {
                    O::Sensitivity => {
                        if n == 0.0 {
                            lang.tr("options.sensitivity.min").to_owned()
                        } else if n == 1.0 {
                            lang.tr("options.sensitivity.max").to_owned()
                        } else {
                            format!("{}%", (n * 200.0) as i32)
                        }
                    }
                    O::Fov => {
                        if raw == 70.0 {
                            lang.tr("options.fov.min").to_owned()
                        } else if raw == 110.0 {
                            lang.tr("options.fov.max").to_owned()
                        } else {
                            format!("{}", raw as i32)
                        }
                    }
                    O::FramerateLimit => {
                        if raw as i32 == FRAMERATE_UNLIMITED {
                            lang.tr("options.framerateLimit.max").to_owned()
                        } else {
                            format!("{} fps", raw as i32)
                        }
                    }
                    O::Gamma => {
                        if n == 0.0 {
                            lang.tr("options.gamma.min").to_owned()
                        } else if n == 1.0 {
                            lang.tr("options.gamma.max").to_owned()
                        } else {
                            format!("+{}%", (n * 100.0) as i32)
                        }
                    }
                    O::Saturation => format!("{}%", (n * 400.0) as i32),
                    O::ChatOpacity => format!("{}%", (n * 90.0 + 10.0) as i32),
                    O::ChatHeightFocused | O::ChatHeightUnfocused => {
                        format!("{}px", crate::screens::chat::chatbox_height(n))
                    }
                    O::ChatWidth => format!("{}px", crate::screens::chat::chatbox_width(n)),
                    O::RenderDistance => format!("{} chunks", raw as i32),
                    O::MipmapLevels => {
                        if raw == 0.0 {
                            lang.tr("options.off").to_owned()
                        } else {
                            format!("{}", raw as i32)
                        }
                    }
                    _ => {
                        if n == 0.0 {
                            lang.tr("options.off").to_owned()
                        } else {
                            format!("{}%", (n * 100.0) as i32)
                        }
                    }
                };
                prefix + &tail
            }
            OptionKind::Boolean => {
                let key = if self.bool_value(s) {
                    "options.on"
                } else {
                    "options.off"
                };
                prefix + lang.tr(key)
            }
            OptionKind::Cycle { .. } => {
                let table: &[&str] = match self {
                    O::GuiScale => &GUI_SCALES,
                    O::Particles => &PARTICLES,
                    O::AmbientOcclusion => &AMBIENT_OCCLUSIONS,
                    O::RenderClouds => &CLOUDS_TYPES,
                    O::ChatVisibility => &CHAT_VISIBILITIES,
                    O::Graphics => &["options.graphics.fast", "options.graphics.fancy"],
                    _ => &[],
                };
                if table.is_empty() {
                    prefix
                } else {
                    prefix + lang.tr(translation_of(table, self.cycle_value(s)))
                }
            }
        }
    }
}

fn toggle_model_part(settings: &mut GameSettings, name: &'static str) {
    let current = settings.model_parts.get(name).copied().unwrap_or(true);
    settings.model_parts.insert(name.to_owned(), !current);
}

#[cfg(test)]
// Each assertion below is its own scenario: writing one settings field at a time reads far
// better than a fresh struct-update literal per case.
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn lang() -> Language {
        let mut l = Language::new("en_US");
        for (k, v) in [
            ("options.sensitivity", "Sensitivity"),
            ("options.sensitivity.min", "*yawn*"),
            ("options.sensitivity.max", "HYPERSPEED!!!"),
            ("options.fov", "FOV"),
            ("options.fov.min", "Normal"),
            ("options.fov.max", "Quake Pro"),
            ("options.framerateLimit", "Max Framerate"),
            ("options.framerateLimit.max", "Unlimited"),
            ("options.gamma", "Brightness"),
            ("options.gamma.min", "Moody"),
            ("options.gamma.max", "Bright"),
            ("options.renderDistance", "Render Distance"),
            ("options.mipmapLevels", "Mipmap Levels"),
            ("options.chat.opacity", "Chat Opacity"),
            ("options.chat.width", "Chat Width"),
            ("options.chat.height.focused", "Focused Height"),
            ("options.saturation", "Saturation"),
            ("options.off", "OFF"),
            ("options.on", "ON"),
            ("options.guiScale", "GUI Scale"),
            ("options.guiScale.auto", "Auto"),
            ("options.guiScale.small", "Small"),
            ("options.guiScale.normal", "Normal"),
            ("options.guiScale.large", "Large"),
            ("options.graphics", "Graphics"),
            ("options.graphics.fast", "Fast"),
            ("options.graphics.fancy", "Fancy"),
            ("options.ao", "Smooth Lighting"),
            ("options.ao.off", "OFF"),
            ("options.ao.min", "Minimum"),
            ("options.ao.max", "Maximum"),
            ("options.viewBobbing", "View Bobbing"),
            ("options.renderClouds", "Clouds"),
            ("options.particles", "Particles"),
            ("options.particles.all", "All"),
            ("options.particles.decreased", "Decreased"),
            ("options.particles.minimal", "Minimal"),
            ("options.chat.visibility", "Chat"),
            ("options.chat.visibility.full", "Shown"),
            ("options.chat.visibility.system", "Commands Only"),
            ("options.chat.visibility.hidden", "Hidden"),
        ] {
            l.insert(k, v);
        }
        l
    }

    /// The five thresholds `getKeyBinding` special-cases, checked against the Java's own
    /// arithmetic rather than against this implementation.
    #[test]
    fn the_slider_labels_reproduce_vanillas_exact_arithmetic() {
        let l = lang();
        let mut s = GameSettings::default();

        s.mouse_sensitivity = 0.0;
        assert_eq!(
            OptionId::Sensitivity.display_string(&s, &l),
            "Sensitivity: *yawn*"
        );
        s.mouse_sensitivity = 1.0;
        assert_eq!(
            OptionId::Sensitivity.display_string(&s, &l),
            "Sensitivity: HYPERSPEED!!!"
        );
        // 0.5 * 200 = 100 exactly.
        s.mouse_sensitivity = 0.5;
        assert_eq!(
            OptionId::Sensitivity.display_string(&s, &l),
            "Sensitivity: 100%"
        );
        // 0.7 * 200 = 140.00001 in f32; `(int)` truncates to 140, not 139.
        s.mouse_sensitivity = 0.7;
        assert_eq!(
            OptionId::Sensitivity.display_string(&s, &l),
            "Sensitivity: 140%"
        );

        s.fov = 70.0;
        assert_eq!(OptionId::Fov.display_string(&s, &l), "FOV: Normal");
        s.fov = 110.0;
        assert_eq!(OptionId::Fov.display_string(&s, &l), "FOV: Quake Pro");
        s.fov = 90.0;
        assert_eq!(OptionId::Fov.display_string(&s, &l), "FOV: 90");

        s.limit_framerate = 260;
        assert_eq!(
            OptionId::FramerateLimit.display_string(&s, &l),
            "Max Framerate: Unlimited"
        );
        s.limit_framerate = 120;
        assert_eq!(
            OptionId::FramerateLimit.display_string(&s, &l),
            "Max Framerate: 120 fps"
        );

        s.gamma = 0.0;
        assert_eq!(OptionId::Gamma.display_string(&s, &l), "Brightness: Moody");
        s.gamma = 1.0;
        assert_eq!(OptionId::Gamma.display_string(&s, &l), "Brightness: Bright");
        s.gamma = 0.5;
        assert_eq!(OptionId::Gamma.display_string(&s, &l), "Brightness: +50%");

        s.render_distance_chunks = 12;
        assert_eq!(
            OptionId::RenderDistance.display_string(&s, &l),
            "Render Distance: 12 chunks"
        );
        s.mipmap_levels = 0;
        assert_eq!(
            OptionId::MipmapLevels.display_string(&s, &l),
            "Mipmap Levels: OFF"
        );
        s.mipmap_levels = 3;
        assert_eq!(
            OptionId::MipmapLevels.display_string(&s, &l),
            "Mipmap Levels: 3"
        );

        // chatOpacity's formula is `n * 90 + 10`, so 1.0 is 100% and 0.0 is 10%.
        s.chat_opacity = 1.0;
        assert_eq!(
            OptionId::ChatOpacity.display_string(&s, &l),
            "Chat Opacity: 100%"
        );
        s.chat_opacity = 0.0;
        assert_eq!(
            OptionId::ChatOpacity.display_string(&s, &l),
            "Chat Opacity: 10%"
        );
        // Saturation multiplies by 400, not 100.
        s.saturation = 0.25;
        assert_eq!(
            OptionId::Saturation.display_string(&s, &l),
            "Saturation: 100%"
        );
    }

    #[test]
    fn boolean_and_cycle_labels_use_the_right_tables() {
        let l = lang();
        let mut s = GameSettings::default();
        s.view_bobbing = true;
        assert_eq!(
            OptionId::ViewBobbing.display_string(&s, &l),
            "View Bobbing: ON"
        );
        s.view_bobbing = false;
        assert_eq!(
            OptionId::ViewBobbing.display_string(&s, &l),
            "View Bobbing: OFF"
        );

        s.gui_scale = 2;
        assert_eq!(
            OptionId::GuiScale.display_string(&s, &l),
            "GUI Scale: Normal"
        );
        s.particles = 1;
        assert_eq!(
            OptionId::Particles.display_string(&s, &l),
            "Particles: Decreased"
        );
        s.ambient_occlusion = 2;
        assert_eq!(
            OptionId::AmbientOcclusion.display_string(&s, &l),
            "Smooth Lighting: Maximum"
        );
        s.clouds = 0;
        assert_eq!(OptionId::RenderClouds.display_string(&s, &l), "Clouds: OFF");
        s.fancy_graphics = true;
        assert_eq!(OptionId::Graphics.display_string(&s, &l), "Graphics: Fancy");
        s.fancy_graphics = false;
        assert_eq!(OptionId::Graphics.display_string(&s, &l), "Graphics: Fast");
        s.chat_visibility = 2;
        assert_eq!(
            OptionId::ChatVisibility.display_string(&s, &l),
            "Chat: Hidden"
        );
    }

    /// The non-vanilla fullbright toggle: an ON/OFF button over `GameSettings::fullbright`, off
    /// by default, labelled without a language key, and absent from every vanilla screen.
    #[test]
    fn fullbright_is_a_default_off_toggle_outside_the_vanilla_screens() {
        let l = lang();
        let mut s = GameSettings::default();
        assert_eq!(OptionId::Fullbright.kind(), OptionKind::Boolean);
        assert!(!OptionId::Fullbright.bool_value(&s));
        assert_eq!(
            OptionId::Fullbright.display_string(&s, &l),
            "Fullbright: OFF"
        );
        assert!(!OptionId::Fullbright.cycle(&mut s, 1), "no relayout");
        assert!(s.fullbright);
        assert_eq!(
            OptionId::Fullbright.display_string(&s, &l),
            "Fullbright: ON"
        );
        OptionId::Fullbright.cycle(&mut s, 1);
        assert!(!s.fullbright);

        assert!(!crate::screens::options::VIDEO_OPTIONS.contains(&OptionId::Fullbright));
        assert!(!crate::screens::options::CHAT_OPTIONS.contains(&OptionId::Fullbright));
    }

    /// Out-of-range enum indices fall back to element 0 instead of throwing, which is what
    /// keeps a hand-edited `options.txt` from crashing the options screen.
    #[test]
    fn an_out_of_range_cycle_index_falls_back_to_the_first_label() {
        let l = lang();
        let mut s = GameSettings::default();
        s.gui_scale = 99;
        assert_eq!(OptionId::GuiScale.display_string(&s, &l), "GUI Scale: Auto");
        s.particles = -4;
        assert_eq!(OptionId::Particles.display_string(&s, &l), "Particles: All");
        assert_eq!(translation_of(&GUI_SCALES, 3), "options.guiScale.large");
        assert_eq!(translation_of(&GUI_SCALES, -1), "options.guiScale.auto");
        assert_eq!(translation_of(&GUI_SCALES, 4), "options.guiScale.auto");
    }

    #[test]
    fn gui_scale_wraps_with_a_mask_and_the_others_with_a_modulo() {
        let mut s = GameSettings::default();
        s.gui_scale = 3;
        assert!(OptionId::GuiScale.cycle(&mut s, 1), "a relayout is needed");
        assert_eq!(s.gui_scale, 0);
        s.gui_scale = 0;
        OptionId::GuiScale.cycle(&mut s, -1);
        assert_eq!(s.gui_scale, 3, "the mask wraps backwards too");

        s.particles = 2;
        assert!(!OptionId::Particles.cycle(&mut s, 1));
        assert_eq!(s.particles, 0);
        s.clouds = 0;
        OptionId::RenderClouds.cycle(&mut s, -1);
        assert_eq!(s.clouds, 2, "rem_euclid, not the truncating %");
    }

    #[test]
    fn toggling_the_unicode_font_asks_for_a_relayout() {
        let mut s = GameSettings::default();
        assert!(OptionId::ForceUnicodeFont.cycle(&mut s, 1));
        assert!(s.force_unicode_font);
        assert!(!OptionId::ViewBobbing.cycle(&mut s, 1));
    }

    #[test]
    fn slider_snapping_rounds_to_the_step_and_clamps() {
        assert_eq!(OptionId::FramerateLimit.snap_to_step_clamp(114.0), 110.0);
        assert_eq!(OptionId::FramerateLimit.snap_to_step_clamp(115.0), 120.0);
        assert_eq!(OptionId::FramerateLimit.snap_to_step_clamp(1.0), 10.0);
        assert_eq!(OptionId::FramerateLimit.snap_to_step_clamp(9999.0), 260.0);
        assert_eq!(OptionId::RenderDistance.snap_to_step_clamp(7.4), 7.0);
        assert_eq!(OptionId::RenderDistance.snap_to_step_clamp(7.5), 8.0);
        // A step-free option passes straight through, then clamps.
        assert_eq!(OptionId::Gamma.snap_to_step_clamp(0.37), 0.37);
        assert_eq!(OptionId::Gamma.snap_to_step_clamp(2.0), 1.0);
    }

    #[test]
    fn extended_framerate_slider_reaches_every_cap_and_unlimited() {
        let option = OptionId::FramerateLimit;
        let mut settings = GameSettings::default();
        for (index, &cap) in FRAMERATE_CHOICES.iter().enumerate() {
            let position = index as f32 / (FRAMERATE_CHOICES.len() - 1) as f32;
            assert_eq!(option.denormalize(position), cap as f32);
            option.set_float_value(&mut settings, cap as f32);
            assert!((option.normalize(option.float_value(&settings)) - position).abs() < 1e-6);
            assert_eq!(
                settings.framerate_limit(),
                (cap != FRAMERATE_UNLIMITED).then_some(cap as u32)
            );
        }
        assert_eq!(option.denormalize(1.0), FRAMERATE_UNLIMITED as f32);
        assert_eq!(FRAMERATE_CHOICES[FRAMERATE_CHOICES.len() - 2], 2000);
    }

    #[test]
    fn render_distance_kind_supports_both_vanilla_capability_branches() {
        for max in [16, 32] {
            assert_eq!(
                OptionId::RenderDistance.kind_for_render_distance_max(max),
                OptionKind::Float {
                    min: 2.0,
                    max: max as f32,
                    step: 1.0,
                }
            );
        }
    }

    #[test]
    fn render_distance_slider_consumes_the_native_capability() {
        let max = native_render_distance_max() as f32;
        assert_eq!(
            OptionId::RenderDistance.kind(),
            OptionId::RenderDistance.kind_for_render_distance_max(native_render_distance_max())
        );
        assert_eq!(OptionId::RenderDistance.denormalize(1.0), max);
        assert_eq!(OptionId::RenderDistance.normalize(max), 1.0);
        assert_eq!(OptionId::RenderDistance.snap_to_step_clamp(100.0), max);
    }

    #[test]
    fn normalize_and_denormalize_round_trip_through_the_slider() {
        for (opt, value) in [
            (OptionId::Fov, 90.0f32),
            (OptionId::RenderDistance, 8.0),
            (OptionId::FramerateLimit, 120.0),
            (OptionId::MipmapLevels, 2.0),
            (OptionId::Gamma, 0.25),
        ] {
            let n = opt.normalize(value);
            assert!((0.0..=1.0).contains(&n), "{opt:?} normalised to {n}");
            assert!(
                (opt.denormalize(n) - value).abs() < 1e-3,
                "{opt:?} did not round-trip: {value} -> {n} -> {}",
                opt.denormalize(n)
            );
        }
        assert_eq!(OptionId::Fov.normalize(30.0), 0.0);
        assert_eq!(OptionId::Fov.normalize(110.0), 1.0);
        assert_eq!(OptionId::Fov.denormalize(0.5), 70.0);
    }

    #[test]
    fn every_float_option_reads_and_writes_the_same_settings_field() {
        let mut s = GameSettings::default();
        for opt in [
            OptionId::Sensitivity,
            OptionId::Fov,
            OptionId::Gamma,
            OptionId::Saturation,
            OptionId::ChatOpacity,
            OptionId::ChatScale,
            OptionId::ChatWidth,
            OptionId::ChatHeightFocused,
            OptionId::ChatHeightUnfocused,
            OptionId::FramerateLimit,
            OptionId::MipmapLevels,
            OptionId::RenderDistance,
        ] {
            let target = opt.denormalize(0.5);
            opt.set_float_value(&mut s, target);
            assert!(
                (opt.float_value(&s) - target).abs() < 1e-4,
                "{opt:?} lost its value"
            );
        }
    }


    /// The percentage the sensitivity button shows for a stored value.
    fn sensitivity_label(value: f32) -> String {
        let mut s = GameSettings::default();
        s.mouse_sensitivity = value;
        OptionId::Sensitivity.display_string(&s, &lang())
    }

    fn sensitivity_percent(label: &str) -> i32 {
        match label.trim_start_matches("Sensitivity: ") {
            "*yawn*" => 0,
            "HYPERSPEED!!!" => 200,
            other => other.trim_end_matches('%').parse().expect(label),
        }
    }

    /// "Some sensitivity numbers in options sliders (specific exact percentages) can't be
    /// reached": every whole percentage 0..=200 has a slider position whose stored value shows
    /// exactly that percentage, and the stored value is `k / 200` (to the float: 53 % is stored
    /// as 0.26500002, because 0.265's nearest float, 0.26499998, would read as 52 % on the
    /// truncating label).
    #[test]
    fn every_whole_sensitivity_percentage_has_exactly_one_snapped_value() {
        for k in 0..=200 {
            let position = k as f32 / 200.0;
            let value = OptionId::Sensitivity.denormalize(position);
            assert!((value - k as f32 / 200.0).abs() < 1e-7, "{k}%: {value}");
            assert_eq!(sensitivity_percent(&sensitivity_label(value)), k, "{k}%");
            // Anywhere within half a step snaps to the same value.
            for nudge in [-0.0024, 0.0024] {
                let near = (position + nudge).clamp(0.0, 1.0);
                assert_eq!(
                    OptionId::Sensitivity.denormalize(near),
                    value,
                    "{k}% {nudge}"
                );
            }
        }
        // The other continuous sliders snap to their labels too.
        assert!((OptionId::Gamma.denormalize(0.374) - 0.37).abs() < 1e-7);
        assert_eq!(OptionId::ChatOpacity.display_steps(), Some(90.0));
        // Stepped options keep vanilla's own snapping (FOV: whole degrees).
        assert_eq!(OptionId::Fov.display_steps(), Some(80.0));
        assert_eq!(OptionId::Fov.denormalize(0.5), 70.0);
    }

    /// The pointer: vanilla maps a 150-pixel slider from whole GUI pixels (142 of travel, so at
    /// most 143 positions for 201 values). With the physical-pixel cursor every percentage is
    /// reachable by mouse at GUI scale 2 and above; at scale 1 (142 physical pixels) the arrow
    /// keys reach the rest (next test).
    #[test]
    fn every_sensitivity_percentage_is_reachable_with_the_mouse_at_gui_scale_two() {
        use crate::widget::{set_precise_cursor_x, Slider};
        let slider = Slider::new(0, 100, 20, "", 0.0, 1.0, 0.5);
        let percentages_at = |scale: i32, precise: bool| {
            let mut seen = std::collections::BTreeSet::new();
            let rect = slider.button.rect;
            let (left, right) = (rect.x * scale, (rect.x + rect.width) * scale);
            for px in left - 4..right + 4 {
                let gui_x = px as f32 / scale as f32;
                set_precise_cursor_x(precise.then_some(gui_x));
                let position = slider.position_for(gui_x.floor() as i32);
                let value = OptionId::Sensitivity.denormalize(position);
                seen.insert(sensitivity_percent(&sensitivity_label(value)));
            }
            set_precise_cursor_x(None);
            seen
        };
        let vanilla = percentages_at(2, false);
        assert!(
            vanilla.len() < 201,
            "whole GUI pixels miss some: {}",
            vanilla.len()
        );
        for scale in [2, 3, 4] {
            let seen = percentages_at(scale, true);
            assert_eq!(seen.len(), 201, "scale {scale}: {seen:?}");
            assert_eq!(seen.first(), Some(&0));
            assert_eq!(seen.last(), Some(&200));
        }
        // A stale sub-pixel value (another frame's pointer) is ignored.
        set_precise_cursor_x(Some(500.7));
        assert_eq!(slider.position_for(104), 0.0);
        set_precise_cursor_x(None);
    }

    /// Fine adjustment: one arrow press is one label step, so from 0 % every percentage is
    /// visited in order, at any GUI scale.
    #[test]
    fn arrow_steps_visit_every_sensitivity_percentage_in_order() {
        use crate::widget::Slider;
        let steps = OptionId::Sensitivity.display_steps().unwrap();
        let mut slider = Slider::new(0, 0, 0, "", 0.0, 1.0, 0.0);
        for k in 1..=200 {
            slider.step_position(1, steps);
            let value = OptionId::Sensitivity.denormalize(slider.position());
            assert_eq!(
                sensitivity_percent(&sensitivity_label(value)),
                k,
                "step {k}"
            );
        }
        slider.step_position(1, steps);
        assert_eq!(slider.position(), 1.0, "clamped");
        slider.step_position(-10, steps);
        assert_eq!(
            sensitivity_percent(&sensitivity_label(
                OptionId::Sensitivity.denormalize(slider.position())
            )),
            190
        );
    }
}
