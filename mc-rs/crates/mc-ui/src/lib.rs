#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Font rendering, the widget framework, and every screen.
//!
//! Its modules:
//!
//! | Module | What |
//! |---|---|
//! | [`font`], [`widget`], [`screen_stack`] | font and GUI framework |
//! | `screens/**` (excluding `screens/container/**`) | menus, HUD, chat, F3 |
//! | [`screens::container`] | inventory and containers |
//! | [`nameplate`] | the `mc-ui` half of the entity-nameplate font seam |
//!
//! # What S15 gives you
//!
//! Everything here is **pure CPU geometry plus one render phase**. Layout, measurement,
//! hit-testing, wrapping and widget state are ordinary arithmetic on ordinary structs, so they
//! are testable without a display; only [`widget::UiPhase`] touches wgpu.
//!
//! ```text
//!   text + widgets ──> DrawList (UiVertex triangles, batched by texture)
//!                          │
//!                          └──> UiPhase (PhaseOrder::UI = 800) ──> wgpu
//! ```
//!
//! * [`font::FontRenderer`] — vanilla's `FontRenderer`: per-glyph widths, the `§` state
//!   machine, drop shadow, word wrap, and glyph-quad emission.
//! * [`font::component`] — the chat component tree (`{"text":…}` / `{"translate":…}`) with
//!   styles, click and hover events.
//! * [`widget::ScaledResolution`] — vanilla's GUI scaling and window↔GUI coordinate mapping.
//! * [`widget`] — `DrawList` drawing helpers (rects, gradients, blits, nine-slice) plus
//!   [`Button`](widget::Button), [`Slider`](widget::Slider),
//!   [`TextField`](widget::TextField), [`Checkbox`](widget::Checkbox) and
//!   [`ScrollList`](widget::ScrollList).
//! * [`screen_stack::Screen`] and [`screen_stack::ScreenCtx`] — the low-level screen contract
//!   used by individual screen implementations. [`screen_stack::ScreenStack`] remains available
//!   as a library/test harness for generic screen behavior, but is intentionally not a production
//!   client dispatcher: `mc-client`'s `Ui` enum owns dispatch because actions also mutate session,
//!   network, and inventory state.
//!
//! # Hazard H9
//!
//! Vanilla takes glyph widths from two baked tables, never from measuring ink at draw time:
//! `font/glyph_sizes.bin` for the unicode pages, and — for the 256 `ascii.png` glyphs — a
//! **one-shot** alpha-column scan performed at load, with vanilla's exact `+1` padding and
//! `(int)(0.5 + col * 8/cell)` rounding. Both are ported byte-for-byte in
//! [`font::FontData`]; re-deriving either from the texture at layout time drifts a pixel or
//! two per character and misaligns every screen built on top. See
//! [`font::measure_ascii_widths`].
//!
//! # Example
//!
//! ```no_run
//! use mc_ui::font::FontRenderer;
//! use mc_ui::widget::{DrawList, ScaledResolution};
//! # fn demo(assets: &mc_assets::ResourceManager) -> Result<(), mc_ui::UiError> {
//! let mut font = FontRenderer::load(assets, false, false)?;
//! let res = ScaledResolution::new(1920, 1080, 0, false);
//!
//! let mut draw = DrawList::new();
//! draw.rect(0, 0, res.scaled_width(), res.scaled_height(), 0x8000_0000);
//! font.draw_string_with_shadow(&mut draw, "Hello, world!", 10.0, 10.0, 0x00ff_ffff);
//! # Ok(())
//! # }
//! ```

pub mod font;
pub mod frame_timer;
pub mod map;
// Marker pins, the waypoint label's sibling.
pub mod marker_label;
// The mod HUD pipeline (HudData, HudElement, ModHud).
pub mod modhud;
// The custom crosshair and the Shiny Pots square glint.
pub mod modvisual;
pub mod nameplate;
pub mod screen_stack;
pub mod screens;
// Waypoint labels drawn through walls at a constant size on screen.
pub mod waypoint_label;
pub mod widget;

use thiserror::Error;

/// Errors returned by `mc-ui`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum UiError {
    /// A code path that S15/S16/S17 have not implemented yet.
    #[error("mc-ui: {0} is not implemented yet")]
    Unimplemented(&'static str),

    /// A font resource could not be read out of the resource packs.
    #[error("mc-ui: font asset {0} is unavailable: {1}")]
    FontAsset(String, #[source] mc_assets::AssetError),

    /// A font texture decoded, but is not shaped the way the font code requires.
    #[error("mc-ui: font texture {location} is malformed: {reason}")]
    FontTexture {
        /// The resource that failed.
        location: String,
        /// What was wrong with it.
        reason: String,
    },

    /// `font/glyph_sizes.bin` was not the expected 65536 bytes.
    #[error("mc-ui: glyph_sizes.bin is {0} bytes, expected 65536")]
    GlyphSizes(usize),

    /// A chat component could not be parsed from JSON.
    ///
    /// Chat arrives from the server, so this is untrusted input: it is always an error value,
    /// never a panic.
    #[error("mc-ui: malformed chat component: {0}")]
    Component(String),

    /// `servers.dat` could not be read or written.
    ///
    /// The file is user-editable and its favicons arrive over the network, so like
    /// [`UiError::Component`] this is always an error value and never a panic. Appended by
    /// S16.
    #[error("mc-ui: servers.dat: {0}")]
    ServerList(String),
}
