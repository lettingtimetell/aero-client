//! The four-line sign editor.
//!
//! java: `client/gui/inventory/GuiEditSign.java`
//!
//! The host owns the live tile entity and networking.  This model owns the otherwise rather
//! easy-to-miss GUI semantics: moving the edit cursor before applying a key, replacing the
//! selected component on *every* key press, and emitting a C12 update even when the user never
//! changed a character.

use glam::{Mat4, Vec3};
use mc_assets::Language;
use mc_math::BlockPos;

use crate::font::component::ChatComponent;
use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{LABEL_COLOR, TITLE_COLOR};
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::{draw_centered_string, draw_string, Button, TextField, Transform, UiTexture};

const DONE_BUTTON_ID: i32 = 0;
/// `GuiEditSign.keyTyped` limits rendered line width, not character count.
pub const MAX_LINE_WIDTH: i32 = 90;

/// The texture bound by vanilla's [`TileEntitySignRenderer`].
// java: `TileEntitySignRenderer.SIGN_TEXTURE`
pub const SIGN_TEXTURE: &str = "textures/entity/sign.png";

const SIGN_TEXTURE_WIDTH: f32 = 64.0;
const SIGN_TEXTURE_HEIGHT: f32 = 32.0;
const SIGN_GUI_SCALE: f32 = 93.75;
const SIGN_RENDER_SCALE: f32 = 0.666_666_7;
const SIGN_MODEL_SCALE: f32 = 0.0625;

/// Which block-state orientation the sign editor's preview represents.
///
/// `GuiEditSign` receives the real tile entity, so it repeats the standing-sign metadata angle
/// or the four wall-sign angles before delegating to `TileEntityRendererDispatcher`.  The editor
/// model normally starts with a standing sign at metadata zero (the placement path); callers that
/// open a wall sign can select its exact metadata with [`EditSignScreen::set_preview_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignPreviewKind {
    /// `Blocks.standing_sign`; metadata is the 16-way rotation index.
    Standing {
        /// The block metadata, as read by `TileEntitySign.getBlockMetadata()`.
        metadata: u8,
    },
    /// `Blocks.wall_sign`; only metadata 2, 4 and 5 have a special renderer angle.
    Wall {
        /// The block metadata, as read by `TileEntitySign.getBlockMetadata()`.
        metadata: u8,
    },
}

impl Default for SignPreviewKind {
    fn default() -> Self {
        Self::Standing { metadata: 0 }
    }
}

impl SignPreviewKind {
    /// Construct a standing-sign preview for the given 16-way metadata.
    #[must_use]
    pub const fn standing(metadata: u8) -> Self {
        Self::Standing { metadata }
    }

    /// Construct a wall-sign preview for the given facing metadata.
    #[must_use]
    pub const fn wall(metadata: u8) -> Self {
        Self::Wall { metadata }
    }

    /// Whether this preview includes the standing sign's support stick.
    #[must_use]
    pub const fn is_standing(self) -> bool {
        matches!(self, Self::Standing { .. })
    }

    /// The metadata-specific angle in degrees used by both Java matrix stacks.
    #[must_use]
    fn rotation_degrees(self) -> f32 {
        match self {
            Self::Standing { metadata } => f32::from(metadata) * 360.0 / 16.0,
            Self::Wall { metadata } => match metadata {
                2 => 180.0,
                4 => 90.0,
                5 => -90.0,
                _ => 0.0,
            },
        }
    }
}

/// One CPU-projected vertex of the sign tile model.
///
/// `position` retains the transformed z coordinate even though [`DrawList`] is an orthographic
/// 2-D list.  That value is the painter-sort key which substitutes for the depth buffer the
/// vanilla GUI keeps enabled while `TileEntitySignRenderer` draws its six-sided boxes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignPreviewVertex {
    /// Fully transformed model position in GUI pixels plus painter depth.
    pub position: [f32; 3],
    /// Normalised coordinate in `textures/entity/sign.png`.
    pub uv: [f32; 2],
}

/// A textured face in the CPU sign preview.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignPreviewFace {
    /// Whether this face belongs to the board or the standing support stick.
    pub part: SignPreviewPart,
    /// The four source-order corners from Java's `ModelBox` face table.
    pub vertices: [SignPreviewVertex; 4],
    /// Mean transformed z, used for back-to-front painter ordering.
    pub depth: f32,
}

/// The two model boxes making up a vanilla sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignPreviewPart {
    /// `ModelSign.signBoard`, a 24×12×2 box.
    Board,
    /// `ModelSign.signStick`, a 2×14×2 box, omitted for wall signs.
    Stick,
}

/// CPU geometry for one centred/scaled `GuiEditSign` tile preview.
///
/// The renderer has one fixed orthographic UI projection and cannot submit a second 3-D camera
/// just for this modal.  The geometry therefore follows Java's matrix calls on the CPU, drops the
/// transformed z only at [`Self::draw`], and sorts all box faces back-to-front before emitting
/// textured triangles.  This keeps the visual seam deterministic and gives headless tests direct
/// access to the transform, model dimensions, face order, and texture UVs.
#[derive(Debug, Clone, PartialEq)]
pub struct SignPreviewGeometry {
    /// The block orientation used to build this model.
    pub kind: SignPreviewKind,
    /// All visible model-box faces in painter order.
    pub faces: Vec<SignPreviewFace>,
}

impl SignPreviewGeometry {
    /// Emit the projected sign model with vanilla's sign texture.
    pub fn draw(&self, d: &mut DrawList) {
        let texture = UiTexture::gui(SIGN_TEXTURE);
        for face in &self.faces {
            let positions = face
                .vertices
                .map(|vertex| [vertex.position[0], vertex.position[1]]);
            let uv = face.vertices.map(|vertex| vertex.uv);
            d.quad(&texture, positions, uv, [[1.0; 4]; 4]);
        }
    }
}

/// Build the exact GUI-space sign model used by `GuiEditSign.drawScreen`.
///
/// `width` is the scaled GUI width.  Java computes `width / 2` as an integer before converting it
/// to a float for `GlStateManager.translate`, so odd widths intentionally anchor on the lower
/// integer half rather than at the mathematical midpoint.
#[must_use]
pub fn sign_preview_geometry(width: i32, kind: SignPreviewKind) -> SignPreviewGeometry {
    let matrices = SignPreviewMatrices::new(width, kind);
    let mut faces = model_box_faces(
        SignPreviewPart::Board,
        [-12.0, -14.0, -1.0],
        [24.0, 12.0, 2.0],
        [0, 0],
        matrices.model,
    );
    if kind.is_standing() {
        faces.extend(model_box_faces(
            SignPreviewPart::Stick,
            [-1.0, -2.0, -1.0],
            [2.0, 14.0, 2.0],
            [0, 14],
            matrices.model,
        ));
    }
    // Vanilla's GUI projection leaves depth testing enabled.  DrawList is a flat painter list,
    // so the near faces must be submitted later.  `total_cmp` keeps NaN-free model arithmetic
    // explicit and preserves the source order for equal-depth coplanar faces.
    faces.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    SignPreviewGeometry { kind, faces }
}

#[derive(Debug, Clone, Copy)]
struct SignPreviewMatrices {
    model: Mat4,
    text: Mat4,
}

impl SignPreviewMatrices {
    fn new(width: i32, kind: SignPreviewKind) -> Self {
        let angle = kind.rotation_degrees();
        // GuiEditSign.drawScreen's outer stack, in Java call order.  Mat4 multiplication is
        // deliberately written in the same order: the last operation is the first one applied to
        // a model vertex, just as OpenGL's right-multiplied current matrix is.
        let mut gui = Mat4::from_translation(Vec3::new((width / 2) as f32, 0.0, 50.0));
        gui *= Mat4::from_scale(Vec3::splat(-SIGN_GUI_SCALE));
        gui *= Mat4::from_rotation_y(180.0f32.to_radians());
        gui *= Mat4::from_rotation_y(angle.to_radians());
        gui *= Mat4::from_translation(Vec3::new(0.0, -1.0625, 0.0));

        // TileEntitySignRenderer.renderTileEntityAt(sign, -0.5, -0.75, -0.5): the first
        // translation lands at (0, -0.25, 0) after adding the renderer's 0.5/0.75*f offsets.
        let mut tile = gui * Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0));
        tile *= Mat4::from_rotation_y((-angle).to_radians());
        if !kind.is_standing() {
            tile *= Mat4::from_translation(Vec3::new(0.0, -0.3125, -0.4375));
        }

        let model =
            tile * Mat4::from_scale(Vec3::new(
                SIGN_RENDER_SCALE,
                -SIGN_RENDER_SCALE,
                -SIGN_RENDER_SCALE,
            )) * Mat4::from_scale(Vec3::splat(SIGN_MODEL_SCALE));

        // TileEntitySignRenderer removes the model scale before moving into its text pass, then
        // translates by 0.5*f and uses f3 = 0.015625*f with a flipped y axis.
        let text_scale = 0.015625 * SIGN_RENDER_SCALE;
        let text =
            tile * Mat4::from_translation(Vec3::new(
                0.0,
                0.5 * SIGN_RENDER_SCALE,
                0.07 * SIGN_RENDER_SCALE,
            )) * Mat4::from_scale(Vec3::new(text_scale, -text_scale, text_scale));
        Self { model, text }
    }

    fn text_transform(self) -> Transform {
        let origin = self.text.transform_point3(Vec3::ZERO);
        let x = self.text.transform_point3(Vec3::X) - origin;
        let y = self.text.transform_point3(Vec3::Y) - origin;
        Transform {
            m: [[x.x, y.x, origin.x], [x.y, y.y, origin.y]],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelBoxSpec {
    part: SignPreviewPart,
    min: [f32; 3],
    max: [f32; 3],
    texture: [i32; 2],
}

fn model_box_faces(
    part: SignPreviewPart,
    min: [f32; 3],
    size: [f32; 3],
    texture: [i32; 2],
    transform: Mat4,
) -> Vec<SignPreviewFace> {
    let spec = ModelBoxSpec {
        part,
        min,
        max: [min[0] + size[0], min[1] + size[1], min[2] + size[2]],
        texture,
    };
    let [x1, y1, z1] = spec.min;
    let [x2, y2, z2] = spec.max;
    let points = [
        Vec3::new(x1, y1, z1), // positiontexturevertex7
        Vec3::new(x2, y1, z1), // positiontexturevertex
        Vec3::new(x2, y2, z1), // positiontexturevertex1
        Vec3::new(x1, y2, z1), // positiontexturevertex2
        Vec3::new(x1, y1, z2), // positiontexturevertex3
        Vec3::new(x2, y1, z2), // positiontexturevertex4
        Vec3::new(x2, y2, z2), // positiontexturevertex5
        Vec3::new(x1, y2, z2), // positiontexturevertex6
    ];
    let [texture_x, texture_y] = spec.texture;
    let (w, h, depth) = (size[0] as i32, size[1] as i32, size[2] as i32);
    let tx = texture_x;
    let ty = texture_y;
    let d = depth;
    let faces = [
        // ModelBox's six TexturedQuad constructions, in source order.
        (
            [4, 1, 2, 5],
            [
                [tx + d + w + d, ty + d],
                [tx + d + w, ty + d],
                [tx + d + w, ty + d + h],
                [tx + d + w + d, ty + d + h],
            ],
        ),
        (
            [0, 4, 7, 3],
            [
                [tx + d, ty + d],
                [tx, ty + d],
                [tx, ty + d + h],
                [tx + d, ty + d + h],
            ],
        ),
        (
            [5, 4, 0, 1],
            [
                [tx + d + w, ty],
                [tx + d, ty],
                [tx + d, ty + d],
                [tx + d + w, ty + d],
            ],
        ),
        (
            [1, 3, 7, 6],
            [
                [tx + d + w + w, ty + d],
                [tx + d + w, ty + d],
                [tx + d + w, ty],
                [tx + d + w + w, ty],
            ],
        ),
        (
            [1, 0, 3, 2],
            [
                [tx + d + w, ty + d],
                [tx + d, ty + d],
                [tx + d, ty + d + h],
                [tx + d + w, ty + d + h],
            ],
        ),
        (
            [4, 5, 6, 7],
            [
                [tx + d + w + d + w, ty + d],
                [tx + d + w + d, ty + d],
                [tx + d + w + d, ty + d + h],
                [tx + d + w + d + w, ty + d + h],
            ],
        ),
    ];
    faces
        .into_iter()
        .map(|(indices, uv)| {
            let vertices = core::array::from_fn(|index| {
                let p = transform.transform_point3(points[indices[index]]);
                SignPreviewVertex {
                    position: p.to_array(),
                    uv: [
                        uv[index][0] as f32 / SIGN_TEXTURE_WIDTH,
                        uv[index][1] as f32 / SIGN_TEXTURE_HEIGHT,
                    ],
                }
            });
            let depth = vertices.iter().map(|v| v.position[2]).sum::<f32>() / 4.0;
            SignPreviewFace {
                part: spec.part,
                vertices,
                depth,
            }
        })
        .collect()
}

/// A host-side operation requested by [`EditSignScreen`].
///
/// The `lines` in [`SignAction::SendUpdateSign`] are already exactly the four canonical JSON
/// component strings consumed by `c2s::UpdateSign`; the app must not wrap them in another object
/// or apply chat-character filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignAction {
    /// Mirror `TileEntitySign.setEditable` while this modal editor is open or closes.
    SetEditable(bool),
    /// `GuiEditSign.actionPerformed` marks the tile dirty for Done and Escape only.
    MarkDirty,
    /// Send the unconditional `C12PacketUpdateSign` close reply.
    SendUpdateSign {
        /// The sign's block position.
        position: BlockPos,
        /// One serialised `IChatComponent` per sign row, top to bottom.
        lines: [String; 4],
    },
    /// Remove the editor after its Done/Escape path completed.
    Close,
}

#[derive(Debug, Clone)]
struct EditableLine {
    /// `IChatComponent.getUnformattedText()`, the text which `GuiEditSign` presents to editing.
    raw: String,
    /// `IChatComponent.getFormattedText()`, which `TileEntitySignRenderer` draws on the tile.
    formatted: String,
    /// The component which C12 must serialise if this row has not been touched.  A key event
    /// replaces it with a bare `ChatComponentText`, exactly like Java assigns `signText[i]`.
    component_json: String,
}

impl EditableLine {
    fn plain(raw: String) -> Self {
        let component_json = ChatComponent::text(&raw).to_json();
        Self {
            formatted: raw.clone(),
            raw,
            component_json,
        }
    }

    fn replace_from_key(&mut self, raw: String) {
        *self = Self::plain(raw);
    }
}

/// Pure editing state for `GuiEditSign`.
///
/// Call [`Self::init`] on each screen layout, drain [`Self::take_actions`] into the tile/network
/// bridge, and call [`Self::on_gui_closed`] if some other screen replaces this one.  Done and
/// Escape finish directly and queue [`SignAction::Close`] themselves.
#[derive(Debug, Clone)]
pub struct EditSignScreen {
    position: BlockPos,
    lines: [EditableLine; 4],
    preview_kind: SignPreviewKind,
    update_counter: i32,
    edit_line: usize,
    done_button: Button,
    finished: bool,
    actions: Vec<SignAction>,
}

impl EditSignScreen {
    /// Start with the four plain text rows used by a newly placed/fabricated sign.
    #[must_use]
    pub fn new(position: BlockPos, lines: [String; 4]) -> Self {
        Self {
            position,
            lines: lines.map(EditableLine::plain),
            preview_kind: SignPreviewKind::default(),
            update_counter: 0,
            edit_line: 0,
            done_button: Button::new(DONE_BUTTON_ID, 0, 0, ""),
            finished: false,
            actions: Vec::new(),
        }
    }

    /// Start from the actual tile components, retaining untouched component style/`extra` data
    /// for the close packet.  `GuiEditSign` shows `getUnformattedText`, then destroys the
    /// selected component's styling as soon as *any* key event targets that row.
    #[must_use]
    pub fn from_components(
        position: BlockPos,
        components: [ChatComponent; 4],
        lang: &Language,
    ) -> Self {
        let lines = components.map(|component| EditableLine {
            raw: component.unformatted_text(lang),
            formatted: component.formatted_text(lang),
            component_json: component.to_json(),
        });
        Self {
            position,
            lines,
            preview_kind: SignPreviewKind::default(),
            update_counter: 0,
            edit_line: 0,
            done_button: Button::new(DONE_BUTTON_ID, 0, 0, ""),
            finished: false,
            actions: Vec::new(),
        }
    }

    /// Parse components received as packet JSON.  The fallback matches a `TileEntitySign` that
    /// had to turn malformed persisted text into `new ChatComponentText(s)`.
    #[must_use]
    pub fn from_component_json(
        position: BlockPos,
        components: [String; 4],
        lang: &Language,
    ) -> Self {
        let lines =
            components.map(
                |component_json| match ChatComponent::from_json(&component_json) {
                    // C12 does not write the source spelling back out: Java has already turned the
                    // line into an IChatComponent, and `PacketBuffer.writeChatComponent` calls
                    // `IChatComponent.Serializer.componentToJson` on close.  Reuse the same parser
                    // and serializer here so whitespace, key order, primitive-vs-object shape and
                    // Gson escaping all reach the wire in Java's canonical form.  The parsed tree
                    // still supplies the editor's unformatted/formatted text, so this does not alter
                    // what the user sees or the four-row editing state.
                    Ok(component) => EditableLine {
                        raw: component.unformatted_text(lang),
                        formatted: component.formatted_text(lang),
                        component_json: component.to_json(),
                    },
                    Err(_) => EditableLine::plain(component_json),
                },
            );
        Self {
            position,
            lines,
            preview_kind: SignPreviewKind::default(),
            update_counter: 0,
            edit_line: 0,
            done_button: Button::new(DONE_BUTTON_ID, 0, 0, ""),
            finished: false,
            actions: Vec::new(),
        }
    }

    /// Recreate the sole 200×20 Done button and enter non-editable tile state.
    // java: GuiEditSign.java:43-48
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        self.done_button = Button::new(
            DONE_BUTTON_ID,
            ctx.width / 2 - 100,
            ctx.height / 4 + 120,
            lang.tr("gui.done"),
        );
        if !self.finished {
            self.actions.push(SignAction::SetEditable(false));
        }
    }

    /// The live Done button for rendering and hit testing.
    #[must_use]
    pub const fn done_button(&self) -> &Button {
        &self.done_button
    }

    /// The driver renders the vanilla `GuiScreen` button list, while this model owns the
    /// button's layout and hit-testing state.
    pub fn done_button_mut(&mut self) -> &mut Button {
        &mut self.done_button
    }

    /// Which of the four lines is currently edited, zero-based.
    #[must_use]
    pub const fn edit_line(&self) -> usize {
        self.edit_line
    }

    /// The tile position this editor was opened for.  The host uses this to mirror the
    /// transient editable/caret state onto a real world tile when one exists; an S36 may also
    /// carry a position with no sign tile, in which case the editor owns the fabricated state.
    #[must_use]
    pub const fn position(&self) -> BlockPos {
        self.position
    }

    /// The editable raw text for each row, rather than its stored chat-component representation.
    #[must_use]
    pub fn raw_lines(&self) -> [String; 4] {
        self.lines.clone().map(|line| line.raw)
    }

    /// The current text for one row.
    #[must_use]
    pub fn line(&self, index: usize) -> Option<&str> {
        self.lines.get(index).map(|line| line.raw.as_str())
    }

    /// The block orientation currently used by the tile preview.
    #[must_use]
    pub const fn preview_kind(&self) -> SignPreviewKind {
        self.preview_kind
    }

    /// Select the standing- or wall-sign orientation used by the tile preview.
    ///
    /// This changes only the modal's CPU draw helper; the host remains responsible for reading
    /// the actual tile's block type and metadata and may update this value whenever that state is
    /// available.  The default is a standing sign with metadata zero, matching a newly placed
    /// sign before the world bridge has supplied a tile.
    pub fn set_preview_kind(&mut self, kind: SignPreviewKind) {
        self.preview_kind = kind;
    }

    /// Build the centered/scaled tile geometry for this editor at a GUI width.
    #[must_use]
    pub fn preview_geometry(&self, width: i32) -> SignPreviewGeometry {
        sign_preview_geometry(width, self.preview_kind)
    }

    /// Draw the vanilla sign tile and its four centered text rows into a [`DrawList`].
    ///
    /// The host should call this after the modal background/title and before its Done button.
    /// Geometry and text are emitted in one balanced helper so the sign's 3-D transform cannot
    /// leak into the rest of the screen.  The font remains the ordinary GUI renderer: Java's
    /// tile renderer scales an already laid-out 90-pixel line, so wrapping and width measurement
    /// intentionally happen before the CPU transform is pushed.
    pub fn draw_preview(&self, d: &mut DrawList, font: &mut FontRenderer, width: i32) {
        self.preview_geometry(width).draw(d);
        let matrices = SignPreviewMatrices::new(width, self.preview_kind);
        let line_being_edited = self.line_being_edited();
        d.with_transform(matrices.text_transform(), |d| {
            for (index, line) in self.lines.iter().enumerate() {
                let mut text = font
                    .wrap_to_lines(&line.formatted, MAX_LINE_WIDTH)
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                if index as i32 == line_being_edited {
                    text = format!("> {text} <");
                }
                let x = (-font.string_width(&text) / 2) as f32;
                let y = (index as i32 * 10 - 20) as f32;
                font.draw_string(d, &text, x, y, 0, false);
            }
        });
    }

    /// `TileEntitySign.lineBeingEdited` for this render tick: `-1` while the vanilla caret is
    /// blinked off.  The host writes this transiently to its sign render state.
    #[must_use]
    pub fn line_being_edited(&self) -> i32 {
        if !self.finished && self.update_counter / 6 % 2 == 0 {
            self.edit_line as i32
        } else {
            -1
        }
    }

    /// Advance `GuiEditSign.updateCounter` once per client tick.
    pub fn tick(&mut self) {
        self.update_counter = self.update_counter.wrapping_add(1);
    }

    /// Drain host operations in their Java occurrence order.
    pub fn take_actions(&mut self) -> Vec<SignAction> {
        std::mem::take(&mut self.actions)
    }

    /// Handle a keyboard event.  Up moves first; down, Return and keypad Return move first;
    /// then Java replaces that selected row with a new plain `ChatComponentText`.
    // java: GuiEditSign.java:92-122
    pub fn key_typed(&mut self, typed: Option<char>, key: i32, font: &FontRenderer) {
        if self.finished {
            return;
        }
        if key == keys::UP {
            self.edit_line = (self.edit_line + 3) & 3;
        }
        if matches!(key, keys::DOWN | keys::ENTER | keys::NUMPAD_ENTER) {
            self.edit_line = (self.edit_line + 1) & 3;
        }

        let mut text = self.lines[self.edit_line].raw.clone();
        if key == keys::BACKSPACE && !text.is_empty() {
            // Java removes one UTF-16 code unit. Rust's input boundary is scalar-value based, so
            // the unpaired-surrogate case cannot be represented here; for every valid scalar,
            // pop is the closest lossless operation available to this UI architecture.
            text.pop();
        }
        if let Some(character) = typed.filter(|character| keys::is_allowed_character(*character)) {
            let mut candidate = text.clone();
            candidate.push(character);
            if font.string_width(&candidate) <= MAX_LINE_WIDTH {
                text = candidate;
            }
        }
        // This occurs even for arrows, rejected input, an empty-backspace, and Escape.
        self.lines[self.edit_line].replace_from_key(text);

        if key == keys::ESCAPE {
            self.finish(true, true);
        }
    }

    /// Process mouse input.  Only a left click on the enabled Done button exits the editor.
    // java: GuiScreen.mouseClicked -> GuiEditSign.actionPerformed
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) {
        if !self.finished && button == 0 && self.done_button.mouse_pressed(x, y) {
            self.finish(true, true);
        }
    }

    /// Mirror `GuiEditSign.onGuiClosed` for a parent-driven screen replacement.
    ///
    /// This has no `MarkDirty` or `Close`: Java only marks the tile in its Done button handler,
    /// and this method is called after the host has already selected another screen.
    // java: GuiEditSign.java:52-63
    pub fn on_gui_closed(&mut self) {
        self.finish(false, false);
    }

    fn finish(&mut self, mark_dirty: bool, close: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        if mark_dirty {
            self.actions.push(SignAction::MarkDirty);
        }
        self.actions.push(SignAction::SendUpdateSign {
            position: self.position,
            lines: self.lines.clone().map(|line| line.component_json),
        });
        self.actions.push(SignAction::SetEditable(true));
        if close {
            self.actions.push(SignAction::Close);
        }
    }
}

// ---- command-block editor -----------------------------------------------------------------

/// The command target encoded in `MC|AdvCdm`'s first byte and following identifier.
///
/// java: `CommandBlockLogic.func_145751_f` plus the two `func_145757_a` implementations in
/// `TileEntityCommandBlock` and `EntityMinecartCommandBlock`.  A block target carries three
/// independent big-endian `i32`s; it is deliberately not the packed protocol `Position` long.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBlockTarget {
    /// A placed `TileEntityCommandBlock` (payload mode `0`).
    Block {
        /// The command block's world position.
        position: BlockPos,
    },
    /// An `EntityMinecartCommandBlock` (payload mode `1`).
    Minecart {
        /// The minecart's network entity id.
        entity_id: i32,
    },
}

/// A host-side operation requested by [`CommandBlockScreen`].
///
/// The screen does not own the world or network session.  The client drains these operations,
/// applies the local `trackOutput`/`lastOutput` mutations, and turns [`Self::Send`] into the
/// `MC|AdvCdm` `C17PacketCustomPayload` using its outbound bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandBlockAction {
    /// Mirror `CommandBlockLogic.setTrackOutput` on the O/X toggle and on Cancel.
    SetTrackOutput(bool),
    /// Send the command and target in the exact vanilla custom-payload shape.
    Send {
        /// The block or command minecart being edited.
        target: CommandBlockTarget,
        /// The current command-field text, including leading/trailing spaces.
        command: String,
        /// The current `TrackOutput` toggle.
        track_output: bool,
    },
    /// Mirror vanilla's `setLastOutput(null)` after a Done with output tracking disabled.
    ClearLastOutput,
    /// Remove the editor after the terminal operation has completed.
    Close,
}

const COMMAND_DONE_ID: i32 = 0;
const COMMAND_CANCEL_ID: i32 = 1;
const COMMAND_TEXT_ID: i32 = 2;
const COMMAND_OUTPUT_TEXT_ID: i32 = 3;
const COMMAND_TRACK_OUTPUT_ID: i32 = 4;
const COMMAND_TEXT_MAX_LENGTH: i32 = 32767;

/// Pure state and rendering model for vanilla's `GuiCommandBlock`.
///
/// `previous_output` is the already-unformatted display text of `CommandBlockLogic.lastOutput`.
/// [`Self::from_state`] is the convenient constructor for the raw component JSON retained by
/// `mc_world::tileent::CommandBlockState`; malformed JSON is treated like an unavailable output
/// and displays vanilla's `-` placeholder rather than panicking on untrusted server data.
#[derive(Debug, Clone)]
pub struct CommandBlockScreen {
    target: CommandBlockTarget,
    initial_command: String,
    previous_output: Option<String>,
    initial_track_output: bool,
    track_output: bool,
    /// The scaled viewport used by the last layout.  The client draws screens every frame, but
    /// vanilla only rebuilds `GuiTextField`s from `initGui` on display/resize; retaining this
    /// lets a live command edit survive the ordinary frame redraw.
    layout: Option<(i32, i32)>,
    command_text_field: TextField,
    previous_output_text_field: TextField,
    done_button: Button,
    cancel_button: Button,
    track_output_button: Button,
    finished: bool,
    actions: Vec<CommandBlockAction>,
    /// The live logic's raw values (command, `trackOutput`,
    /// `lastOutput` JSON) as last mirrored by [`Self::sync_live_logic`], so an unchanged frame
    /// costs only comparisons.
    live_raw: Option<(String, bool, Option<String>)>,
}

impl CommandBlockScreen {
    /// Construct an editor from the command and an optional unformatted previous-output string.
    #[must_use]
    pub fn new(
        target: CommandBlockTarget,
        command: impl Into<String>,
        track_output: bool,
        previous_output: Option<String>,
    ) -> Self {
        Self {
            target,
            initial_command: command.into(),
            previous_output,
            initial_track_output: track_output,
            track_output,
            layout: None,
            command_text_field: TextField::new(COMMAND_TEXT_ID, 0, 0, 0, 0),
            previous_output_text_field: TextField::new(COMMAND_OUTPUT_TEXT_ID, 0, 0, 0, 0),
            done_button: Button::sized(COMMAND_DONE_ID, 0, 0, 150, 20, ""),
            cancel_button: Button::sized(COMMAND_CANCEL_ID, 0, 0, 150, 20, ""),
            track_output_button: Button::sized(COMMAND_TRACK_OUTPUT_ID, 0, 0, 20, 20, "O"),
            finished: false,
            actions: Vec::new(),
            live_raw: None,
        }
    }

    /// Mirror the live `CommandBlockLogic` that vanilla's
    /// `GuiCommandBlock` holds by reference: the command block's tile state, or a command
    /// minecart's watcher-fed logic.
    ///
    /// java: an `S35PacketUpdateTileEntity` (a minecart: a watcher update) rewrites that logic
    /// while the editor is open, and the editor reads it back at three moments only: `initGui`
    /// on a resize (the command, `trackOutput` as Cancel's baseline, and `lastOutput`), the O/X
    /// toggle (`setTrackOutput(!shouldTrackOutput())`, then `lastOutput`), and Done (the
    /// `trackOutput` written into `MC|AdvCdm`). What is on screen is not refreshed in between.
    /// The host calls this before each input pass; only a changed value parses the JSON.
    pub fn sync_live_logic(
        &mut self,
        command: &str,
        track_output: bool,
        last_output_json: Option<&str>,
        lang: &mc_assets::Language,
    ) {
        if let Some((live_command, live_track, live_output)) = &self.live_raw {
            if live_command == command
                && *live_track == track_output
                && live_output.as_deref() == last_output_json
            {
                return;
            }
        }
        self.initial_command = command.to_owned();
        self.track_output = track_output;
        self.previous_output = last_output_json.and_then(|json| {
            ChatComponent::from_json(json)
                .ok()
                .map(|component| component.unformatted_text(lang))
        });
        self.live_raw = Some((
            command.to_owned(),
            track_output,
            last_output_json.map(ToOwned::to_owned),
        ));
    }

    /// Construct an editor from the typed command-block state delivered by S35 action `2`.
    ///
    /// java: `GuiCommandBlock.initGui` calls `getCommand`, `shouldTrackOutput`, and
    /// `getLastOutput().getUnformattedText()`.  The world state deliberately stores the raw
    /// component JSON, so parsing belongs here at the UI boundary.
    #[must_use]
    pub fn from_state(
        target: CommandBlockTarget,
        state: &mc_world::tileent::CommandBlockState,
        lang: &mc_assets::Language,
    ) -> Self {
        let previous_output = state.last_output.as_deref().and_then(|json| {
            ChatComponent::from_json(json)
                .ok()
                .map(|component| component.unformatted_text(lang))
        });
        Self::new(
            target,
            state.command.clone(),
            state.track_output,
            previous_output,
        )
    }

    /// Construct an editor from a command-minecart watcher snapshot.
    ///
    /// `EntityMinecartCommandBlock` keeps its `lastOutput` as watcher JSON and Java parses it
    /// only while opening `GuiCommandBlock`; malformed or empty text therefore becomes the
    /// ordinary unavailable-output placeholder rather than rejecting the GUI open.
    #[must_use]
    pub fn from_minecart(
        target: CommandBlockTarget,
        command: impl Into<String>,
        track_output: bool,
        last_output_json: &str,
        lang: &mc_assets::Language,
    ) -> Self {
        let previous_output = ChatComponent::from_json(last_output_json)
            .ok()
            .map(|component| component.unformatted_text(lang));
        Self::new(target, command, track_output, previous_output)
    }

    /// The command block or command minecart being edited.
    #[must_use]
    pub const fn target(&self) -> CommandBlockTarget {
        self.target
    }

    /// The command field's current text, including whitespace.
    #[must_use]
    pub fn command(&self) -> String {
        self.command_text_field.text()
    }

    /// Whether this editor currently tracks command output.
    #[must_use]
    pub const fn track_output(&self) -> bool {
        self.track_output
    }

    /// The command field, for the host's live input/caret bridge.
    #[must_use]
    pub const fn command_text_field(&self) -> &TextField {
        &self.command_text_field
    }

    /// The disabled previous-output field, for rendering and geometry tests.
    #[must_use]
    pub const fn previous_output_text_field(&self) -> &TextField {
        &self.previous_output_text_field
    }

    /// The Done button, including its vanilla enabled gate.
    #[must_use]
    pub const fn done_button(&self) -> &Button {
        &self.done_button
    }

    /// The Cancel button.
    #[must_use]
    pub const fn cancel_button(&self) -> &Button {
        &self.cancel_button
    }

    /// The `O`/`X` output-tracking toggle button.
    #[must_use]
    pub const fn track_output_button(&self) -> &Button {
        &self.track_output_button
    }

    /// Whether Done/Cancel has already queued a terminal operation.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Drain host operations in their Java occurrence order.
    pub fn take_actions(&mut self) -> Vec<CommandBlockAction> {
        std::mem::take(&mut self.actions)
    }

    /// Recreate the fields and three buttons at the current scaled resolution.
    ///
    /// java: `GuiCommandBlock.initGui` (`:35-61`).  The screen's command field is focused and
    /// both text fields use the 32767-character limit; the previous-output field is disabled.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &mc_assets::Language, font: &FontRenderer) {
        let center = ctx.width / 2;
        let button_y = ctx.height / 4 + 120 + 12;
        let layout = (ctx.width, ctx.height);
        if self.layout != Some(layout) {
            // A resize runs `initGui` again in Java, rereading `localCommandBlock.getCommand()`;
            // a normal redraw does not rebuild the field.  The initial command is therefore
            // intentionally restored only on a layout change, never once per frame.
            let command = self.initial_command.clone();
            // `field_175389_t` is also recaptured from the live logic on every Java `initGui`.
            // This matters when the player toggles output, resizes, then presses Cancel: the
            // resize is the new baseline, not the value from the first open.
            self.initial_track_output = self.track_output;
            self.done_button = Button::sized(
                COMMAND_DONE_ID,
                center - 4 - 150,
                button_y,
                150,
                20,
                lang.tr("gui.done"),
            );
            self.cancel_button = Button::sized(
                COMMAND_CANCEL_ID,
                center + 4,
                button_y,
                150,
                20,
                lang.tr("gui.cancel"),
            );
            self.track_output_button = Button::sized(
                COMMAND_TRACK_OUTPUT_ID,
                center + 150 - 20,
                150,
                20,
                20,
                if self.track_output { "O" } else { "X" },
            );

            self.command_text_field = TextField::new(COMMAND_TEXT_ID, center - 150, 50, 300, 20);
            self.command_text_field
                .set_max_length(COMMAND_TEXT_MAX_LENGTH);
            self.command_text_field.set_focused(true);
            self.command_text_field.set_text(&command, font);

            self.previous_output_text_field =
                TextField::new(COMMAND_OUTPUT_TEXT_ID, center - 150, 150, 276, 20);
            self.previous_output_text_field
                .set_max_length(COMMAND_TEXT_MAX_LENGTH);
            self.previous_output_text_field.enabled = false;
            // `initGui` writes `-`, then `func_175388_a`: the O/X
            // label and the output field show the live logic as of now, and stay so until the
            // next `initGui` or toggle, whatever packets change meanwhile.
            self.previous_output_text_field.set_text("-", font);
            self.refresh_previous_output(font);
            self.layout = Some(layout);
        } else {
            // Keep translated labels current without recreating the text widgets and losing the
            // command field's caret/selection state.
            self.done_button.text = lang.tr("gui.done").to_owned();
            self.cancel_button.text = lang.tr("gui.cancel").to_owned();
        }
        self.refresh_done_button();
    }

    /// Advance the focused command field's cursor blink.
    ///
    /// java: `GuiCommandBlock.updateScreen` (`:29-32`).
    pub fn tick(&mut self) {
        self.command_text_field.tick();
    }

    /// Handle a key event, including Ctrl clipboard operations in the command field.
    ///
    /// java: `GuiCommandBlock.keyTyped` (`:115-139`).  Return and keypad Return submit Done;
    /// Escape submits Cancel; every other key is left to the text fields.
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
    ) {
        if self.finished {
            return;
        }
        self.command_text_field
            .key_typed(typed, key, mods, clipboard, font);
        self.previous_output_text_field
            .key_typed(typed, key, mods, clipboard, font);
        self.refresh_done_button();

        match key {
            keys::ENTER | keys::NUMPAD_ENTER => self.finish_done(),
            keys::ESCAPE => self.finish_cancel(),
            _ => {}
        }
    }

    /// Handle a mouse press on the buttons or either text field.
    ///
    /// java: `GuiCommandBlock.actionPerformed` followed by `GuiCommandBlock.mouseClicked`.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32, font: &FontRenderer) {
        if self.finished {
            return;
        }
        if button == 0 && self.done_button.mouse_pressed(x, y) {
            self.finish_done();
            return;
        }
        if button == 0 && self.cancel_button.mouse_pressed(x, y) {
            self.finish_cancel();
            return;
        }
        if button == 0 && self.track_output_button.mouse_pressed(x, y) {
            self.track_output = !self.track_output;
            // java: `localCommandBlock.setTrackOutput(...)` writes
            // the live logic itself, so a later packet that leaves `TrackOutput` out keeps it.
            self.actions
                .push(CommandBlockAction::SetTrackOutput(self.track_output));
            self.track_output_button.text = if self.track_output { "O" } else { "X" }.to_owned();
            self.refresh_previous_output(font);
            return;
        }
        self.command_text_field.mouse_clicked(x, y, button, font);
        self.previous_output_text_field
            .mouse_clicked(x, y, button, font);
    }

    /// Draw the command-block editor over a world or dirt background.
    ///
    /// java: `GuiCommandBlock.drawScreen` (`:150-190`) and inherited
    /// `GuiScreen.drawDefaultBackground`.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        in_world: bool,
        lang: &mc_assets::Language,
    ) {
        crate::screens::common::draw_world_background(d, ctx.width, ctx.height, in_world);
        draw_centered_string(
            d,
            font,
            lang.tr("advMode.setCommand"),
            ctx.width / 2,
            20,
            TITLE_COLOR,
        );
        draw_string(
            d,
            font,
            lang.tr("advMode.command"),
            ctx.width / 2 - 150,
            37,
            LABEL_COLOR,
        );
        self.command_text_field.draw(d, font);

        let suggestions = [
            lang.tr("advMode.nearestPlayer"),
            lang.tr("advMode.randomPlayer"),
            lang.tr("advMode.allPlayers"),
            lang.tr("advMode.allEntities"),
            "",
        ];
        let suggestion_y = 75;
        for (index, suggestion) in suggestions.iter().enumerate() {
            draw_string(
                d,
                font,
                suggestion,
                ctx.width / 2 - 150,
                suggestion_y + index as i32 * font.font_height(),
                LABEL_COLOR,
            );
        }

        if !self.previous_output_text_field.is_empty() {
            let output_y = suggestion_y + suggestions.len() as i32 * font.font_height() + 16;
            draw_string(
                d,
                font,
                lang.tr("advMode.previousOutput"),
                ctx.width / 2 - 150,
                output_y,
                LABEL_COLOR,
            );
            self.previous_output_text_field.draw(d, font);
        }

        self.done_button.draw(d, font, mouse.0, mouse.1);
        self.cancel_button.draw(d, font, mouse.0, mouse.1);
        self.track_output_button.draw(d, font, mouse.0, mouse.1);
    }

    fn refresh_done_button(&mut self) {
        self.done_button.enabled = !self
            .command_text_field
            .text()
            .trim_matches(|c: char| c <= '\u{20}')
            .is_empty();
    }

    /// java: `func_175388_a`'s field half: the last output while tracking, `-` while not. With
    /// tracking on and no last output it leaves the field's text alone, so `initGui` writes `-`
    /// before calling it.
    fn refresh_previous_output(&mut self, font: &FontRenderer) {
        if self.track_output {
            if let Some(output) = self.previous_output.as_deref() {
                self.previous_output_text_field.set_text(output, font);
            }
        } else {
            self.previous_output_text_field.set_text("-", font);
        }
    }

    fn finish_done(&mut self) {
        if self.finished || !self.done_button.enabled {
            return;
        }
        self.finished = true;
        self.actions.push(CommandBlockAction::Send {
            target: self.target,
            command: self.command(),
            track_output: self.track_output,
        });
        if !self.track_output {
            self.actions.push(CommandBlockAction::ClearLastOutput);
        }
        self.actions.push(CommandBlockAction::Close);
    }

    fn finish_cancel(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.actions.push(CommandBlockAction::SetTrackOutput(
            self.initial_track_output,
        ));
        self.actions.push(CommandBlockAction::Close);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::ChatStyle;
    use crate::font::TextFormat;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;
    use mc_protocol::{Packet, PacketBuffer};

    fn position() -> BlockPos {
        BlockPos::new(12, 64, -7)
    }

    fn lang() -> Language {
        let mut lang = Language::new("en_US");
        lang.insert("gui.done", "Done");
        lang
    }

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn wire_lines(lines: [String; 4]) -> [String; 4] {
        let packet = mc_protocol::play::c2s::UpdateSign {
            position: position(),
            lines,
        };
        let mut encoded = PacketBuffer::new();
        packet.write(&mut encoded);
        assert!(encoded.finish_write().is_ok());

        let mut decoded = PacketBuffer::from_slice(encoded.as_slice());
        assert_eq!(decoded.read_block_pos().unwrap(), position());
        let lines = core::array::from_fn(|_| decoded.read_string(384).unwrap());
        assert!(decoded.is_empty());
        lines
    }

    #[test]
    fn init_uses_vanillas_button_and_locks_the_tile() {
        let mut screen = EditSignScreen::new(position(), Default::default());
        screen.init(&ctx(), &lang());
        let done = screen.done_button();
        assert_eq!(done.id, DONE_BUTTON_ID);
        assert_eq!((done.rect.x, done.rect.y), (854 / 2 - 100, 480 / 4 + 120));
        assert_eq!((done.rect.width, done.rect.height), (200, 20));
        assert_eq!(screen.take_actions(), vec![SignAction::SetEditable(false)]);
    }

    #[test]
    fn keys_wrap_before_editing_and_every_key_replaces_the_component() {
        let font = uniform_font();
        let styled = ChatComponent::text("north").with_style(ChatStyle {
            color: Some(TextFormat::Red),
            ..ChatStyle::default()
        });
        let mut screen = EditSignScreen::from_components(
            position(),
            [
                styled,
                ChatComponent::text("east"),
                ChatComponent::text("south"),
                ChatComponent::text("west"),
            ],
            &lang(),
        );

        screen.key_typed(None, keys::UP, &font);
        assert_eq!(screen.edit_line(), 3);
        assert_eq!(screen.line(3), Some("west"));
        screen.key_typed(None, keys::ENTER, &font);
        assert_eq!(screen.edit_line(), 0, "Return wraps 3 -> 0 before editing");
        screen.on_gui_closed();

        let actions = screen.take_actions();
        let SignAction::SendUpdateSign { lines, .. } = &actions[0] else {
            panic!("C12 action missing: {actions:?}");
        };
        assert_eq!(
            lines[0], "\"north\"",
            "Return stripped the original red style"
        );
        assert_eq!(lines[3], "\"west\"", "Up also rewrote its target row");
        assert_eq!(lines[1], "\"east\"");
        assert_eq!(lines[2], "\"south\"");
    }

    #[test]
    fn typing_uses_allowed_characters_and_a_ninety_pixel_width_not_a_char_limit() {
        let font = uniform_font(); // every ASCII glyph advances five pixels
        let mut screen = EditSignScreen::new(position(), Default::default());
        for _ in 0..18 {
            screen.key_typed(Some('i'), 23, &font);
        }
        assert_eq!(
            screen.line(0).unwrap().chars().count(),
            18,
            "18*5 == 90 fits"
        );
        screen.key_typed(Some('i'), 23, &font);
        assert_eq!(
            screen.line(0).unwrap().chars().count(),
            18,
            "19*5 exceeds 90"
        );

        screen.key_typed(Some('\u{a7}'), 0, &font);
        screen.key_typed(Some('\t'), 0, &font);
        screen.key_typed(Some('\u{7f}'), 0, &font);
        assert_eq!(screen.line(0).unwrap().chars().count(), 18);

        // A rejected character is still a key event: it replaces a styled target component.
        let styled = ChatComponent::text("x").with_style(ChatStyle {
            bold: Some(true),
            ..ChatStyle::default()
        });
        let mut styled_screen = EditSignScreen::from_components(
            position(),
            [
                styled,
                ChatComponent::text(""),
                ChatComponent::text(""),
                ChatComponent::text(""),
            ],
            &lang(),
        );
        styled_screen.key_typed(Some('\u{a7}'), 0, &font);
        styled_screen.on_gui_closed();
        let SignAction::SendUpdateSign { lines, .. } = &styled_screen.take_actions()[0] else {
            unreachable!();
        };
        assert_eq!(
            lines[0], "\"x\"",
            "the rejected section sign still removes styling"
        );
    }

    #[test]
    fn close_always_emits_exact_c12_component_strings_once() {
        let mut screen = EditSignScreen::new(
            position(),
            [
                "hello".into(),
                "".into(),
                "quote: \"".into(),
                "newline\\n".into(),
            ],
        );
        screen.on_gui_closed();
        screen.on_gui_closed();
        assert_eq!(
            screen.take_actions(),
            vec![
                SignAction::SendUpdateSign {
                    position: position(),
                    lines: [
                        "\"hello\"".into(),
                        "\"\"".into(),
                        "\"quote: \\\"\"".into(),
                        "\"newline\\\\n\"".into(),
                    ],
                },
                SignAction::SetEditable(true),
            ],
            "even an untouched blank sign sends one C12; plain ChatComponentText serialises as a JSON string primitive"
        );
    }

    #[test]
    fn untouched_rows_are_canonicalized_for_c12_while_editor_text_and_order_survive() {
        let mut screen = EditSignScreen::from_component_json(
            position(),
            [
                // Same component as row two, but deliberately different whitespace/key order.
                r#"{ "text" : "same", "bold" : true }"#.into(),
                r#"{"bold":true,"text":"same"}"#.into(),
                // A bare JSON string is the Java serializer's plain ChatComponentText shape.
                r#""plain text""#.into(),
                // S33/C12 normally reject this before the screen sees it.  Keep the screen's
                // non-panicking fallback and make its outgoing representation canonical too.
                "{not valid json".into(),
            ],
            &lang(),
        );

        assert_eq!(
            screen.raw_lines(),
            [
                "same".to_owned(),
                "same".to_owned(),
                "plain text".to_owned(),
                "{not valid json".to_owned(),
            ],
            "parsing/canonicalizing a row must not change the text presented by GuiEditSign"
        );

        screen.on_gui_closed();
        let SignAction::SendUpdateSign { lines, .. } = &screen.take_actions()[0] else {
            panic!("C12 action missing")
        };
        let expected = [
            r#"{"bold":true,"text":"same"}"#.to_owned(),
            r#"{"bold":true,"text":"same"}"#.to_owned(),
            r#""plain text""#.to_owned(),
            r#""{not valid json""#.to_owned(),
        ];
        assert_eq!(
            lines, &expected,
            "each row must use Java's canonical component JSON"
        );
        assert_eq!(
            wire_lines(lines.clone()),
            expected,
            "C12 wire rows retain canonical order"
        );
    }

    #[test]
    fn c12_canonicalization_matches_gson_quotes_html_escapes_and_non_bmp() {
        // This is the live editor boundary where Java's C12 packet would invoke
        // IChatComponent.Serializer.componentToJson.  Keep actual characters in the input: the
        // expected strings below are the exact Gson spelling, not merely semantically equivalent
        // JSON.  The supplementary scalar must remain a scalar (Gson does not force it into a
        // surrogate escape), while HTML-sensitive characters and U+2028/U+2029 are escaped.
        let mut screen = EditSignScreen::from_component_json(
            position(),
            [
                r#"{ "text": "quote: \" and slash: \\", "bold": true }"#.into(),
                r#"{ "text": "<>&='" }"#.into(),
                "{\"text\":\"astral 😀\\u2028line\\u2029\"}".into(),
                r#"{ "text": "ordered", "extra": ["tail"], "italic": false, "color": "gold" }"#
                    .into(),
            ],
            &lang(),
        );

        assert_eq!(
            screen.raw_lines(),
            [
                "quote: \" and slash: \\".to_owned(),
                "<>&='".to_owned(),
                "astral 😀\u{2028}line\u{2029}".to_owned(),
                "orderedtail".to_owned(),
            ]
        );

        screen.on_gui_closed();
        let SignAction::SendUpdateSign { lines, .. } = &screen.take_actions()[0] else {
            panic!("C12 action missing")
        };
        let expected = [
            r#"{"bold":true,"text":"quote: \" and slash: \\"}"#.to_owned(),
            r#""\u003c\u003e\u0026\u003d\u0027""#.to_owned(),
            "\"astral 😀\\u2028line\\u2029\"".to_owned(),
            r#"{"italic":false,"color":"gold","extra":["tail"],"text":"ordered"}"#.to_owned(),
        ];
        assert_eq!(lines, &expected);
        assert_eq!(wire_lines(lines.clone()), expected);
    }

    #[test]
    fn edited_non_bmp_text_becomes_a_plain_json_string() {
        // A key event replaces the selected component with ChatComponentText, just as Java's
        // GuiEditSign does.  This control protects both the primitive JSON shape and UTF-8
        // preservation for a supplementary scalar.
        let mut screen = EditSignScreen::new(position(), Default::default());
        screen.key_typed(Some('😀'), 0, &uniform_font());
        screen.on_gui_closed();
        let SignAction::SendUpdateSign { lines, .. } = &screen.take_actions()[0] else {
            panic!("C12 action missing")
        };
        assert_eq!(lines[0], "\"😀\"");
        assert_eq!(wire_lines(lines.clone())[0], "\"😀\"");
    }

    #[test]
    fn done_and_escape_mark_dirty_send_and_close_but_replacement_does_not_mark_dirty() {
        let font = uniform_font();
        let mut done = EditSignScreen::new(position(), Default::default());
        done.mouse_clicked(100, 100, 1);
        assert!(
            done.take_actions().is_empty(),
            "right click is a negative control"
        );
        done.init(&ctx(), &lang());
        let x = done.done_button().rect.x + 1;
        let y = done.done_button().rect.y + 1;
        done.mouse_clicked(x, y, 0);
        assert!(matches!(
            done.take_actions().as_slice(),
            [
                SignAction::SetEditable(false),
                SignAction::MarkDirty,
                SignAction::SendUpdateSign { .. },
                SignAction::SetEditable(true),
                SignAction::Close
            ]
        ));

        let mut escape = EditSignScreen::new(position(), Default::default());
        escape.key_typed(None, keys::ESCAPE, &font);
        assert!(matches!(
            escape.take_actions().as_slice(),
            [
                SignAction::MarkDirty,
                SignAction::SendUpdateSign { .. },
                SignAction::SetEditable(true),
                SignAction::Close
            ]
        ));
    }

    #[test]
    fn caret_blinks_six_ticks_on_then_six_ticks_off() {
        let mut screen = EditSignScreen::new(position(), Default::default());
        assert_eq!(screen.line_being_edited(), 0);
        for _ in 0..6 {
            screen.tick();
        }
        assert_eq!(screen.line_being_edited(), -1);
        for _ in 0..6 {
            screen.tick();
        }
        assert_eq!(screen.line_being_edited(), 0);
    }

    #[test]
    fn sign_preview_is_centered_scaled_and_bound_to_vanillas_texture() {
        let width = 855;
        let geometry = sign_preview_geometry(width, SignPreviewKind::standing(0));
        assert_eq!(geometry.faces.len(), 12, "six board + six stick faces");
        assert_eq!(
            geometry
                .faces
                .iter()
                .filter(|face| face.part == SignPreviewPart::Board)
                .count(),
            6
        );
        assert_eq!(
            geometry
                .faces
                .iter()
                .filter(|face| face.part == SignPreviewPart::Stick)
                .count(),
            6
        );

        let board = geometry
            .faces
            .iter()
            .filter(|face| face.part == SignPreviewPart::Board)
            .flat_map(|face| face.vertices.iter());
        let (min_x, max_x) = board
            .clone()
            .map(|vertex| vertex.position[0])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
                (min.min(x), max.max(x))
            });
        let centre = (width / 2) as f32;
        assert!((min_x - (centre - 46.875)).abs() < 1e-4);
        assert!((max_x - (centre + 46.875)).abs() < 1e-4);

        let mut draw = DrawList::new();
        geometry.draw(&mut draw);
        assert_eq!(draw.vertices().len(), 12 * 6);
        assert_eq!(draw.batches().len(), 1);
        assert_eq!(draw.batches()[0].texture, UiTexture::gui(SIGN_TEXTURE));
    }

    #[test]
    fn sign_preview_reproduces_model_sign_atlas_uvs_and_text_depth_offset() {
        let geometry = sign_preview_geometry(854, SignPreviewKind::standing(0));
        let front = geometry
            .faces
            .iter()
            .find(|face| {
                face.part == SignPreviewPart::Board
                    && face.vertices[0].uv == [26.0 / 64.0, 2.0 / 32.0]
            })
            .expect("ModelSign board front face must use the 2..26 by 2..14 atlas region");
        assert_eq!(
            front.vertices.map(|vertex| vertex.uv),
            [
                [26.0 / 64.0, 2.0 / 32.0],
                [2.0 / 64.0, 2.0 / 32.0],
                [2.0 / 64.0, 14.0 / 32.0],
                [26.0 / 64.0, 14.0 / 32.0],
            ]
        );
        assert!(geometry
            .faces
            .windows(2)
            .all(|faces| faces[0].depth <= faces[1].depth));

        let matrices = SignPreviewMatrices::new(854, SignPreviewKind::standing(0));
        let transform = matrices.text_transform();
        let [text_x, text_y] = transform.apply([0.0, 0.0]);
        assert!((text_x - 427.0).abs() < 1e-4, "text origin x={text_x}");
        assert!((text_y - 91.796875).abs() < 1e-4, "text origin y={text_y}");
        let text_origin_3d = matrices.text.transform_point3(Vec3::ZERO);
        assert!(
            (text_origin_3d.z - 54.375).abs() < 1e-4,
            "the renderer's 0.07*f text-depth offset was dropped: z={}",
            text_origin_3d.z
        );
    }

    #[test]
    fn sign_preview_draws_untouched_component_formatting_but_key_replacement_is_plain() {
        let styled = ChatComponent::text("A").with_style(ChatStyle {
            color: Some(TextFormat::Red),
            ..ChatStyle::default()
        });
        let mut screen = EditSignScreen::from_components(
            position(),
            [
                styled,
                ChatComponent::text(""),
                ChatComponent::text(""),
                ChatComponent::text(""),
            ],
            &lang(),
        );
        let mut draw = DrawList::new();
        let mut font = uniform_font();
        screen.draw_preview(&mut draw, &mut font, 854);
        let red = [1.0, 85.0 / 255.0, 85.0 / 255.0, 1.0];
        assert!(draw.vertices().iter().any(|vertex| vertex.color == red));

        screen.key_typed(Some('!'), 0, &uniform_font());
        let mut plain_draw = DrawList::new();
        let mut plain_font = uniform_font();
        screen.draw_preview(&mut plain_draw, &mut plain_font, 854);
        assert!(!plain_draw
            .vertices()
            .iter()
            .any(|vertex| vertex.color == red));
    }

    #[test]
    fn wall_sign_preview_uses_java_facing_angles_and_omits_the_stick() {
        assert_eq!(SignPreviewKind::wall(2).rotation_degrees(), 180.0);
        assert_eq!(SignPreviewKind::wall(4).rotation_degrees(), 90.0);
        assert_eq!(SignPreviewKind::wall(5).rotation_degrees(), -90.0);
        assert_eq!(SignPreviewKind::wall(3).rotation_degrees(), 0.0);

        let geometry = sign_preview_geometry(854, SignPreviewKind::wall(5));
        assert_eq!(geometry.faces.len(), 6);
        assert!(geometry
            .faces
            .iter()
            .all(|face| face.part == SignPreviewPart::Board));
        assert!(geometry
            .faces
            .windows(2)
            .all(|faces| faces[0].depth <= faces[1].depth));
        assert_ne!(
            geometry,
            sign_preview_geometry(854, SignPreviewKind::wall(4)),
            "metadata 5 must use the Java -90 degree wall rotation, not metadata 4's +90"
        );
        assert_ne!(
            geometry,
            sign_preview_geometry(854, SignPreviewKind::standing(0)),
            "wall renderer's support translation/orientation is not a standing sign"
        );
    }

    #[test]
    fn sign_preview_draws_caret_text_after_model_with_a_balanced_transform() {
        let screen =
            EditSignScreen::new(position(), ["A".into(), "B".into(), "C".into(), "D".into()]);
        let mut draw = DrawList::new();
        let mut font = uniform_font();
        screen.draw_preview(&mut draw, &mut font, 854);

        assert_eq!(draw.transform_depth(), 0);
        assert_eq!(draw.batches()[0].texture, UiTexture::gui(SIGN_TEXTURE));
        assert!(
            draw.batches().len() >= 2,
            "font text must follow the sign tile"
        );
        assert!(draw
            .batches()
            .iter()
            .skip(1)
            .any(|batch| matches!(batch.texture, UiTexture::Font(_))));
        assert!(draw
            .vertices()
            .iter()
            .skip(12 * 6)
            .any(|vertex| vertex.pos[0] != 427.0));
    }
}

#[cfg(test)]
mod command_block_tests {
    use super::*;
    use crate::widget::keys::MemoryClipboard;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> mc_assets::Language {
        let mut lang = mc_assets::Language::new("en_US");
        lang.insert("gui.done", "Done");
        lang.insert("gui.cancel", "Cancel");
        lang.insert("advMode.setCommand", "Set Command");
        lang.insert("advMode.command", "Command");
        lang.insert("advMode.previousOutput", "Previous Output");
        lang
    }

    #[test]
    fn init_matches_vanillas_command_fields_buttons_and_trim_gate() {
        let font = uniform_font();
        let mut screen = CommandBlockScreen::new(
            CommandBlockTarget::Block {
                position: BlockPos::new(12, 64, -7),
            },
            "  say hi  ",
            true,
            Some("ran".to_owned()),
        );
        screen.init(&ctx(), &lang(), &font);

        assert_eq!(
            screen.command_text_field().rect,
            crate::widget::draw::Rect::new(277, 50, 300, 20)
        );
        assert_eq!(screen.command_text_field().max_length(), 32767);
        assert!(screen.command_text_field().is_focused());
        assert_eq!(screen.command(), "  say hi  ");
        assert_eq!(
            screen.previous_output_text_field().rect,
            crate::widget::draw::Rect::new(277, 150, 276, 20)
        );
        assert!(!screen.previous_output_text_field().enabled);
        assert_eq!(screen.previous_output_text_field().text(), "ran");
        assert!(screen.done_button().enabled, "trimmed command is non-empty");
        assert_eq!(
            screen.done_button().rect,
            crate::widget::draw::Rect::new(273, 252, 150, 20)
        );
        assert_eq!(
            screen.cancel_button().rect,
            crate::widget::draw::Rect::new(431, 252, 150, 20)
        );
        assert_eq!(
            screen.track_output_button().rect,
            crate::widget::draw::Rect::new(557, 150, 20, 20)
        );
        assert_eq!(screen.track_output_button().text, "O");
    }

    #[test]
    fn enter_sends_the_current_command_once_and_closes() {
        let font = uniform_font();
        let mut clipboard = MemoryClipboard::default();
        let target = CommandBlockTarget::Minecart { entity_id: 41 };
        let mut screen = CommandBlockScreen::new(target, "say", true, None);
        screen.init(&ctx(), &lang(), &font);
        screen.key_typed(Some('!'), 23, Modifiers::NONE, &mut clipboard, &font);
        assert_eq!(screen.command(), "say!");
        screen.key_typed(None, keys::ENTER, Modifiers::NONE, &mut clipboard, &font);
        assert_eq!(
            screen.take_actions(),
            vec![
                CommandBlockAction::Send {
                    target,
                    command: "say!".to_owned(),
                    track_output: true,
                },
                CommandBlockAction::Close,
            ]
        );
        screen.key_typed(None, keys::ENTER, Modifiers::NONE, &mut clipboard, &font);
        assert!(screen.take_actions().is_empty());
    }

    #[test]
    fn toggle_refreshes_previous_output_and_cancel_only_commits_track_output() {
        let font = uniform_font();
        let mut screen = CommandBlockScreen::new(
            CommandBlockTarget::Block {
                position: BlockPos::ORIGIN,
            },
            "  ",
            true,
            Some("last".to_owned()),
        );
        screen.init(&ctx(), &lang(), &font);
        assert!(!screen.done_button().enabled);

        let toggle = screen.track_output_button().rect;
        screen.mouse_clicked(toggle.x + 1, toggle.y + 1, 0, &font);
        assert!(!screen.track_output());
        assert_eq!(screen.track_output_button().text, "X");
        assert_eq!(screen.previous_output_text_field().text(), "-");
        // java: the toggle is `localCommandBlock.setTrackOutput`,
        // a write to the live logic, not a value held by the screen.
        assert_eq!(
            screen.take_actions(),
            vec![CommandBlockAction::SetTrackOutput(false)]
        );

        let cancel = screen.cancel_button().rect;
        screen.mouse_clicked(cancel.x + 1, cancel.y + 1, 0, &font);
        assert_eq!(
            screen.take_actions(),
            vec![
                // `GuiCommandBlock` restores the value captured in initGui on Cancel.
                CommandBlockAction::SetTrackOutput(true),
                CommandBlockAction::Close
            ]
        );
    }

    #[test]
    fn resize_recaptures_cancel_baseline_and_rereads_command() {
        let font = uniform_font();
        let mut clipboard = MemoryClipboard::default();
        let mut screen = CommandBlockScreen::new(
            CommandBlockTarget::Block {
                position: BlockPos::ORIGIN,
            },
            "say",
            true,
            None,
        );
        screen.init(&ctx(), &lang(), &font);
        screen.key_typed(Some('!'), 0, Modifiers::NONE, &mut clipboard, &font);
        assert_eq!(screen.command(), "say!");

        let toggle = screen.track_output_button().rect;
        screen.mouse_clicked(toggle.x + 1, toggle.y + 1, 0, &font);
        assert!(!screen.track_output());

        let resized = ScreenCtx::new(ScaledResolution::new(640, 360, 1, false), 0.0);
        screen.init(&resized, &lang(), &font);
        assert_eq!(screen.command(), "say");
        let cancel = screen.cancel_button().rect;
        screen.mouse_clicked(cancel.x + 1, cancel.y + 1, 0, &font);
        assert_eq!(
            screen.take_actions(),
            vec![
                // The toggle's own write to the live logic, then Cancel's restore of the
                // baseline the resize recaptured.
                CommandBlockAction::SetTrackOutput(false),
                CommandBlockAction::SetTrackOutput(false),
                CommandBlockAction::Close,
            ]
        );
    }

    /// java: `GuiCommandBlock` holds the live logic. A packet
    /// that rewrites it while the editor is open changes nothing on screen until a resize
    /// (`initGui`) reads it back; Done always sends the live `trackOutput`.
    #[test]
    fn live_logic_changes_show_after_a_resize_and_done_sends_the_live_track_output() {
        let font = uniform_font();
        let lang = lang();
        let mut clipboard = MemoryClipboard::default();
        let target = CommandBlockTarget::Block {
            position: BlockPos::ORIGIN,
        };
        let mut screen = CommandBlockScreen::new(target, "say old", true, Some("old".to_owned()));
        screen.sync_live_logic("say old", true, Some(r#"{"text":"old"}"#), &lang);
        screen.init(&ctx(), &lang, &font);
        assert_eq!(screen.command(), "say old");
        assert_eq!(screen.previous_output_text_field().text(), "old");

        // An S35 rewrites the logic: the editor keeps showing what initGui read.
        screen.sync_live_logic("say new", true, Some(r#"{"text":"new"}"#), &lang);
        screen.init(&ctx(), &lang, &font);
        assert_eq!(screen.command(), "say old");
        assert_eq!(screen.previous_output_text_field().text(), "old");

        // A resize runs initGui again over the live logic.
        let resized = ScreenCtx::new(ScaledResolution::new(640, 360, 1, false), 0.0);
        screen.init(&resized, &lang, &font);
        assert_eq!(screen.command(), "say new");
        assert_eq!(screen.previous_output_text_field().text(), "new");

        // A later packet turns tracking off (1.8.9's `readDataFromNBT` then keeps the old
        // output): the screen still shows "O", but Done writes the live bit and clears the
        // output.
        screen.sync_live_logic("say new", false, Some(r#"{"text":"new"}"#), &lang);
        screen.init(&resized, &lang, &font);
        assert_eq!(screen.track_output_button().text, "O");
        assert_eq!(screen.previous_output_text_field().text(), "new");
        screen.key_typed(None, keys::ENTER, Modifiers::NONE, &mut clipboard, &font);
        assert_eq!(
            screen.take_actions(),
            vec![
                CommandBlockAction::Send {
                    target,
                    command: "say new".to_owned(),
                    track_output: false,
                },
                CommandBlockAction::ClearLastOutput,
                CommandBlockAction::Close,
            ]
        );
    }

    /// java: the toggle flips the live bit and `func_175388_a`
    /// reads the live output, so a new output that arrived meanwhile shows on the way back.
    #[test]
    fn the_toggle_flips_the_live_track_output_and_reads_the_live_output() {
        let font = uniform_font();
        let lang = lang();
        let mut screen = CommandBlockScreen::new(
            CommandBlockTarget::Minecart { entity_id: 9 },
            "say",
            true,
            Some("a".to_owned()),
        );
        screen.sync_live_logic("say", true, Some(r#"{"text":"a"}"#), &lang);
        screen.init(&ctx(), &lang, &font);
        assert_eq!(screen.previous_output_text_field().text(), "a");

        // A new output arrives: nothing on screen changes yet.
        screen.sync_live_logic("say", true, Some(r#"{"text":"b"}"#), &lang);
        let toggle = screen.track_output_button().rect;
        screen.mouse_clicked(toggle.x + 1, toggle.y + 1, 0, &font);
        assert!(!screen.track_output());
        assert_eq!(screen.track_output_button().text, "X");
        assert_eq!(screen.previous_output_text_field().text(), "-");
        assert_eq!(
            screen.take_actions(),
            vec![CommandBlockAction::SetTrackOutput(false)]
        );

        // The host wrote that bit into the live logic; the next toggle reads the live output.
        screen.sync_live_logic("say", false, Some(r#"{"text":"b"}"#), &lang);
        screen.mouse_clicked(toggle.x + 1, toggle.y + 1, 0, &font);
        assert!(screen.track_output());
        assert_eq!(screen.track_output_button().text, "O");
        assert_eq!(screen.previous_output_text_field().text(), "b");
        assert_eq!(
            screen.take_actions(),
            vec![CommandBlockAction::SetTrackOutput(true)]
        );
    }

    #[test]
    fn from_state_reduces_last_output_component_to_unformatted_text() {
        let font = uniform_font();
        let state = mc_world::tileent::CommandBlockState {
            command: "say hi".to_owned(),
            last_output: Some(r#"{"text":"ran","bold":true}"#.to_owned()),
            ..Default::default()
        };
        let mut screen = CommandBlockScreen::from_state(
            CommandBlockTarget::Block {
                position: BlockPos::new(1, 2, 3),
            },
            &state,
            &lang(),
        );
        screen.init(&ctx(), &lang(), &font);
        assert_eq!(screen.previous_output_text_field().text(), "ran");
    }

    #[test]
    fn from_minecart_parses_raw_watcher_output_and_keeps_empty_as_placeholder() {
        let font = uniform_font();
        let target = CommandBlockTarget::Minecart { entity_id: 41 };
        let mut screen = CommandBlockScreen::from_minecart(
            target,
            "say cart",
            true,
            r#"{"text":"ran","italic":true}"#,
            &lang(),
        );
        screen.init(&ctx(), &lang(), &font);
        assert_eq!(screen.target(), target);
        assert_eq!(screen.command(), "say cart");
        assert_eq!(screen.previous_output_text_field().text(), "ran");

        let mut empty = CommandBlockScreen::from_minecart(target, "", true, "", &lang());
        empty.init(&ctx(), &lang(), &font);
        assert_eq!(empty.previous_output_text_field().text(), "-");
    }
}
