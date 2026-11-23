//! The chat component tree.
//!
//! java: `util/IChatComponent.java`, `ChatComponentStyle.java`, `ChatComponentText.java`,
//! `ChatComponentTranslation.java`, `ChatComponentScore.java`, `ChatComponentSelector.java`,
//! `ChatStyle.java`, `event/ClickEvent.java`, `event/HoverEvent.java`
//!
//! Every chat line, sign, book page, scoreboard title and disconnect reason arrives as this
//! JSON tree. It is **untrusted server input**, so parsing never panics and never recurses
//! without a depth bound.
//!
//! # No parent pointers
//!
//! Vanilla's `ChatStyle` holds a mutable pointer to its parent and resolves each property by
//! walking up the chain. The arena-index rule rules that shape out, and it is not needed: the parent
//! style is always exactly the style of the node you descended through, so
//! [`ChatComponent::spans`] threads a [`ResolvedStyle`] down the traversal instead. The
//! observable result — including the exact `§`-code sequence [`ChatComponent::formatted_text`]
//! produces — is identical.
//!
//! # Traversal shape
//!
//! Vanilla's iterator yields *the node itself* then each sibling's full sub-iteration — except
//! for a translation component, which yields **its expanded children** instead of itself. That
//! is why a nested translation never double-counts its own text, and it is reproduced here.

use std::fmt;

use mc_assets::Language;
use serde_json::{Map, Value};

use super::format::{TextFormat, SECTION_SIGN};
use crate::UiError;

/// How deep a server-supplied component tree may nest before the parser stops descending.
///
/// # This is a stack bound, not a strictness policy
///
/// It used to be `64` and to produce a **hard error**, on the reasoning that vanilla has no limit
/// and a hostile server could nest a component ten thousand deep for free. That reasoning was
/// backwards on both halves and is why this doc is long.
///
/// * `64` is not a hostile depth. Vanilla renders a depth-1000 tree without complaint -- Gson's
///   own recursion and `IChatComponent`'s are both plain recursion on a default JVM stack -- and
///   a server that decorates each word of a long sentence with its own `extra` node reaches
///   tens of levels in ordinary use.
/// * A hard error is not "safer". `UiError::Component` propagates out of the `S02PacketChat`
///   handler and into `self.disconnect()`, so under the old rule **one chat message could
///   disconnect the player**, which is a strictly worse outcome than rendering the message
///   wrong. `from_value` now truncates instead: past this depth the subtree is dropped and an
///   empty text node takes its place, and no error is ever raised for depth alone.
///
/// A bound still has to exist, because in Rust the danger is not a catchable
/// `StackOverflowError`: [`ChatComponent`] owns `Vec<ChatComponent>`, so both parsing *and the
/// recursive `Drop`* descend the tree, and a stack overflow is an immediate `SIGSEGV`/abort with
/// no unwind. Capping construction is what keeps the destructor bounded too. `1024` sits above
/// anything vanilla renders and roughly two orders of magnitude below the depth that threatens
/// an 8 MiB stack.
pub const MAX_DEPTH: usize = 1024;

/// How many nested JSON *containers* [`ChatComponent::from_json`] will hand to `serde_json`.
///
/// `serde_json` has its own `RECURSION_LIMIT = 128` on nested `{`/`[`, and exceeding it is a
/// **parse error**, not a truncation -- which lands on the same `self.disconnect()` path
/// [`MAX_DEPTH`] describes. A component level is two containers (`{"extra":[ … ]}`), so that
/// limit bites at roughly 64 component levels: the audit blamed the old `MAX_DEPTH = 64` for
/// the disconnect, but this is the constraint that was actually binding, and the two numbers
/// coinciding is why raising `MAX_DEPTH` alone changes nothing.
///
/// `serde_json` can be told to drop its limit (`Deserializer::disable_recursion_limit`), but
/// only behind its `unbounded_depth` feature, which this workspace does not enable -- and
/// unbounded recursion over hostile input trades a disconnect for a `SIGSEGV`. So instead
/// [`clamp_json_depth`] rewrites anything deeper into a shallow-but-valid document before
/// `serde_json` ever sees it, and the client renders what it can.
const MAX_JSON_CONTAINERS: usize = 96;

/// The most work [`ChatComponent::spans`] does for one component:
/// one unit per node visited, plus one per byte of translation format read and of text emitted.
///
/// A translation argument can be referenced any number of times (`%1$s%1$s`), and a missing key
/// is its own format, so a server can nest `{"translate":"%1$s%1$s","with":[ ... ]}` and double
/// the flattened text at every level: forty levels fit in one chat packet or server-list MOTD
/// and flatten to 2^40 spans. Vanilla walks the same tree with `createDeepCopyIterator` and hangs
/// until it runs out of memory. Here the walk stops when the budget is spent and keeps the text
/// it has. A component that does not reuse arguments stays far below it: a 32767-character
/// packet holds at most about 1600 translation nodes, and each costs a few units per byte of its
/// format, which is a few hundred bytes for the longest 1.8.9 string.
pub const MAX_FLATTEN_WORK: usize = 1 << 22;

/// Charge one span of `len` bytes to the flatten budget; `false` (and the budget spent) if it
/// no longer covers it.
fn charge_span(work: &mut usize, len: usize) -> bool {
    let cost = len.saturating_add(1);
    if cost > *work {
        *work = 0;
        false
    } else {
        *work -= cost;
        true
    }
}

/// What clicking styled text does.
// java: event/ClickEvent.java:80 (Action)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClickAction {
    /// Open a URL in the system browser (behind a confirmation prompt).
    OpenUrl,
    /// Open a local file. **Never** honoured for server-sent components.
    OpenFile,
    /// Send the value as a chat message.
    RunCommand,
    /// Twitch chat moderation. The service is dead; kept so round-tripping is lossless.
    TwitchUserInfo,
    /// Put the value in the chat input box.
    SuggestCommand,
    /// Turn to a page of a written book.
    ChangePage,
    /// Not vanilla: put the PNG at the value's path on the clipboard as an image. The F2 chat
    /// line's `[COPY]` link. Client-only like `OpenFile`:
    /// never honoured for server-sent components (see [`ClickAction::allowed_in_chat`]).
    CopyImage,
}

impl ClickAction {
    /// The name used in JSON.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            ClickAction::OpenUrl => "open_url",
            ClickAction::OpenFile => "open_file",
            ClickAction::RunCommand => "run_command",
            ClickAction::TwitchUserInfo => "twitch_user_info",
            ClickAction::SuggestCommand => "suggest_command",
            ClickAction::ChangePage => "change_page",
            ClickAction::CopyImage => "copy_image",
        }
    }

    /// Parse a JSON action name.
    ///
    /// [`ClickAction::CopyImage`] is deliberately absent: no JSON (a server's or anyone's) can
    /// name it, so only a component the client builds itself carries it.
    #[must_use]
    pub fn from_canonical(name: &str) -> Option<Self> {
        [
            ClickAction::OpenUrl,
            ClickAction::OpenFile,
            ClickAction::RunCommand,
            ClickAction::TwitchUserInfo,
            ClickAction::SuggestCommand,
            ClickAction::ChangePage,
        ]
        .into_iter()
        .find(|a| a.canonical_name() == name)
    }

    /// Whether a *server-sent* component may carry this action.
    ///
    /// `open_file` and `twitch_user_info` are client-only: vanilla drops them at deserialise
    /// time so a server cannot make the client open arbitrary local paths.
    // java: ClickEvent.java:81 (the `allowedInChat` column)
    #[must_use]
    pub const fn allowed_in_chat(self) -> bool {
        matches!(
            self,
            ClickAction::OpenUrl
                | ClickAction::RunCommand
                | ClickAction::SuggestCommand
                | ClickAction::ChangePage
        )
    }
}

/// A click action plus its argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickEvent {
    /// What to do.
    pub action: ClickAction,
    /// The URL, command or page number.
    pub value: String,
}

/// What hovering over styled text shows.
// java: event/HoverEvent.java:80 (Action)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoverAction {
    /// A tooltip of literal text.
    ShowText,
    /// An achievement or statistic, named by id.
    ShowAchievement,
    /// An item, as stringified NBT.
    ShowItem,
    /// An entity, as stringified NBT.
    ShowEntity,
}

impl HoverAction {
    /// The name used in JSON.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            HoverAction::ShowText => "show_text",
            HoverAction::ShowAchievement => "show_achievement",
            HoverAction::ShowItem => "show_item",
            HoverAction::ShowEntity => "show_entity",
        }
    }

    /// Parse a JSON action name.
    #[must_use]
    pub fn from_canonical(name: &str) -> Option<Self> {
        [
            HoverAction::ShowText,
            HoverAction::ShowAchievement,
            HoverAction::ShowItem,
            HoverAction::ShowEntity,
        ]
        .into_iter()
        .find(|a| a.canonical_name() == name)
    }

    /// All four hover actions are permitted from servers.
    #[must_use]
    pub const fn allowed_in_chat(self) -> bool {
        true
    }
}

/// A hover action plus the component describing what to show.
#[derive(Debug, Clone)]
pub struct HoverEvent {
    /// What to show.
    pub action: HoverAction,
    /// The tooltip body. For `show_item`/`show_entity` its unformatted text is stringified NBT.
    pub value: Box<ChatComponent>,
}

/// A partially-specified style: every field that is `None` inherits from the parent.
// java: util/ChatStyle.java
#[derive(Debug, Clone, Default)]
pub struct ChatStyle {
    /// `§0`–`§f`, or `§r` for an explicit reset.
    pub color: Option<TextFormat>,
    /// `§l`
    pub bold: Option<bool>,
    /// `§o`
    pub italic: Option<bool>,
    /// `§n`
    pub underlined: Option<bool>,
    /// `§m`
    pub strikethrough: Option<bool>,
    /// `§k`
    pub obfuscated: Option<bool>,
    /// Runs when the text is clicked.
    pub click_event: Option<ClickEvent>,
    /// Shown when the text is hovered.
    pub hover_event: Option<HoverEvent>,
    /// Inserted into the chat box when the text is shift-clicked.
    pub insertion: Option<String>,
}

impl ChatStyle {
    /// Whether this style overrides nothing at all.
    ///
    /// Vanilla's `isEmpty` deliberately ignores `insertion`, so a style that sets *only* an
    /// insertion still serialises to `null`. Reproduced.
    // java: ChatStyle.java:177 (isEmpty)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.color.is_none()
            && self.bold.is_none()
            && self.italic.is_none()
            && self.underlined.is_none()
            && self.strikethrough.is_none()
            && self.obfuscated.is_none()
            && self.click_event.is_none()
            && self.hover_event.is_none()
    }

    /// Resolve against an inherited style.
    #[must_use]
    pub fn inherit(&self, parent: &ResolvedStyle) -> ResolvedStyle {
        ResolvedStyle {
            color: self.color.or(parent.color),
            bold: self.bold.unwrap_or(parent.bold),
            italic: self.italic.unwrap_or(parent.italic),
            underlined: self.underlined.unwrap_or(parent.underlined),
            strikethrough: self.strikethrough.unwrap_or(parent.strikethrough),
            obfuscated: self.obfuscated.unwrap_or(parent.obfuscated),
            click_event: self
                .click_event
                .clone()
                .or_else(|| parent.click_event.clone()),
            hover_event: self
                .hover_event
                .clone()
                .or_else(|| parent.hover_event.clone()),
            insertion: self.insertion.clone().or_else(|| parent.insertion.clone()),
        }
    }
}

/// A style with every property decided.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedStyle {
    /// The effective colour, or `None` for "whatever the caller passed to `drawString`".
    pub color: Option<TextFormat>,
    /// `§l`
    pub bold: bool,
    /// `§o`
    pub italic: bool,
    /// `§n`
    pub underlined: bool,
    /// `§m`
    pub strikethrough: bool,
    /// `§k`
    pub obfuscated: bool,
    /// Runs when the text is clicked.
    pub click_event: Option<ClickEvent>,
    /// Shown when the text is hovered.
    pub hover_event: Option<HoverEvent>,
    /// Inserted into chat on shift-click.
    pub insertion: Option<String>,
}

impl ResolvedStyle {
    /// The `§` prefix that reproduces this style in a flat string.
    ///
    /// Order is vanilla's, and it is not alphabetical: colour, bold, italic, underline,
    /// **obfuscated**, strikethrough.
    // java: ChatStyle.java:306 (getFormattingCode)
    #[must_use]
    pub fn formatting_code(&self) -> String {
        let mut s = String::new();
        if let Some(c) = self.color {
            s.push(SECTION_SIGN);
            s.push(c.code());
        }
        for (flag, code) in [
            (self.bold, TextFormat::Bold),
            (self.italic, TextFormat::Italic),
            (self.underlined, TextFormat::Underline),
            (self.obfuscated, TextFormat::Obfuscated),
            (self.strikethrough, TextFormat::Strikethrough),
        ] {
            if flag {
                s.push(SECTION_SIGN);
                s.push(code.code());
            }
        }
        s
    }
}

impl PartialEq for ChatStyle {
    fn eq(&self, other: &Self) -> bool {
        // ChatStyle.equals compares every effective getter, not the nullable override fields.
        // Stand-alone styles inherit vanilla's root style; component equality supplies the real
        // parent styles recursively below.
        let root = ResolvedStyle::default();
        self.inherit(&root) == other.inherit(&root)
    }
}

/// What kind of component this is.
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentKind {
    /// Literal text.
    Text(String),
    /// A translation key plus its arguments.
    Translate {
        /// The `.lang` key.
        key: String,
        /// Arguments substituted for `%s` / `%N$s`.
        with: Vec<ChatComponent>,
    },
    /// A scoreboard score. The client only ever renders the pre-resolved `value`.
    Score {
        /// Player name or selector.
        name: String,
        /// Objective name.
        objective: String,
        /// The value the server already resolved, if it sent one.
        value: Option<String>,
    },
    /// An entity selector. The client cannot resolve these, so it renders the raw selector.
    Selector(String),
}

/// One node of a chat component tree.
#[derive(Debug, Clone)]
pub struct ChatComponent {
    /// The node's own content.
    pub kind: ComponentKind,
    /// Style overrides applied to this node and inherited by its children and siblings.
    pub style: ChatStyle,
    /// `"extra"` — appended after this node, inheriting its style.
    pub extra: Vec<ChatComponent>,
}

/// One leaf of a laid-out component tree: a run of text with a fully decided style.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// The literal text of this run.
    pub text: String,
    /// The style in force for it.
    pub style: ResolvedStyle,
}

impl PartialEq for HoverEvent {
    fn eq(&self, other: &Self) -> bool {
        self.action == other.action && self.value == other.value
    }
}

impl PartialEq for ChatComponent {
    fn eq(&self, other: &Self) -> bool {
        let root = ResolvedStyle::default();
        self.eq_with_parents(other, &root, &root)
    }
}

impl ChatComponent {
    fn eq_with_parents(
        &self,
        other: &Self,
        self_parent: &ResolvedStyle,
        other_parent: &ResolvedStyle,
    ) -> bool {
        let self_style = self.style.inherit(self_parent);
        let other_style = other.style.inherit(other_parent);

        if self_style != other_style {
            return false;
        }

        let kinds_equal = match (&self.kind, &other.kind) {
            (ComponentKind::Text(left), ComponentKind::Text(right)) => left == right,
            (
                ComponentKind::Translate {
                    key: left_key,
                    with: left_args,
                },
                ComponentKind::Translate {
                    key: right_key,
                    with: right_args,
                },
            ) => {
                left_key == right_key
                    && left_args.len() == right_args.len()
                    && left_args
                        .iter()
                        .zip(right_args)
                        .all(|(left, right)| left.eq_with_parents(right, &self_style, &other_style))
            }
            (
                ComponentKind::Score {
                    name: left_name,
                    objective: left_objective,
                    // ChatComponentScore.equals deliberately ignores its mutable display
                    // value; Java compares only name/objective and the base style/siblings.
                    // The value is server-resolved presentation state, not component identity.
                    ..
                },
                ComponentKind::Score {
                    name: right_name,
                    objective: right_objective,
                    ..
                },
            ) => left_name == right_name && left_objective == right_objective,
            (ComponentKind::Selector(left), ComponentKind::Selector(right)) => left == right,
            _ => false,
        };

        kinds_equal
            && self.extra.len() == other.extra.len()
            && self
                .extra
                .iter()
                .zip(&other.extra)
                .all(|(left, right)| left.eq_with_parents(right, &self_style, &other_style))
    }

    /// A plain literal-text component.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: ComponentKind::Text(text.into()),
            style: ChatStyle::default(),
            extra: Vec::new(),
        }
    }

    /// A translation component.
    #[must_use]
    pub fn translate(key: impl Into<String>, with: Vec<ChatComponent>) -> Self {
        Self {
            kind: ComponentKind::Translate {
                key: key.into(),
                with,
            },
            style: ChatStyle::default(),
            extra: Vec::new(),
        }
    }

    /// Append a sibling, builder style.
    #[must_use]
    pub fn with_extra(mut self, sibling: ChatComponent) -> Self {
        self.extra.push(sibling);
        self
    }

    /// Replace the style, builder style.
    #[must_use]
    pub fn with_style(mut self, style: ChatStyle) -> Self {
        self.style = style;
        self
    }

    /// Flatten to text runs with resolved styles, in render order.
    ///
    /// `lang` supplies translations; a missing key renders as the key itself, which is what
    /// [`mc_assets::Language::tr`] already does.
    #[must_use]
    pub fn spans(&self, lang: &Language) -> Vec<Span> {
        let mut out = Vec::new();
        let mut work = MAX_FLATTEN_WORK;
        self.visit(&ResolvedStyle::default(), lang, 0, &mut out, &mut work);
        out
    }

    fn visit(
        &self,
        parent: &ResolvedStyle,
        lang: &Language,
        depth: usize,
        out: &mut Vec<Span>,
        work: &mut usize,
    ) {
        // Every node visited costs one unit; see MAX_FLATTEN_WORK.
        if depth > MAX_DEPTH || *work == 0 {
            return;
        }
        *work -= 1;
        let resolved = self.style.inherit(parent);
        match &self.kind {
            ComponentKind::Text(t) => {
                if charge_span(work, t.len()) {
                    out.push(Span {
                        text: t.clone(),
                        style: resolved.clone(),
                    });
                }
            }
            ComponentKind::Selector(s) => {
                if charge_span(work, s.len()) {
                    out.push(Span {
                        text: s.clone(),
                        style: resolved.clone(),
                    });
                }
            }
            ComponentKind::Score { value, .. } => {
                if charge_span(work, value.as_deref().map_or(0, str::len)) {
                    out.push(Span {
                        text: value.clone().unwrap_or_default(),
                        style: resolved.clone(),
                    });
                }
            }
            ComponentKind::Translate { key, with } => {
                let (selected, fallback) = (lang.tr(key), lang.tr_fallback(key));
                // Expanding reads both formats in full, so they are charged before it runs.
                let reading = selected.len() + fallback.len();
                if reading >= *work {
                    *work = 0;
                    return;
                }
                *work -= reading;
                // Vanilla yields the *expanded* children instead of the translation node
                // itself, so the node contributes no span of its own.
                for piece in expand_translation_with_fallback(selected, fallback, with) {
                    if *work == 0 {
                        return;
                    }
                    match piece {
                        Piece::Literal(t) => {
                            if charge_span(work, t.len()) {
                                out.push(Span {
                                    text: t,
                                    style: resolved.clone(),
                                });
                            }
                        }
                        Piece::Arg(i) => {
                            if let Some(arg) = with.get(i) {
                                arg.visit(&resolved, lang, depth + 1, out, work);
                            }
                        }
                    }
                }
            }
        }
        for sibling in &self.extra {
            if *work == 0 {
                return;
            }
            sibling.visit(&resolved, lang, depth + 1, out, work);
        }
    }

    /// The tree's text with every formatting code stripped.
    // java: ChatComponentStyle.java:72 (getUnformattedText)
    #[must_use]
    pub fn unformatted_text(&self, lang: &Language) -> String {
        self.spans(lang).into_iter().map(|s| s.text).collect()
    }

    /// The tree flattened to one string with `§` codes, each run terminated by `§r`.
    // java: ChatComponentStyle.java:87 (getFormattedText)
    #[must_use]
    pub fn formatted_text(&self, lang: &Language) -> String {
        let mut out = String::new();
        for span in self.spans(lang) {
            out.push_str(&span.style.formatting_code());
            out.push_str(&span.text);
            out.push_str(&TextFormat::Reset.control_string());
        }
        out
    }

    /// Parse a component from JSON.
    ///
    /// # Gson is always lenient, and this has to be too
    ///
    /// Vanilla parses with `GSON.fromJson(json, IChatComponent.class)`, and `Gson.fromJson`
    /// unconditionally does `reader.setLenient(true)` before reading. So a vanilla client
    /// accepts `//` and `/* */` comments, `'single quoted'` strings, unquoted names/values and
    /// alternate separators, all of which `serde_json` rejects. A rejection here is not cosmetic: it
    /// becomes `UiError::Component`, which the `S02PacketChat` path turns into
    /// `self.disconnect()` -- so a plugin whose message JSON has a comment in it would kick
    /// every mc-rs player on the server.
    ///
    /// Strict parsing is tried first and is the only path valid JSON ever takes, so the relaxer
    /// cannot affect well-formed input. Only input that would otherwise be fatal is retried
    /// through [`relax_json`], and if that fails too the *original* strict error is reported.
    ///
    /// # Errors
    /// [`UiError::Component`] for JSON that is malformed even under Gson's rules, or for an
    /// unrecognised node shape. Never for nesting depth -- see [`MAX_DEPTH`].
    pub fn from_json(json: &str) -> Result<Self, UiError> {
        Self::from_value(&Self::json_value(json)?, 0)
    }

    /// Parse a packet field whose consumer permits a null component.
    ///
    /// Gson returns null for a root JSON `null`, an empty document, or an empty component
    /// array (including nested empty arrays). For example, vanilla's sign renderer skips a
    /// null line. Keep that result distinct from a literal `"null"` text component, and from
    /// [`Self::from_json`], whose callers require a component. Nested null members still follow
    /// the component adapter's normal rules: `[null]` and `{"text":null}` remain errors.
    ///
    /// # Errors
    /// [`UiError::Component`] for malformed JSON or an unrecognised node shape. Accepting a
    /// nullable root does not suppress those errors or relax other packet fields.
    pub fn from_json_nullable(json: &str) -> Result<Option<Self>, UiError> {
        // Gson treats a document containing only whitespace/comments (or its initial BOM)
        // as empty. Reuse the lenient lexer's exact whitespace/comment rules rather than
        // Rust's broader Unicode `trim`, which would discard valid literal text.
        let root = json
            .strip_prefix('\u{feff}')
            .unwrap_or(json)
            .trim_start_matches([' ', '\t', '\r', '\n']);
        if root.is_empty() {
            return Ok(None);
        }
        if root.starts_with(['/', '#']) {
            let chars: Vec<char> = root.chars().collect();
            let mut cursor = 0;
            if skip_lenient_ignorable(&chars, &mut cursor).is_some() && cursor == chars.len() {
                return Ok(None);
            }
        }
        let value = match Self::json_value(json) {
            Ok(value) => value,
            // Only on the error path, so valid JSON is untouched.
            Err(_) if gson_reads_null_first(json) => return Ok(None),
            Err(error) => return Err(error),
        };
        if value.is_null() {
            Ok(None)
        } else {
            Self::from_value_nullable(&value, 0)
        }
    }

    fn json_value(json: &str) -> Result<Value, UiError> {
        // Gson strips a UTF-8 BOM before reading, lenient or not (`JsonReader.nextNonWhitespace`
        // consumes `\ufeff` at position 0).
        let json = json.strip_prefix('\u{feff}').unwrap_or(json);
        let value: Value = match serde_json::from_str(json) {
            Ok(value) => value,
            Err(strict) => {
                // Gson's `JsonReader` turns any `\uXXXX` into that `char`, so it accepts the
                // lone-surrogate escape `serde_json` rejects (failing here disconnects).
                let json = lone_surrogate_escapes_replaced(json);
                let relaxed = relax_json(&json);
                match serde_json::from_str(&relaxed) {
                    Ok(value) => value,
                    // Last resort: the document may simply be nested past `serde_json`'s own
                    // recursion limit. Truncating the deep part renders the message wrong;
                    // returning `Err` here disconnects the player. See `MAX_JSON_CONTAINERS`.
                    Err(_) => {
                        serde_json::from_str(&clamp_json_depth(&relaxed, MAX_JSON_CONTAINERS))
                            .map_err(|_| UiError::Component(strict.to_string()))?
                    }
                }
            }
        };
        Ok(value)
    }

    /// Parse a component from an already-decoded `serde_json::Value`.
    ///
    /// # Errors
    /// [`UiError::Component`] for an unrecognised node shape. **Never for nesting depth** --
    /// past [`MAX_DEPTH`] the subtree is silently dropped, because this error reaches
    /// `self.disconnect()`; see that constant's doc.
    // java: IChatComponent.java:61 (Serializer.deserialize)
    pub fn from_value(value: &Value, depth: usize) -> Result<Self, UiError> {
        if depth > MAX_DEPTH {
            return Ok(Self::text(String::new()));
        }
        match value {
            // A bare primitive is literal text. Vanilla accepts numbers and booleans here too,
            // via Gson's `getAsString`.
            Value::String(s) => Ok(Self::text(s.clone())),
            Value::Number(n) => Ok(Self::text(n.to_string())),
            Value::Bool(b) => Ok(Self::text(b.to_string())),
            Value::Array(_) => {
                // Gson's array adapter returns null for an empty array. A root component
                // cannot represent that null, so retain the existing error at this API
                // boundary; nullable call sites (translation arguments and hover values) use
                // `from_value_nullable` below and preserve the distinction.
                Self::from_value_nullable(value, depth)?
                    .ok_or_else(|| UiError::Component("empty component array".into()))
            }
            Value::Object(obj) => Self::from_object(obj, depth),
            other => Err(UiError::Component(format!(
                "don't know how to turn {other} into a component"
            ))),
        }
    }

    /// Deserialize a value at a call site where vanilla's component adapter may return null.
    ///
    /// `IChatComponent.Serializer.deserialize` returns null for an empty JSON array. Its array
    /// loop also ignores leading null elements until it sees a real component, while a null
    /// sibling after a real component reaches `appendSibling(null)` and fails. The public
    /// `from_value` API intentionally keeps its non-null component contract; this helper retains
    /// the nullable adapter result for translation arguments and hover payloads.
    fn from_value_nullable(value: &Value, depth: usize) -> Result<Option<Self>, UiError> {
        if depth > MAX_DEPTH {
            return Ok(Some(Self::text(String::new())));
        }
        let Value::Array(items) = value else {
            return Self::from_value(value, depth).map(Some);
        };

        let mut root = None;
        for item in items {
            let child = Self::from_value_nullable(item, depth + 1)?;
            match (&mut root, child) {
                (None, child) => root = child,
                (Some(_), None) => {
                    return Err(UiError::Component(
                        "null component cannot be appended as a sibling".into(),
                    ));
                }
                (Some(parent), Some(child)) => parent.extra.push(child),
            }
        }
        Ok(root)
    }

    fn from_object(obj: &Map<String, Value>, depth: usize) -> Result<Self, UiError> {
        let kind = if let Some(t) = obj.get("text") {
            // IChatComponent.Serializer calls JsonElement.getAsString directly here.  Gson
            // accepts every JsonPrimitive, but not nulls or containers; `to_string()` would
            // silently turn a hostile object into text instead of making the packet fail.
            ComponentKind::Text(gson_primitive_string(t, "text")?)
        } else if let Some(k) = obj.get("translate") {
            let mut with = Vec::new();
            if let Some(with_value) = obj.get("with") {
                let args = gson_array(with_value, "with")?;
                for a in args {
                    // `IChatComponent.Serializer.deserialize([])` returns null. Translation
                    // keeps that null format argument and later renders it as the string
                    // "null"; a text component is the equivalent representation here.
                    let mut argument = Self::from_value_nullable(a, depth + 1)?
                        .unwrap_or_else(|| Self::text("null"));
                    // IChatComponent.Serializer turns a text argument whose style is empty and
                    // which has no siblings into its raw String. ChatStyle.isEmpty deliberately
                    // ignores insertion, so an insertion-only argument loses that insertion at
                    // parse time in vanilla. Our representation keeps every argument as a
                    // ChatComponent; clearing this one field reproduces the observable result.
                    if matches!(&argument.kind, ComponentKind::Text(_))
                        && argument.style.is_empty()
                        && argument.extra.is_empty()
                    {
                        argument.style.insertion = None;
                    }
                    with.push(argument);
                }
            }
            ComponentKind::Translate {
                key: gson_primitive_string(k, "translate")?,
                with,
            }
        } else if let Some(score_value) = obj.get("score") {
            let score = gson_object(score_value, "score")?;
            let (Some(name), Some(objective)) = (score.get("name"), score.get("objective")) else {
                return Err(UiError::Component(
                    "a score component needs at least a name and an objective".into(),
                ));
            };
            ComponentKind::Score {
                // These three go through JsonUtils.getString, whose only type gate is
                // JsonPrimitive (numbers and booleans deliberately remain valid).
                name: gson_primitive_string(name, "name")?,
                objective: gson_primitive_string(objective, "objective")?,
                value: score
                    .get("value")
                    .map(|value| gson_primitive_string(value, "value"))
                    .transpose()?,
            }
        } else if let Some(s) = obj.get("selector") {
            ComponentKind::Selector(gson_primitive_string(s, "selector")?)
        } else {
            return Err(UiError::Component(
                "component has none of text/translate/score/selector".into(),
            ));
        };

        let mut extra = Vec::new();
        match obj.get("extra") {
            Some(Value::Array(items)) if items.is_empty() => {
                return Err(UiError::Component(
                    "unexpected empty array of components".into(),
                ));
            }
            Some(Value::Array(items)) => {
                for item in items {
                    extra.push(Self::from_value(item, depth + 1)?);
                }
            }
            Some(_) => return Err(UiError::Component("\"extra\" is not an array".into())),
            None => {}
        }

        Ok(Self {
            kind,
            style: style_from_object(obj, depth)?,
            extra,
        })
    }

    /// Serialise back to vanilla's JSON shape.
    ///
    /// A style-less, sibling-less text component collapses to a bare JSON string, exactly as
    /// vanilla's serialiser does, so `from_json`/`to_json` round-trips are stable.
    // java: IChatComponent.java:196 (Serializer.serialize)
    #[must_use]
    pub fn to_value(&self) -> Value {
        if let ComponentKind::Text(t) = &self.kind {
            if self.style.is_empty() && self.extra.is_empty() {
                return Value::String(t.clone());
            }
        }
        let mut obj = Map::new();
        // ChatStyle.Serializer is guarded by ChatStyle.isEmpty. Since that predicate ignores
        // insertion, an insertion-only style is intentionally omitted for every non-text node.
        if !self.style.is_empty() {
            style_into_object(&self.style, &mut obj);
        }
        if !self.extra.is_empty() {
            obj.insert(
                "extra".into(),
                Value::Array(self.extra.iter().map(ChatComponent::to_value).collect()),
            );
        }
        match &self.kind {
            ComponentKind::Text(t) => {
                obj.insert("text".into(), Value::String(t.clone()));
            }
            ComponentKind::Translate { key, with } => {
                obj.insert("translate".into(), Value::String(key.clone()));
                if !with.is_empty() {
                    obj.insert(
                        "with".into(),
                        Value::Array(with.iter().map(ChatComponent::to_value).collect()),
                    );
                }
            }
            ComponentKind::Score {
                name,
                objective,
                value,
            } => {
                let mut score = Map::new();
                score.insert("name".into(), Value::String(name.clone()));
                score.insert("objective".into(), Value::String(objective.clone()));
                score.insert(
                    "value".into(),
                    Value::String(value.clone().unwrap_or_default()),
                );
                obj.insert("score".into(), Value::Object(score));
            }
            ComponentKind::Selector(s) => {
                obj.insert("selector".into(), Value::String(s.clone()));
            }
        }
        Value::Object(obj)
    }

    /// Serialise to a JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        write_component_json(self, &mut out);
        out
    }
}

impl fmt::Display for ChatComponent {
    /// Renders with an empty language table, i.e. translation keys show as themselves.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lang = Language::new("en_US");
        f.write_str(&self.unformatted_text(&lang))
    }
}

/// Gson's `JsonPrimitive.getAsBoolean`: strings use `Boolean.parseBoolean`, and numeric
/// primitives therefore become an explicit `false` rather than an absent property.
fn gson_boolean(v: &Value, field: &str) -> Result<bool, UiError> {
    match v {
        Value::Bool(value) => Ok(*value),
        Value::String(value) => Ok(value.eq_ignore_ascii_case("true")),
        Value::Number(_) => Ok(false),
        _ => Err(expected_json_type(field, "a Boolean", v)),
    }
}

/// Gson's `JsonPrimitive.getAsString`, restricted to the three JSON primitive types on which
/// vanilla can call it successfully. This is also JsonUtils.getString's type gate.
fn gson_primitive_string(v: &Value, field: &str) -> Result<String, UiError> {
    match v {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        _ => Err(expected_json_type(field, "a string", v)),
    }
}

/// `JsonObject.getAsJsonObject` / `getAsJsonArray` are typed accessors, not optional casts.
/// A present wrong-shaped member throws while Gson is deserialising the component.
fn gson_object<'a>(v: &'a Value, field: &str) -> Result<&'a Map<String, Value>, UiError> {
    v.as_object()
        .ok_or_else(|| expected_json_type(field, "a JsonObject", v))
}

fn gson_array<'a>(v: &'a Value, field: &str) -> Result<&'a Vec<Value>, UiError> {
    v.as_array()
        .ok_or_else(|| expected_json_type(field, "a JsonArray", v))
}

fn expected_json_type(field: &str, expected: &str, actual: &Value) -> UiError {
    UiError::Component(format!(
        "expected {field} to be {expected}, was {}",
        json_type_name(actual)
    ))
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

// java: ChatStyle.java:486 (Serializer.deserialize)
fn style_from_object(obj: &Map<String, Value>, depth: usize) -> Result<ChatStyle, UiError> {
    let mut style = ChatStyle {
        bold: obj
            .get("bold")
            .map(|value| gson_boolean(value, "bold"))
            .transpose()?,
        italic: obj
            .get("italic")
            .map(|value| gson_boolean(value, "italic"))
            .transpose()?,
        underlined: obj
            .get("underlined")
            .map(|value| gson_boolean(value, "underlined"))
            .transpose()?,
        strikethrough: obj
            .get("strikethrough")
            .map(|value| gson_boolean(value, "strikethrough"))
            .transpose()?,
        obfuscated: obj
            .get("obfuscated")
            .map(|value| gson_boolean(value, "obfuscated"))
            .transpose()?,
        color: obj.get("color").map(gson_color).transpose()?.flatten(),
        insertion: obj
            .get("insertion")
            .map(|value| gson_primitive_string(value, "insertion"))
            .transpose()?,
        ..ChatStyle::default()
    };

    if let Some(click_value) = obj.get("clickEvent") {
        let click = gson_object(click_value, "clickEvent")?;
        let action = click
            .get("action")
            .map(|value| gson_primitive_string(value, "clickEvent.action"))
            .transpose()?
            .and_then(|name| ClickAction::from_canonical(&name));
        let value = click
            .get("value")
            .map(|value| gson_primitive_string(value, "clickEvent.value"))
            .transpose()?;
        if let (Some(action), Some(value)) = (action, value) {
            // Vanilla drops disallowed actions silently rather than erroring, which is the
            // safe thing to do with a hostile server: `open_file` never survives the wire.
            if action.allowed_in_chat() {
                style.click_event = Some(ClickEvent { action, value });
            }
        }
    }

    if let Some(hover_value) = obj.get("hoverEvent") {
        let hover = gson_object(hover_value, "hoverEvent")?;
        let action = hover
            .get("action")
            .map(|value| gson_primitive_string(value, "hoverEvent.action"))
            .transpose()?
            .and_then(|name| HoverAction::from_canonical(&name));

        // Gson's deserialisation context returns null for an absent, explicit JSON-null, or
        // empty-array hover value. It does, however, still attempt malformed non-null component
        // values even when the action itself is unknown.
        let parsed_value = match hover.get("value") {
            None | Some(Value::Null) => None,
            Some(raw) => ChatComponent::from_value_nullable(raw, depth + 1)?,
        };
        if let (Some(action), Some(value)) = (action, parsed_value) {
            if action.allowed_in_chat() {
                style.hover_event = Some(HoverEvent {
                    action,
                    value: Box::new(value),
                });
            }
        }
    }

    Ok(style)
}

/// `EnumTypeAdapterFactory` consumes either a JSON string or number with `nextString`, accepts
/// JSON null as no colour, and rejects every other shape before its exact enum lookup.
fn gson_color(v: &Value) -> Result<Option<TextFormat>, UiError> {
    let Some(name) = (match v {
        Value::Null => None,
        Value::String(name) => Some(name.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => return Err(expected_json_type("color", "a string", v)),
    }) else {
        return Ok(None);
    };
    // ChatStyle's Gson enum adapter does an exact map lookup. The more permissive
    // `TextFormat::from_name` belongs to command/NBT lookups, not component JSON.
    Ok(TextFormat::ALL
        .into_iter()
        .find(|format| format.friendly_name() == name))
}

/// Replace every JSON value nested deeper than `max_depth` containers with `""`, so the result
/// is valid JSON that `serde_json` will accept.
///
/// `""` and not `null`, because an empty string is a legal component (a text node with no text)
/// while `null` is the one JSON value `ChatComponent::from_value` rejects -- swapping one
/// disconnect for another would defeat the point.
///
/// Input is assumed to already be strict JSON (the relaxer runs first). A `"` inside a string
/// literal is tracked so a brace in someone's chat text cannot shift the depth count.
#[must_use]
pub fn clamp_json_depth(json: &str, max_depth: usize) -> String {
    let mut out = String::with_capacity(json.len());
    let mut depth = 0usize;
    let mut chars = json.chars().peekable();
    // How many containers deep inside a clamped subtree we currently are; `Some(0)` means the
    // next `}`/`]` closes the subtree we replaced.
    let mut skipping: Option<usize> = None;

    while let Some(c) = chars.next() {
        if let Some(inner) = skipping {
            match c {
                '"' => {
                    // Skip a whole string literal so its braces do not count.
                    while let Some(sc) = chars.next() {
                        if sc == '\\' {
                            chars.next();
                        } else if sc == '"' {
                            break;
                        }
                    }
                }
                '{' | '[' => skipping = Some(inner + 1),
                '}' | ']' => {
                    if inner == 0 {
                        skipping = None;
                        depth -= 1;
                    } else {
                        skipping = Some(inner - 1);
                    }
                }
                _ => {}
            }
            continue;
        }

        match c {
            '"' => {
                out.push(c);
                while let Some(sc) = chars.next() {
                    out.push(sc);
                    if sc == '\\' {
                        if let Some(esc) = chars.next() {
                            out.push(esc);
                        }
                    } else if sc == '"' {
                        break;
                    }
                }
            }
            '{' | '[' => {
                depth += 1;
                if depth > max_depth {
                    out.push_str("\"\"");
                    skipping = Some(0);
                } else {
                    out.push(c);
                }
            }
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Rewrite Gson 2.2.4's lenient JSON dialect into strict JSON.
///
/// java: `Gson.fromJson(String, Type)` -> `fromJson(JsonReader, Type)`, whose first two
/// statements are `boolean oldLenient = reader.isLenient(); reader.setLenient(true);`. Vanilla
/// therefore accepts all of this, and [`ChatComponent::from_json`] must not disconnect over it.
///
/// This follows `JsonReader.doPeek`: comments and the non-execute prefix are skipped, names and
/// values may be unquoted, `=` / `=>` may replace `:`, `;` may replace `,`, keywords are
/// case-insensitive, quoted strings may contain raw control characters and unknown escapes, and
/// missing array values become `null`. An object trailing separator remains an error; an array
/// trailing separator is a final `null`. Those two similar-looking cases are intentionally
/// different in Gson 2.2.4.
///
/// Only ever called on input strict parsing has already rejected, so it cannot damage
/// well-formed JSON.
#[must_use]
pub fn relax_json(json: &str) -> String {
    try_relax_json(json).unwrap_or_else(|| json.to_owned())
}

/// Rewrite each `\uXXXX` escape naming a surrogate without its partner as the code point
/// `mc_nbt` uses for a lone surrogate (`U+F0000 + unit`, written raw). Escaped pairs and every
/// other escape are copied unchanged.
///
/// java: Gson's `JsonReader.readEscapeCharacter` returns `(char) hex` with no surrogate check,
/// so `{"text":"\ud83d"}` (half an emoji, cut by a server-side length limit and written by a
/// JSON library that escapes non-ASCII) is an ordinary one-`char` string in 1.8.9, drawn as
/// nothing. The escape code point also draws as nothing here: the font walks UTF-16 units, and
/// both halves of a supplementary code point are zero-width surrogate glyphs.
fn lone_surrogate_escapes_replaced(json: &str) -> std::borrow::Cow<'_, str> {
    fn escaped_unit(chars: &[char], at: usize) -> Option<u16> {
        if chars.get(at) != Some(&'\\') || chars.get(at + 1) != Some(&'u') {
            return None;
        }
        let hex = chars.get(at + 2..at + 6)?;
        hex.iter().try_fold(0_u16, |unit, digit| {
            Some((unit << 4) | u16::try_from(digit.to_digit(16)?).ok()?)
        })
    }

    if !json.contains("\\u") {
        return std::borrow::Cow::Borrowed(json);
    }
    let chars: Vec<char> = json.chars().collect();
    let mut out = String::with_capacity(json.len());
    let mut changed = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let Some(unit) = escaped_unit(&chars, i) else {
            // `\\`, `\"` and friends: copy both characters so the second is never rescanned.
            out.extend(chars.get(i..(i + 2).min(chars.len())).unwrap_or_default());
            i += 2;
            continue;
        };
        let paired = (0xD800..=0xDBFF).contains(&unit)
            && matches!(escaped_unit(&chars, i + 6), Some(0xDC00..=0xDFFF));
        if paired {
            out.extend(&chars[i..i + 12]);
            i += 12;
        } else if (0xD800..=0xDFFF).contains(&unit) {
            let escape = u32::from(unit) + mc_nbt::modified_utf8::LONE_SURROGATE_ESCAPE;
            out.push(char::from_u32(escape).unwrap_or(char::REPLACEMENT_CHARACTER));
            changed = true;
            i += 6;
        } else {
            out.extend(&chars[i..i + 6]);
            i += 6;
        }
    }
    if changed {
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(json)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LenientState {
    RootValue,
    RootEnd,
    ObjectFirstKey,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectEnd,
    ArrayFirstValue,
    ArrayValue,
    ArrayEnd,
}

fn try_relax_json(json: &str) -> Option<String> {
    let bytes: Vec<char> = json.chars().collect();
    let mut out = String::with_capacity(json.len());
    let mut i = 0;
    let mut stack = vec![LenientState::RootValue];

    skip_lenient_ignorable(&bytes, &mut i)?;
    // JsonReader consumes this XSSI prefix only at the beginning of a lenient document.
    if bytes.get(i..i + 5) == Some(&[')', ']', '}', '\'', '\n']) {
        i += 5;
    }

    while let Some(state) = stack.last().copied() {
        skip_lenient_ignorable(&bytes, &mut i)?;
        match state {
            LenientState::RootValue
            | LenientState::ObjectValue
            | LenientState::ArrayFirstValue
            | LenientState::ArrayValue => {
                if state == LenientState::ArrayFirstValue && bytes.get(i) == Some(&']') {
                    out.push(']');
                    i += 1;
                    stack.pop();
                    continue;
                }
                if matches!(
                    state,
                    LenientState::ArrayFirstValue | LenientState::ArrayValue
                ) && matches!(bytes.get(i), Some(',' | ';'))
                {
                    out.push_str("null,");
                    i += 1;
                    *stack.last_mut()? = LenientState::ArrayValue;
                    continue;
                }
                if state == LenientState::ArrayValue && bytes.get(i) == Some(&']') {
                    out.push_str("null]");
                    i += 1;
                    stack.pop();
                    continue;
                }
                begin_lenient_value(&bytes, &mut i, &mut out, &mut stack)?;
            }
            LenientState::RootEnd => {
                if i != bytes.len() {
                    return None;
                }
                stack.pop();
            }
            LenientState::ObjectFirstKey | LenientState::ObjectKey => {
                if state == LenientState::ObjectFirstKey && bytes.get(i) == Some(&'}') {
                    out.push('}');
                    i += 1;
                    stack.pop();
                    continue;
                }
                let key = read_lenient_name(&bytes, &mut i)?;
                push_json_string(&mut out, &key);
                *stack.last_mut()? = LenientState::ObjectColon;
            }
            LenientState::ObjectColon => {
                match bytes.get(i)? {
                    ':' => i += 1,
                    '=' => {
                        i += 1;
                        if bytes.get(i) == Some(&'>') {
                            i += 1;
                        }
                    }
                    _ => return None,
                }
                out.push(':');
                *stack.last_mut()? = LenientState::ObjectValue;
            }
            LenientState::ObjectEnd => match bytes.get(i)? {
                '}' => {
                    out.push('}');
                    i += 1;
                    stack.pop();
                }
                ',' | ';' => {
                    out.push(',');
                    i += 1;
                    *stack.last_mut()? = LenientState::ObjectKey;
                }
                _ => return None,
            },
            LenientState::ArrayEnd => match bytes.get(i)? {
                ']' => {
                    out.push(']');
                    i += 1;
                    stack.pop();
                }
                ',' | ';' => {
                    out.push(',');
                    i += 1;
                    *stack.last_mut()? = LenientState::ArrayValue;
                }
                _ => return None,
            },
        }
    }

    (i == bytes.len()).then_some(out)
}

fn begin_lenient_value(
    bytes: &[char],
    i: &mut usize,
    out: &mut String,
    stack: &mut Vec<LenientState>,
) -> Option<()> {
    let parent = stack.last_mut()?;
    *parent = match *parent {
        LenientState::RootValue => LenientState::RootEnd,
        LenientState::ObjectValue => LenientState::ObjectEnd,
        LenientState::ArrayFirstValue | LenientState::ArrayValue => LenientState::ArrayEnd,
        _ => return None,
    };

    match *bytes.get(*i)? {
        '{' => {
            out.push('{');
            *i += 1;
            stack.push(LenientState::ObjectFirstKey);
        }
        '[' => {
            out.push('[');
            *i += 1;
            stack.push(LenientState::ArrayFirstValue);
        }
        quote @ ('"' | '\'') => write_lenient_string(bytes, i, quote, out)?,
        _ => {
            let value = read_lenient_literal(bytes, i)?;
            if value.eq_ignore_ascii_case("true") {
                out.push_str("true");
            } else if value.eq_ignore_ascii_case("false") {
                out.push_str("false");
            } else if value.eq_ignore_ascii_case("null") {
                out.push_str("null");
            } else {
                // JsonReader exposes every other unquoted token as a string. Quoting numeric
                // tokens too is safe for chat components: every primitive is consumed through
                // JsonPrimitive.getAsString (including style booleans).
                push_json_string(out, &value);
            }
        }
    }
    Some(())
}

fn read_lenient_name(bytes: &[char], i: &mut usize) -> Option<String> {
    match *bytes.get(*i)? {
        quote @ ('"' | '\'') => {
            let mut encoded = String::new();
            write_lenient_string(bytes, i, quote, &mut encoded)?;
            serde_json::from_str(&encoded).ok()
        }
        _ => read_lenient_literal(bytes, i),
    }
}

fn read_lenient_literal(bytes: &[char], i: &mut usize) -> Option<String> {
    let start = *i;
    while let Some(&c) = bytes.get(*i) {
        if matches!(
            c,
            ' ' | '\t'
                | '\r'
                | '\n'
                | '/'
                | '\\'
                | ';'
                | '#'
                | '='
                | '{'
                | '}'
                | '['
                | ']'
                | ':'
                | ','
        ) {
            break;
        }
        *i += 1;
    }
    (*i > start).then(|| bytes[start..*i].iter().collect())
}

fn write_lenient_string(
    bytes: &[char],
    i: &mut usize,
    quote: char,
    out: &mut String,
) -> Option<()> {
    *i += 1;
    out.push('"');
    loop {
        let c = *bytes.get(*i)?;
        *i += 1;
        if c == quote {
            out.push('"');
            return Some(());
        }
        if c != '\\' {
            push_json_char(out, c);
            continue;
        }

        let escaped = *bytes.get(*i)?;
        *i += 1;
        if escaped == 'u' {
            let digits = bytes.get(*i..*i + 4)?;
            if !digits.iter().all(|c| c.is_ascii_hexdigit()) {
                return None;
            }
            out.push_str("\\u");
            out.extend(digits);
            *i += 4;
        } else if matches!(escaped, 'b' | 'f' | 'n' | 'r' | 't') {
            out.push('\\');
            out.push(escaped);
        } else {
            // JsonReader.readEscapeCharacter returns unknown escapes as the escaped character.
            push_json_char(out, escaped);
        }
    }
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        push_json_char(out, c);
    }
    out.push('"');
}

fn push_json_char(out: &mut String, c: char) {
    match c {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\u{08}' => out.push_str("\\b"),
        '\u{0c}' => out.push_str("\\f"),
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        c if c <= '\u{1f}' => {
            use std::fmt::Write as _;
            write!(out, "\\u{:04x}", c as u32).expect("writing to String cannot fail");
        }
        _ => out.push(c),
    }
}

/// Parse a whole JSON document the way `Gson.fromJson` reads it:
/// strict JSON first, then Gson's lenient dialect (comments, unquoted names and values, single
/// quotes, raw control characters in strings, lone-surrogate escapes). For the other documents
/// vanilla reads with Gson, such as the server-list status response.
///
/// # Errors
/// [`UiError::Component`] for a document that is malformed even under Gson's rules.
pub fn gson_json_value(json: &str) -> Result<Value, UiError> {
    ChatComponent::json_value(json)
}

/// Whether Gson reads a null component from the *start* of `json`,
/// in which case it never looks at the rest.
///
/// java: `Gson.fromJson(String, Type)` checks that the document was fully consumed only when the
/// result is non-null (`assertFullConsumption`: `obj != null && reader.peek() != END_DOCUMENT`).
/// The component adapter returns null for a root `null` keyword (`TreeTypeAdapter`) and for an
/// array whose elements are all such arrays (`IChatComponent.Serializer.deserialize`), and
/// `fromJson` returns null for a document that is empty after the lenient non-execute prefix
/// `)]}'\n` and comments. So `null{"text":...`, `[]junk` and `)]}'\n//junk` decode to a null
/// component in vanilla (a chat handler then logs a NullPointerException), while a parser that
/// insists on one whole document refused them, which disconnected the player.
fn gson_reads_null_first(json: &str) -> bool {
    let chars: Vec<char> = json
        .strip_prefix('\u{feff}')
        .unwrap_or(json)
        .chars()
        .collect();
    let mut i = 0;
    if skip_lenient_ignorable(&chars, &mut i).is_none() {
        return false;
    }
    // java: `JsonReader.consumeNonExecutePrefix`, lenient only, at the document start.
    if chars[i..].starts_with(&[')', ']', '}', '\'', '\n']) {
        i += 5;
        if skip_lenient_ignorable(&chars, &mut i).is_none() {
            return false;
        }
    }
    let rest = &chars[i..];
    if rest.is_empty() {
        return true;
    }
    // java: `JsonReader.peekKeyword`: `null` in any letter case, then the end or a character
    // `isLiteral` rejects (in lenient mode `/ \ ; # =` as well as the structural ones).
    if rest.len() >= 4
        && rest[..4]
            .iter()
            .zip("null".chars())
            .all(|(&c, k)| c.to_ascii_lowercase() == k)
    {
        return rest.get(4).is_none_or(|&c| {
            matches!(
                c,
                '/' | '\\'
                    | ';'
                    | '#'
                    | '='
                    | '{'
                    | '}'
                    | '['
                    | ']'
                    | ':'
                    | ','
                    | ' '
                    | '\t'
                    | '\u{c}'
                    | '\r'
                    | '\n'
            )
        });
    }
    if rest[0] != '[' {
        return false;
    }
    // The first value is an array: find where it ends (outside strings and comments), parse just
    // that value, and ask the component adapter whether it is null.
    let mut depth = 0usize;
    let mut j = 0;
    while j < rest.len() {
        match rest[j] {
            '[' | '{' => depth += 1,
            ']' | '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let first: String = rest[..=j].iter().collect();
                    return ChatComponent::json_value(&first).is_ok_and(|value| {
                        value.is_array()
                            && matches!(ChatComponent::from_value_nullable(&value, 0), Ok(None))
                    });
                }
            }
            quote @ ('"' | '\'') => {
                j += 1;
                while j < rest.len() && rest[j] != quote {
                    j += if rest[j] == '\\' { 2 } else { 1 };
                }
            }
            '/' | '#' => {
                let mut k = j;
                if skip_lenient_ignorable(rest, &mut k).is_none() {
                    return false;
                }
                if k > j {
                    j = k;
                    continue;
                }
            }
            _ => {}
        }
        j += 1;
    }
    false
}

fn skip_lenient_ignorable(bytes: &[char], i: &mut usize) -> Option<()> {
    loop {
        while matches!(bytes.get(*i), Some(' ' | '\t' | '\r' | '\n')) {
            *i += 1;
        }
        match (bytes.get(*i), bytes.get(*i + 1)) {
            (Some('/'), Some('/')) => {
                *i += 2;
                while !matches!(bytes.get(*i), None | Some('\n' | '\r')) {
                    *i += 1;
                }
            }
            (Some('#'), _) => {
                *i += 1;
                while !matches!(bytes.get(*i), None | Some('\n' | '\r')) {
                    *i += 1;
                }
            }
            (Some('/'), Some('*')) => {
                *i += 2;
                while !matches!((bytes.get(*i), bytes.get(*i + 1)), (Some('*'), Some('/'))) {
                    if *i >= bytes.len() {
                        return None;
                    }
                    *i += 1;
                }
                *i += 2;
            }
            _ => return Some(()),
        }
    }
}

// java: ChatStyle.java:578 (Serializer.serialize)
fn style_into_object(style: &ChatStyle, obj: &mut Map<String, Value>) {
    for (key, flag) in [
        ("bold", style.bold),
        ("italic", style.italic),
        ("underlined", style.underlined),
        ("strikethrough", style.strikethrough),
        ("obfuscated", style.obfuscated),
    ] {
        if let Some(v) = flag {
            obj.insert(key.into(), Value::Bool(v));
        }
    }
    if let Some(c) = style.color {
        obj.insert("color".into(), Value::String(c.friendly_name().to_owned()));
    }
    if let Some(i) = &style.insertion {
        obj.insert("insertion".into(), Value::String(i.clone()));
    }
    if let Some(c) = &style.click_event {
        let mut m = Map::new();
        m.insert(
            "action".into(),
            Value::String(c.action.canonical_name().to_owned()),
        );
        m.insert("value".into(), Value::String(c.value.clone()));
        obj.insert("clickEvent".into(), Value::Object(m));
    }
    if let Some(h) = &style.hover_event {
        let mut m = Map::new();
        m.insert(
            "action".into(),
            Value::String(h.action.canonical_name().to_owned()),
        );
        m.insert("value".into(), h.value.to_value());
        obj.insert("hoverEvent".into(), Value::Object(m));
    }
}

/// Write component JSON in Gson's insertion order and with Gson's default HTML escaping.
///
/// `serde_json::Map` is key-sorted in this workspace, so serialising
/// [`ChatComponent::to_value`] cannot
/// preserve the order in which vanilla's adapters add properties. Order is observable on the
/// sign/book wire, even though it is semantically irrelevant after parsing.
fn write_component_json(component: &ChatComponent, out: &mut String) {
    if let ComponentKind::Text(text) = &component.kind {
        if component.style.is_empty() && component.extra.is_empty() {
            push_gson_string(out, text);
            return;
        }
    }

    out.push('{');
    let mut first = true;
    if !component.style.is_empty() {
        write_style_json(&component.style, out, &mut first);
    }
    if !component.extra.is_empty() {
        push_json_field(out, &mut first, "extra");
        out.push('[');
        for (index, child) in component.extra.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            write_component_json(child, out);
        }
        out.push(']');
    }

    match &component.kind {
        ComponentKind::Text(text) => {
            push_json_field(out, &mut first, "text");
            push_gson_string(out, text);
        }
        ComponentKind::Translate { key, with } => {
            push_json_field(out, &mut first, "translate");
            push_gson_string(out, key);
            if !with.is_empty() {
                push_json_field(out, &mut first, "with");
                out.push('[');
                for (index, argument) in with.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    write_component_json(argument, out);
                }
                out.push(']');
            }
        }
        ComponentKind::Score {
            name,
            objective,
            value,
        } => {
            push_json_field(out, &mut first, "score");
            out.push('{');
            let mut score_first = true;
            push_json_field(out, &mut score_first, "name");
            push_gson_string(out, name);
            push_json_field(out, &mut score_first, "objective");
            push_gson_string(out, objective);
            push_json_field(out, &mut score_first, "value");
            push_gson_string(out, value.as_deref().unwrap_or_default());
            out.push('}');
        }
        ComponentKind::Selector(selector) => {
            push_json_field(out, &mut first, "selector");
            push_gson_string(out, selector);
        }
    }
    out.push('}');
}

fn write_style_json(style: &ChatStyle, out: &mut String, first: &mut bool) {
    for (key, value) in [
        ("bold", style.bold),
        ("italic", style.italic),
        ("underlined", style.underlined),
        ("strikethrough", style.strikethrough),
        ("obfuscated", style.obfuscated),
    ] {
        if let Some(value) = value {
            push_json_field(out, first, key);
            out.push_str(if value { "true" } else { "false" });
        }
    }
    if let Some(color) = style.color {
        push_json_field(out, first, "color");
        push_gson_string(out, color.friendly_name());
    }
    if let Some(insertion) = &style.insertion {
        push_json_field(out, first, "insertion");
        push_gson_string(out, insertion);
    }
    if let Some(click) = &style.click_event {
        push_json_field(out, first, "clickEvent");
        out.push_str("{\"action\":");
        push_gson_string(out, click.action.canonical_name());
        out.push_str(",\"value\":");
        push_gson_string(out, &click.value);
        out.push('}');
    }
    if let Some(hover) = &style.hover_event {
        push_json_field(out, first, "hoverEvent");
        out.push_str("{\"action\":");
        push_gson_string(out, hover.action.canonical_name());
        out.push_str(",\"value\":");
        write_component_json(&hover.value, out);
        out.push('}');
    }
}

fn push_json_field(out: &mut String, first: &mut bool, name: &str) {
    if !*first {
        out.push(',');
    }
    *first = false;
    push_gson_string(out, name);
    out.push(':');
}

fn push_gson_string(out: &mut String, value: &str) {
    // serde_json provides the JSON/control-character escaping. Gson's default writer also
    // escapes five HTML-sensitive ASCII characters plus the two JavaScript line separators.
    let escaped = serde_json::to_string(value).expect("serialising a string is infallible");
    for ch in escaped.chars() {
        match ch {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '=' => out.push_str("\\u003d"),
            '\'' => out.push_str("\\u0027"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            _ => out.push(ch),
        }
    }
}

/// One piece of an expanded translation format string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Literal(String),
    Arg(usize),
}

/// Expand a `.lang` format string into literals and argument references.
///
/// Vanilla's pattern is `%(?:(\d+)\$)?([A-Za-z%]|$)`: `%%` is a literal percent, `%s` takes the
/// next argument, `%N$s` takes argument `N` (1-based), and any other conversion is a format
/// error. An out-of-range argument index is *skipped* rather than rendered, which is vanilla's
/// behaviour and keeps a short `"with"` array from being a crash.
///
/// The live component path retries the immutable built-in `en_US` value after an unsupported
/// conversion, matching `ChatComponentTranslation.ensureInitialized`. If both tables are
/// malformed, the selected run is emitted literally instead of reproducing vanilla's final
/// server-triggerable throw.
// java: ChatComponentTranslation.java:75 (initializeFromFormat)
fn expand_translation(format: &str, args: &[ChatComponent]) -> Vec<Piece> {
    expand_translation_impl(format, args, false).expect("lossy expansion cannot fail")
}

fn expand_translation_with_fallback(
    selected: &str,
    fallback: &str,
    args: &[ChatComponent],
) -> Vec<Piece> {
    try_expand_translation(selected, args)
        .or_else(|()| try_expand_translation(fallback, args))
        .unwrap_or_else(|()| expand_translation(selected, args))
}

fn try_expand_translation(format: &str, args: &[ChatComponent]) -> Result<Vec<Piece>, ()> {
    expand_translation_impl(format, args, true)
}

fn expand_translation_impl(
    format: &str,
    args: &[ChatComponent],
    strict: bool,
) -> Result<Vec<Piece>, ()> {
    let chars: Vec<char> = format.chars().collect();
    let mut out: Vec<Piece> = Vec::new();
    let mut literal = String::new();
    let mut implicit = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] != '%' {
            literal.push(chars[i]);
            i += 1;
            continue;
        }

        // Try to match `%(\d+\$)?[A-Za-z%]`.
        let mut j = i + 1;
        let digits_start = j;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        let explicit = if j > digits_start && j < chars.len() && chars[j] == '$' {
            let n: usize = chars[digits_start..j]
                .iter()
                .collect::<String>()
                .parse()
                .unwrap_or(0);
            j += 1;
            Some(n)
        } else {
            j = i + 1;
            None
        };

        let conversion = chars.get(j).copied();
        match conversion {
            Some('%') if explicit.is_none() => {
                if !literal.is_empty() {
                    out.push(Piece::Literal(std::mem::take(&mut literal)));
                }
                // Java constructs a new ChatComponentText for every `%%` match. Keeping the
                // percent as its own piece preserves the otherwise-observable reset boundary in
                // getFormattedText().
                out.push(Piece::Literal("%".into()));
                i = j + 1;
            }
            Some('s') => {
                let index = match explicit {
                    Some(n) => n.wrapping_sub(1),
                    None => {
                        let n = implicit;
                        implicit += 1;
                        n
                    }
                };
                if index < args.len() {
                    // An out-of-range argument is skipped before Java creates any child, so it
                    // must not split the surrounding literal into separate formatted runs.
                    if !literal.is_empty() {
                        out.push(Piece::Literal(std::mem::take(&mut literal)));
                    }
                    out.push(Piece::Arg(index));
                }
                i = j + 1;
            }
            _ => {
                if strict {
                    return Err(());
                }
                // Both selected and fallback formats were invalid. Keep the server-controlled
                // text visible without reproducing vanilla's client-crashing final throw.
                literal.push('%');
                i += 1;
            }
        }
    }

    if !literal.is_empty() {
        out.push(Piece::Literal(literal));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{Cursor, Read};

    use mc_assets::{ResourceLocation, ResourceManager, ResourcePack};

    struct TestLangPack {
        name: &'static str,
        files: BTreeMap<String, Vec<u8>>,
    }

    impl TestLangPack {
        fn boxed(name: &'static str, files: &[(&str, &str)]) -> Box<dyn ResourcePack> {
            Box::new(Self {
                name,
                files: files
                    .iter()
                    .map(|(path, text)| ((*path).to_owned(), text.as_bytes().to_vec()))
                    .collect(),
            })
        }
    }

    impl ResourcePack for TestLangPack {
        fn open(&self, location: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&location.to_string())
                .map(|bytes| Box::new(Cursor::new(bytes.clone())) as Box<dyn Read>)
        }

        fn contains(&self, location: &ResourceLocation) -> bool {
            self.files.contains_key(&location.to_string())
        }

        fn name(&self) -> &str {
            self.name
        }

        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }
    }

    fn lang() -> Language {
        let mut l = Language::new("en_US");
        l.insert("chat.type.text", "<%s> %s");
        l.insert("death.attack.explosion", "%1$s was blown up");
        l.insert("gui.done", "Done");
        l.insert("percent.test", "100%% sure");
        l
    }

    /// java: Gson's `readEscapeCharacter` makes a `char` of any `\uXXXX`, so a lone surrogate
    /// escape is ordinary text in 1.8.9 (drawn as nothing). `serde_json` rejects it, and a chat
    /// component that fails to parse fails its packet. The lone half takes `mc_nbt`'s spelling.
    #[test]
    fn a_lone_surrogate_escape_parses_the_way_gson_does() {
        let lone_high = r#"{"text":"a\ud83db"}"#;
        assert!(
            serde_json::from_str::<Value>(lone_high).is_err(),
            "negative control: serde_json alone rejects it"
        );
        let parsed = ChatComponent::from_json(lone_high).expect("Gson accepts it");
        assert_eq!(parsed.unformatted_text(&lang()), "a\u{FD83D}b");

        for (json, text) in [
            (r#"{"text":"\ude00 first"}"#, "\u{FDE00} first"),
            (r#"{"text":"pair 😀"}"#, "pair \u{1F600}"),
            (r#"{"text":"\ud83d😀"}"#, "\u{FD83D}\u{1F600}"),
            (r#"["", {"text":"x\udbff"}]"#, "x\u{FDBFF}"),
            // An escaped backslash before `u` is not an escape: nothing to rewrite.
            (r#"{"text":"\\ud83d \ud83d"}"#, "\\ud83d \u{FD83D}"),
        ] {
            let parsed = ChatComponent::from_json(json).expect(json);
            assert_eq!(parsed.unformatted_text(&lang()), text, "{json}");
        }
        assert!(
            ChatComponent::from_json(r#"{"text":"\ud8"}"#).is_err(),
            "still malformed"
        );
    }

    #[test]
    fn a_bare_string_is_a_text_component() {
        let c = ChatComponent::from_json("\"hello\"").expect("parse");
        assert_eq!(c, ChatComponent::text("hello"));
        assert_eq!(c.unformatted_text(&lang()), "hello");
        assert_eq!(
            c.to_json(),
            "\"hello\"",
            "and it collapses back to a string"
        );
    }

    #[test]
    fn styles_inherit_down_into_extras() {
        let json = r#"{"text":"a","bold":true,"color":"red","extra":[{"text":"b"},{"text":"c","bold":false}]}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        let spans = c.spans(&lang());
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].style.color, Some(TextFormat::Red));
        assert!(spans[1].style.bold, "b inherits bold");
        assert_eq!(
            spans[1].style.color,
            Some(TextFormat::Red),
            "and the colour"
        );
        assert!(!spans[2].style.bold, "c overrides it back off");
        assert_eq!(spans[2].style.color, Some(TextFormat::Red));
    }

    /// `ChatStyle.Serializer` delegates colour to `EnumTypeAdapterFactory`, whose bytecode calls
    /// `Map.get(JsonReader.nextString())`; it does not use EnumChatFormatting's lenient command
    /// lookup. This is deliberately case-sensitive and punctuation-sensitive.
    #[test]
    fn component_json_color_names_are_exact() {
        for canonical in TextFormat::ALL {
            let name = canonical.friendly_name();
            let json = format!(r#"{{"text":"x","color":"{name}"}}"#);
            let component = ChatComponent::from_json(&json).expect("canonical enum name");
            assert_eq!(
                component.style.color,
                Some(canonical),
                "the Gson enum adapter has an entry for {name:?}"
            );
        }

        for noncanonical in [
            "RED",
            "DARK_BLUE",
            "Dark_Blue",
            "dark blue",
            "dark-blue",
            "darkblue",
        ] {
            let json = format!(r#"{{"text":"x","color":"{noncanonical}"}}"#);
            let component = ChatComponent::from_json(&json).expect("noncanonical color is ignored");
            assert_eq!(
                component.style.color, None,
                "the Gson enum adapter has no entry for {noncanonical:?}"
            );
            assert_eq!(component.formatted_text(&lang()), "x§r");
        }
    }

    /// `JsonPrimitive.getAsBoolean` applies `Boolean.parseBoolean(getAsString())` to every
    /// non-boolean primitive. In particular, numbers are explicit false values and must block
    /// inheritance rather than disappearing into `None`.
    #[test]
    fn component_json_style_flags_use_gson_primitive_coercion() {
        let json = r#"{"text":"p","bold":true,"extra":[{"text":"x","bold":1,"italic":"TRUE","underlined":"false","strikethrough":0,"obfuscated":"true","insertion":7}]}"#;
        let component = ChatComponent::from_json(json).expect("coerced style");
        let child = &component.extra[0];
        assert_eq!(child.style.bold, Some(false), "numeric true is still false");
        assert_eq!(child.style.italic, Some(true));
        assert_eq!(child.style.underlined, Some(false));
        assert_eq!(child.style.strikethrough, Some(false));
        assert_eq!(child.style.obfuscated, Some(true));
        assert_eq!(child.style.insertion.as_deref(), Some("7"));

        let spans = component.spans(&lang());
        assert!(spans[0].style.bold);
        assert!(
            !spans[1].style.bold,
            "Some(false) must override the parent's bold style"
        );
    }

    #[test]
    fn formatted_text_emits_vanillas_code_order() {
        let json = r#"{"text":"x","color":"gold","bold":true,"obfuscated":true,"strikethrough":true,"italic":true,"underlined":true}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        // colour, bold, italic, underline, obfuscated, strikethrough — then the trailing reset.
        assert_eq!(c.formatted_text(&lang()), "§6§l§o§n§k§mx§r");
    }

    #[test]
    fn a_plain_component_still_gets_a_trailing_reset() {
        let c = ChatComponent::text("hi");
        assert_eq!(c.formatted_text(&lang()), "hi§r");
    }

    #[test]
    fn translation_substitutes_positional_and_implicit_arguments() {
        let c = ChatComponent::translate(
            "chat.type.text",
            vec![ChatComponent::text("Steve"), ChatComponent::text("hello")],
        );
        assert_eq!(c.unformatted_text(&lang()), "<Steve> hello");

        let c =
            ChatComponent::translate("death.attack.explosion", vec![ChatComponent::text("Alex")]);
        assert_eq!(c.unformatted_text(&lang()), "Alex was blown up");
    }

    #[test]
    fn one_component_retranslates_against_every_supplied_language() {
        let component =
            ChatComponent::translate("cache.probe", vec![ChatComponent::text("argument")]);
        let mut first = Language::new("first");
        first.insert("cache.probe", "old %s");
        let mut second = Language::new("second");
        second.insert("cache.probe", "%s new");

        assert_eq!(component.unformatted_text(&first), "old argument");
        assert_eq!(component.unformatted_text(&second), "argument new");
        assert_eq!(
            component.unformatted_text(&first),
            "old argument",
            "translation expansion is state-free and follows the current language"
        );
    }

    #[test]
    fn a_double_percent_is_a_literal_percent() {
        let c = ChatComponent::translate("percent.test", vec![]);
        assert_eq!(c.unformatted_text(&lang()), "100% sure");
    }

    #[test]
    fn a_missing_translation_key_renders_as_the_key() {
        let c = ChatComponent::translate("no.such.key", vec![]);
        assert_eq!(c.unformatted_text(&lang()), "no.such.key");
    }

    #[test]
    fn malformed_selected_translation_retries_the_immutable_builtin_english() {
        let mut manager = ResourceManager::new();
        manager.push_pack(TestLangPack::boxed(
            "jar",
            &[(
                "minecraft:lang/en_US.lang",
                "fallback.probe=Fallback %s\nselected.probe=Base %x\nboth.bad=Base %x",
            )],
        ));
        manager.push_pack(TestLangPack::boxed(
            "resource pack",
            &[(
                "minecraft:lang/en_US.lang",
                "fallback.probe=Broken %x\nselected.probe=Selected %s\nboth.bad=Top %x",
            )],
        ));
        let language = Language::load(&manager, &[mc_assets::DEFAULT_LANGUAGE]);
        let args = vec![ChatComponent::text("argument")];

        assert_eq!(
            ChatComponent::translate("fallback.probe", args.clone()).unformatted_text(&language),
            "Fallback argument",
            "an invalid selected conversion retries the built-in table"
        );
        assert_eq!(
            ChatComponent::translate("selected.probe", args.clone()).unformatted_text(&language),
            "Selected argument",
            "a valid selected translation wins without consulting fallback"
        );
        assert_eq!(
            ChatComponent::translate("both.bad", args).unformatted_text(&language),
            "Top %x",
            "a doubly-invalid server-controlled format remains visible instead of crashing"
        );
    }

    #[test]
    fn a_missing_argument_is_skipped_not_panicked_on() {
        let c = ChatComponent::translate("chat.type.text", vec![ChatComponent::text("only")]);
        assert_eq!(c.unformatted_text(&lang()), "<only> ");
    }

    #[test]
    fn translation_arguments_inherit_the_translations_style() {
        let json = r#"{"translate":"chat.type.text","color":"aqua","with":[{"text":"S","bold":true},"m"]}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        let spans = c.spans(&lang());
        let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["<", "S", "> ", "m"]);
        assert!(spans
            .iter()
            .all(|s| s.style.color == Some(TextFormat::Aqua)));
        assert!(spans[1].style.bold);
        assert!(!spans[0].style.bold);
    }

    #[test]
    fn a_json_array_is_a_component_with_siblings() {
        let c = ChatComponent::from_json(r#"["a","b","c"]"#).expect("parse");
        assert_eq!(c.unformatted_text(&lang()), "abc");
        assert_eq!(c.extra.len(), 2);
    }

    #[test]
    fn click_and_hover_events_round_trip() {
        let json = r#"{"text":"click","clickEvent":{"action":"run_command","value":"/help"},"hoverEvent":{"action":"show_text","value":"tip"}}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        let click = c.style.click_event.clone().expect("click");
        assert_eq!(click.action, ClickAction::RunCommand);
        assert_eq!(click.value, "/help");
        let hover = c.style.hover_event.clone().expect("hover");
        assert_eq!(hover.action, HoverAction::ShowText);
        assert_eq!(hover.value.unformatted_text(&lang()), "tip");
        let resolved_hover = c.spans(&lang())[0]
            .style
            .hover_event
            .clone()
            .expect("hover survives into the resolved span consumed by chat hit-testing");
        assert_eq!(resolved_hover, hover);

        let round = ChatComponent::from_json(&c.to_json()).expect("reparse");
        assert_eq!(round, c);
    }

    /// `ChatStyle.Serializer` obtains click values as a JsonPrimitive and then calls
    /// `getAsString`, so a natural unquoted page number is a live event in 1.8.9.
    #[test]
    fn click_event_values_use_gson_primitive_coercion() {
        let numeric = ChatComponent::from_json(
            r#"{"text":"next","clickEvent":{"action":"change_page","value":7}}"#,
        )
        .expect("numeric click value");
        let click = numeric.style.click_event.expect("live change-page event");
        assert_eq!(click.action, ClickAction::ChangePage);
        assert_eq!(click.value, "7");

        let boolean = ChatComponent::from_json(
            r#"{"text":"run","clickEvent":{"action":"run_command","value":false}}"#,
        )
        .expect("boolean click value");
        assert_eq!(
            boolean.style.click_event.expect("live run event").value,
            "false"
        );
    }

    /// The component serializer uses Gson's typed accessors for every nested member.  In
    /// particular, a present object is not a stringified `"{}"`, and a present `with` member
    /// cannot be quietly ignored just because it is not an array.  These errors propagate out
    /// of packet parsing in Java, so accepting any of them changes the visible disconnect path.
    #[test]
    fn typed_json_accessors_reject_present_wrong_shaped_component_members() {
        for json in [
            r#"{"text":{}}"#,
            r#"{"text":null}"#,
            r#"{"translate":[]}"#,
            r#"{"translate":"key","with":{}}"#,
            r#"{"score":[],"selector":"@p"}"#,
            r#"{"score":{"name":{},"objective":"kills"}}"#,
            r#"{"score":{"name":"Steve","objective":null}}"#,
            r#"{"selector":[]}"#,
            r#"{"text":"x","bold":{}}"#,
            r#"{"text":"x","color":true}"#,
            r#"{"text":"x","insertion":[]}"#,
            r#"{"text":"x","clickEvent":[]}"#,
            r#"{"text":"x","clickEvent":{"action":{},"value":"/x"}}"#,
            r#"{"text":"x","clickEvent":{"action":"run_command","value":[]}}"#,
            r#"{"text":"x","hoverEvent":[]}"#,
            r#"{"text":"x","hoverEvent":{"action":{},"value":"tip"}}"#,
            r#"{"text":"x","hoverEvent":{"action":"unknown","value":{}}}"#,
        ] {
            assert!(ChatComponent::from_json(json).is_err(), "accepted {json}");
        }
    }

    /// Gson's component adapter uses `null` as the result for an empty array. The root API
    /// remains non-null (and therefore rejects `[]`), but translation arguments and hover values
    /// have nullable adapter contexts in vanilla and must not turn that result into a disconnect.
    #[test]
    fn empty_component_arrays_keep_gsons_nullable_context_semantics() {
        let mut language = lang();
        language.insert("null.argument", "%s");

        let translation = ChatComponent::from_json(r#"{"translate":"null.argument","with":[[]]}"#)
            .expect("an empty-array translation argument is a nullable Java format argument");
        assert_eq!(translation.unformatted_text(&language), "null");
        assert_eq!(
            translation.to_json(),
            r#"{"translate":"null.argument","with":["null"]}"#
        );

        let hover = ChatComponent::from_json(
            r#"{"text":"x","hoverEvent":{"action":"show_text","value":[]}}"#,
        )
        .expect("an empty-array hover value is discarded by ChatStyle.Serializer");
        assert!(hover.style.hover_event.is_none());
        assert_eq!(hover.unformatted_text(&language), "x");

        // The same null result can be nested in a one-element array. A leading null is ignored
        // by Gson's array loop until a real component arrives; a trailing null would call
        // appendSibling(null) and remains an error.
        assert_eq!(
            ChatComponent::from_json(r#"[[],"x"]"#)
                .expect("leading null array entry")
                .unformatted_text(&language),
            "x"
        );
        assert!(ChatComponent::from_json(r#"["x",[]]"#).is_err());
        assert!(ChatComponent::from_json("[]").is_err());
    }

    #[test]
    fn nullable_packet_components_preserve_vanillas_absent_root() {
        // Oracle: the installed 1.8.9 client serializer (`eu$a.a(String)`) with Gson 2.2.4.
        // The non-null API must retain its existing contract for other packet consumers.
        for json in [
            "null",
            " \t null\r\n",
            "NULL",
            "[]",
            "[[]]",
            "[[[]]]",
            "",
            " \t\r\n",
            "\u{feff}",
            "\u{feff}null",
            "// comment only",
            "/* comment only */",
            "# comment only",
        ] {
            assert!(
                ChatComponent::from_json_nullable(json)
                    .expect("Gson returns a null root")
                    .is_none(),
                "input: {json:?}"
            );
            assert!(ChatComponent::from_json(json).is_err(), "input: {json:?}");
        }
    }

    #[test]
    fn nullable_packet_components_do_not_hide_malformed_nested_values() {
        for json in [
            "{",
            "{}",
            "/* unterminated",
            r#"{"text":null}"#,
            r#"{"text":"x","extra":[]}"#,
            "[null]",
            r#"[null,"x"]"#,
            r#"["x",[]]"#,
            r#"{"translate":"key","with":[null]}"#,
        ] {
            assert!(
                ChatComponent::from_json_nullable(json).is_err(),
                "nullable root accepted malformed input: {json:?}"
            );
        }
        assert_eq!(
            ChatComponent::from_json_nullable(r#"[[],"x"]"#)
                .expect("leading empty component array")
                .expect("text component")
                .unformatted_text(&lang()),
            "x"
        );
    }

    #[test]
    fn nullable_packet_components_keep_real_sign_text_and_literal_null() {
        // A real server's S33 sign ended with JSON null, after these three text rows.
        for (json, expected) in [
            (r#""Have fun""#, "Have fun"),
            (r#""at""#, "at"),
            (r#""play.test  !""#, "play.test  !"),
            (r#""null""#, "null"),
            (r#"{"text":""}"#, ""),
        ] {
            let component = ChatComponent::from_json_nullable(json)
                .expect("valid text")
                .expect("a text node is not absent");
            assert_eq!(component.unformatted_text(&lang()), expected);
            assert_eq!(component, ChatComponent::from_json(json).unwrap());
        }
        assert!(ChatComponent::from_json_nullable("null").unwrap().is_none());
    }

    #[test]
    fn jsonutils_string_members_still_coerce_all_json_primitives() {
        let c = ChatComponent::from_json(
            r#"{"text":true,"score":{"name":7,"objective":false,"value":2}}"#,
        )
        .expect("Gson JsonPrimitive.getAsString accepts booleans and numbers");
        assert_eq!(c.unformatted_text(&lang()), "true");

        let score = ChatComponent::from_json(r#"{"score":{"name":7,"objective":false,"value":2}}"#)
            .expect("JsonUtils.getString has the same primitive gate");
        assert_eq!(score.unformatted_text(&lang()), "2");
    }

    /// A server must not be able to make the client open a local file.
    #[test]
    fn disallowed_click_actions_are_dropped() {
        let json = r#"{"text":"x","clickEvent":{"action":"open_file","value":"C:/secret"}}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        assert!(c.style.click_event.is_none());
        assert!(!ClickAction::OpenFile.allowed_in_chat());
        assert!(!ClickAction::TwitchUserInfo.allowed_in_chat());
    }

    /// The screenshot line's `[COPY]` action is client-only twice
    /// over. No JSON name parses to it, and it is not allowed in chat either, so a server can
    /// never make the client read a local file onto the clipboard.
    #[test]
    fn the_copy_image_action_never_comes_from_json() {
        assert!(!ClickAction::CopyImage.allowed_in_chat());
        assert_eq!(ClickAction::from_canonical("copy_image"), None);
        let json = r#"{"text":"x","clickEvent":{"action":"copy_image","value":"C:/secret.png"}}"#;
        let c = ChatComponent::from_json(json).expect("parse");
        assert!(c.style.click_event.is_none());
    }

    /// `copy_to_clipboard` was added after 1.8.9. It must stay unknown to the chat-component
    /// parser even though the app-level `screens::Action::CopyToClipboard` is live for login UI.
    /// The valid action control proves this exercises the same click-event decoding path rather
    /// than merely rejecting every click event.
    #[test]
    fn modern_copy_to_clipboard_is_not_a_vanilla_click_action() {
        let modern = ChatComponent::from_json(
            r#"{"text":"copy","clickEvent":{"action":"copy_to_clipboard","value":"secret"}}"#,
        )
        .expect("unknown click actions are ignored by vanilla's ChatStyle serializer");
        assert!(modern.style.click_event.is_none());
        assert!(ClickAction::from_canonical("copy_to_clipboard").is_none());

        let vanilla = ChatComponent::from_json(
            r#"{"text":"run","clickEvent":{"action":"run_command","value":"/help"}}"#,
        )
        .expect("a real 1.8.9 click action still decodes");
        assert_eq!(
            vanilla.style.click_event.as_ref().map(|event| event.action),
            Some(ClickAction::RunCommand)
        );
    }

    #[test]
    fn click_events_hit_test_by_span() {
        let json = r#"[{"text":"aa","clickEvent":{"action":"run_command","value":"/a"}},{"text":"bb","clickEvent":{"action":"run_command","value":"/b"}}]"#;
        let c = ChatComponent::from_json(json).expect("parse");
        let spans = c.spans(&lang());
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].style.click_event.as_ref().unwrap().value, "/a");
        assert_eq!(spans[1].style.click_event.as_ref().unwrap().value, "/b");
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(ChatComponent::from_json("{").is_err());
        assert!(ChatComponent::from_json("{}").is_err(), "no recognised key");
        assert!(ChatComponent::from_json("[]").is_err(), "empty array");
        assert!(
            ChatComponent::from_json(r#"{"text":"a","extra":[]}"#).is_err(),
            "vanilla rejects an empty extra array"
        );
        assert!(
            ChatComponent::from_json(r#"{"score":{"name":"n"}}"#).is_err(),
            "a score needs an objective"
        );
        assert!(ChatComponent::from_json("null").is_err());
    }

    /// Build `{"text":"a","extra":[ … ]}` nested `levels` deep with `"x"` at the bottom.
    fn nested(levels: usize) -> String {
        let mut json = String::new();
        for _ in 0..levels {
            json.push_str(r#"{"text":"a","extra":["#);
        }
        json.push_str(r#""x""#);
        for _ in 0..levels {
            json.push_str("]}");
        }
        json
    }

    /// **The disconnect vector.** `UiError::Component` propagates to `self.disconnect()`, so
    /// every depth here used to be one `S02PacketChat` that kicks the player. This test replaces
    /// `deep_nesting_is_bounded`, which asserted exactly that (`from_json(500-deep).unwrap_err()`)
    /// and so could never fail while the behaviour was wrong.
    #[test]
    fn no_nesting_depth_can_produce_an_error() {
        for levels in [1usize, 10, 40, 60, 64, 65, 100, 500, 1000, 5000] {
            let parsed = ChatComponent::from_json(&nested(levels));
            assert!(
                parsed.is_ok(),
                "depth {levels} must not disconnect the player"
            );
        }
    }

    /// `from_value` is `pub` and takes `depth` as a parameter, so a host can reach the
    /// [`MAX_DEPTH`] guard without going through `from_json` (whose own ceiling is the lower
    /// [`MAX_JSON_CONTAINERS`]). Past the limit it must **truncate**, not `Err`: the error
    /// reaches `self.disconnect()`.
    #[test]
    fn from_value_truncates_past_max_depth_instead_of_erroring() {
        let value = serde_json::json!({"text": "deep"});
        let ok = ChatComponent::from_value(&value, MAX_DEPTH).expect("at the limit");
        assert_eq!(ok.unformatted_text(&lang()), "deep");
        let truncated = ChatComponent::from_value(&value, MAX_DEPTH + 1)
            .expect("past the limit must not be an error");
        assert_eq!(truncated.unformatted_text(&lang()), "");
    }

    /// An array-nested component costs **one** JSON container per level, not the two that
    /// `{"extra":[…]}` costs, so `serde_json` is happy well past where the old `MAX_DEPTH = 64`
    /// used to return `Err` -- which is the band where raising the constant is what does the
    /// work, rather than the clamp.
    #[test]
    fn an_array_nested_component_past_the_old_limit_is_not_an_error() {
        let json = format!("{}{}{}", "[".repeat(80), r#""x""#, "]".repeat(80));
        let c = ChatComponent::from_json(&json).expect("80 nested arrays");
        assert_eq!(c.unformatted_text(&lang()), "x");
    }

    /// Below the clamp the tree is rendered in full, so the fallback cannot be hiding a parser
    /// that silently drops everything.
    #[test]
    fn a_moderately_deep_tree_still_renders_every_level() {
        let c = ChatComponent::from_json(&nested(40)).expect("40 deep");
        let text = c.unformatted_text(&lang());
        assert_eq!(text, format!("{}x", "a".repeat(40)));
    }

    /// Past the clamp the deep tail is dropped rather than fatal: the levels that fit still
    /// render, and the innermost `"x"` is gone.
    #[test]
    fn a_tree_past_the_clamp_renders_its_shallow_levels_and_drops_the_rest() {
        let c = ChatComponent::from_json(&nested(1000)).expect("1000 deep");
        let text = c.unformatted_text(&lang());
        assert!(!text.is_empty(), "the shallow levels still render");
        assert!(
            text.len() < 1000,
            "and the deep tail is truncated, not rendered: {}",
            text.len()
        );
        assert!(text.chars().all(|ch| ch == 'a'), "the innermost x is gone");
    }

    /// java: `Gson.fromJson` sets `reader.setLenient(true)` unconditionally, so a vanilla client
    /// renders all of these. Each one used to be a disconnect.
    #[test]
    fn gsons_lenient_dialect_does_not_disconnect_the_player() {
        for (label, json) in [
            ("line comment", "{\"text\":\"hi\"} // trailing note"),
            ("hash comment", "# note\n{\"text\":\"hi\"}"),
            ("block comment", "{/* why */\"text\":\"hi\"}"),
            ("single-quoted value", "{\"text\":'hi'}"),
            ("single-quoted key", "{'text':\"hi\"}"),
            ("bare key", "{text:\"hi\"}"),
            ("bare value", "{text:hi}"),
            ("equals separator", "{text=>hi}"),
            ("semicolon separator", "{text:ignored;text:hi}"),
            ("non-execute prefix", ")]}'\n{text:hi}"),
            ("utf-8 BOM", "\u{feff}{\"text\":\"hi\"}"),
        ] {
            let c = ChatComponent::from_json(json)
                .unwrap_or_else(|e| panic!("{label}: {json:?} -> {e}"));
            assert_eq!(c.unformatted_text(&lang()), "hi", "{label}");
        }
    }

    /// Gson's lenient reader accepts raw newlines and treats an unknown escape as the escaped
    /// character. A pass-through quote copier leaves both forms invalid for `serde_json`.
    #[test]
    fn lenient_quoted_strings_follow_json_reader_escape_rules() {
        let c = ChatComponent::from_json("{text:'h\\qi\nthere'}").expect("lenient string");
        assert_eq!(c.unformatted_text(&lang()), "hqi\nthere");
    }

    /// `JsonReader.peekKeyword` is ASCII-case-insensitive. This matters for style flags because
    /// a boolean primitive takes precedence over inherited style in `ChatStyle.Serializer`.
    #[test]
    fn lenient_keywords_are_case_insensitive() {
        let c = ChatComponent::from_json("{text:hi;bold:TRUE}").expect("lenient keywords");
        assert_eq!(c.style.bold, Some(true));
    }

    /// In Gson 2.2.4 array separators stand in for missing `null` elements. Object separators do
    /// not: a separator immediately before `}` is a syntax error. The old comma trimmer reversed
    /// both behaviours.
    #[test]
    fn lenient_trailing_separator_semantics_match_gson_2_2_4() {
        assert_eq!(relax_json("[x,,y,]"), r#"["x",null,"y",null]"#);
        assert!(ChatComponent::from_json("{text:hi,}").is_err());
        assert!(ChatComponent::from_json("{text:hi;}").is_err());
    }

    #[test]
    fn unterminated_lenient_tokens_remain_errors() {
        assert!(ChatComponent::from_json("{text:'hi}").is_err());
        assert!(ChatComponent::from_json("{text:hi/* no end}").is_err());
    }

    /// The relaxer only ever runs on input strict parsing rejected, so it cannot touch a
    /// well-formed document -- including one whose *text* contains the characters it rewrites.
    #[test]
    fn valid_json_never_reaches_the_relaxer() {
        let json = r#"{"text":"a // b /* c */ 'd' {e:1},"}"#;
        let c = ChatComponent::from_json(json).expect("valid");
        assert_eq!(c.unformatted_text(&lang()), "a // b /* c */ 'd' {e:1},");
    }

    /// Something malformed under Gson's rules too is still an error -- this port does not
    /// swap one divergence for another by making `from_json` infallible.
    #[test]
    fn genuinely_malformed_json_is_still_an_error() {
        assert!(ChatComponent::from_json("{\"text\":").is_err());
        assert!(ChatComponent::from_json("not json at all {{{").is_err());
    }

    #[test]
    fn numbers_and_booleans_are_accepted_as_text() {
        assert_eq!(
            ChatComponent::from_json("42")
                .expect("n")
                .unformatted_text(&lang()),
            "42"
        );
        assert_eq!(
            ChatComponent::from_json("true")
                .expect("b")
                .unformatted_text(&lang()),
            "true"
        );
    }

    #[test]
    fn score_and_selector_components_render_their_payload() {
        let c = ChatComponent::from_json(
            r#"{"score":{"name":"Steve","objective":"kills","value":"7"}}"#,
        )
        .expect("parse");
        assert_eq!(c.unformatted_text(&lang()), "7");
        let c = ChatComponent::from_json(r#"{"selector":"@p"}"#).expect("parse");
        assert_eq!(c.unformatted_text(&lang()), "@p");
    }

    #[test]
    fn an_empty_style_serialises_away() {
        let c = ChatComponent::text("plain");
        assert_eq!(c.to_json(), "\"plain\"");
        let styled = ChatComponent::text("x").with_style(ChatStyle {
            bold: Some(true),
            ..ChatStyle::default()
        });
        assert_eq!(styled.to_json(), r#"{"bold":true,"text":"x"}"#);
    }

    #[test]
    fn expand_translation_handles_the_awkward_cases() {
        let args = vec![ChatComponent::text("A"), ChatComponent::text("B")];
        assert_eq!(
            expand_translation("%s and %s", &args),
            vec![Piece::Arg(0), Piece::Literal(" and ".into()), Piece::Arg(1)]
        );
        assert_eq!(
            expand_translation("%2$s then %1$s", &args),
            vec![
                Piece::Arg(1),
                Piece::Literal(" then ".into()),
                Piece::Arg(0)
            ]
        );
        // Explicit indices do not advance the implicit counter, matching java.util.Formatter.
        assert_eq!(
            expand_translation("%2$s%s", &args),
            vec![Piece::Arg(1), Piece::Arg(0)]
        );
        assert_eq!(
            expand_translation("100%%", &args),
            vec![Piece::Literal("100".into()), Piece::Literal("%".into())]
        );
        assert_eq!(
            expand_translation("a%9$sb", &args),
            vec![Piece::Literal("ab".into())],
            "a skipped argument creates no child boundary"
        );
        assert_eq!(
            expand_translation("%d unsupported", &args),
            vec![Piece::Literal("%d unsupported".into())],
            "kept verbatim instead of throwing"
        );
        assert_eq!(
            expand_translation("trailing %", &args),
            vec![Piece::Literal("trailing %".into())]
        );
        assert_eq!(expand_translation("", &args), vec![]);
    }

    /// `ChatComponentStyle.getFormattedText` terminates every child with reset, making the exact
    /// child boundaries observable even though the inserted resets render as a no-op.
    #[test]
    fn translation_formatted_text_preserves_java_child_boundaries() {
        let mut language = lang();
        language.insert("skip.test", "a%9$sb");

        assert_eq!(
            ChatComponent::translate("percent.test", vec![]).formatted_text(&language),
            format!(
                "100{}%{} sure{}",
                TextFormat::Reset.control_string(),
                TextFormat::Reset.control_string(),
                TextFormat::Reset.control_string()
            )
        );
        assert_eq!(
            ChatComponent::translate("skip.test", vec![]).formatted_text(&language),
            format!("ab{}", TextFormat::Reset.control_string())
        );
    }

    /// Vanilla throws `ChatComponentTranslationFormatException` (and for bad indices sometimes
    /// `ArrayIndexOutOfBoundsException` / `NumberFormatException`) for these server-controlled
    /// translation keys. Keeping the malformed run literal or safely skipping an impossible
    /// index is an intentional hardening: a chat packet must not be able to crash the client.
    #[test]
    fn hostile_translation_formats_are_safe_literal_fallbacks() {
        let language = Language::new("empty");
        let args = vec![ChatComponent::text("A"), ChatComponent::text("B")];

        for format in [
            "%d", "%f", "%x", "%S", "%1$%", "%1$1$s", "%1s", "%$s", "%1$", "tail %", "% x", "%5",
            "%-", "%\n", "%\t",
        ] {
            assert_eq!(
                ChatComponent::translate(format, args.clone()).unformatted_text(&language),
                format,
                "unsupported format stays visible instead of crashing"
            );
        }

        for impossible_index in ["%0$s", "%00$s", "%99999999999999$s", "%2147483648$s"] {
            assert_eq!(
                ChatComponent::translate(impossible_index, args.clone())
                    .unformatted_text(&language),
                "",
                "an impossible index is safely skipped"
            );
        }
    }

    #[test]
    fn resolved_style_default_has_no_codes() {
        assert_eq!(ResolvedStyle::default().formatting_code(), "");
    }

    #[test]
    fn component_equality_compares_effective_styles_like_java() {
        let implicit_defaults = ChatComponent::text("same");
        let explicit_defaults = ChatComponent::text("same").with_style(ChatStyle {
            bold: Some(false),
            italic: Some(false),
            ..ChatStyle::default()
        });
        assert_eq!(
            implicit_defaults, explicit_defaults,
            "ChatStyle.equals calls the effective boolean getters"
        );

        let inherited_child = ChatComponent::text("parent")
            .with_style(ChatStyle {
                bold: Some(true),
                ..ChatStyle::default()
            })
            .with_extra(ChatComponent::text("child"));
        let explicit_child = ChatComponent::text("parent")
            .with_style(ChatStyle {
                bold: Some(true),
                ..ChatStyle::default()
            })
            .with_extra(ChatComponent::text("child").with_style(ChatStyle {
                bold: Some(true),
                ..ChatStyle::default()
            }));
        assert_eq!(
            inherited_child, explicit_child,
            "sibling styles are compared after inheriting their owning component's style"
        );

        let overridden_child = ChatComponent::text("parent")
            .with_style(ChatStyle {
                bold: Some(true),
                ..ChatStyle::default()
            })
            .with_extra(ChatComponent::text("child").with_style(ChatStyle {
                bold: Some(false),
                ..ChatStyle::default()
            }));
        assert_ne!(inherited_child, overridden_child);
    }

    /// `ChatComponentScore.equals` compares only the score identity and base component state;
    /// its mutable, server-resolved display value is intentionally omitted.
    /// java: `ChatComponentScore.java:84-98`
    #[test]
    fn score_equality_ignores_the_server_resolved_display_value_like_java() {
        let left = ChatComponent {
            kind: ComponentKind::Score {
                name: "Notch".into(),
                objective: "kills".into(),
                value: Some("12".into()),
            },
            style: ChatStyle::default(),
            extra: Vec::new(),
        };
        let right = ChatComponent {
            kind: ComponentKind::Score {
                name: "Notch".into(),
                objective: "kills".into(),
                value: Some("13".into()),
            },
            style: ChatStyle::default(),
            extra: Vec::new(),
        };
        assert_eq!(left, right);

        // Negative control: name/objective still identify the score component.
        let different_objective = ChatComponent {
            kind: ComponentKind::Score {
                name: "Notch".into(),
                objective: "deaths".into(),
                value: Some("13".into()),
            },
            style: ChatStyle::default(),
            extra: Vec::new(),
        };
        assert_ne!(left, different_objective);
    }

    #[test]
    fn translation_arguments_and_hover_values_use_resolved_component_equality() {
        let inherited_argument = ChatComponent::translate("key", vec![ChatComponent::text("arg")])
            .with_style(ChatStyle {
                underlined: Some(true),
                ..ChatStyle::default()
            });
        let explicit_argument = ChatComponent::translate(
            "key",
            vec![ChatComponent::text("arg").with_style(ChatStyle {
                underlined: Some(true),
                ..ChatStyle::default()
            })],
        )
        .with_style(ChatStyle {
            underlined: Some(true),
            ..ChatStyle::default()
        });
        assert_eq!(inherited_argument, explicit_argument);

        assert_eq!(
            HoverEvent {
                action: HoverAction::ShowText,
                value: Box::new(inherited_argument),
            },
            HoverEvent {
                action: HoverAction::ShowText,
                value: Box::new(explicit_argument),
            },
            "HoverEvent.equals delegates to the component value's Java equality"
        );
    }

    #[test]
    fn is_empty_ignores_insertion_like_vanilla() {
        let s = ChatStyle {
            insertion: Some("hi".into()),
            ..ChatStyle::default()
        };
        assert!(s.is_empty(), "vanilla's isEmpty does not look at insertion");
    }

    #[test]
    fn insertion_only_translation_arguments_collapse_to_plain_strings() {
        let mut language = lang();
        language.insert("test.arg", "[%s]");
        let component = ChatComponent::from_json(
            r#"{"translate":"test.arg","with":[{"text":"name","insertion":"secret"}]}"#,
        )
        .expect("translation");
        let spans = component.spans(&language);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].text, "name");
        assert_eq!(
            spans[1].style.insertion, None,
            "vanilla replaced this argument component with its raw String"
        );
        assert_eq!(
            component.to_json(),
            r#"{"translate":"test.arg","with":["name"]}"#
        );
    }

    #[test]
    fn insertion_only_styles_are_omitted_by_vanillas_empty_style_gate() {
        let text =
            ChatComponent::from_json(r#"{"text":"x","insertion":"ins"}"#).expect("text component");
        assert_eq!(text.style.insertion.as_deref(), Some("ins"));
        assert_eq!(text.to_json(), r#""x""#);

        let selector = ChatComponent::from_json(r#"{"selector":"@p","insertion":"ins"}"#)
            .expect("selector component");
        assert_eq!(selector.to_json(), r#"{"selector":"@p"}"#);

        let extra =
            ChatComponent::from_json(r#"{"text":"a","extra":[{"text":"b","insertion":"ins"}]}"#)
                .expect("extra component");
        assert_eq!(
            extra.extra[0].style.insertion.as_deref(),
            Some("ins"),
            "extra parsing keeps the component; only serialisation collapses it"
        );
        assert_eq!(extra.to_json(), r#"{"extra":["b"],"text":"a"}"#);
    }

    #[test]
    fn json_writer_uses_vanillas_adapter_insertion_order() {
        let component = ChatComponent::from_json(
            r#"{"text":"x","color":"gold","bold":true,"italic":false,"underlined":true,"strikethrough":false,"obfuscated":true,"insertion":"i","clickEvent":{"action":"run_command","value":"/x"},"hoverEvent":{"action":"show_text","value":"tip"},"extra":["z"]}"#,
        )
        .expect("all style fields");
        assert_eq!(
            component.to_json(),
            r#"{"bold":true,"italic":false,"underlined":true,"strikethrough":false,"obfuscated":true,"color":"gold","insertion":"i","clickEvent":{"action":"run_command","value":"/x"},"hoverEvent":{"action":"show_text","value":"tip"},"extra":["z"],"text":"x"}"#
        );

        // Negative control: serde_json's workspace-default Map is a BTreeMap and therefore
        // alphabetises these keys. This proves the fixture reaches the ordering divergence.
        let sorted = component.to_value().to_string();
        assert_ne!(sorted, component.to_json());
        assert!(sorted.starts_with(r#"{"bold":true,"clickEvent"#));
    }

    /// `levels` translations, each a missing key that names its one argument `refs` times,
    /// around `{"text":"a"}` (or around nothing when `leaf` is false). Unbudgeted, it flattens
    /// to `refs^levels` spans.
    fn reused_argument_tower(levels: usize, refs: usize, leaf: bool) -> String {
        let format = "%1$s".repeat(refs);
        let mut json = if leaf {
            r#"{"text":"a"}"#.to_owned()
        } else {
            format!(r#"{{"translate":"{format}"}}"#)
        };
        for _ in 0..levels {
            json = format!(r#"{{"translate":"{format}","with":[{json}]}}"#);
        }
        json
    }

    /// A chat packet (or a server-list MOTD) that reuses translation
    /// arguments at every level used to flatten to `2^40` spans: a hang, then an allocation
    /// failure. It now stops at the budget, quickly, and keeps the text it reached.
    #[test]
    fn reused_translation_arguments_cannot_make_flattening_explode() {
        let lang = Language::new("en_US");
        let json = reused_argument_tower(40, 2, true);
        assert!(
            json.len() < 32_767,
            "fits in one chat packet ({} bytes)",
            json.len()
        );
        let component = ChatComponent::from_json(&json).expect("valid JSON");
        let started = std::time::Instant::now();
        let spans = component.spans(&lang);
        let text = component.unformatted_text(&lang);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "flattening took {:?}",
            started.elapsed()
        );
        assert!(!spans.is_empty() && spans.len() <= MAX_FLATTEN_WORK);
        assert!(text.len() <= MAX_FLATTEN_WORK && text.bytes().all(|b| b == b'a'));

        // A tower that emits nothing at all must stop too: every visit is charged.
        let silent = ChatComponent::from_json(&reused_argument_tower(40, 2, false)).unwrap();
        let started = std::time::Instant::now();
        assert!(silent.spans(&lang).is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    /// Gson skips its "fully consumed" check when the component is
    /// null, so these all decode to a null component in vanilla (the Java decode oracle accepts
    /// every one) and used to disconnect here. Anything non-null still must be the whole document.
    #[test]
    fn a_leading_null_component_ignores_the_rest_as_gson_does() {
        for json in [
            "null{\"text\":\"x\"}",
            "null=>",
            "NuLl;anything",
            "null# comment\n{\"text\":\"x\"}",
            "/* block */ null{\"text\":\"x\"}#",
            "null[\"a\",,\"b\"]",
            "null\r1e999999",
            "null}",
            "[]\"a\",,\"b\"]",
            "[[],[]]{\"text\":\"x\"}",
            ")]}'\n//{\"text\":\"x\"}",
            ")]}'\n#\u{0}{\"text\":\"x\"}",
        ] {
            assert!(
                matches!(ChatComponent::from_json_nullable(json), Ok(None)),
                "{json:?}"
            );
        }
        // Negative controls: a non-null first value must still be the whole document, and a
        // `null` that runs into a literal character is the text "nullx", not the keyword.
        for json in [
            "nullx junk",
            "[\"a\"]junk",
            "[null]x",
            "{\"text\":\"x\"}junk",
            "\"a\" \"b\"",
            "/* unterminated null",
        ] {
            assert!(ChatComponent::from_json_nullable(json).is_err(), "{json:?}");
        }
        // Valid documents are untouched.
        assert!(matches!(
            ChatComponent::from_json_nullable("null"),
            Ok(None)
        ));
        assert!(matches!(
            ChatComponent::from_json_nullable(")]}'\n{\"text\":\"x\"}"),
            Ok(Some(_))
        ));
    }

    /// Reuse that a real message could contain is never shortened:
    /// the same tower at twelve levels flattens to exactly its 4096 spans, as before the budget.
    #[test]
    fn moderate_argument_reuse_flattens_in_full() {
        let lang = Language::new("en_US");
        let component = ChatComponent::from_json(&reused_argument_tower(12, 2, true)).unwrap();
        let spans = component.spans(&lang);
        assert_eq!(spans.len(), 4096);
        assert_eq!(component.unformatted_text(&lang), "a".repeat(4096));
        // A long run of literal text is kept whole.
        let long = ChatComponent::text("x".repeat(32_000));
        assert_eq!(long.unformatted_text(&lang).len(), 32_000);
    }

    #[test]
    fn json_writer_uses_gsons_default_html_escaping() {
        let component = ChatComponent::text("a < b > c & d = e ' f\u{2028}g\u{2029}h");
        assert_eq!(
            component.to_json(),
            r#""a \u003c b \u003e c \u0026 d \u003d e \u0027 f\u2028g\u2029h""#
        );

        let nested = ChatComponent::text("x").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/msg <b> &y".into(),
            }),
            ..ChatStyle::default()
        });
        assert_eq!(
            nested.to_json(),
            r#"{"clickEvent":{"action":"run_command","value":"/msg \u003cb\u003e \u0026y"},"text":"x"}"#
        );
    }
}
