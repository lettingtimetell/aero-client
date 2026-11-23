//! The mod menu's search: one ranked list of mods and settings.
//!
//! The engine is plain text logic over this crate's data (the
//! catalog, [`crate::menu_layout`]'s summaries and keywords, dynamic rows), so it lives here and
//! is tested here; `mc-ui`'s `modmenu/search.rs` is the menu's adapter.
//!
//! # What matches
//!
//! * **Mods** match on their name, their summary and their keywords (so "color" lists Color
//!   Grading as a mod, and the mods that merely mention colour show up through their colour
//!   settings instead). The long descriptions are searched only when nothing else matches at all.
//! * **Settings** match on their label, description, choice labels (a button's text too), their
//!   own keywords ([`crate::menu_layout::OPTION_KEYWORDS`]) and words implied by their kind: "key",
//!   "bind" or "hotkey" for a key setting, "color" and colour names ("red") for a colour setting.
//!   A term may also match the setting's mod ("zoom key", "crosshair red"), but at least one term
//!   must match the setting itself.
//! * **Dynamic rows** (a waypoint's name) match on their label, like a setting.
//!
//! Every term must match. Text is compared lower case, in US spelling (colour = color, armour =
//! armor, centre = center, grey = gray...), with spaces and punctuation ignored ("togglesprint",
//! "boss bar", "f3+b"). A half-typed UK spelling still matches ("colou").
//!
//! # Ranking
//!
//! Each term scores the best field it matches (exact name, then start of name, then keyword,
//! then a word of the name...; a setting's label is worth less than a mod's name, so a mod and
//! its own setting of the same name list the mod first). A result's score is its weakest term,
//! then the sum: a result that matches every term well beats one that matches one term
//! perfectly and another only in passing. Equal results keep the menu's order.
//!
//! The index is built once per menu ([`SearchIndex::new`]); [`SearchIndex::search_into`] reuses
//! the result vectors, so typing does not allocate once they have grown.

use crate::descriptor::{ModDescriptor, OptionKind};
use crate::dynamic::DynamicRows;
use crate::menu_layout::{self, MenuCategory};

/// UK spellings and their US forms, longest first where one contains another.
const SPELLINGS: [(&str, &str); 12] = [
    ("centred", "centered"),
    ("centre", "center"),
    ("colour", "color"),
    ("armour", "armor"),
    ("favourite", "favorite"),
    ("behaviour", "behavior"),
    ("neighbour", "neighbor"),
    ("grey", "gray"),
    ("metre", "meter"),
    ("customise", "customize"),
    ("optimise", "optimize"),
    ("recognise", "recognize"),
];

/// Words a key setting answers to.
const KEY_WORDS: [&str; 7] = [
    "key", "keys", "keybind", "keybinds", "bind", "binding", "hotkey",
];

/// Words a colour setting answers to: "color" and the colours people name.
const COLOR_WORDS: [&str; 22] = [
    "color", "colors", "red", "green", "blue", "yellow", "orange", "pink", "purple", "magenta",
    "cyan", "aqua", "white", "black", "gray", "gold", "lime", "brown", "teal", "violet", "rgb",
    "hex",
];

// ---- strengths: what one term is worth in each field --------------------------------------
// Mod fields.
const NAME_EXACT: u16 = 100;
const NAME_PREFIX: u16 = 85;
const KEYWORD_EXACT: u16 = 80;
const NAME_WORD: u16 = 75;
const KEYWORD_PREFIX: u16 = 60;
const KEYWORD_WORD: u16 = 55;
const NAME_SUBSTR: u16 = 45;
const SUMMARY_WORD: u16 = 40;
const KEYWORD_SUBSTR: u16 = 30;
const SUMMARY_SUBSTR: u16 = 25;
const DESCRIPTION_WORD: u16 = 15;
const DESCRIPTION_SUBSTR: u16 = 10;
// Setting fields. A label is worth a little less than a mod's name.
const LABEL_EXACT: u16 = 70;
const LABEL_PREFIX: u16 = 65;
const OPTION_KEYWORD_EXACT: u16 = 60;
const LABEL_WORD: u16 = 58;
const KIND_WORD: u16 = 52;
const CHOICE_EXACT: u16 = 50;
const OPTION_KEYWORD_PREFIX: u16 = 48;
const CHOICE_WORD: u16 = 45;
const LABEL_SUBSTR: u16 = 35;
const OPTION_DESCRIPTION_WORD: u16 = 28;
const CHOICE_SUBSTR: u16 = 25;
const OPTION_DESCRIPTION_SUBSTR: u16 = 15;
/// A term that names the setting's mod rather than the setting ("zoom" in "zoom key").
const CONTEXT_NAME: u16 = 30;
const CONTEXT_KEYWORD: u16 = 22;
/// A term shorter than this matches only whole words or word starts, never inside a word ("red"
/// must not find "Centered Inventory").
const MIN_SUBSTRING_LEN: usize = 4;

/// One piece of text, normalised: lower case, US spelling, letters and digits only.
#[derive(Debug, Clone, Default)]
struct Text {
    compact: String,
    /// Byte offsets in `compact` where a word starts.
    words: Vec<u32>,
}

impl Text {
    fn new(source: &str) -> Text {
        let mut text = Text::default();
        let mut word = String::new();
        text.set(source, &mut word);
        text
    }

    /// Normalise `source` into `self`, reusing both buffers.
    fn set(&mut self, source: &str, word: &mut String) {
        self.compact.clear();
        self.words.clear();
        let mut chars = source.chars().peekable();
        while chars.peek().is_some() {
            word.clear();
            while let Some(c) = chars.next_if(|c| c.is_alphanumeric()) {
                word.extend(c.to_lowercase());
            }
            if word.is_empty() {
                chars.next();
                continue;
            }
            self.words.push(self.compact.len() as u32);
            push_us_spelling(word, &mut self.compact);
        }
    }

    /// How well `term` (normalised) fits this text.
    fn fit(&self, term: &str) -> Option<Fit> {
        if term.is_empty() || self.compact.is_empty() {
            return None;
        }
        if self.compact == term {
            Some(Fit::Exact)
        } else if self.compact.starts_with(term) {
            Some(Fit::Prefix)
        } else if self
            .words
            .iter()
            .any(|&at| self.compact[at as usize..].starts_with(term))
        {
            Some(Fit::Word)
        } else if term.len() >= MIN_SUBSTRING_LEN && self.compact.contains(term) {
            Some(Fit::Substring)
        } else {
            None
        }
    }
}

/// Append `word` (lower case) to `out` in US spelling.
fn push_us_spelling(word: &str, out: &mut String) {
    let mut rest = word;
    'outer: while !rest.is_empty() {
        for (uk, us) in SPELLINGS {
            if rest.starts_with(uk) {
                out.push_str(us);
                rest = &rest[uk.len()..];
                continue 'outer;
            }
        }
        let c = rest.chars().next().unwrap_or(' ');
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
}

/// A half-typed UK spelling at the end of a term ("colou", "armou", "centr") becomes the part
/// both spellings share ("colo"), so the results don't vanish halfway through typing "colour".
fn trim_partial_uk_tail(term: &mut String) {
    for (uk, us) in SPELLINGS {
        let shared = uk
            .bytes()
            .zip(us.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        if uk.len() < 5 {
            continue;
        }
        for len in (shared + 1..uk.len()).rev() {
            if term.ends_with(&uk[..len]) {
                let keep = term.len() - len + shared;
                term.truncate(keep);
                return;
            }
        }
    }
}

/// How a term fits a piece of text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Fit {
    Substring,
    Word,
    Prefix,
    Exact,
}

fn best_fit<'a>(texts: impl IntoIterator<Item = &'a Text>, term: &str) -> Option<Fit> {
    texts.into_iter().filter_map(|text| text.fit(term)).max()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindWords {
    None,
    Key,
    Color,
}

#[derive(Debug, Clone)]
struct OptionEntry {
    index: usize,
    label: Text,
    description: Text,
    /// Choice labels, or a button's text.
    choices: Vec<Text>,
    keywords: Vec<Text>,
    kind: KindWords,
}

#[derive(Debug, Clone)]
struct ModEntry {
    name: Text,
    summary: Text,
    keywords: Vec<Text>,
    description: Text,
    /// Position in the menu's window order (equal results keep it).
    rank: u32,
    options: Vec<OptionEntry>,
}

/// The catalog's text, normalised once for searching.
#[derive(Debug, Clone)]
pub struct SearchIndex {
    catalog: &'static [&'static ModDescriptor],
    mods: Vec<ModEntry>,
}

/// A mod that matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModHit {
    /// Index into the catalog.
    pub mod_index: usize,
    /// Higher is better.
    pub score: u32,
}

/// What a setting result points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitTarget {
    /// An option, by index into the mod's `options`.
    Option(usize),
    /// A dynamic row, by its [`crate::DynamicRow::key`].
    Row(u64),
}

/// A setting (or dynamic row) that matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SettingHit {
    /// Index into the catalog of the mod the setting belongs to.
    pub mod_index: usize,
    /// The setting.
    pub target: HitTarget,
    /// Higher is better.
    pub score: u32,
}

/// One result of either kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hit {
    /// A mod.
    Mod(ModHit),
    /// A setting of a mod.
    Setting(SettingHit),
}

impl Hit {
    /// The catalog index of the mod the result belongs to.
    #[must_use]
    pub fn mod_index(self) -> usize {
        match self {
            Hit::Mod(hit) => hit.mod_index,
            Hit::Setting(hit) => hit.mod_index,
        }
    }
}

/// The results of one search, ranked. Keep one and pass it to every
/// [`SearchIndex::search_into`]: its vectors are reused.
#[derive(Debug, Clone, Default)]
pub struct SearchResults {
    /// Matching mods, best first.
    pub mods: Vec<ModHit>,
    /// Matching settings, grouped by mod: the mods in order of their best setting, each mod's
    /// settings best first.
    pub settings: Vec<SettingHit>,
    catalog: &'static [&'static ModDescriptor],
    terms: Vec<String>,
    joined: String,
    row_text: Text,
    word: String,
    best_of_mod: Vec<(u32, u32)>,
}

/// A combined score: the weakest term first, then the sum of all of them.
fn combine(min: u16, sum: u32) -> u32 {
    (u32::from(min) << 16) | sum.min(0xFFFF)
}

/// Per-term strengths folded into `(min, sum)`; `None` once a term misses.
#[derive(Debug, Clone, Copy)]
struct Fold {
    min: u16,
    sum: u32,
}

impl Fold {
    const START: Fold = Fold {
        min: u16::MAX,
        sum: 0,
    };

    fn add(self, strength: u16) -> Option<Fold> {
        (strength > 0).then(|| Fold {
            min: self.min.min(strength),
            sum: self.sum + u32::from(strength),
        })
    }

    fn score(self, joined: u16, terms: usize) -> u32 {
        // The whole query as one word ("toggle sprint" = "togglesprint") can fit better than
        // its terms do one by one.
        if terms > 1 && joined > self.min {
            combine(joined, u32::from(joined) * terms as u32)
        } else {
            combine(self.min, self.sum)
        }
    }
}

impl SearchIndex {
    /// Normalise the text of every mod in `catalog` (once per menu).
    #[must_use]
    pub fn new(catalog: &'static [&'static ModDescriptor]) -> SearchIndex {
        let order: Vec<&str> = menu_layout::menu_order().collect();
        let mods = catalog
            .iter()
            .enumerate()
            .map(|(index, descriptor)| {
                let meta = menu_layout::meta(descriptor.id);
                let rank = order
                    .iter()
                    .position(|id| *id == descriptor.id)
                    .map_or(10_000 + index as u32, |at| at as u32);
                ModEntry {
                    name: Text::new(descriptor.name),
                    summary: Text::new(meta.map_or("", |meta| meta.summary)),
                    keywords: meta
                        .map_or(&[][..], |meta| meta.keywords)
                        .iter()
                        .map(|keyword| Text::new(keyword))
                        .collect(),
                    description: Text::new(descriptor.description),
                    rank,
                    options: descriptor
                        .options
                        .iter()
                        .enumerate()
                        .filter(|(_, option)| option.kind.is_visible())
                        .map(|(index, option)| OptionEntry {
                            index,
                            label: Text::new(option.label),
                            description: Text::new(option.description),
                            choices: match option.kind {
                                OptionKind::Choice { choices } => {
                                    choices.iter().map(|choice| Text::new(choice)).collect()
                                }
                                OptionKind::Button { label } => vec![Text::new(label)],
                                _ => Vec::new(),
                            },
                            keywords: menu_layout::option_keywords(descriptor.id, option.id)
                                .iter()
                                .map(|keyword| Text::new(keyword))
                                .collect(),
                            kind: match option.kind {
                                OptionKind::Keybind => KindWords::Key,
                                OptionKind::Color { .. } => KindWords::Color,
                                _ => KindWords::None,
                            },
                        })
                        .collect(),
                }
            })
            .collect();
        SearchIndex { catalog, mods }
    }

    /// The catalog this index searches.
    #[must_use]
    pub fn catalog(&self) -> &'static [&'static ModDescriptor] {
        self.catalog
    }

    /// Search for `query`. `rows` are this frame's dynamic rows per catalog index (may be
    /// shorter than the catalog). An empty query gives no results.
    #[must_use]
    pub fn search(&self, query: &str, rows: &[DynamicRows]) -> SearchResults {
        let mut results = SearchResults::default();
        self.search_into(query, rows, &mut results);
        results
    }

    /// [`SearchIndex::search`] into `out`, reusing its vectors.
    pub fn search_into(&self, query: &str, rows: &[DynamicRows], out: &mut SearchResults) {
        out.mods.clear();
        out.settings.clear();
        out.catalog = self.catalog;
        out.terms.clear();
        out.joined.clear();
        for raw in query.split_whitespace() {
            out.row_text.set(raw, &mut out.word);
            if out.row_text.compact.is_empty() {
                continue;
            }
            let mut term = out.row_text.compact.clone();
            trim_partial_uk_tail(&mut term);
            out.joined.push_str(&term);
            out.terms.push(term);
        }
        if out.terms.is_empty() {
            return;
        }
        let terms = &out.terms;
        let joined = out.joined.as_str();
        for (mod_index, entry) in self.mods.iter().enumerate() {
            for option in &entry.options {
                if let Some(score) = option_score(entry, option, terms, joined) {
                    out.settings.push(SettingHit {
                        mod_index,
                        target: HitTarget::Option(option.index),
                        score,
                    });
                }
            }
            for row in rows.get(mod_index).map_or(&[][..], DynamicRows::rows) {
                out.row_text.set(&row.label, &mut out.word);
                if let Some(score) = row_score(entry, &out.row_text, terms, joined) {
                    out.settings.push(SettingHit {
                        mod_index,
                        target: HitTarget::Row(row.key),
                        score,
                    });
                }
            }
            if let Some(score) = mod_score(entry, terms, joined, false) {
                out.mods.push(ModHit { mod_index, score });
            }
        }
        // Last resort: when nothing else matched, the mods' descriptions (a word only they use).
        if out.mods.is_empty() && out.settings.is_empty() {
            for (mod_index, entry) in self.mods.iter().enumerate() {
                if let Some(score) = mod_score(entry, terms, joined, true) {
                    out.mods.push(ModHit { mod_index, score });
                }
            }
        }
        let mods = &self.mods;
        out.mods.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then(mods[a.mod_index].rank.cmp(&mods[b.mod_index].rank))
        });
        // Group the settings by mod: mods by their best setting, then each mod's settings.
        out.best_of_mod.clear();
        out.best_of_mod.resize(mods.len(), (0, u32::MAX));
        for hit in &out.settings {
            let best = &mut out.best_of_mod[hit.mod_index];
            best.0 = best.0.max(hit.score);
            best.1 = mods[hit.mod_index].rank;
        }
        let best_of_mod = &out.best_of_mod;
        out.settings.sort_by(|a, b| {
            let (best_a, rank_a) = best_of_mod[a.mod_index];
            let (best_b, rank_b) = best_of_mod[b.mod_index];
            best_b
                .cmp(&best_a)
                .then(rank_a.cmp(&rank_b))
                .then(a.mod_index.cmp(&b.mod_index))
                .then(b.score.cmp(&a.score))
                .then(target_order(a.target).cmp(&target_order(b.target)))
        });
    }
}

/// Settings before dynamic rows, each in its own order.
fn target_order(target: HitTarget) -> (u8, u64) {
    match target {
        HitTarget::Option(index) => (0, index as u64),
        HitTarget::Row(key) => (1, key),
    }
}

fn name_strength(fit: Option<Fit>) -> u16 {
    match fit {
        Some(Fit::Exact) => NAME_EXACT,
        Some(Fit::Prefix) => NAME_PREFIX,
        Some(Fit::Word) => NAME_WORD,
        Some(Fit::Substring) => NAME_SUBSTR,
        None => 0,
    }
}

fn keyword_strength(fit: Option<Fit>) -> u16 {
    match fit {
        Some(Fit::Exact) => KEYWORD_EXACT,
        Some(Fit::Prefix) => KEYWORD_PREFIX,
        Some(Fit::Word) => KEYWORD_WORD,
        Some(Fit::Substring) => KEYWORD_SUBSTR,
        None => 0,
    }
}

fn summary_strength(fit: Option<Fit>) -> u16 {
    match fit {
        Some(Fit::Exact | Fit::Prefix | Fit::Word) => SUMMARY_WORD,
        Some(Fit::Substring) => SUMMARY_SUBSTR,
        None => 0,
    }
}

fn label_strength(fit: Option<Fit>) -> u16 {
    match fit {
        Some(Fit::Exact) => LABEL_EXACT,
        Some(Fit::Prefix) => LABEL_PREFIX,
        Some(Fit::Word) => LABEL_WORD,
        Some(Fit::Substring) => LABEL_SUBSTR,
        None => 0,
    }
}

/// One term against a mod's own fields (name, keywords, summary; the description too when
/// `with_description`).
fn mod_term(entry: &ModEntry, term: &str, with_description: bool) -> u16 {
    let mut best = name_strength(entry.name.fit(term))
        .max(keyword_strength(best_fit(&entry.keywords, term)))
        .max(summary_strength(entry.summary.fit(term)));
    if with_description {
        best = best.max(match entry.description.fit(term) {
            Some(Fit::Exact | Fit::Prefix | Fit::Word) => DESCRIPTION_WORD,
            Some(Fit::Substring) => DESCRIPTION_SUBSTR,
            None => 0,
        });
    }
    best
}

fn mod_score(
    entry: &ModEntry,
    terms: &[String],
    joined: &str,
    with_description: bool,
) -> Option<u32> {
    let mut fold = Fold::START;
    for term in terms {
        fold = fold.add(mod_term(entry, term, with_description))?;
    }
    let joined_strength = if terms.len() > 1 {
        name_strength(entry.name.fit(joined))
            .max(keyword_strength(best_fit(&entry.keywords, joined)))
    } else {
        0
    };
    Some(fold.score(joined_strength, terms.len()))
}

/// One term against the setting's mod: "zoom" in "zoom key".
fn context_term(entry: &ModEntry, term: &str) -> u16 {
    let name = match entry.name.fit(term) {
        Some(Fit::Exact | Fit::Prefix | Fit::Word) => CONTEXT_NAME,
        Some(Fit::Substring) => CONTEXT_KEYWORD,
        None => 0,
    };
    let keyword = if best_fit(&entry.keywords, term).is_some_and(|fit| fit >= Fit::Word) {
        CONTEXT_KEYWORD
    } else {
        0
    };
    name.max(keyword)
}

/// One term against a setting's own fields.
fn option_term(option: &OptionEntry, term: &str) -> u16 {
    let keyword = match best_fit(&option.keywords, term) {
        Some(Fit::Exact) => OPTION_KEYWORD_EXACT,
        Some(Fit::Prefix | Fit::Word) => OPTION_KEYWORD_PREFIX,
        _ => 0,
    };
    let kind = match option.kind {
        KindWords::Key if KEY_WORDS.contains(&term) => KIND_WORD,
        KindWords::Color if COLOR_WORDS.contains(&term) => KIND_WORD,
        _ => 0,
    };
    let choice = match best_fit(&option.choices, term) {
        Some(Fit::Exact) => CHOICE_EXACT,
        Some(Fit::Prefix | Fit::Word) => CHOICE_WORD,
        Some(Fit::Substring) => CHOICE_SUBSTR,
        None => 0,
    };
    let description = match option.description.fit(term) {
        Some(Fit::Exact | Fit::Prefix | Fit::Word) => OPTION_DESCRIPTION_WORD,
        Some(Fit::Substring) => OPTION_DESCRIPTION_SUBSTR,
        None => 0,
    };
    label_strength(option.label.fit(term))
        .max(keyword)
        .max(kind)
        .max(choice)
        .max(description)
}

fn option_score(
    entry: &ModEntry,
    option: &OptionEntry,
    terms: &[String],
    joined: &str,
) -> Option<u32> {
    let mut fold = Fold::START;
    let mut own = false;
    for term in terms {
        let strength = option_term(option, term);
        own |= strength > 0;
        fold = fold.add(strength.max(context_term(entry, term)))?;
    }
    if !own {
        return None;
    }
    let joined_strength = if terms.len() > 1 {
        label_strength(option.label.fit(joined))
    } else {
        0
    };
    Some(fold.score(joined_strength, terms.len()))
}

fn row_score(entry: &ModEntry, label: &Text, terms: &[String], joined: &str) -> Option<u32> {
    let mut fold = Fold::START;
    let mut own = false;
    for term in terms {
        let strength = label_strength(label.fit(term));
        own |= strength > 0;
        fold = fold.add(strength.max(context_term(entry, term)))?;
    }
    if !own {
        return None;
    }
    let joined_strength = if terms.len() > 1 {
        label_strength(label.fit(joined))
    } else {
        0
    };
    Some(fold.score(joined_strength, terms.len()))
}

impl SearchResults {
    /// Nothing matched (or nothing was searched).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mods.is_empty() && self.settings.is_empty()
    }

    /// The best result: the one the list highlights and Enter opens. A setting beats a mod only
    /// with a strictly better score ("zoom key" → Zoom's key; "color" → Color Grading).
    #[must_use]
    pub fn best(&self) -> Option<Hit> {
        // Both lists are sorted best first (the settings by their mod's best setting, then by
        // score), so each list's first entry is its best, ties resolved in menu order.
        match (self.mods.first(), self.settings.first()) {
            (Some(m), Some(s)) if s.score > m.score => Some(Hit::Setting(*s)),
            (Some(m), _) => Some(Hit::Mod(*m)),
            (None, Some(s)) => Some(Hit::Setting(*s)),
            (None, None) => None,
        }
    }

    /// Whether mod `mod_index` has any result (as a mod or through a setting).
    #[must_use]
    pub fn shows_mod(&self, mod_index: usize) -> bool {
        self.mods.iter().any(|hit| hit.mod_index == mod_index)
            || self.settings.iter().any(|hit| hit.mod_index == mod_index)
    }

    /// Whether mod `mod_index` matched as a mod.
    #[must_use]
    pub fn is_mod_hit(&self, mod_index: usize) -> bool {
        self.mods.iter().any(|hit| hit.mod_index == mod_index)
    }

    /// Whether option `option_index` of mod `mod_index` matched (a setting found by search is
    /// never hidden: see `menu_rules::OptionState::for_search_hit`).
    #[must_use]
    pub fn option_hit(&self, mod_index: usize, option_index: usize) -> bool {
        self.settings
            .iter()
            .any(|hit| hit.mod_index == mod_index && hit.target == HitTarget::Option(option_index))
    }

    /// Whether dynamic row `key` of mod `mod_index` matched.
    #[must_use]
    pub fn row_hit(&self, mod_index: usize, key: u64) -> bool {
        self.settings
            .iter()
            .any(|hit| hit.mod_index == mod_index && hit.target == HitTarget::Row(key))
    }

    /// The matching settings of mod `mod_index`, best first.
    pub fn settings_of(&self, mod_index: usize) -> impl Iterator<Item = &SettingHit> + '_ {
        self.settings
            .iter()
            .filter(move |hit| hit.mod_index == mod_index)
    }

    /// The mods that have matching settings, in the settings list's order, each once.
    pub fn setting_mods(&self) -> impl Iterator<Item = usize> + '_ {
        self.settings
            .iter()
            .enumerate()
            .filter(|(i, hit)| *i == 0 || self.settings[i - 1].mod_index != hit.mod_index)
            .map(|(_, hit)| hit.mod_index)
    }

    /// How many different mods have a result ("All mods" while searching).
    #[must_use]
    pub fn mod_count(&self) -> usize {
        let from_settings = self.setting_mods().filter(|&m| !self.is_mod_hit(m)).count();
        self.mods.len() + from_settings
    }

    /// How many different mods with a result `category` lists (second listings included): the
    /// count beside each category in window 1 while searching.
    #[must_use]
    pub fn count_in(&self, category: &MenuCategory) -> usize {
        let listed = |m: usize| {
            self.catalog
                .get(m)
                .is_some_and(|descriptor| category.lists(descriptor.id))
        };
        let from_mods = self.mods.iter().filter(|hit| listed(hit.mod_index)).count();
        let from_settings = self
            .setting_mods()
            .filter(|&m| listed(m) && !self.is_mod_hit(m))
            .count();
        from_mods + from_settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;
    use crate::dynamic::DynamicRows;

    /// The expected best result of a phrase: a mod, or a mod's setting.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Want {
        Mod(&'static str),
        Setting(&'static str, &'static str),
    }
    use Want::{Mod, Setting};

    fn index() -> SearchIndex {
        SearchIndex::new(catalog::all())
    }

    fn describe(hit: Option<Hit>) -> Option<Want> {
        let all = catalog::all();
        hit.map(|hit| match hit {
            Hit::Mod(m) => Mod(all[m.mod_index].id),
            Hit::Setting(s) => {
                let d = all[s.mod_index];
                match s.target {
                    HitTarget::Option(o) => Setting(d.id, d.options[o].id),
                    HitTarget::Row(_) => Setting(d.id, "<row>"),
                }
            }
        })
    }

    fn mod_ids(results: &SearchResults) -> Vec<&'static str> {
        results
            .mods
            .iter()
            .map(|hit| catalog::all()[hit.mod_index].id)
            .collect()
    }

    /// About forty phrases pinned to their best result. When a keyword, a summary or a
    /// label changes, this says which phrases moved.
    #[test]
    fn golden_phrases_find_their_result() {
        let golden: &[(&str, Want)] = &[
            ("color", Mod("color_grading")),
            ("colour", Mod("color_grading")),
            ("gamma", Mod("fullbright")),
            ("coords", Mod("coordinates")),
            ("xyz", Mod("coordinates")),
            ("togglesprint", Mod("toggle_sprint")),
            ("toggle sprint", Mod("toggle_sprint")),
            ("crosshair red", Setting("crosshair", "color")),
            ("zoom key", Setting("zoom", "key")),
            ("f3+b", Mod("hitboxes")),
            ("ctm", Mod("connected_textures")),
            ("ssao", Mod("ambient_occlusion")),
            ("aa", Mod("anti_aliasing")),
            ("taa", Mod("anti_aliasing")),
            ("fxaa", Mod("anti_aliasing")),
            ("potion", Mod("potion_status")),
            ("music", Mod("spotify")),
            ("spotify", Mod("spotify")),
            ("now playing", Mod("spotify")),
            ("blockhit", Mod("old_animations")),
            ("sidebar", Mod("scoreboard")),
            ("shaders", Mod("shaders_classic")),
            ("shader", Mod("shaders_classic")),
            ("bossbar", Mod("boss_bar")),
            ("boss bar", Mod("boss_bar")),
            ("fps", Mod("fps")),
            ("fog", Mod("fog")),
            ("noon", Mod("time_changer")),
            ("hitbox", Mod("hitboxes")),
            ("macro", Mod("chat_macros")),
            ("red numbers", Setting("scoreboard", "hide_numbers")),
            ("hide scoreboard", Setting("scoreboard", "hide")),
            ("armour", Mod("armor_status")),
            ("centre", Mod("centered_inventory")),
            ("nametag", Mod("nametags")),
            ("name tags", Mod("nametags")),
            ("fov", Mod("dynamic_fov")),
            ("fullscreen", Mod("borderless_fullscreen")),
            ("record", Mod("replay")),
            ("loading screen", Mod("aero_title")),
            ("blur", Mod("motion_blur")),
            ("crits", Mod("particles")),
            ("hitbox width", Setting("hitboxes", "width")),
            ("chat background", Setting("chat", "background_opacity")),
            ("sharpen", Setting("anti_aliasing", "sharpness")),
            ("mouse", Mod("raw_input")),
            ("time", Mod("time_changer")),
            ("zoom", Mod("zoom")),
            ("gg", Mod("auto_gg")),
            // One-word spellings players type find the two-word name or keyword (spaces ignored).
            ("autogg", Mod("auto_gg")),
            ("lowfire", Mod("fire_height")),
            ("autotext", Mod("chat_macros")),
            ("durability", Mod("armor_status")),
            ("clouds", Mod("volumetric_clouds")),
            ("logo", Mod("aero_title")),
            ("TOGGLE   Sprint", Mod("toggle_sprint")),
            // The plan's findability tasks (§5.5) and this wave's new settings.
            ("thicker", Setting("hitboxes", "width")),
            ("keystrokes", Mod("keystrokes")),
            ("eat", Setting("old_animations", "use_stops_mining")),
            ("chat history", Mod("chat")),
            ("cloud height", Setting("volumetric_clouds", "height")),
            // The shore is a title-screen scene now.
            ("shore", Setting("aero_title", "menu_scene")),
            ("clip key", Setting("replay", "key_clip")),
        ];
        let index = index();
        let mut results = SearchResults::default();
        let mut wrong = Vec::new();
        for (phrase, want) in golden {
            index.search_into(phrase, &[], &mut results);
            let got = describe(results.best());
            if got != Some(*want) {
                wrong.push(format!("{phrase:?}: want {want:?}, got {got:?}"));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} phrases moved:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        assert!(golden.len() >= 40);
    }

    /// Every mod is the best result for its own name, typed as shown or without spaces (so no
    /// keyword or setting can hijack another mod's name).
    #[test]
    fn every_mod_is_found_by_its_name() {
        let index = index();
        for (m, descriptor) in catalog::all().iter().enumerate() {
            let squashed: String = descriptor.name.split_whitespace().collect();
            for query in [descriptor.name, squashed.as_str()] {
                let results = index.search(query, &[]);
                assert_eq!(
                    results.best().map(Hit::mod_index),
                    Some(m),
                    "{query:?} → {:?}",
                    describe(results.best())
                );
                assert!(matches!(results.best(), Some(Hit::Mod(_))), "{query:?}");
            }
        }
    }

    /// "color" and "colour" both put Color Grading first, then the colour settings.
    #[test]
    fn both_spellings_give_the_same_results() {
        let index = index();
        let color = index.search("color", &[]);
        let colour = index.search("colour", &[]);
        assert_eq!(color.mods, colour.mods);
        assert_eq!(color.settings, colour.settings);
        assert_eq!(mod_ids(&color)[0], "color_grading");
        let fps = catalog::index_of("fps").unwrap();
        let text_color = catalog::all()[fps].option_index("text_color").unwrap();
        assert!(color.option_hit(fps, text_color), "FPS › Text color");
        assert!(
            !color.is_mod_hit(fps),
            "FPS is a setting result, not a mod result"
        );
        // Half typed: "colou" still finds the colour settings.
        let partial = index.search("colou", &[]);
        assert!(partial.option_hit(fps, text_color));
        for (uk, us) in [("armour", "armor"), ("centre", "center"), ("grey", "gray")] {
            let a = index.search(uk, &[]);
            let b = index.search(us, &[]);
            assert_eq!(a.mods, b.mods, "{uk}/{us}");
            assert_eq!(a.settings, b.settings, "{uk}/{us}");
        }
    }

    #[test]
    fn keywords_list_every_mod_they_name() {
        let index = index();
        let potion = mod_ids(&index.search("potion", &[]));
        for id in ["potion_status", "shiny_pots", "particles"] {
            assert!(potion.contains(&id), "potion → {id}: {potion:?}");
        }
        let shaders = mod_ids(&index.search("shaders", &[]));
        assert_eq!(
            &shaders[..3],
            ["shaders_classic", "shaders_dreamy", "shaders_underwater"]
        );
        assert!(shaders.contains(&"super_secret_settings"));
    }

    /// Spaces are optional, case and punctuation don't matter.
    #[test]
    fn spaces_case_and_punctuation_are_ignored() {
        let index = index();
        for (a, b) in [
            ("togglesprint", "Toggle Sprint"),
            ("bossbar", "boss bar"),
            ("F3+B", "f3b"),
            ("anti aliasing", "Anti-Aliasing"),
            ("1.7", "17"),
        ] {
            let x = index.search(a, &[]);
            let y = index.search(b, &[]);
            assert_eq!(describe(x.best()), describe(y.best()), "{a} / {b}");
            assert!(!x.is_empty(), "{a}");
        }
    }

    /// A mod's long description is a last resort: it never adds noise to a search that found
    /// something ("crosshair" does not list Tall Grass, whose description mentions the
    /// crosshair), but a word only a description has still finds its mod.
    #[test]
    fn descriptions_are_a_last_resort() {
        let index = index();
        let results = index.search("crosshair", &[]);
        assert_eq!(mod_ids(&results), ["crosshair"]);
        assert_eq!(
            describe(index.search("blindness", &[]).best()),
            Some(Mod("fog"))
        );
    }

    /// Short terms match words and word starts only: "red" finds the red settings, not
    /// "Centered Inventory"; the best result is the first one listed.
    #[test]
    fn short_terms_do_not_match_inside_words() {
        let index = index();
        let results = index.search("red", &[]);
        assert!(!mod_ids(&results).contains(&"centered_inventory"));
        let scoreboard = catalog::index_of("scoreboard").unwrap();
        let hide_numbers = catalog::all()[scoreboard]
            .option_index("hide_numbers")
            .unwrap();
        assert!(results.option_hit(scoreboard, hide_numbers));
        let first = results.settings[0];
        match results.best() {
            Some(Hit::Setting(best)) => assert_eq!(best, first),
            Some(Hit::Mod(best)) => assert!(best.score >= first.score),
            None => panic!("no result for red"),
        }
    }

    #[test]
    fn nothing_matches_nothing() {
        let index = index();
        for query in ["", "   ", "qqqxx", "zoom qqqxx"] {
            let results = index.search(query, &[]);
            assert!(results.is_empty(), "{query:?}");
            assert_eq!(results.best(), None);
            assert_eq!(results.mod_count(), 0);
        }
    }

    /// Every term must match: "zoom key" lists Zoom's key, not its amount slider.
    #[test]
    fn every_term_must_match_and_settings_need_one_own_term() {
        let index = index();
        let zoom = catalog::index_of("zoom").unwrap();
        let d = catalog::all()[zoom];
        let results = index.search("zoom key", &[]);
        assert!(results.option_hit(zoom, d.option_index("key").unwrap()));
        assert!(!results.option_hit(zoom, d.option_index("factor").unwrap()));
        // "fps" alone: the FPS mod, but its name does not make every FPS setting a result.
        let fps = catalog::index_of("fps").unwrap();
        let d = catalog::all()[fps];
        let results = index.search("fps", &[]);
        assert!(results.is_mod_hit(fps));
        assert!(
            results.option_hit(fps, d.option_index("display").unwrap()),
            "\"Show\" has the choice FPS"
        );
        assert!(!results.option_hit(fps, d.option_index("text_color").unwrap()));
        // "fps color": FPS's colour settings, through the mod's name.
        let results = index.search("fps color", &[]);
        assert!(results.option_hit(fps, d.option_index("text_color").unwrap()));
        assert!(!results.is_mod_hit(fps));
    }

    /// Choice labels and button text are searched; hidden settings are not.
    #[test]
    fn choices_buttons_and_hidden_settings() {
        let index = index();
        let armor = catalog::index_of("armor_status").unwrap();
        let layout = catalog::all()[armor].option_index("layout").unwrap();
        assert!(index.search("horizontal", &[]).option_hit(armor, layout));
        let particles = catalog::index_of("particles").unwrap();
        let table = catalog::all()[particles].option_index("table").unwrap();
        let results = index.search("group values", &[]);
        assert!(!results.option_hit(particles, table), "a hidden setting");
        let replay = catalog::index_of("replay").unwrap();
        let viewer = catalog::all()[replay].option_index("open_viewer").unwrap();
        assert!(index.search("replay open", &[]).option_hit(replay, viewer));
    }

    /// Dynamic rows (a waypoint's name) are results like settings.
    #[test]
    fn dynamic_rows_are_searched() {
        let index = index();
        let waypoints = catalog::index_of("waypoints").unwrap();
        let mut rows = vec![DynamicRows::default(); catalog::all().len()];
        rows[waypoints].begin();
        rows[waypoints].push(7).label.push_str("Spawn point");
        rows[waypoints].push(8).label.push_str("Diamonds");
        let results = index.search("spawn", &rows);
        assert!(results.row_hit(waypoints, 7));
        assert!(!results.row_hit(waypoints, 8));
        assert_eq!(results.best(), Some(Hit::Setting(results.settings[0])));
        assert!(index
            .search("waypoints diamonds", &rows)
            .row_hit(waypoints, 8));
    }

    /// Counts per category include second listings; "All mods" counts each mod once.
    #[test]
    fn counts_per_category_and_in_all() {
        let index = index();
        let results = index.search("crosshair", &[]);
        let hud = menu_layout::category("hud").unwrap();
        let pvp = menu_layout::category("pvp").unwrap();
        let world = menu_layout::category("world").unwrap();
        assert_eq!(results.count_in(hud), 1);
        assert_eq!(results.count_in(pvp), 1, "Also here");
        assert_eq!(results.count_in(world), 0);
        assert_eq!(results.mod_count(), 1);
        let results = index.search("color", &[]);
        let total: usize = menu_layout::CATEGORIES
            .iter()
            .map(|c| results.count_in(c))
            .sum();
        assert!(results.mod_count() > 5);
        assert!(total >= results.mod_count());
    }

    /// The settings list is grouped: each mod's settings sit together.
    #[test]
    fn settings_are_grouped_by_mod() {
        let index = index();
        let results = index.search("color", &[]);
        let order: Vec<usize> = results.setting_mods().collect();
        let mut seen = std::collections::HashSet::new();
        for m in &order {
            assert!(seen.insert(*m), "mod {m} appears in two runs");
        }
        let total: usize = order.iter().map(|&m| results.settings_of(m).count()).sum();
        assert_eq!(total, results.settings.len());
    }

    /// Searching again into the same results does not grow or leak old hits.
    #[test]
    fn results_are_reused() {
        let index = index();
        let mut results = SearchResults::default();
        index.search_into("color", &[], &mut results);
        let capacity = results.settings.capacity();
        index.search_into("fps", &[], &mut results);
        assert_eq!(describe(results.best()), Some(Mod("fps")));
        index.search_into("color", &[], &mut results);
        assert_eq!(results.settings.capacity(), capacity, "no reallocation");
        index.search_into("", &[], &mut results);
        assert!(results.is_empty());
    }

    /// A tuning aid, not a check: prints the ranked results of some phrases.
    /// `cargo test -p mc-mods --lib dump_search -- --ignored --nocapture`, or set
    /// `MC_SEARCH_DUMP="phrase|phrase"` for your own.
    #[test]
    #[ignore = "prints results for tuning; run by hand"]
    fn dump_search() {
        let phrases = std::env::var("MC_SEARCH_DUMP").unwrap_or_else(|_| {
            "color|potion|shaders|zoom key|crosshair red|red|key|opacity|fog|time".to_owned()
        });
        let started = std::time::Instant::now();
        let index = index();
        println!("index built in {:?}", started.elapsed());
        let mut reused = SearchResults::default();
        let started = std::time::Instant::now();
        let rounds = 200;
        for _ in 0..rounds {
            for phrase in phrases.split('|') {
                index.search_into(phrase, &[], &mut reused);
            }
        }
        let per = started.elapsed() / (rounds * phrases.split('|').count() as u32);
        println!("one search: {per:?} (this build's profile; provisional)");
        let all = catalog::all();
        for phrase in phrases.split('|') {
            let results = index.search(phrase, &[]);
            println!("== {phrase:?}: best {:?}", describe(results.best()));
            for hit in results.mods.iter().take(8) {
                println!("   mod {:>8x} {}", hit.score, all[hit.mod_index].id);
            }
            for hit in results.settings.iter().take(12) {
                let d = all[hit.mod_index];
                let what = match hit.target {
                    HitTarget::Option(o) => d.options[o].label.to_owned(),
                    HitTarget::Row(key) => format!("row {key}"),
                };
                println!("   set {:>8x} {} › {what}", hit.score, d.id);
            }
        }
    }

    #[test]
    fn normalising_text() {
        let text = Text::new("Text Colour (Centred), F3+B");
        assert_eq!(text.compact, "textcolorcenteredf3b");
        assert_eq!(text.fit("color"), Some(Fit::Word));
        assert_eq!(text.fit("textcolor"), Some(Fit::Prefix));
        assert_eq!(text.fit("olor"), Some(Fit::Substring));
        assert_eq!(text.fit("olo"), None, "short terms only match word starts");
        let mut term = String::from("colou");
        trim_partial_uk_tail(&mut term);
        assert_eq!(term, "colo");
        let mut term = String::from("armou");
        trim_partial_uk_tail(&mut term);
        assert_eq!(term, "armo");
        let mut term = String::from("color");
        trim_partial_uk_tail(&mut term);
        assert_eq!(term, "color");
    }
}
