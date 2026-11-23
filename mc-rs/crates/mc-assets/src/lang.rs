//! Language files and translation.
//!
//! 1.8.9 ships `key=value` `.lang` files, not the JSON that arrived in later versions.
//!
// java: Locale.java, Language.java, I18n.java

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{AssetError, ResourceLocation, ResourceManager};

/// The language every client falls back to, and the only one inside `1.8.9.jar`.
pub const DEFAULT_LANGUAGE: &str = "en_US";

/// The largest pack language file merged: vanilla's `en_US.lang` is 122 KB.
pub const MAX_LANG_BYTES: usize = 8 << 20;

/// A loaded translation table.
///
/// Vanilla splits this across two classes: `Locale` holds the `key -> value` map and
/// `Language` describes a selectable language. The frozen section 7 API calls the table
/// `Language`, so that is what this is; the descriptor is [`LanguageInfo`].
///
/// # Examples
/// ```
/// use mc_assets::Language;
///
/// let mut lang = Language::new("en_US");
/// lang.merge_lang_str("gui.done=Done\nchat.type.text=<%s> %s\n");
///
/// assert_eq!(lang.tr("gui.done"), "Done");
/// assert_eq!(lang.format("chat.type.text", &["Steve", "hi"]), "<Steve> hi");
/// // A key with no translation renders as itself, exactly as vanilla does.
/// assert_eq!(lang.tr("nope.missing"), "nope.missing");
/// ```
#[derive(Debug, Clone, Default)]
pub struct Language {
    code: String,
    entries: BTreeMap<String, String>,
    fallback_entries: BTreeMap<String, String>,
    unicode: bool,
}

impl Language {
    /// An empty table for the given language code.
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            entries: BTreeMap::new(),
            fallback_entries: BTreeMap::new(),
            unicode: false,
        }
    }

    /// Loads `lang/<code>.lang` for each code, from every namespace of every pack.
    ///
    /// Ordering mirrors `Locale.loadLocaleDataFiles`: codes are applied in the order given
    /// (so pass the fallback first and the selected language last), namespaces within a code,
    /// and packs bottom-to-top within a namespace. Later writes win at every level. A code
    /// that no pack provides is skipped silently, as in vanilla.
    ///
    /// The returned code is the effective `LanguageManager.getCurrentLanguage()` code. If the
    /// requested code is not advertised by the active pack stack, vanilla keeps loading that
    /// raw code's translations but exposes the advertised `en_US` descriptor as the current
    /// language. Keeping those two decisions separate is important: the descriptor supplies
    /// bidi/font direction, while `Locale` still receives the same code list.
    // java: Locale.java:27 (loadLocaleDataFiles)
    pub fn load(manager: &ResourceManager, codes: &[&str]) -> Self {
        mc_core::profile_scope!("assets.lang.load");
        let requested_code = codes.last().copied().unwrap_or(DEFAULT_LANGUAGE);
        let descriptors = manager.language_descriptors();
        let effective_code = descriptors
            .iter()
            .find(|language| language.code == requested_code)
            .or_else(|| {
                descriptors
                    .iter()
                    .find(|language| language.code == DEFAULT_LANGUAGE)
            })
            .map_or(requested_code, |language| language.code.as_str());
        let mut out = Self::new(effective_code);

        // java: StatCollector.fallbackTranslator is constructed directly from the classpath
        // `assets/minecraft/lang/en_US.lang` and is never replaced by the selected locale or a
        // user resource pack. The resource stack is bottom-to-top, so its first minecraft en_US
        // resource is the built-in table while later resources remain ordinary overrides.
        let fallback = ResourceLocation::new("minecraft", format!("lang/{DEFAULT_LANGUAGE}.lang"));
        if let Some(resource) = manager.all_resources(&fallback).into_iter().next() {
            let text = String::from_utf8_lossy(&resource.data);
            merge_lang_str_into(&mut out.fallback_entries, &text);
        }

        for code in codes {
            let path = format!("lang/{code}.lang");
            for namespace in manager.namespaces() {
                let loc = ResourceLocation::new(namespace, path.clone());
                for resource in manager.all_resources(&loc) {
                    // The table stays in memory, at ten or more times the
                    // text for short lines: a pack's file past the cap is skipped.
                    if resource.data.len() > MAX_LANG_BYTES {
                        tracing::warn!(
                            pack = %resource.pack_name,
                            location = %loc,
                            bytes = resource.data.len(),
                            "language file larger than any real one; skipped"
                        );
                        continue;
                    }
                    out.merge_lang_bytes(&resource.data);
                }
            }
        }

        out.recompute_unicode();
        tracing::debug!(code = %out.code, keys = out.entries.len(), unicode = out.unicode, "loaded language");
        out
    }

    /// The language code this table was loaded for.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Whether more than 10% of the loaded characters are outside Latin-1.
    ///
    /// Vanilla uses this to switch the font renderer to the unicode glyph pages; S15 will
    /// want it.
    // java: Locale.java:56 (checkUnicode)
    pub fn is_unicode(&self) -> bool {
        self.unicode
    }

    /// Number of translation keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the table has this key.
    // java: I18n.java (hasKey)
    pub fn has_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Iterates `(key, value)` pairs in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// The translation for `key`, or `key` itself when it is not translated.
    ///
    /// Returning the key is vanilla behaviour and downstream code relies on it — an
    /// untranslated tooltip shows `item.foo.name` rather than an empty string.
    ///
    /// The section 7 contract writes this as `fn tr(&self, key: &str) -> &str`. Returning
    /// the borrowed key on a miss means the two input lifetimes have to unify, which
    /// elision would not do; the signature is otherwise unchanged and every ordinary call
    /// site (`lang.tr("gui.done")`) compiles the same way.
    // java: Locale.java:121 (translateKeyPrivate)
    pub fn tr<'a>(&'a self, key: &'a str) -> &'a str {
        self.entries.get(key).map_or(key, String::as_str)
    }

    /// The built-in `en_US` translation for `key`, or `key` itself when it is absent.
    ///
    /// This table is captured from the bottom built-in resource before ordinary resource-pack
    /// overrides are merged. It is the `StatCollector.translateToFallback` path used when a
    /// selected chat translation contains an invalid format conversion.
    // java: StatCollector.java:33 (translateToFallback)
    pub fn tr_fallback<'a>(&'a self, key: &'a str) -> &'a str {
        self.fallback_entries.get(key).map_or(key, String::as_str)
    }

    /// The translation for `key` with `%s` / `%1$s` arguments substituted.
    ///
    /// See [`format_template`] for the exact substitution rules.
    // java: Locale.java:130 (formatMessage)
    pub fn format(&self, key: &str, args: &[&str]) -> String {
        format_template(self.tr(key), args)
    }

    /// Merges a `.lang` file's bytes. Later merges overwrite earlier ones.
    ///
    /// Decoding is lossy UTF-8, matching `IOUtils.readLines(in, UTF_8)`.
    // java: Locale.java:100 (loadLocaleData)
    pub fn merge_lang_bytes(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.merge_lang_str(&text);
    }

    /// Merges `.lang` text. Later merges overwrite earlier ones.
    pub fn merge_lang_str(&mut self, text: &str) {
        merge_lang_str_into(&mut self.entries, text);
    }

    /// Inserts or overwrites a single key. Handy for tests and for hard-coded fallbacks.
    pub fn insert(&mut self, key: impl Into<String>, value: impl AsRef<str>) {
        self.entries
            .insert(key.into(), normalize_specifiers(value.as_ref()));
    }

    /// Recomputes [`is_unicode`](Language::is_unicode) after manual edits.
    // java: Locale.java:56
    pub fn recompute_unicode(&mut self) {
        let mut wide = 0usize;
        let mut total = 0usize;
        for value in self.entries.values() {
            // Java's String.length()/charAt() operate on UTF-16 code units. Astral characters
            // therefore contribute two wide units, not one Rust `char`.
            for c in value.encode_utf16() {
                total += 1;
                if c >= 256 {
                    wide += 1;
                }
            }
        }
        self.unicode = total > 0 && (wide as f32 / total as f32) as f64 > 0.1;
    }
}

fn merge_lang_str_into(entries: &mut BTreeMap<String, String>, text: &str) {
    // BufferedReader/IOUtils.readLines recognises LF, CRLF, and a lone CR. `str::lines`
    // recognises only the first two, which made old-Mac-style resource-pack files parse as
    // one giant value.
    for line in text.split(['\r', '\n']) {
        // java: Locale.java:104 — blank lines and '#' comments are skipped.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Splitter.on('=').limit(2): everything after the first '=' is the value.
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        entries.insert(key.to_owned(), normalize_specifiers(value));
    }
}

/// Rewrites `%d`, `%f` and their width/precision forms to `%s`, preserving any explicit
/// argument index.
///
/// Vanilla does this once at load time with the regex `%(\d+\$)?[\d\.]*[df]` → `%$1s`, so
/// that every argument can be passed as a string. This reproduces that scan, including its
/// blindness to `%%`.
// java: Locale.java:20
fn normalize_specifiers(value: &str) -> String {
    let b = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut i = 0;

    while i < b.len() {
        if b[i] != b'%' {
            let start = i;
            while i < b.len() && b[i] != b'%' {
                i += 1;
            }
            out.push_str(&value[start..i]);
            continue;
        }

        // `%` — try to match (\d+\$)?[\d.]*[df]
        let mut j = i + 1;
        let digits_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        let index_group = if j > digits_start && b.get(j) == Some(&b'$') {
            let g = &value[digits_start..=j];
            j += 1;
            g
        } else {
            j = i + 1;
            ""
        };
        while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
            j += 1;
        }
        if matches!(b.get(j), Some(b'd') | Some(b'f')) {
            out.push('%');
            out.push_str(index_group);
            out.push('s');
            i = j + 1;
        } else {
            out.push('%');
            i += 1;
        }
    }

    out
}

/// Substitutes `%s` and its `%N$s` explicit-index form into `template`.
///
/// The rules match `java.util.Formatter`, which lang files depend on:
///
/// * `%%` emits a literal `%`.
/// * `%s` takes the *next* implicit argument. The implicit counter is **not** advanced by
///   explicit-index specifiers, so `"%s%2$s again %s"` with `A, B, C` yields `AB again B`.
/// * `%N$s` takes argument `N`, one-based, and can repeat.
/// * Width and precision on string conversions are applied after substitution. Precision is a
///   count of Java UTF-16 `char` units, so a limit can split a supplementary character.
/// * A malformed conversion or a missing argument makes vanilla replace the whole result with
///   `"Format error: "` followed by the untranslated template; this does the same.
///
/// # Examples
/// ```
/// use mc_assets::format_template;
///
/// assert_eq!(
///     format_template("Prefix, %s%2$s again %s and %1$s lastly %s and also %1$s again!",
///                     &["A", "B", "C", "D", "E"]),
///     "Prefix, AB again B and A lastly C and also A again!"
/// );
/// assert_eq!(format_template("100%% done", &[]), "100% done");
/// ```
pub fn format_template(template: &str, args: &[&str]) -> String {
    try_format_template(template, args).unwrap_or_else(|()| format!("Format error: {template}"))
}

/// How much text a format may add to its template (arguments and
/// padding). A language file's template decides it: `%2000000000s` asked for a 2 GB string, and
/// `%1$s` repeated over a long argument (a book author, an item name: server text) multiplied it.
/// Past this the result is vanilla's `Format error:` text; nothing a real language file holds
/// comes near it.
const MAX_FORMAT_EXPANSION: usize = 1 << 20;

fn try_format_template(template: &str, args: &[&str]) -> Result<String, ()> {
    let b = template.as_bytes();
    let budget = template.len().saturating_add(MAX_FORMAT_EXPANSION);
    let mut out = String::with_capacity(template.len() + 16);
    let mut i = 0;
    let mut next_implicit = 0usize;
    let mut previous_argument = None;

    while i < b.len() {
        if b[i] != b'%' {
            let start = i;
            while i < b.len() && b[i] != b'%' {
                i += 1;
            }
            out.push_str(&template[start..i]);
            continue;
        }

        let mut j = i + 1;

        // Explicit one-based argument index: digits immediately followed by '$'.
        let digits_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        let explicit: Option<usize> = if j > digits_start && b.get(j) == Some(&b'$') {
            let parsed = template[digits_start..j].parse::<usize>().map_err(|_| ())?;
            j += 1;
            Some(parsed)
        } else {
            j = i + 1;
            None
        };

        let mut left_justify = false;
        let mut reuse_previous = false;
        while matches!(b.get(j), Some(b'-' | b'<')) {
            match b[j] {
                b'-' if !left_justify => left_justify = true,
                b'<' if !reuse_previous => reuse_previous = true,
                _ => return Err(()),
            }
            j += 1;
        }

        let width_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        // A leading zero is Formatter's `0` flag, not part of the width, and that flag is
        // illegal for string conversions.
        if b.get(width_start) == Some(&b'0') {
            return Err(());
        }
        let width = if j > width_start {
            Some(template[width_start..j].parse::<usize>().map_err(|_| ())?)
        } else {
            None
        };

        let precision = if b.get(j) == Some(&b'.') {
            j += 1;
            let precision_start = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j == precision_start {
                return Err(());
            }
            Some(
                template[precision_start..j]
                    .parse::<usize>()
                    .map_err(|_| ())?,
            )
        } else {
            None
        };

        match b.get(j) {
            Some(b'%')
                if explicit.is_none_or(|index| index >= 1)
                    && !reuse_previous
                    && precision.is_none() =>
            {
                if left_justify && width.is_none() {
                    return Err(());
                }
                push_padded(&mut out, "%", width, left_justify, budget)?;
                i = j + 1;
            }
            Some(b'n')
                if explicit.is_none_or(|index| index >= 1)
                    && !left_justify
                    && !reuse_previous
                    && width.is_none()
                    && precision.is_none() =>
            {
                out.push_str(if cfg!(windows) { "\r\n" } else { "\n" });
                i = j + 1;
            }
            Some(conversion @ (b's' | b'S')) => {
                if left_justify && width.is_none() {
                    return Err(());
                }
                if reuse_previous && explicit.is_some() {
                    return Err(());
                }

                let arg_index = if reuse_previous {
                    previous_argument
                } else {
                    match explicit {
                        Some(n) if n >= 1 => Some(n - 1),
                        Some(_) => None,
                        None => {
                            let n = next_implicit;
                            next_implicit += 1;
                            Some(n)
                        }
                    }
                };
                let arg_index = arg_index.ok_or(())?;
                let value = args.get(arg_index).ok_or(())?;
                previous_argument = Some(arg_index);

                let mut value = if *conversion == b'S' {
                    value.to_uppercase()
                } else {
                    (*value).to_owned()
                };
                if let Some(limit) = precision {
                    // `Formatter` applies an `s` precision through String.length()/substring,
                    // both of which count UTF-16 code units. Keep this crate's established
                    // lossy boundary for a lone surrogate: Rust strings cannot represent the
                    // half that Java can leave behind, so it becomes U+FFFD at the API edge.
                    value = String::from_utf16_lossy(
                        &value.encode_utf16().take(limit).collect::<Vec<_>>(),
                    );
                }
                push_padded(&mut out, &value, width, left_justify, budget)?;
                i = j + 1;
            }
            _ => return Err(()),
        }
    }

    Ok(out)
}

/// Appends `value` padded to `width`, or `Err` if `out` would grow past `budget` bytes.
fn push_padded(
    out: &mut String,
    value: &str,
    width: Option<usize>,
    left_justify: bool,
    budget: usize,
) -> Result<(), ()> {
    let padding = width
        .unwrap_or(0)
        .saturating_sub(value.encode_utf16().count());
    if out
        .len()
        .saturating_add(padding)
        .saturating_add(value.len())
        > budget
    {
        return Err(());
    }
    if !left_justify {
        out.extend(std::iter::repeat_n(' ', padding));
    }
    out.push_str(value);
    if left_justify {
        out.extend(std::iter::repeat_n(' ', padding));
    }
    Ok(())
}

/// A selectable language, as declared in the `"language"` section of `pack.mcmeta`.
// java: Language.java
#[derive(Debug, Clone)]
pub struct LanguageInfo {
    /// Language code, e.g. `en_GB`.
    pub code: String,
    /// Region name, e.g. `United Kingdom`.
    pub region: String,
    /// Display name, e.g. `English`.
    pub name: String,
    /// Whether the script runs right-to-left.
    pub bidirectional: bool,
}

// java: Language.java:33-49
//
// Vanilla treats the language code as the identity of a descriptor.  This matters when a
// resource-pack stack advertises the same code more than once: a HashSet/TreeSet must not
// distinguish entries merely because a pack supplied different display text.
impl PartialEq for LanguageInfo {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Eq for LanguageInfo {}

impl std::hash::Hash for LanguageInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.code.hash(state);
    }
}

impl PartialOrd for LanguageInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LanguageInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.code.cmp(&other.code)
    }
}

impl std::fmt::Display for LanguageInfo {
    // java: Language.java:29
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name, self.region)
    }
}

/// Parses the `"language"` section of a `pack.mcmeta`, sorted by code.
///
/// Vanilla rejects the whole section when any entry has the wrong shape or is missing a
/// required, non-empty `region` or `name`.  As with Gson's `JsonUtils`, primitive region/name
/// values are stringified and primitive bidirectional values are accepted.
// java: LanguageMetadataSectionSerializer.java
pub fn parse_language_sections(mcmeta: &[u8]) -> Result<Vec<LanguageInfo>, AssetError> {
    let root: Value =
        serde_json::from_slice(mcmeta).map_err(|e| AssetError::json("pack.mcmeta", e))?;
    // The Java metadata entry point receives a JsonObject.  Reject a top-level array, scalar,
    // or null instead of silently treating it as metadata with no language section.
    let root = root
        .as_object()
        .ok_or(AssetError::BadPackMeta("pack.mcmeta must be an object"))?;
    let Some(raw_section) = root.get("language") else {
        return Ok(Vec::new());
    };
    let section = raw_section
        .as_object()
        .ok_or(AssetError::BadPackMeta("invalid \"language\" section"))?;

    let mut out = Vec::with_capacity(section.len());
    for (code, entry) in section {
        let entry = entry
            .as_object()
            .ok_or(AssetError::BadPackMeta("invalid language entry"))?;
        // JsonUtils.getString accepts every JSON primitive (not just JSON strings), because
        // Gson's JsonPrimitive#getAsString stringifies numbers and booleans too.  Preserve that
        // slightly surprising 1.8.9 behavior while still rejecting null, arrays, and objects.
        let region = entry
            .get("region")
            .and_then(gson_primitive_string)
            .filter(|region| !region.is_empty())
            .ok_or(AssetError::BadPackMeta(
                "language entry has an invalid region",
            ))?;
        let name = entry
            .get("name")
            .and_then(gson_primitive_string)
            .filter(|name| !name.is_empty())
            .ok_or(AssetError::BadPackMeta(
                "language entry has an invalid name",
            ))?;
        let bidirectional = match entry.get("bidirectional") {
            None => false,
            Some(value) => match gson_primitive_bool(value) {
                Some(value) => value,
                None => {
                    return Err(AssetError::BadPackMeta(
                        "language entry has an invalid bidirectional flag",
                    ));
                }
            },
        };
        out.push(LanguageInfo {
            code: code.clone(),
            region,
            name,
            bidirectional,
        });
    }
    out.sort();
    Ok(out)
}

/// Gson's `JsonPrimitive#getAsString`, for the subset of primitives accepted by the language
/// metadata serializer.
fn gson_primitive_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Gson's `JsonPrimitive#getAsBoolean`: booleans retain their value, while every other
/// primitive is parsed through `Boolean.parseBoolean(getAsString())`.
fn gson_primitive_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::String(value) => Some(value.eq_ignore_ascii_case("true")),
        Value::Number(_) => Some(false),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};

    use crate::ResourcePack;

    struct TestLangPack {
        name: &'static str,
        files: BTreeMap<String, Vec<u8>>,
        languages: Vec<LanguageInfo>,
    }

    impl TestLangPack {
        fn boxed(name: &'static str, files: &[(&str, &str)]) -> Box<dyn ResourcePack> {
            Box::new(Self {
                name,
                files: files
                    .iter()
                    .map(|(path, text)| ((*path).to_owned(), text.as_bytes().to_vec()))
                    .collect(),
                languages: Vec::new(),
            })
        }

        fn boxed_with_languages(
            name: &'static str,
            files: &[(&str, &str)],
            languages: Vec<LanguageInfo>,
        ) -> Box<dyn ResourcePack> {
            Box::new(Self {
                name,
                files: files
                    .iter()
                    .map(|(path, text)| ((*path).to_owned(), text.as_bytes().to_vec()))
                    .collect(),
                languages,
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

        fn language_descriptors(&self) -> &[LanguageInfo] {
            &self.languages
        }
    }

    #[test]
    fn parses_key_equals_value() {
        let mut l = Language::new("en_US");
        l.merge_lang_str("gui.done=Done\r\n#a comment\n\ngui.cancel=Cancel\n");
        assert_eq!(l.tr("gui.done"), "Done");
        assert_eq!(l.tr("gui.cancel"), "Cancel");
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn lone_carriage_returns_are_line_endings_like_java_readline() {
        let mut l = Language::new("en_US");
        l.merge_lang_str("a=one\rb=two\r#comment\rc=three");
        assert_eq!(l.tr("a"), "one");
        assert_eq!(l.tr("b"), "two");
        assert_eq!(l.tr("c"), "three");
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn only_the_first_equals_splits() {
        let mut l = Language::new("en_US");
        l.merge_lang_str("key=a=b=c");
        assert_eq!(l.tr("key"), "a=b=c");
    }

    #[test]
    fn missing_key_returns_the_key() {
        let l = Language::new("en_US");
        assert_eq!(l.tr("item.nope.name"), "item.nope.name");
        assert_eq!(l.format("item.nope.name", &["x"]), "item.nope.name");
    }

    #[test]
    fn later_merges_override() {
        let mut l = Language::new("en_US");
        l.merge_lang_str("gui.done=Done");
        l.merge_lang_str("gui.done=Fertig");
        assert_eq!(l.tr("gui.done"), "Fertig");
    }

    #[test]
    fn built_in_english_fallback_is_separate_from_resource_pack_overrides() {
        let mut manager = ResourceManager::new();
        manager.push_pack(TestLangPack::boxed(
            "jar",
            &[(
                "minecraft:lang/en_US.lang",
                "probe=Built in %s\nnumeric=Took %d ms",
            )],
        ));
        manager.push_pack(TestLangPack::boxed(
            "resource pack",
            &[(
                "minecraft:lang/en_US.lang",
                "probe=Broken %x\nnumeric=Pack %.2f seconds",
            )],
        ));

        let language = Language::load(&manager, &[DEFAULT_LANGUAGE]);
        assert_eq!(language.tr("probe"), "Broken %x");
        assert_eq!(language.tr_fallback("probe"), "Built in %s");
        assert_eq!(language.tr("numeric"), "Pack %s seconds");
        assert_eq!(language.tr_fallback("numeric"), "Took %s ms");
        assert_eq!(language.tr_fallback("missing.key"), "missing.key");
    }

    /// A pack's language file past [`MAX_LANG_BYTES`] (millions of short
    /// keys stay in memory at ten or more times the text) is skipped; the others still merge.
    #[test]
    fn an_oversized_pack_language_file_is_skipped() {
        let mut huge = String::with_capacity(MAX_LANG_BYTES + 64);
        let mut n = 0u32;
        while huge.len() <= MAX_LANG_BYTES {
            huge.push_str(&format!("k{n}=v\n"));
            n += 1;
        }
        huge.push_str("probe=Huge\n");
        let mut manager = ResourceManager::new();
        manager.push_pack(TestLangPack::boxed(
            "jar",
            &[("minecraft:lang/en_US.lang", "probe=Built in\n")],
        ));
        manager.push_pack(TestLangPack::boxed(
            "small pack",
            &[("minecraft:lang/en_US.lang", "other=Small\n")],
        ));
        manager.push_pack(TestLangPack::boxed(
            "huge pack",
            &[("minecraft:lang/en_US.lang", huge.as_str())],
        ));
        let language = Language::load(&manager, &[DEFAULT_LANGUAGE]);
        assert_eq!(
            language.tr("probe"),
            "Built in",
            "the huge file is not merged"
        );
        assert_eq!(language.tr("other"), "Small");
        assert_eq!(language.tr("k1"), "k1");
    }

    #[test]
    fn unknown_selection_uses_the_advertised_english_descriptor_code() {
        let english = LanguageInfo {
            code: DEFAULT_LANGUAGE.to_owned(),
            region: "United States".to_owned(),
            name: "English".to_owned(),
            bidirectional: true,
        };
        let mut manager = ResourceManager::new();
        manager.push_pack(TestLangPack::boxed_with_languages(
            "described",
            &[
                ("minecraft:lang/en_US.lang", "probe=English\n"),
                ("minecraft:lang/unknown.lang", "probe=Unknown\n"),
            ],
            vec![english],
        ));

        // Java still loads the raw current code, but LanguageManager.getCurrentLanguage()
        // resolves its descriptor to advertised en_US. The Rust table exposes that same
        // effective code so the font-direction bridge can apply the descriptor's bidi bit.
        let language = Language::load(&manager, &[DEFAULT_LANGUAGE, "unknown"]);
        assert_eq!(language.code(), DEFAULT_LANGUAGE);
        assert_eq!(language.tr("probe"), "Unknown");
        let descriptor = manager
            .language_descriptors()
            .into_iter()
            .find(|language| language.code == DEFAULT_LANGUAGE)
            .expect("advertised en_US descriptor");
        assert!(descriptor.bidirectional);
    }

    #[test]
    fn numeric_specifiers_are_normalised_at_load() {
        // java: Locale.java:111 — `%(\d+\$)?[\d\.]*[df]` -> `%$1s`
        let mut l = Language::new("en_US");
        l.merge_lang_str("k=Took %d ms and %.2f seconds, %2$d again");
        assert_eq!(l.tr("k"), "Took %s ms and %s seconds, %2$s again");
        // Verified against JDK 21: String.format on the normalised string.
        assert_eq!(
            l.format("k", &["7", "1.5"]),
            "Took 7 ms and 1.5 seconds, 1.5 again"
        );
    }

    #[test]
    fn implicit_counter_ignores_explicit_indices() {
        // Golden vector: `translation.test.complex` from the real en_US.lang, with the
        // output taken from JDK 21's String.format.
        let t = "Prefix, %s%2$s again %s and %1$s lastly %s and also %1$s again!";
        assert_eq!(
            format_template(t, &["A", "B", "C", "D", "E"]),
            "Prefix, AB again B and A lastly C and also A again!"
        );
    }

    #[test]
    fn explicit_positional_arguments() {
        // java: demo.help.movement from en_US.lang
        assert_eq!(
            format_template(
                "Use %1$s, %2$s, %3$s, %4$s and the mouse to move around",
                &["W", "A", "S", "D"]
            ),
            "Use W, A, S, D and the mouse to move around"
        );
    }

    #[test]
    fn double_percent_is_a_literal() {
        assert_eq!(format_template("100%% done %s", &["x"]), "100% done x");
    }

    #[test]
    fn missing_arguments_replace_the_whole_message_with_vanillas_format_error() {
        assert_eq!(
            format_template("a %s b %2$s c", &[]),
            "Format error: a %s b %2$s c"
        );
        assert_eq!(format_template("a %s b", &["1", "2"]), "a 1 b");
    }

    #[test]
    fn malformed_formats_match_the_real_en_us_test_vectors() {
        // These are Mojang's deliberately-invalid entries in the real en_US.lang. Java's
        // Locale catches the IllegalFormatException and prefixes the original template.
        assert_eq!(format_template("hi %", &["A"]), "Format error: hi %");
        assert_eq!(format_template("hi %  s", &["A"]), "Format error: hi %  s");
        assert_eq!(
            format_template("50% off, %s!", &["now"]),
            "Format error: 50% off, %s!"
        );
    }

    #[test]
    fn string_width_precision_case_and_relative_index_match_formatter() {
        assert_eq!(
            format_template("[%5s][%-5s]", &["x", "y"]),
            "[    x][y    ]"
        );
        assert_eq!(format_template("%.3s", &["abcdef"]), "abc");
        assert_eq!(format_template("%S", &["Abc"]), "ABC");
        assert_eq!(format_template("%2$s %<s %s", &["A", "B"]), "B B A");

        // Negative controls: Java rejects these rather than treating all punctuation as a
        // harmless flag, which the old scanner did.
        assert_eq!(format_template("% s", &["x"]), "Format error: % s");
        assert_eq!(format_template("%05s", &["x"]), "Format error: %05s");
        assert_eq!(format_template("%-s", &["x"]), "Format error: %-s");
    }

    #[test]
    fn percent_and_platform_newline_conversions_accept_explicit_indices() {
        // java.util.Formatter permits an argument index on conversions that consume no
        // argument.  The index is syntactically checked but never looked up.
        assert_eq!(format_template("%1$5%", &[]), "    %");
        assert_eq!(format_template("%999$%", &[]), "%");
        assert_eq!(
            format_template("%1$n", &[]),
            if cfg!(windows) { "\r\n" } else { "\n" }
        );

        // Zero and an overflowing explicit index are still malformed even though these
        // conversions do not consume an argument.
        assert_eq!(format_template("%0$%", &[]), "Format error: %0$%");
        assert_eq!(
            format_template("%999999999999999999999999999999$%", &[]),
            "Format error: %999999999999999999999999999999$%"
        );
    }

    #[test]
    fn string_precision_counts_utf16_units_like_java_formatter() {
        // JDK 21: String.format("[%.1s][%.2s][%.3s][%.4s]", value, value, value, value)
        // has UTF-16 units `[a][a, high-surrogate][a, surrogate-pair][a, pair, b]`.
        // The public Rust string necessarily exposes the dangling surrogate as U+FFFD.
        assert_eq!(
            format_template(
                "[%.1s][%.2s][%.3s][%.4s]",
                &["a\u{1f600}b", "a\u{1f600}b", "a\u{1f600}b", "a\u{1f600}b"],
            ),
            "[a][a\u{fffd}][a\u{1f600}][a\u{1f600}b]"
        );

        // This reaches the normal UI translation path: a resource pack can narrow a signed
        // book's `book.byAuthor` text. Scalar-count truncation used to retain the full emoji.
        let mut language = Language::new("custom");
        language.insert("book.byAuthor", "By %.2s");
        assert_eq!(
            language.format("book.byAuthor", &["a\u{1f600}b"]),
            "By a\u{fffd}"
        );
    }

    #[test]
    fn multibyte_text_survives_formatting() {
        assert_eq!(format_template("é%sü", &["—"]), "é—ü");
    }

    #[test]
    fn unicode_ratio_matches_vanilla_rule() {
        let mut ascii = Language::new("en_US");
        ascii.merge_lang_str("a=hello world");
        ascii.recompute_unicode();
        assert!(!ascii.is_unicode());

        let mut wide = Language::new("zh_CN");
        wide.merge_lang_str("a=你好世界你好世界");
        wide.recompute_unicode();
        assert!(wide.is_unicode());
    }

    #[test]
    fn unicode_ratio_counts_utf16_units_like_java() {
        let mut l = Language::new("emoji");
        // Java sees the astral character as two wide UTF-16 units: 2 / 11 > 10%. Counting
        // Rust scalar values instead gives 1 / 10 == 10%, the old false result.
        l.merge_lang_str("a=😀aaaaaaaaa");
        l.recompute_unicode();
        assert!(l.is_unicode());

        let mut boundary = Language::new("emoji");
        boundary.merge_lang_str("a=😀aaaaaaaaaaaaaaaaaaa");
        boundary.recompute_unicode();
        assert!(!boundary.is_unicode(), "2 / 21 is below 10%");
    }

    #[test]
    fn language_sections_parse() {
        let meta = br#"{"pack":{"pack_format":1,"description":"d"},
            "language":{
                "en_GB":{"region":"United Kingdom","name":"English"},
                "he_IL":{"region":"Israel","name":"Hebrew","bidirectional":true}
            }}"#;
        let langs = parse_language_sections(meta).unwrap();
        assert_eq!(langs.len(), 2);
        assert_eq!(langs[0].code, "en_GB");
        assert_eq!(langs[0].to_string(), "English (United Kingdom)");
        assert!(langs[1].bidirectional);
    }

    #[test]
    fn language_info_identity_and_order_use_only_the_code() {
        use std::collections::{BTreeSet, HashSet};

        let english_a = LanguageInfo {
            code: "en_US".into(),
            region: "United States".into(),
            name: "English".into(),
            bidirectional: false,
        };
        let english_b = LanguageInfo {
            code: "en_US".into(),
            region: "different metadata".into(),
            name: "Different English".into(),
            bidirectional: true,
        };
        let arabic = LanguageInfo {
            code: "ar_SA".into(),
            region: "Saudi Arabia".into(),
            name: "Arabic".into(),
            bidirectional: true,
        };

        assert_eq!(english_a, english_b);
        assert_eq!(english_a.cmp(&english_b), std::cmp::Ordering::Equal);
        assert!(arabic < english_a);

        let mut ordered = BTreeSet::new();
        ordered.insert(english_a.clone());
        ordered.insert(english_b.clone());
        assert_eq!(ordered.len(), 1);

        let mut hashed = HashSet::new();
        hashed.insert(english_a);
        hashed.insert(english_b);
        assert_eq!(hashed.len(), 1);
    }

    #[test]
    fn language_metadata_uses_gson_primitive_coercions() {
        let meta = br#"{"language":{
            "numeric":{"region":42,"name":true,"bidirectional":"TrUe"},
            "boolean":{"region":false,"name":3.5,"bidirectional":0}
        }}"#;
        let langs = parse_language_sections(meta).unwrap();
        assert_eq!(langs.len(), 2);
        assert_eq!(langs[0].code, "boolean");
        assert_eq!(langs[0].region, "false");
        assert_eq!(langs[0].name, "3.5");
        assert!(!langs[0].bidirectional);
        assert_eq!(langs[1].code, "numeric");
        assert_eq!(langs[1].region, "42");
        assert_eq!(langs[1].name, "true");
        assert!(langs[1].bidirectional);
    }

    #[test]
    fn one_malformed_language_rejects_the_whole_section_like_vanilla() {
        let bad_entries: &[&[u8]] = &[
            br#"{"language":{"good":{"region":"Here","name":"Good"},"bad":{"region":"","name":"Bad"}}}"#,
            br#"{"language":{"bad":{"region":"Here"}}}"#,
            br#"{"language":{"bad":"not an object"}}"#,
            br#"{"language":{"bad":{"region":"Here","name":"Bad","bidirectional":[]}}}"#,
            br#"{"language":[]}"#,
            br#"[]"#,
        ];
        for meta in bad_entries {
            assert!(
                matches!(
                    parse_language_sections(meta),
                    Err(AssetError::BadPackMeta(_))
                ),
                "malformed language metadata must not contribute its good siblings: {}",
                String::from_utf8_lossy(meta)
            );
        }
    }

    #[test]
    fn language_sections_absent_is_not_an_error() {
        assert!(parse_language_sections(br#"{"pack":{"pack_format":1}}"#)
            .unwrap()
            .is_empty());
    }
}
