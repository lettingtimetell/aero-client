//! The 1.8.9 `JsonToNBT` (Mojangson) text parser.
//!
//! This is deliberately **not** modern SNBT.  It exists for the string payload of
//! chat `show_item` and `show_entity` hover events, which in 1.8.9 is parsed by
//! `JsonToNBT.getTagFromJson`.  In particular, bracketed integer sequences become
//! `TAG_Int_Array`, numeric suffix handling retains Mojang's literal `|` regex typo,
//! and heterogeneous list elements are silently discarded.
//!
//! The splitting is a line-for-line port of `JsonToNBT`'s scanners rather than a grammar,
//! because the game's own output depends on it: `NBTTagList.toString` writes `[0:a,1:b]`,
//! which servers put into item hovers verbatim, and only `JsonToNBT`'s "anything before a
//! colon is a name" rule reads that back.  The same scanners give 1.8.9's leniencies (a
//! trailing comma, an empty value, unquoted text with spaces, keys that keep their quotes)
//! and its rejections (`{ }`, `{a:1, }`).  `tests/jsontonbt_oracle.rs` checks the port
//! against expectations produced by the Java class itself.

use std::fmt;

use crate::{Nbt, NbtCompound};

/// A malformed 1.8.9 Mojangson value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnbtError {
    message: String,
    position: usize,
}

impl SnbtError {
    fn new(message: impl Into<String>, position: usize) -> Self {
        Self {
            message: message.into(),
            position,
        }
    }

    /// The byte offset at which parsing stopped.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }
}

impl fmt::Display for SnbtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mojangson at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for SnbtError {}

/// Parse the compound-only form accepted by 1.8.9's
/// `JsonToNBT.getTagFromJson`.
///
/// The vanilla entry point rejects a root list or scalar even though nested values
/// can be any tag type.  Prefer this function for a chat hover payload.
pub fn parse_snbt_compound(input: &str) -> Result<NbtCompound, SnbtError> {
    let chars: Vec<char> = input.chars().collect();
    let parser = JsonToNbt { root: &chars };
    // java: `jsonString = jsonString.trim()`, then the '{' and single-top-tag checks.
    let json = java_trim(&chars);
    if json.first() != Some(&'{') {
        return Err(parser.error(json, "Invalid tag encountered, expected '{' as first char."));
    }
    if parser.count_top_tags(json)? != 1 {
        return Err(parser.error(json, "Encountered multiple top tags, only one expected"));
    }
    match parser.value(json, 0)? {
        Nbt::Compound(compound) => Ok(compound),
        // `func_150316_a` turns every '{'-prefixed source into a compound.
        _ => Err(parser.error(json, "expected compound root")),
    }
}

/// Parse one Mojangson tag.
///
/// This wider helper is useful for tests and nested consumers; 1.8.9's public
/// `getTagFromJson` entry point itself is compound-only, so hover callers should
/// use [`parse_snbt_compound`].  It applies the same whole-source bracket and quote
/// validation, then parses the value the way `JsonToNBT` parses a nested one.
pub fn parse_snbt(input: &str) -> Result<Nbt, SnbtError> {
    let chars: Vec<char> = input.chars().collect();
    let parser = JsonToNbt { root: &chars };
    let json = java_trim(&chars);
    if parser.count_top_tags(json)? != 1 {
        return Err(parser.error(json, "Encountered multiple top tags, only one expected"));
    }
    parser.value(json, 0)
}

/// Java has no nesting limit here: a deep enough payload overflows the thread's stack, which no
/// caller catches.  This bound turns that into an ordinary parse error.
const MAX_DEPTH: usize = 512;

/// `JsonToNBT`'s static methods over the `char`s of the whole input.
///
/// Java indexes UTF-16 code units, but every character the algorithm compares against is ASCII
/// or BMP whitespace, so a supplementary character (two Java `char`s) splits exactly like one
/// Rust `char`.  Every Java `substring` here is a sub-slice of `root`, which is how an error
/// reports its byte offset.
struct JsonToNbt<'a> {
    root: &'a [char],
}

impl JsonToNbt<'_> {
    /// java: `NBTException`; `at` is the (sub)string Java was looking at when it threw.
    fn error(&self, at: &[char], message: impl Into<String>) -> SnbtError {
        let start = (at.as_ptr() as usize).saturating_sub(self.root.as_ptr() as usize)
            / std::mem::size_of::<char>();
        let position = self.root[..start.min(self.root.len())]
            .iter()
            .map(|ch| ch.len_utf8())
            .sum();
        SnbtError::new(message, position)
    }

    /// java: `func_150310_b`: check quotes and brackets over the whole source and count its
    /// top-level tags (a source with no brackets at all is one tag).
    fn count_top_tags(&self, source: &[char]) -> Result<usize, SnbtError> {
        let mut count = 0;
        let mut quoted = false;
        let mut stack = Vec::new();
        for (index, &ch) in source.iter().enumerate() {
            if ch == '"' {
                if is_escaped(source, index) {
                    if !quoted {
                        return Err(self.error(source, "Illegal use of \\\""));
                    }
                } else {
                    quoted = !quoted;
                }
            } else if !quoted {
                if ch == '{' || ch == '[' {
                    if stack.is_empty() {
                        count += 1;
                    }
                    stack.push(ch);
                } else if ch == '}' && stack.pop() != Some('{') {
                    return Err(self.error(source, "Unbalanced curly brackets {}"));
                } else if ch == ']' && stack.pop() != Some('[') {
                    return Err(self.error(source, "Unbalanced square brackets []"));
                }
            }
        }
        if quoted {
            return Err(self.error(source, "Unbalanced quotation"));
        }
        if !stack.is_empty() {
            return Err(self.error(source, "Unbalanced brackets"));
        }
        Ok(if count == 0 && !source.is_empty() {
            1
        } else {
            count
        })
    }

    /// java: `func_150316_a(name, value)`, then `Any.parse()` on what it built.
    ///
    /// Java builds the whole `Any` tree before parsing any of it, but parsing can only fail where
    /// the game itself crashes, so one recursive pass accepts and rejects the same inputs.
    fn value(&self, source: &[char], depth: usize) -> Result<Nbt, SnbtError> {
        let source = java_trim(source);
        match source.first() {
            // `depth` counts the containers already open around this value.
            Some('{' | '[') if depth >= MAX_DEPTH => Err(self.error(source, "nesting too deep")),
            Some('{') => self.compound(source, depth).map(Nbt::Compound),
            Some('[') if !is_int_array_source(source) => self.list(source, depth),
            _ => primitive(source).ok_or_else(|| self.error(source, "lone quote")),
        }
    }

    /// java: `func_150316_a`'s compound branch and `JsonToNBT.Compound.parse`.
    ///
    /// The loop is written out here and in [`Self::list`] (rather than shared through a
    /// closure) so that each nesting level costs only this frame and [`Self::value`]'s: the
    /// recursion must fit [`MAX_DEPTH`] levels into the UI thread's 1 MiB stack in any build.
    fn compound(&self, source: &[char], depth: usize) -> Result<NbtCompound, SnbtError> {
        let mut compound = NbtCompound::new();
        let mut rest = self.body(source)?;
        while !rest.is_empty() {
            let (entry, next) = self.step(rest, true)?;
            if let Some((name, value)) = entry {
                let tag = self.value(value, depth + 1)?;
                // `setTag`: a repeated name keeps its last value.
                compound.insert(name.iter().collect::<String>(), tag);
            }
            rest = next;
        }
        Ok(compound)
    }

    /// java: `func_150316_a`'s list branch and `JsonToNBT.List.parse`.  An entry's name (the
    /// index `NBTTagList.toString` writes before a colon) is read and ignored.
    fn list(&self, source: &[char], depth: usize) -> Result<Nbt, SnbtError> {
        let mut items: Vec<Nbt> = Vec::new();
        let mut rest = self.body(source)?;
        while !rest.is_empty() {
            let (entry, next) = self.step(rest, false)?;
            if let Some((_index, value)) = entry {
                let tag = self.value(value, depth + 1)?;
                // `NBTTagList.appendTag` logs and drops a tag whose type differs from the first.
                if items
                    .first()
                    .is_none_or(|first| first.tag_type() == tag.tag_type())
                {
                    items.push(tag);
                }
            }
            rest = next;
        }
        Ok(Nbt::list(items))
    }

    /// java: `substring(1, length() - 1)`, which throws StringIndexOutOfBoundsException for a
    /// lone bracket; nothing catches that.
    fn body<'s>(&self, source: &'s [char]) -> Result<&'s [char], SnbtError> {
        if source.len() < 2 {
            return Err(self.error(source, "lone bracket"));
        }
        Ok(&source[1..source.len() - 1])
    }

    /// One turn of the loop both branches of `func_150316_a` share: cut the next entry off the
    /// front, split it into name and value unless it is empty, and check the one separator
    /// character after it. Returns what is left (empty once the entry ran to the end).
    ///
    /// Java builds the entry's value before it checks the separator; both orders reject the
    /// same inputs, and checking first keeps the separator error off the recursion.
    #[allow(clippy::type_complexity)]
    fn step<'s>(
        &self,
        rest: &'s [char],
        compound: bool,
    ) -> Result<(Option<(&'s [char], &'s [char])>, &'s [char]), SnbtError> {
        let entry = self.next_entry(rest, compound)?;
        let split = if entry.is_empty() {
            None
        } else {
            Some(self.split_entry(entry, !compound)?)
        };
        let Some(&separator) = rest.get(entry.len()) else {
            return Ok((split, &rest[rest.len()..]));
        };
        if !matches!(separator, ',' | '{' | '}' | '[' | ']') {
            return Err(self.error(
                &rest[entry.len()..],
                format!("Unexpected token '{separator}'"),
            ));
        }
        Ok((split, &rest[entry.len() + 1..]))
    }

    /// java: `func_150314_a(s, isCompound)`: where the next entry ends.  A compound entry needs a
    /// name separator ahead of any comma.  A list entry keeps its colon only when a comma comes
    /// later: Java compares the raw indices, so a missing comma (-1) also drops the colon.
    fn next_entry<'s>(&self, rest: &'s [char], compound: bool) -> Result<&'s [char], SnbtError> {
        // Java runs both `func_150312_a` scans over the whole remaining
        // text for every entry, which is quadratic: a 32767-character list of short entries in
        // a chat hover took over a second per parse, and the hover parses on every frame. Only
        // the separators before the first unquoted comma or bracket can change the outcome, so
        // the scan stops there and the answers are exactly the two full scans'.
        let (mut colon, comma) = first_separators(rest);
        if compound {
            if colon == -1 {
                // The two errors differ only in their message; the full scan picks it.
                if comma != -1 && find_unquoted(rest, ':') != -1 {
                    return Err(self.error(rest, "Name error"));
                }
                return Err(self.error(rest, "Unable to locate name/value separator"));
            }
        } else if comma == -1 {
            // Java compares raw indices, so a missing comma (-1) drops any colon.
            colon = -1;
        }
        self.entry_end(rest, colon)
    }

    /// java: `func_179269_a(s, separatorIndex)`: scan from just past the separator.  The entry
    /// ends at a comma outside brackets and quotes, or, once a value that *starts* with a quote
    /// has closed it, at the next non-whitespace character.
    fn entry_end<'s>(&self, rest: &'s [char], separator: isize) -> Result<&'s [char], SnbtError> {
        let mut stack = Vec::new();
        let mut quoted = false;
        let mut quote_first = false;
        let mut seen_text = false;
        let mut quote_end = 0;
        let start = usize::try_from(separator + 1).unwrap_or(0);
        for (index, &ch) in rest.iter().enumerate().skip(start) {
            if ch == '"' {
                if is_escaped(rest, index) {
                    if !quoted {
                        return Err(self.error(rest, "Illegal use of \\\""));
                    }
                } else {
                    quoted = !quoted;
                    if quoted && !seen_text {
                        quote_first = true;
                    }
                    if !quoted {
                        quote_end = index;
                    }
                }
            } else if !quoted {
                if ch == '{' || ch == '[' {
                    stack.push(ch);
                } else if ch == '}' && stack.pop() != Some('{') {
                    return Err(self.error(rest, "Unbalanced curly brackets {}"));
                } else if ch == ']' && stack.pop() != Some('[') {
                    return Err(self.error(rest, "Unbalanced square brackets []"));
                } else if ch == ',' && stack.is_empty() {
                    return Ok(&rest[..index]);
                }
            }
            if !java_is_whitespace(ch) {
                if !quoted && quote_first && quote_end != index {
                    return Ok(&rest[..=quote_end]);
                }
                seen_text = true;
            }
        }
        Ok(rest)
    }

    /// java: `func_179270_a(s, isList)`: `func_150313_b` gives the name and `func_150311_c` the
    /// value.  The name keeps any quotes: 1.8.9 has no quoted-name syntax.
    fn split_entry<'s>(
        &self,
        entry: &'s [char],
        list: bool,
    ) -> Result<(&'s [char], &'s [char]), SnbtError> {
        let entry = if list { java_trim(entry) } else { entry };
        if list && matches!(entry.first(), Some('{' | '[')) {
            return Ok((&entry[..0], entry));
        }
        match usize::try_from(find_unquoted(entry, ':')) {
            Ok(colon) => Ok((java_trim(&entry[..colon]), java_trim(&entry[colon + 1..]))),
            Err(_) if list => Ok((&entry[..0], entry)),
            Err(_) => Err(self.error(entry, "Unable to locate name/value separator")),
        }
    }
}

/// java: `func_150312_a(s, ch)`: the index of the first `target` outside quotes, or -1, which is
/// also the answer as soon as an unquoted `{` or `[` comes first.
fn find_unquoted(source: &[char], target: char) -> isize {
    let mut outside = true;
    for (index, &ch) in source.iter().enumerate() {
        if ch == '"' {
            if !is_escaped(source, index) {
                outside = !outside;
            }
        } else if outside {
            if ch == target {
                return isize::try_from(index).unwrap_or(isize::MAX);
            }
            if ch == '{' || ch == '[' {
                return -1;
            }
        }
    }
    -1
}

/// `(find_unquoted(source, ':'), find_unquoted(source, ','))` as far as
/// [`JsonToNbt::next_entry`] needs them: the scan stops at the first unquoted `,`, `{` or `[`.
/// The comma is always exact. The colon is exact when it comes before that stop, and `-1` when
/// the full scan would put it after the comma, where neither branch of `next_entry` uses it.
fn first_separators(source: &[char]) -> (isize, isize) {
    let mut outside = true;
    let mut colon = -1;
    for (index, &ch) in source.iter().enumerate() {
        if ch == '"' {
            if !is_escaped(source, index) {
                outside = !outside;
            }
        } else if outside {
            let at = isize::try_from(index).unwrap_or(isize::MAX);
            match ch {
                ':' if colon == -1 => colon = at,
                ',' => return (colon, at),
                '{' | '[' => return (colon, -1),
                _ => {}
            }
        }
    }
    (colon, -1)
}

/// java: `func_179271_b(s, i)`, its recursion unrolled: a character is escaped when an odd run
/// of backslashes precedes it.
fn is_escaped(source: &[char], index: usize) -> bool {
    source[..index]
        .iter()
        .rev()
        .take_while(|&&ch| ch == '\\')
        .count()
        % 2
        == 1
}

/// java: `String.trim()`: drops leading and trailing characters up to U+0020.  That is not
/// Unicode whitespace: a no-break space survives, a control character does not.
fn java_trim(source: &[char]) -> &[char] {
    let start = source
        .iter()
        .position(|&ch| ch > ' ')
        .unwrap_or(source.len());
    let end = source
        .iter()
        .rposition(|&ch| ch > ' ')
        .map_or(start, |last| last + 1);
    &source[start..end]
}

/// java 8: `Character.isWhitespace(char)`: Unicode 6.2's space, line and paragraph separators
/// except the three no-break spaces, plus U+0009-U+000D and U+001C-U+001F.
fn java_is_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\t'..='\r'
            | '\u{1c}'..='\u{20}'
            | '\u{1680}'
            | '\u{180e}'
            | '\u{2000}'..='\u{2006}'
            | '\u{2008}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// java: `field_179273_b`, `\[[-+\d|,\s]+\]` as a whole match (`\d` and `\s` are ASCII-only in
/// `java.util.regex` by default).  `func_150316_a` hands such a source to `Primitive`.
fn is_int_array_source(source: &[char]) -> bool {
    source.len() > 2
        && source[0] == '['
        && source[source.len() - 1] == ']'
        && source[1..source.len() - 1].iter().all(|&ch| {
            matches!(
                ch,
                '-' | '+' | '0'..='9' | '|' | ',' | ' ' | '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r'
            )
        })
}

/// java: `JsonToNBT.Primitive.parse`.  `None` where Java throws an exception that nothing
/// catches.
fn primitive(source: &[char]) -> Option<Nbt> {
    let value: String = source.iter().collect();
    if let Some(tag) = numeric_or_boolean(&value) {
        return Some(tag);
    }
    if value.len() >= 2 && value.starts_with('[') && value.ends_with(']') {
        // `Splitter.on(',').omitEmptyStrings()`, then `Integer.parseInt(part.trim())`; any
        // failure keeps the whole source as a string.
        let mut ints = Vec::new();
        for part in value[1..value.len() - 1]
            .split(',')
            .filter(|part| !part.is_empty())
        {
            let Ok(int) = part.trim_matches(|ch: char| ch <= ' ').parse::<i32>() else {
                return Some(Nbt::String(value));
            };
            ints.push(int);
        }
        return Some(Nbt::IntArray(ints));
    }
    let mut text = value.as_str();
    if text.starts_with('"') && text.ends_with('"') {
        // java: `substring(1, length() - 1)` throws for a lone quote.
        text = text.get(1..text.len() - 1)?;
    }
    // java: `replaceAll("\\\\\"", "\"")`, then every `\\` pair becomes one backslash.
    let text = text.replace("\\\"", "\"");
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'\\') {
            chars.next();
        }
        out.push(ch);
    }
    Some(Nbt::String(out))
}

/// `Primitive.parse`'s seven numeric patterns in order, then its boolean test.  The suffix
/// classes really are `[d|D]` and friends, so `|` is an accepted suffix.  A matched number that
/// overflows takes Java's NumberFormatException branch: the whole source as a string.
fn numeric_or_boolean(value: &str) -> Option<Nbt> {
    fn parsed<T: std::str::FromStr>(value: &str, body: &str, tag: fn(T) -> Nbt) -> Nbt {
        body.parse()
            .map_or_else(|_| Nbt::String(value.to_owned()), tag)
    }

    // DOUBLE `[-+]?[0-9]*\.?[0-9]+[d|D]` and FLOAT `...[f|F]`.
    if let Some(body) = value
        .strip_suffix(['d', '|', 'D'])
        .filter(|b| is_decimal(b))
    {
        return Some(parsed(value, body, Nbt::Double));
    }
    if let Some(body) = value
        .strip_suffix(['f', '|', 'F'])
        .filter(|b| is_decimal(b))
    {
        return Some(parsed(value, body, Nbt::Float));
    }
    // BYTE `[-+]?[0-9]+[b|B]`, LONG `...[l|L]`, SHORT `...[s|S]`.
    if let Some(body) = value
        .strip_suffix(['b', '|', 'B'])
        .filter(|b| is_integer(b))
    {
        return Some(parsed(value, body, Nbt::Byte));
    }
    if let Some(body) = value
        .strip_suffix(['l', '|', 'L'])
        .filter(|b| is_integer(b))
    {
        return Some(parsed(value, body, Nbt::Long));
    }
    if let Some(body) = value
        .strip_suffix(['s', '|', 'S'])
        .filter(|b| is_integer(b))
    {
        return Some(parsed(value, body, Nbt::Short));
    }
    // INTEGER `[-+]?[0-9]+`, then DOUBLE_UNTYPED `[-+]?[0-9]*\.?[0-9]+`.
    if is_integer(value) {
        return Some(parsed(value, value, Nbt::Int));
    }
    if is_decimal(value) {
        return Some(parsed(value, value, Nbt::Double));
    }
    if java_equals_ignore_case(value, "true") {
        return Some(Nbt::Byte(1));
    }
    if java_equals_ignore_case(value, "false") {
        return Some(Nbt::Byte(0));
    }
    None
}

/// `[-+]?[0-9]+`
fn is_integer(source: &str) -> bool {
    let digits = source.strip_prefix(['-', '+']).unwrap_or(source);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// `[-+]?[0-9]*\.?[0-9]+`: digits with at most one dot, ending in a digit.
fn is_decimal(source: &str) -> bool {
    let source = source.strip_prefix(['-', '+']).unwrap_or(source);
    let mut dots = 0;
    let mut digits = 0;
    for byte in source.bytes() {
        if byte == b'.' {
            dots += 1;
            if dots > 1 {
                return false;
            }
        } else if byte.is_ascii_digit() {
            digits += 1;
        } else {
            return false;
        }
    }
    digits > 0 && !source.ends_with('.')
}

/// java: `String.equalsIgnoreCase`: per UTF-16 unit, equal as is, after
/// `Character.toUpperCase`, or after `toLowerCase` of those.  So `"falſe"` (long s) is false.
fn java_equals_ignore_case(a: &str, b: &str) -> bool {
    fn single(mut mapped: impl Iterator<Item = char>) -> Option<u16> {
        match (mapped.next(), mapped.next()) {
            (Some(one), None) => u16::try_from(u32::from(one)).ok(),
            _ => None,
        }
    }
    fn upper(unit: u16) -> u16 {
        char::from_u32(u32::from(unit))
            .and_then(|ch| single(ch.to_uppercase()))
            .unwrap_or(unit)
    }
    fn lower(unit: u16) -> u16 {
        // Java's simple mapping of U+0130 is plain `i`; Rust exposes the full "i\u{307}".
        if unit == 0x0130 {
            return u16::from(b'i');
        }
        char::from_u32(u32::from(unit))
            .and_then(|ch| single(ch.to_lowercase()))
            .unwrap_or(unit)
    }
    a.encode_utf16().count() == b.encode_utf16().count()
        && a.encode_utf16()
            .zip(b.encode_utf16())
            .all(|(x, y)| x == y || upper(x) == upper(y) || lower(upper(x)) == lower(upper(y)))
}

#[cfg(test)]
mod tests {
    use super::{parse_snbt, parse_snbt_compound};
    use crate::Nbt;

    #[test]
    fn hover_item_compound_preserves_legacy_numeric_dispatch() {
        let parsed = parse_snbt_compound(
            r#"{id:"minecraft:diamond_sword",Count:1b,Damage:7,ench:[{id:16s,lvl:5s}],ok:TrUe}"#,
        )
        .expect("vanilla hover payload parses");
        assert_eq!(parsed.get_str("id"), Some("minecraft:diamond_sword"));
        assert_eq!(parsed.get("Count"), Some(&Nbt::Byte(1)));
        assert_eq!(parsed.get("Damage"), Some(&Nbt::Int(7)));
        assert_eq!(parsed.get("ok"), Some(&Nbt::Byte(1)));
        let enchantments = parsed.get_list("ench").expect("list survives");
        assert_eq!(
            enchantments[0].as_compound().unwrap().get_i16("lvl"),
            Some(5)
        );

        // A modern parser often turns this into an unsigned byte or an integer.
        assert_ne!(parsed.get("Count"), Some(&Nbt::Int(1)));
        assert_eq!(parse_snbt(r#""1""#).unwrap(), Nbt::String("1".to_owned()));
    }

    #[test]
    fn bracketed_integer_form_is_an_int_array_and_mixed_lists_drop_mismatches() {
        assert_eq!(
            parse_snbt("[1,-2,+3]").unwrap(),
            Nbt::IntArray(vec![1, -2, 3])
        );
        assert_eq!(
            parse_snbt("[1,\"wrong\",2]").unwrap(),
            Nbt::list(vec![Nbt::Int(1), Nbt::Int(2)])
        );

        // Treating every bracketed sequence as a list misses JsonToNBT's special form.
        assert_ne!(
            parse_snbt("[1,2]").unwrap(),
            Nbt::list(vec![Nbt::Int(1), Nbt::Int(2)])
        );
    }

    #[test]
    fn mojangs_suffix_typo_and_numeric_overflow_fall_back_like_java() {
        // `|` is accidentally included in Java's suffix character classes.  The
        // double test precedes byte/long/short, so this becomes a double.
        assert_eq!(parse_snbt("1|").unwrap(), Nbt::Double(1.0));
        assert_eq!(
            parse_snbt("99999999999").unwrap(),
            Nbt::String("99999999999".to_owned())
        );
        assert_eq!(parse_snbt("1e3").unwrap(), Nbt::String("1e3".to_owned()));
    }

    #[test]
    fn root_is_compound_only_and_syntax_errors_are_rejected() {
        assert!(parse_snbt_compound("[1,2]").is_err());
        assert!(parse_snbt_compound("{id:1").is_err());
        assert!(parse_snbt_compound("{id:1} trailing").is_err());
        assert!(
            parse_snbt_compound("{a:1}{b:2}").is_err(),
            "two top-level tags"
        );
        // java: `{ }` and `{a:1, }` leave a whitespace-only entry, and a compound entry with no
        // name separator throws "Unable to locate name/value separator".
        assert!(parse_snbt_compound("{ }").is_err());
        assert!(parse_snbt_compound("{a:1, }").is_err());
    }

    /// The parity audit's triggers: each is accepted by 1.8.9's `JsonToNBT` (a strict
    /// recursive-descent parser used to reject them).
    #[test]
    fn jsontonbt_leniencies_are_reproduced() {
        let trailing = parse_snbt_compound("{id:1,}").expect("java skips the empty tail");
        assert_eq!(trailing.get("id"), Some(&Nbt::Int(1)));
        assert_eq!(trailing.len(), 1);

        let list = parse_snbt_compound(r#"{a:[{b:1},]}"#).expect("trailing list comma");
        assert_eq!(list.get_list("a").map(|items| items.len()), Some(1));

        let empty = parse_snbt_compound("{a:}").expect("an empty value is a string");
        assert_eq!(empty.get("a"), Some(&Nbt::String(String::new())));

        let blank = parse_snbt_compound("{a:[ ]}").expect("whitespace-only brackets");
        assert_eq!(
            blank.get("a"),
            Some(&Nbt::String("[ ]".to_owned())),
            "java: Integer.parseInt(\"\") fails, so the source stays a string"
        );

        let spaced = parse_snbt_compound("{foo bar:hello world}").expect("unquoted spaces");
        assert_eq!(
            spaced.get("foo bar"),
            Some(&Nbt::String("hello world".to_owned()))
        );

        let quoted = parse_snbt_compound(r#"{"id":1}"#).expect("parses");
        assert_eq!(
            quoted.get("\"id\""),
            Some(&Nbt::Int(1)),
            "1.8.9 keeps the quotes as part of the name"
        );
        assert!(quoted.get("id").is_none());
    }

    /// Hover parsing runs on the UI thread, whose Windows stack is 1 MiB. The deepest nesting
    /// the parser accepts must fit there even in a debug build, and one level more is an
    /// ordinary error (Java has no bound and overflows its stack instead).
    #[test]
    fn the_nesting_bound_fits_a_one_mebibyte_stack() {
        fn nested(levels: usize) -> String {
            let mut source = "{a:".repeat(levels);
            source.push('1');
            source.push_str(&"}".repeat(levels));
            source
        }
        fn lists(levels: usize) -> String {
            format!("{}{}", "[".repeat(levels), "]".repeat(levels))
        }
        let deepest = super::MAX_DEPTH;
        let result = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                (
                    parse_snbt_compound(&nested(deepest)).is_ok(),
                    parse_snbt_compound(&nested(deepest + 1)).is_err(),
                    parse_snbt(&lists(deepest)).is_ok() && parse_snbt(&lists(deepest + 1)).is_err(),
                )
            })
            .expect("spawn")
            .join()
            .expect("no stack overflow");
        assert_eq!(result, (true, true, true));
    }

    /// `NBTTagCompound.toString` (unquoted names) and `NBTTagList.toString` (`[0:a,1:b]`) are
    /// what servers put into item hovers; `JsonToNBT` reads them back by treating the index as
    /// an ignored name.
    #[test]
    fn the_games_own_list_format_round_trips() {
        let parsed = parse_snbt_compound(
            r#"{id:276s,Count:1b,tag:{ench:[0:{lvl:5s,id:16s},1:{lvl:1s,id:34s}],display:{Lore:[0:"§7Sharp, very",1:"[two]"],Name:"Blade"}},Damage:0s}"#,
        )
        .expect("the game's own toString output parses");
        let tag = parsed.get_compound("tag").expect("tag");
        let ench = tag.get_list("ench").expect("ench");
        assert_eq!(ench.len(), 2);
        assert_eq!(ench[1].as_compound().unwrap().get_i16("id"), Some(34));
        let lore = tag
            .get_compound("display")
            .and_then(|display| display.get_list("Lore"))
            .expect("lore");
        assert_eq!(
            lore.iter().collect::<Vec<_>>(),
            [
                &Nbt::String("§7Sharp, very".to_owned()),
                &Nbt::String("[two]".to_owned())
            ]
        );
    }

    #[test]
    fn quotes_obey_backslash_parity_and_leave_other_escapes_literal() {
        let parsed = parse_snbt_compound(r#"{name:"a\\\"b\nc"}"#).unwrap();
        assert_eq!(parsed.get_str("name"), Some("a\\\"b\\nc"));
    }

    use super::{find_unquoted, first_separators};

    /// `next_entry` as it was: Java's two full scans per entry.
    fn next_entry_by_full_scans(rest: &[char], compound: bool) -> Result<isize, &'static str> {
        let mut colon = find_unquoted(rest, ':');
        let comma = find_unquoted(rest, ',');
        if compound {
            if colon == -1 {
                return Err("Unable to locate name/value separator");
            }
            if comma != -1 && comma < colon {
                return Err("Name error");
            }
        } else if colon == -1 || colon > comma {
            colon = -1;
        }
        Ok(colon)
    }

    /// The early-stopping scan decides every entry exactly as Java's two
    /// full scans do: the same separator, the same error, over every string of up to six
    /// characters from the alphabet that matters (and longer random ones).
    #[test]
    fn the_early_stopping_scan_matches_the_two_full_scans() {
        const ALPHABET: [char; 8] = [':', ',', '{', '[', '"', '\\', 'a', ']'];
        let mut checked = 0usize;
        let mut check = |text: &[char]| {
            for compound in [true, false] {
                let (mut colon, comma) = first_separators(text);
                let fast = if compound {
                    if colon == -1 {
                        if comma != -1 && find_unquoted(text, ':') != -1 {
                            Err("Name error")
                        } else {
                            Err("Unable to locate name/value separator")
                        }
                    } else {
                        Ok(colon)
                    }
                } else {
                    if comma == -1 {
                        colon = -1;
                    }
                    Ok(colon)
                };
                assert_eq!(
                    fast,
                    next_entry_by_full_scans(text, compound),
                    "{:?} compound={compound}",
                    text.iter().collect::<String>()
                );
                checked += 1;
            }
        };
        let mut text = Vec::new();
        fn all(len: usize, text: &mut Vec<char>, check: &mut dyn FnMut(&[char])) {
            check(text);
            if text.len() == len {
                return;
            }
            for ch in ALPHABET {
                text.push(ch);
                all(len, text, check);
                text.pop();
            }
        }
        all(6, &mut text, &mut check);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..20_000 {
            let len = (state % 40) as usize;
            let text: Vec<char> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ALPHABET[(state % 8) as usize]
                })
                .collect();
            check(&text);
        }
        assert!(checked > 600_000, "{checked}");
    }

    /// A hover payload at the chat limit parses in linear time: 8000
    /// list entries took over a second per parse before (and the hover parses every frame).
    #[test]
    fn a_long_list_parses_in_linear_time() {
        let entries = "\"a\",".repeat(8_000);
        let text = format!("{{l:[{}\"a\"]}}", entries);
        let started = std::time::Instant::now();
        let parsed = parse_snbt_compound(&text).expect("a long list");
        let elapsed = started.elapsed();
        assert_eq!(parsed.get_list("l").map(|list| list.len()), Some(8_001));
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "{elapsed:?}"
        );
        let compound = format!(
            "{{{}z:1}}",
            (0..4_000).map(|i| format!("k{i}:1,")).collect::<String>()
        );
        let started = std::time::Instant::now();
        assert_eq!(parse_snbt_compound(&compound).map(|c| c.len()), Ok(4_001));
        assert!(started.elapsed() < std::time::Duration::from_millis(500));
    }
}
