//! `JsonToNBT` parity against the game's own class.
//!
//! `tests/data/jsontonbt_oracle.tsv` was written by `tests/data/GenJsonToNbtOracle.java`, which
//! runs 1.8.9's `JsonToNBT.getTagFromJson` on Java 8 (the runtime 1.8.9 ships with) over
//! hand-written edge cases and over hover payloads built with `NBTTagCompound.toString`. Every
//! line must parse to the same canonical tag here.
//!
//! One deliberate difference: a Java exception other than `NBTException` (`EXC:` in the
//! fixture) escapes `GuiScreen.handleComponentHover`'s catch and crashes the game. This client
//! reports an ordinary parse error instead, which the hover renders as "Invalid Item!".

use mc_nbt::{parse_snbt_compound, Nbt};

/// The fixture's escaping, reversed: `\\`, `\"` and `\uXXXX` (one UTF-16 unit).
fn unescape(text: &str) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(ch.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('"') => units.push(u16::from(b'"')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).expect("four hex digits"));
            }
            other => panic!("bad escape {other:?} in {text:?}"),
        }
    }
    String::from_utf16(&units).expect("fixture inputs are valid UTF-16")
}

/// `GenJsonToNbtOracle.esc`.
fn escape(text: &str) -> String {
    let mut out = String::new();
    for unit in text.encode_utf16() {
        match unit {
            0x5c => out.push_str("\\\\"),
            0x22 => out.push_str("\\\""),
            0x20..=0x7e => out.push(char::from(unit as u8)),
            _ => out.push_str(&format!("\\u{unit:04x}")),
        }
    }
    out
}

/// `GenJsonToNbtOracle.dump`.
fn dump(tag: &Nbt) -> String {
    match tag {
        Nbt::Byte(value) => format!("b{value}"),
        Nbt::Short(value) => format!("s{value}"),
        Nbt::Int(value) => format!("i{value}"),
        Nbt::Long(value) => format!("l{value}"),
        Nbt::Float(value) => format!("f{:x}", value.to_bits()),
        Nbt::Double(value) => format!("d{:x}", value.to_bits()),
        Nbt::String(value) => format!("\"{}\"", escape(value)),
        Nbt::List(list) => {
            let items: Vec<String> = list.iter().map(dump).collect();
            format!("L{}[{}]", list.tag_type().id(), items.join(","))
        }
        Nbt::Compound(compound) => {
            let mut entries: Vec<(&str, &Nbt)> = compound.iter().collect();
            // java: a TreeSet of the names, i.e. String.compareTo over UTF-16 units.
            entries.sort_by(|(a, _), (b, _)| a.encode_utf16().cmp(b.encode_utf16()));
            let entries: Vec<String> = entries
                .into_iter()
                .map(|(name, value)| format!("\"{}\"={}", escape(name), dump(value)))
                .collect();
            format!("{{{}}}", entries.join(","))
        }
        Nbt::IntArray(values) => {
            let values: Vec<String> = values.iter().map(ToString::to_string).collect();
            format!("I[{}]", values.join(","))
        }
        other => format!("?{}", other.id()),
    }
}

fn parsed(input: &str) -> String {
    parse_snbt_compound(input).map_or_else(|_| "ERR".to_owned(), |c| dump(&Nbt::Compound(c)))
}

#[test]
fn every_oracle_input_parses_like_java_8s_jsontonbt() {
    let fixture = include_str!("data/jsontonbt_oracle.tsv");
    let mut checked = 0;
    let mut failures = Vec::new();
    for (line_number, line) in fixture.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (input, java) = line.split_once('\t').expect("input<TAB>result");
        let input = unescape(input);
        let expected = if java.starts_with("EXC:") {
            "ERR"
        } else {
            java
        };
        let actual = parsed(&input);
        if actual != expected {
            failures.push(format!(
                "line {}: {input:?}\n  java: {java}\n  rust: {actual}",
                line_number + 1
            ));
        }
        checked += 1;
    }
    assert!(checked >= 190, "the fixture lost lines: {checked}");
    assert!(
        failures.is_empty(),
        "{} of {checked} inputs differ from Java:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The fixture covers the cases the parity audit named and the game's own list format; if
/// it is regenerated without them, the comparison above stops proving anything about them.
#[test]
fn the_oracle_still_covers_the_audit_triggers() {
    let fixture = include_str!("data/jsontonbt_oracle.tsv");
    for input in [
        "{a:1,}",
        "{a:[1,]}",
        "{a:}",
        "{a:[ ]}",
        "{foo bar:1}",
        "{a:[0:\\\"x\\\",1:\\\"y\\\"]}",
    ] {
        assert!(
            fixture
                .lines()
                .any(|line| line.split('\t').next() == Some(input)),
            "missing {input}"
        );
    }
}
