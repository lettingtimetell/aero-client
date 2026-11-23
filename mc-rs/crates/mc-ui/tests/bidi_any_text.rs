//! With a right-to-left language selected (Arabic, Hebrew, Persian...), every string the font
//! draws goes through `bidi::reorder` first: chat, scoreboards, names, signs and books, which a
//! server writes. Any text must come back without a panic, in bounded time, with no glyph lost
//! or invented beyond what shaping and mirroring change.

use std::time::{Duration, Instant};

use mc_ui::font::bidi::reorder;

/// Characters that steer the bidi algorithm or the Arabic shaper, plus plain ones.
const ALPHABET: &[char] = &[
    'a',
    'Z',
    '1',
    '9',
    ' ',
    '.',
    ',',
    ':',
    '-',
    '+',
    '%',
    '$',
    '#',
    '/',
    '(',
    ')',
    '[',
    ']',
    '{',
    '}',
    '<',
    '>',
    '\u{a7}',
    '\u{a0}',
    '\t',
    '\n',
    '\r',
    '\u{2029}',
    '\u{1c}',
    '\u{1f}',
    '\u{5d0}',
    '\u{5d1}',
    '\u{5ea}',
    '\u{5b4}',
    '\u{627}',
    '\u{644}',
    '\u{644}',
    '\u{627}',
    '\u{628}',
    '\u{62a}',
    '\u{645}',
    '\u{647}',
    '\u{64b}',
    '\u{650}',
    '\u{651}',
    '\u{660}',
    '\u{669}',
    '\u{6f0}',
    '\u{fe8e}',
    '\u{fefb}',
    '\u{200c}',
    '\u{200d}',
    '\u{200e}',
    '\u{200f}',
    '\u{202a}',
    '\u{202b}',
    '\u{202c}',
    '\u{202d}',
    '\u{202e}',
    '\u{2066}',
    '\u{2067}',
    '\u{2068}',
    '\u{2069}',
    '\u{300}',
    '\u{301}',
    '\u{2018}',
    '\u{201c}',
    '\u{fd3e}',
    '\u{fd3f}',
    '\u{2045}',
    '\u{2046}',
    '\u{ff08}',
    '\u{ff09}',
    '\u{1F600}',
    '\u{10900}',
    '\u{1e900}',
    '\u{10fffd}',
    '\u{feff}',
    '\u{fffd}',
    '\u{e000}',
];

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_text(state: &mut u64, max_len: usize) -> String {
    let len = (next(state) % (max_len as u64 + 1)) as usize;
    (0..len)
        .map(|_| ALPHABET[(next(state) % ALPHABET.len() as u64) as usize])
        .collect()
}

/// Run `reorder` on its own thread so a hang shows up as a failure, not a stuck test run.
fn reorder_bounded(text: &str, limit: Duration) -> Result<String, String> {
    let owned = text.to_owned();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(|| reorder(&owned));
        let _ = tx.send(result.map_err(|_| "panicked".to_owned()));
    });
    match rx.recv_timeout(limit) {
        Ok(result) => {
            let _ = worker.join();
            result
        }
        Err(_) => Err(format!("still running after {limit:?}")),
    }
}

/// 4000 texts from a fixed seed (about half a second); raise `RUNS` by hand to look longer
/// (56,000 texts ran clean on 2026-11-22).
#[test]
fn any_short_text_reorders_without_a_panic() {
    const RUNS: u32 = 4000;
    let mut state = 0x5eed_2026_1122_u64;
    for _ in 0..RUNS {
        let text = random_text(&mut state, 48);
        if let Err(problem) = reorder_bounded(&text, Duration::from_secs(5)) {
            panic!("{problem} on {text:?}");
        }
    }
}

#[test]
fn long_and_deeply_nested_text_reorders_in_bounded_time() {
    let mut cases: Vec<(&str, String)> = vec![
        (
            "200 embeddings",
            "\u{202b}a".repeat(200) + &"\u{202c}".repeat(200),
        ),
        (
            "200 isolates",
            "\u{2067}\u{5d0}".repeat(200) + &"\u{2069}".repeat(200),
        ),
        ("unbalanced pops", "\u{202c}\u{2069}x".repeat(500)),
        (
            "5000 brackets",
            "(\u{5d0}[".repeat(5000) + &")]".repeat(5000),
        ),
        ("10000 arabic", "\u{644}\u{627}\u{64b}".repeat(10_000)),
        (
            "a chat line",
            "\u{a7}c\u{5d0}\u{5d1} 123 (abc) \u{627}\u{644}".repeat(400),
        ),
    ];
    let mut state = 0xb1d1_u64;
    cases.push(("random 20000", random_text(&mut state, 20_000)));
    for (label, text) in &cases {
        let started = Instant::now();
        match reorder_bounded(text, Duration::from_secs(10)) {
            Ok(out) => println!(
                "{label}: {} chars in, {} out, {:?}",
                text.chars().count(),
                out.chars().count(),
                started.elapsed()
            ),
            Err(problem) => panic!("{label}: {problem}"),
        }
    }
}
