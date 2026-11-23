//! Secrets this process holds (sign-in tokens), and their removal from text that leaves it.
//!
//! Every token already lives in a type whose `Debug` prints `<redacted>` (`mc_auth::Session`,
//! `mc-client`'s `AccessToken` and the rest), so formatting one by accident shows nothing. This
//! module is the second line of defence: a token's text is registered when it enters the process
//! (the command line, a sign-in, a refresh, the account store), and the places where text leaves
//! the process for files players share replace every registered secret with [`REDACTED`]: the
//! log writer (`logwriter`, whose output the zip's launcher saves as `latest-launch.log`), crash
//! reports, and a panic message printed to stderr. A careless
//! `tracing::info!("{}", token.expose())`, an error that quotes a response body, or a panic
//! message that contains a token is caught there.
//!
//! The cost is paid where text is written, never where it is logged: the log writer scrubs on its
//! own thread, and crash reports and panics are rare. Registering takes a lock and copies the
//! secret, once per sign-in or refresh, never per frame. Until something is registered (no
//! sign-in yet, or an offline session) scrubbing is one atomic load.
//!
//! What this cannot see: a secret that was cut before it reached the writer (only whole secrets
//! are matched, so code that truncates text which may hold a token scrubs it first, as
//! `mc-auth`'s error bodies do), and a secret transformed on the way (percent-encoded, say).
//! Request bodies and headers, where tokens travel encoded, are never logged.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{PoisonError, RwLock};

/// Strings shorter than this are not registered: a short string could be an ordinary word in a
/// log line, and every real token is far longer (Minecraft and Microsoft tokens run to hundreds
/// of characters).
pub const MIN_SECRET_LEN: usize = 16;

/// What a registered secret is replaced with.
pub const REDACTED: &str = "<redacted>";

/// Secrets kept at most. A session registers a handful (the launcher's token, then a few per
/// sign-in or refresh); past this the one registered (or registered again) longest ago is
/// forgotten first.
const MAX_SECRETS: usize = 64;

/// The process's registered secrets.
static SECRETS: RwLock<Registry> = RwLock::new(Registry::new());
/// Set once anything has been registered, so no text is scanned before the first secret exists.
static ANY: AtomicBool = AtomicBool::new(false);

/// Remember `secret` so text written from now on has it replaced by [`REDACTED`].
///
/// Call it once when a secret enters the process; registering the same secret again only marks
/// it as recently used (it is forgotten last). Surrounding whitespace is ignored, and a secret
/// shorter than [`MIN_SECRET_LEN`] is not registered.
pub fn register(secret: &str) {
    let secret = secret.trim();
    if secret.len() < MIN_SECRET_LEN {
        return;
    }
    let inserted = SECRETS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(secret.as_bytes());
    if inserted {
        ANY.store(true, Ordering::Release);
    }
}

/// `text` with every registered secret replaced by [`REDACTED`]; borrowed, without allocating,
/// when it holds none.
#[must_use]
pub fn scrub(text: &str) -> Cow<'_, str> {
    match scrub_bytes(text.as_bytes()) {
        // A registered secret is valid UTF-8, so each match starts and ends on a character
        // boundary of `text`, and the replacement is ASCII: the result is valid UTF-8.
        Some(bytes) => Cow::Owned(
            String::from_utf8(bytes)
                .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned()),
        ),
        None => Cow::Borrowed(text),
    }
}

/// [`scrub`] for bytes (the log writer's batches): `None` when nothing was replaced.
#[must_use]
pub fn scrub_bytes(text: &[u8]) -> Option<Vec<u8>> {
    if !ANY.load(Ordering::Acquire) {
        return None;
    }
    SECRETS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .scrub(text)
}

/// One registered secret, and when it was registered (for the oldest-first eviction).
struct Entry {
    order: u64,
    bytes: Box<[u8]>,
}

/// Registered secrets, longest first: a secret that contains a shorter one (an Xbox identity
/// header contains its token) is replaced whole before the shorter one could split it.
struct Registry {
    entries: Vec<Entry>,
    next_order: u64,
}

impl Registry {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_order: 0,
        }
    }

    /// Add `secret` unless it is already known; `true` when it was added.
    fn insert(&mut self, secret: &[u8]) -> bool {
        if let Some(known) = self
            .entries
            .iter_mut()
            .find(|known| *known.bytes == *secret)
        {
            // Registered again (a session built again from the same token):
            // it is in use, so it is forgotten last. Without this the launcher's token, the
            // first one registered and used for every join, was the first one evicted.
            known.order = self.next_order;
            self.next_order += 1;
            return false;
        }
        if self.entries.len() >= MAX_SECRETS {
            let oldest = (0..self.entries.len()).min_by_key(|&index| self.entries[index].order);
            if let Some(oldest) = oldest {
                self.entries.remove(oldest);
            }
        }
        let at = self
            .entries
            .partition_point(|known| known.bytes.len() >= secret.len());
        self.entries.insert(
            at,
            Entry {
                order: self.next_order,
                bytes: secret.into(),
            },
        );
        self.next_order += 1;
        true
    }

    /// `text` with every secret replaced, or `None` when it holds none.
    fn scrub(&self, text: &[u8]) -> Option<Vec<u8>> {
        let mut current: Cow<'_, [u8]> = Cow::Borrowed(text);
        let mut changed = false;
        for secret in &self.entries {
            if let Some(replaced) = replace_all(&current, &secret.bytes) {
                current = Cow::Owned(replaced);
                changed = true;
            }
        }
        changed.then(|| current.into_owned())
    }
}

/// `text` with every occurrence of `needle` replaced by [`REDACTED`], or `None` when it has none.
fn replace_all(text: &[u8], needle: &[u8]) -> Option<Vec<u8>> {
    let first = *needle.first()?;
    let mut out: Option<Vec<u8>> = None;
    let (mut copied, mut at) = (0, 0);
    while at + needle.len() <= text.len() {
        // The next byte that could start the needle, among the starts that leave room for it.
        let Some(offset) = text[at..=text.len() - needle.len()]
            .iter()
            .position(|&byte| byte == first)
        else {
            break;
        };
        at += offset;
        if text[at..].starts_with(needle) {
            let buffer = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
            buffer.extend_from_slice(&text[copied..at]);
            buffer.extend_from_slice(REDACTED.as_bytes());
            at += needle.len();
            copied = at;
        } else {
            at += 1;
        }
    }
    let mut buffer = out?;
    buffer.extend_from_slice(&text[copied..]);
    Some(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The process-wide registry is shared by tests running in parallel: each test registers
    // secrets of its own and never assumes the registry is empty. Eviction, which would drop
    // other tests' secrets, is tested on a local `Registry`.

    #[test]
    fn a_registered_secret_is_replaced_everywhere_in_the_text() {
        let secret = "eyJhbGciOiJIUzI1NiJ9.ZmFrZS10b2tlbi1mb3ItdGVzdHM.c2ln";
        register(secret);
        let text = format!("{secret} start, middle {secret}{secret} and end {secret}");
        let scrubbed = scrub(&text);
        assert!(!scrubbed.contains(secret), "{scrubbed}");
        assert_eq!(
            scrubbed,
            "<redacted> start, middle <redacted><redacted> and end <redacted>"
        );
        // Bytes, as the log writer passes them.
        let bytes = scrub_bytes(text.as_bytes()).expect("replaced");
        assert_eq!(bytes, scrubbed.as_bytes());
    }

    #[test]
    fn text_without_a_secret_is_borrowed_unchanged() {
        register("fake-secret-for-the-borrowing-test");
        let text = "INFO nothing secret here, eyJ but not a token";
        assert!(matches!(scrub(text), Cow::Borrowed(t) if t == text));
        assert_eq!(scrub_bytes(text.as_bytes()), None);
        assert!(matches!(scrub(""), Cow::Borrowed("")));
    }

    #[test]
    fn short_strings_are_never_registered() {
        // Shorter than MIN_SECRET_LEN once the spaces are trimmed: a common word must not vanish
        // from every log line.
        register("  minecraft  ");
        register("");
        assert!(matches!(scrub("minecraft "), Cow::Borrowed(_)));
        // Surrounding whitespace is not part of the secret.
        register("  padded-fake-secret-value  ");
        assert_eq!(scrub("[padded-fake-secret-value]"), "[<redacted>]");
    }

    #[test]
    fn a_longer_secret_is_replaced_whole_before_a_shorter_one_inside_it() {
        // An Xbox identity header holds its token: registering both must not leave the header's
        // other half behind.
        let token = "fake-xsts-token-for-the-nesting-test";
        let header = format!("XBL3.0 x=1234567890123;{token}");
        register(token);
        register(&header);
        assert_eq!(scrub(&format!("[{header}]")), "[<redacted>]");
        assert_eq!(scrub(&format!("[{token}]")), "[<redacted>]");
    }

    #[test]
    fn text_around_a_secret_stays_valid_utf8() {
        register("fake-secret-between-accents");
        assert_eq!(
            scrub("größe fake-secret-between-accents ✓"),
            "größe <redacted> ✓"
        );
    }

    #[test]
    fn overlapping_and_partial_matches_are_handled() {
        let secret = "aaaaaaaaaaaaaaaab-fake-overlap";
        register(secret);
        // The text's run of 'a' is longer than the secret's: the match starts later.
        assert_eq!(scrub(&format!("aaa{secret}aaa")), "aaa<redacted>aaa");
        // A prefix alone is not the secret (see the module docs on truncation).
        assert!(matches!(
            scrub("aaaaaaaaaaaaaaaab-fake-over"),
            Cow::Borrowed(_)
        ));
    }

    /// Differential check against `str::replace` (non-overlapping, left to right, the same rule),
    /// applied in the registry's longest-first order: 20,000 random texts and secret sets over a
    /// two-letter alphabet, so partial matches and overlaps are everywhere.
    #[test]
    fn scrubbing_matches_std_replace_on_random_inputs() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let word = |next: &mut dyn FnMut(u64) -> u64, len: u64| -> String {
            (0..len)
                .map(|_| if next(2) == 0 { 'a' } else { 'b' })
                .collect()
        };
        for _ in 0..20_000 {
            let mut registry = Registry::new();
            let count = next(4);
            for _ in 0..count {
                let len = 1 + next(6);
                let secret = word(&mut next, len);
                registry.insert(secret.as_bytes());
            }
            let len = next(40);
            let text = word(&mut next, len);
            let mut expected = text.clone();
            for entry in &registry.entries {
                let secret = std::str::from_utf8(&entry.bytes).unwrap();
                expected = expected.replace(secret, REDACTED);
            }
            let got = registry
                .scrub(text.as_bytes())
                .map_or_else(|| text.clone(), |bytes| String::from_utf8(bytes).unwrap());
            assert_eq!(got, expected, "text {text:?}");
        }
    }

    #[test]
    fn a_registry_keeps_one_entry_per_secret_longest_first_and_stays_bounded() {
        let mut registry = Registry::new();
        assert!(registry.insert(b"fake-secret-registered-twice"));
        assert!(!registry.insert(b"fake-secret-registered-twice"));
        assert_eq!(registry.entries.len(), 1);
        for index in 0..(MAX_SECRETS * 2) {
            let secret = format!("fake-bounded-{}", "x".repeat(index % 7 + 3));
            let secret = format!("{secret}-{index:04}");
            registry.insert(secret.as_bytes());
        }
        assert_eq!(registry.entries.len(), MAX_SECRETS);
        assert!(registry
            .entries
            .windows(2)
            .all(|pair| pair[0].bytes.len() >= pair[1].bytes.len()));
        // The oldest went first: the first secret is gone, the newest is still replaced.
        assert_eq!(registry.scrub(b"fake-secret-registered-twice"), None);
        let newest_index = MAX_SECRETS * 2 - 1;
        let newest = format!(
            "fake-bounded-{}-{newest_index:04}",
            "x".repeat(newest_index % 7 + 3)
        );
        assert_eq!(
            registry.scrub(newest.as_bytes()).as_deref(),
            Some(REDACTED.as_bytes())
        );
    }

    /// A secret registered again counts as new: the next secret to come
    /// pushes out the least recently registered one instead (the launcher's token, registered
    /// first and used for every join, was the first one forgotten).
    #[test]
    fn a_secret_registered_again_is_forgotten_last() {
        let mut registry = Registry::new();
        let token = b"fake-launcher-token-registered-at-start";
        assert!(registry.insert(token));
        for index in 0..(MAX_SECRETS - 1) {
            assert!(registry.insert(format!("fake-sign-in-secret-{index:04}").as_bytes()));
        }
        assert!(!registry.insert(token), "known: registered again");
        assert!(registry.insert(b"fake-one-more-secret-after-the-cap"));
        assert_eq!(registry.entries.len(), MAX_SECRETS);
        assert_eq!(
            registry.scrub(token).as_deref(),
            Some(REDACTED.as_bytes()),
            "the token in use stays"
        );
        assert_eq!(
            registry.scrub(b"fake-sign-in-secret-0000"),
            None,
            "the oldest went"
        );
    }
}
