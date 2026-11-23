//! The console writer behind [`init_tracing`](crate::init_tracing): formatted log lines are
//! handed to a background thread instead of being written to stdout on the thread that logged
//! them.
//!
//! `tracing_subscriber::fmt`'s default writer is `io::stdout()`, line-buffered, so every event
//! the filter lets through costs a synchronous `write` on the game thread. On Windows the client
//! is a console-subsystem binary, so that write is a `WriteConsoleW` round trip to the console
//! host; behind a launcher that captures stdout it is a pipe that blocks once the reader falls
//! behind. Several `warn!` sites fire once per packet or per sound (an unknown sound event from a
//! server-side resource pack, a rejected attribute modifier, a failing packet handler), which is
//! exactly when the frame can least afford it.
//!
//! What stays the same: the formatted bytes (the `fmt` layer still formats the event, with the
//! same ANSI setting, on the logging thread), their order (one FIFO channel, one consumer), and
//! the stream (stdout). What changes: the write happens on `mc-log-writer`, consecutive lines are
//! coalesced into one write, and a full channel drops lines rather than stalling the game — the
//! number dropped is reported in-band once the console catches up. [`flush`] drains the queue;
//! `mc-client`'s `main` calls it before printing a crash report and before exiting.
//!
//! One deliberate change to the bytes: the current user's home folder (`C:\Users\<account>` on
//! Windows) is written as `%USERPROFILE%` (`~` elsewhere). These lines are meant to be shared:
//! the zip's launcher saves them as `latest-launch.log` and asks players to send it, and a shared
//! log should not name the Windows account. The worker does
//! the replacing, so the game thread never pays for it; [`redact_home`] gives a crash report the
//! same treatment.
//!
//! Std-only on purpose: `tracing-appender` would do the same job but adds a crate (and its
//! `time`/`crossbeam` dependencies) for ~100 lines of channel plumbing.

use std::borrow::Cow;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tracing_subscriber::fmt::MakeWriter;

/// Lines that may wait for the console before new ones are dropped. A formatted line is roughly
/// 100–200 bytes, so a full queue is about a megabyte.
const CAPACITY: usize = 8192;

/// Upper bound for one coalesced `write`; keeps a backlog from turning into one huge allocation.
const BATCH_BYTES: usize = 64 * 1024;

enum Message {
    Line(Vec<u8>),
    /// Answered once every line queued before it has been written.
    Flush(SyncSender<()>),
}

/// The installed writer's queue, kept so [`flush`] can reach it.
static QUEUE: OnceLock<SyncSender<Message>> = OnceLock::new();
/// Lines dropped because the queue was full; reported in-band by the worker.
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// The current user's home folder, read from the environment once (see the module docs).
static HOME: OnceLock<Option<HomeRedaction>> = OnceLock::new();

fn home() -> Option<&'static HomeRedaction> {
    HOME.get_or_init(HomeRedaction::from_env).as_ref()
}

/// `text` with the current user's home folder written as `%USERPROFILE%` (`~` off Windows), as
/// the log writer writes it: for text meant to be shared that does not pass through the writer,
/// such as a crash report.
#[must_use]
pub fn redact_home(text: &str) -> Cow<'_, str> {
    match home().and_then(|home| home.apply(text.as_bytes())) {
        // Whole folder names are swapped for ASCII, so the text stays valid UTF-8.
        Some(bytes) => Cow::Owned(
            String::from_utf8(bytes)
                .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned()),
        ),
        None => Cow::Borrowed(text),
    }
}

/// `text` as it may be shared: every registered secret (a sign-in token, see
/// [`crate::secrets`]) replaced, then the home folder written as [`redact_home`] writes it. What
/// a crash report goes through, and the same treatment the log writer gives every line.
#[must_use]
pub fn redact_for_sharing(text: &str) -> Cow<'_, str> {
    match crate::secrets::scrub(text) {
        Cow::Borrowed(text) => redact_home(text),
        Cow::Owned(scrubbed) => Cow::Owned(redact_home(&scrubbed).into_owned()),
    }
}

/// The current user's home folder as it can appear in a log line, and what replaces it.
#[derive(Debug)]
struct HomeRedaction {
    /// The folder as written, with every `\` doubled (a path printed with `{:?}`), and with `/`.
    forms: Vec<Vec<u8>>,
    replacement: &'static [u8],
}

impl HomeRedaction {
    /// `USERPROFILE` (Windows), else `HOME`.
    fn from_env() -> Option<Self> {
        if let Some(home) = std::env::var_os("USERPROFILE").filter(|home| !home.is_empty()) {
            return Self::new(&home.to_string_lossy(), "%USERPROFILE%");
        }
        let home = std::env::var_os("HOME").filter(|home| !home.is_empty())?;
        Self::new(&home.to_string_lossy(), "~")
    }

    /// `None` for something too short to be a home folder (a drive root, `/`): redacting that
    /// would mangle every path in the log.
    fn new(home: &str, replacement: &'static str) -> Option<Self> {
        let home = home.trim_end_matches(['\\', '/']);
        let named = home
            .split(['\\', '/'])
            .filter(|part| !part.is_empty() && !part.ends_with(':'))
            .count();
        if named < 2 {
            return None;
        }
        let mut forms: Vec<Vec<u8>> = vec![home.as_bytes().to_vec()];
        for form in [home.replace('\\', "\\\\"), home.replace('\\', "/")] {
            if !forms
                .iter()
                .any(|known| known.eq_ignore_ascii_case(form.as_bytes()))
            {
                forms.push(form.into_bytes());
            }
        }
        Some(Self {
            forms,
            replacement: replacement.as_bytes(),
        })
    }

    /// `text` with every form of the folder replaced, or `None` (no allocation) when it holds
    /// none. ASCII case is ignored; a match must end where the folder's name ends.
    fn apply(&self, text: &[u8]) -> Option<Vec<u8>> {
        let mut current: Cow<'_, [u8]> = Cow::Borrowed(text);
        let mut changed = false;
        for form in &self.forms {
            if let Some(replaced) = replace_form(&current, form, self.replacement) {
                current = Cow::Owned(replaced);
                changed = true;
            }
        }
        changed.then(|| current.into_owned())
    }
}

/// Replace each case-insensitive occurrence of `form` that is not followed by a character that
/// would continue the folder's name (`C:\Users\abc` must not match inside `C:\Users\abcd`).
fn replace_form(text: &[u8], form: &[u8], replacement: &[u8]) -> Option<Vec<u8>> {
    if form.is_empty() || text.len() < form.len() {
        return None;
    }
    let mut out: Option<Vec<u8>> = None;
    let (mut copied, mut at) = (0, 0);
    while at + form.len() <= text.len() {
        let end = at + form.len();
        let continues = text.get(end).is_some_and(|&next| {
            next.is_ascii_alphanumeric() || matches!(next, b'_' | b'-' | 0x80..)
        });
        if !continues && text[at..end].eq_ignore_ascii_case(form) {
            let buffer = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
            buffer.extend_from_slice(&text[copied..at]);
            buffer.extend_from_slice(replacement);
            copied = end;
            at = end;
        } else {
            at += 1;
        }
    }
    let mut buffer = out?;
    buffer.extend_from_slice(&text[copied..]);
    Some(buffer)
}

/// A [`MakeWriter`] that queues each formatted event for the `mc-log-writer` thread.
///
/// If that thread cannot be spawned the writer degrades to the old synchronous stdout write, so
/// logging never silently disappears.
#[derive(Debug, Clone)]
pub struct NonBlockingStdout {
    queue: Option<SyncSender<Message>>,
}

impl NonBlockingStdout {
    /// Spawn the writer thread (once per process) and return a writer feeding it.
    #[must_use]
    pub fn install() -> Self {
        if let Some(queue) = QUEUE.get() {
            return Self {
                queue: Some(queue.clone()),
            };
        }
        let (queue, receiver) = mpsc::sync_channel(CAPACITY);
        let spawned = std::thread::Builder::new()
            .name("mc-log-writer".to_owned())
            .spawn(move || run(&receiver, &mut io::stdout(), &DROPPED, home()));
        match spawned {
            Ok(_) => {
                let queue = QUEUE.get_or_init(|| queue).clone();
                Self { queue: Some(queue) }
            }
            Err(_) => Self { queue: None },
        }
    }
}

impl<'a> MakeWriter<'a> for NonBlockingStdout {
    type Writer = EventWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        EventWriter { owner: self }
    }
}

/// One event's writer. `fmt` formats the whole event first and then calls `write_all` once, so
/// each `write` here is one complete line.
#[derive(Debug)]
pub struct EventWriter<'a> {
    owner: &'a NonBlockingStdout,
}

impl Write for EventWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let Some(queue) = &self.owner.queue else {
            return write_now(buf);
        };
        match queue.try_send(Message::Line(buf.to_vec())) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
            // The worker is gone (it only exits when every sender is dropped, which the
            // process-wide subscriber never does); keep the line rather than lose it.
            Err(TrySendError::Disconnected(_)) => return write_now(buf),
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The synchronous fallback when the worker is not running: the bytes the worker would write.
fn write_now(buf: &[u8]) -> io::Result<usize> {
    // Registered secrets first, then the home folder, as the worker does.
    let scrubbed = crate::secrets::scrub_bytes(buf);
    let text = scrubbed.as_deref().unwrap_or(buf);
    match home().and_then(|home| home.apply(text)) {
        Some(redacted) => io::stdout().write_all(&redacted).map(|()| buf.len()),
        None if scrubbed.is_some() => io::stdout().write_all(text).map(|()| buf.len()),
        None => io::stdout().write(buf),
    }
}

/// The worker: write queued lines in order, coalescing whatever is already waiting. `dropped`
/// is the counter the writers bump on a full queue (a parameter so tests can isolate it), and
/// `home` the folder written as its stand-in (see the module docs).
fn run(
    receiver: &Receiver<Message>,
    out: &mut dyn Write,
    dropped_lines: &AtomicU64,
    home: Option<&HomeRedaction>,
) {
    let mut batch: Vec<u8> = Vec::with_capacity(BATCH_BYTES);
    let mut acks: Vec<SyncSender<()>> = Vec::new();
    let mut reported = 0u64;
    while let Ok(first) = receiver.recv() {
        let mut next = Some(first);
        while let Some(message) = next.take() {
            match message {
                Message::Line(line) => batch.extend_from_slice(&line),
                Message::Flush(ack) => acks.push(ack),
            }
            if batch.len() < BATCH_BYTES {
                next = receiver.try_recv().ok();
            }
        }
        let dropped = dropped_lines.load(Ordering::Relaxed);
        if dropped != reported {
            let _ = writeln!(
                batch,
                "mc-rs: {} log line(s) dropped because the console could not keep up",
                dropped - reported
            );
            reported = dropped;
        }
        if !batch.is_empty() {
            // A registered secret (a sign-in token) never reaches the console,
            // whatever line it was logged in. Scrubbed here, off the logging thread.
            if let Some(scrubbed) = crate::secrets::scrub_bytes(&batch) {
                batch.clear();
                batch.extend_from_slice(&scrubbed);
            }
            if let Some(redacted) = home.and_then(|home| home.apply(&batch)) {
                batch.clear();
                batch.extend_from_slice(&redacted);
            }
            // Errors are ignored exactly as `fmt` ignores them for its default stdout writer:
            // there is nowhere better to report a broken console.
            let _ = out.write_all(&batch);
            let _ = out.flush();
            batch.clear();
        }
        for ack in acks.drain(..) {
            let _ = ack.send(());
        }
    }
}

/// Block until every line logged before this call has been written, or `timeout` elapses.
///
/// A no-op when the background writer is not installed (tests, or a failed spawn).
pub fn flush(timeout: Duration) {
    let Some(queue) = QUEUE.get() else {
        return;
    };
    let deadline = Instant::now() + timeout;
    let (ack, done) = mpsc::sync_channel(1);
    let mut message = Message::Flush(ack);
    loop {
        match queue.try_send(message) {
            Ok(()) => break,
            Err(TrySendError::Full(returned)) => {
                if Instant::now() >= deadline {
                    return;
                }
                message = returned;
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
    let _ = done.recv_timeout(deadline.saturating_duration_since(Instant::now()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines come out in order, coalesced, and a flush is answered only after the lines queued
    /// before it were written.
    #[test]
    fn worker_writes_in_order_and_acknowledges_flush_after_prior_lines() {
        let (queue, receiver) = mpsc::sync_channel(16);
        for i in 0..5 {
            queue
                .send(Message::Line(format!("line {i}\n").into_bytes()))
                .unwrap();
        }
        let (ack, done) = mpsc::sync_channel(1);
        queue.send(Message::Flush(ack)).unwrap();
        queue.send(Message::Line(b"after\n".to_vec())).unwrap();
        drop(queue);

        let mut out: Vec<u8> = Vec::new();
        run(&receiver, &mut out, &AtomicU64::new(0), None);
        done.try_recv().expect("flush acknowledged");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "line 0\nline 1\nline 2\nline 3\nline 4\nafter\n"
        );
    }

    /// A writer whose worker could not start writes straight through instead of losing lines.
    #[test]
    fn writer_without_a_worker_is_synchronous() {
        let writer = NonBlockingStdout { queue: None };
        let mut event = writer.make_writer();
        assert_eq!(event.write(b"").unwrap(), 0);
    }

    /// A full queue drops (and counts) rather than blocking the logging thread.
    #[test]
    fn a_full_queue_drops_instead_of_blocking() {
        let (queue, _receiver) = mpsc::sync_channel(1);
        let writer = NonBlockingStdout { queue: Some(queue) };
        let before = DROPPED.load(Ordering::Relaxed);
        let mut event = writer.make_writer();
        assert_eq!(event.write(b"kept\n").unwrap(), 5);
        assert_eq!(event.write(b"dropped\n").unwrap(), 8);
        assert!(DROPPED.load(Ordering::Relaxed) > before);
    }

    fn redacted(home: &HomeRedaction, text: &str) -> String {
        home.apply(text.as_bytes()).map_or_else(
            || text.to_owned(),
            |bytes| String::from_utf8(bytes).unwrap(),
        )
    }

    /// Every form a log line can hold the home folder in becomes its stand-in.
    #[test]
    fn every_form_of_the_home_folder_is_replaced() {
        let home = HomeRedaction::new(r"C:\Users\Someone", "%USERPROFILE%").unwrap();
        assert_eq!(
            redacted(
                &home,
                r"INFO saved path=C:\Users\Someone\Desktop\Aero Client\game\x.png"
            ),
            r"INFO saved path=%USERPROFILE%\Desktop\Aero Client\game\x.png"
        );
        assert_eq!(
            redacted(&home, r#"path="C:\\Users\\Someone\\AppData\\Roaming""#),
            r#"path="%USERPROFILE%\\AppData\\Roaming""#
        );
        assert_eq!(
            redacted(&home, "c:/users/SOMEONE/x and C:/Users/Someone"),
            "%USERPROFILE%/x and %USERPROFILE%"
        );
        assert_eq!(
            redacted(&home, r"two C:\Users\Someone\a C:\Users\Someone\b"),
            r"two %USERPROFILE%\a %USERPROFILE%\b"
        );
        assert_eq!(
            redacted(&home, r"end C:\Users\Someone."),
            r"end %USERPROFILE%."
        );
    }

    /// Another folder, or a longer name starting the same way, is left alone, and text around a
    /// non-ASCII name stays valid UTF-8.
    #[test]
    fn other_folders_and_longer_names_are_left_alone() {
        let home = HomeRedaction::new(r"C:\Users\Someone\", "%USERPROFILE%").unwrap();
        assert!(home.apply(br"C:\Users\Someone2\x").is_none());
        assert!(home.apply(br"C:\Users\Someone_old\x").is_none());
        assert!(home.apply(br"D:\Games\Aero Client\game").is_none());
        assert!(home.apply(b"").is_none());
        let accented = HomeRedaction::new(r"C:\Users\José", "%USERPROFILE%").unwrap();
        assert_eq!(
            redacted(&accented, r"größe C:\Users\José\Desktop ✓"),
            r"größe %USERPROFILE%\Desktop ✓"
        );
        assert!(accented.apply(r"C:\Users\Joséé\x".as_bytes()).is_none());
    }

    /// A drive root or `/` is never taken for a home folder: it would mangle every path.
    #[test]
    fn a_root_is_never_treated_as_a_home_folder() {
        assert!(HomeRedaction::new(r"C:\", "%USERPROFILE%").is_none());
        assert!(HomeRedaction::new("/", "~").is_none());
        assert!(HomeRedaction::new(r"C:\Users", "%USERPROFILE%").is_none());
        let unix = HomeRedaction::new("/home/someone", "~").unwrap();
        assert_eq!(
            redacted(&unix, "at /home/someone/.minecraft"),
            "at ~/.minecraft"
        );
    }

    /// The worker writes the home folder as its stand-in, in lines it coalesced.
    #[test]
    fn the_worker_writes_the_home_folder_as_its_stand_in() {
        let home = HomeRedaction::new(r"C:\Users\Someone", "%USERPROFILE%").unwrap();
        let (queue, receiver) = mpsc::sync_channel(4);
        queue
            .send(Message::Line(br"saved C:\Users\Someone\a.png".to_vec()))
            .unwrap();
        queue
            .send(Message::Line(b"\nnext C:/Users/Someone/b\n".to_vec()))
            .unwrap();
        drop(queue);
        let mut out: Vec<u8> = Vec::new();
        run(&receiver, &mut out, &AtomicU64::new(0), Some(&home));
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "saved %USERPROFILE%\\a.png\nnext %USERPROFILE%/b\n"
        );
    }

    /// A registered token never reaches the console, in whichever line or
    /// batch it was logged, and the rest of each line is kept. Without the registration (the
    /// negative control) the same worker writes the token as it came.
    #[test]
    fn the_worker_never_writes_a_registered_secret() {
        let token = "eyJhbGciOiJSUzI1NiJ9.fake-logwriter-test-token.c2lnbmF0dXJl";
        let unregistered = "eyJhbGciOiJSUzI1NiJ9.fake-but-never-registered.c2lnbmF0dXJl";
        crate::secrets::register(token);
        let home = HomeRedaction::new(r"C:\Users\Someone", "%USERPROFILE%").unwrap();
        let (queue, receiver) = mpsc::sync_channel(4);
        queue
            .send(Message::Line(
                format!("INFO login token={token}\n").into_bytes(),
            ))
            .unwrap();
        queue
            .send(Message::Line(
                format!("WARN C:\\Users\\Someone\\x {token}{token} {unregistered}\n").into_bytes(),
            ))
            .unwrap();
        drop(queue);
        let mut out: Vec<u8> = Vec::new();
        run(&receiver, &mut out, &AtomicU64::new(0), Some(&home));
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains(token), "{out}");
        assert_eq!(
            out,
            format!(
                "INFO login token=<redacted>\nWARN %USERPROFILE%\\x <redacted><redacted> \
                 {unregistered}\n"
            )
        );
        // What crash reports go through.
        assert_eq!(
            redact_for_sharing(&format!("Caused by: HTTP 401 {token}")),
            "Caused by: HTTP 401 <redacted>"
        );
        assert!(matches!(
            redact_for_sharing("nothing secret"),
            Cow::Borrowed("nothing secret")
        ));
    }
}
