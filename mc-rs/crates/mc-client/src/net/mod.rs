//! The network side of the client: the tokio runtime, the connection task, the channel
//! protocol, and the server-list ping.
//!
//! Nothing in this module ever touches [`mc_world::World`]. The only things that cross the
//! boundary into the game loop are the two enums in [`channel`].

pub mod capture;
pub mod channel;
pub(crate) mod chunk_v5;
pub mod ping;
pub mod replay;
pub mod task;

pub use capture::{CaptureSink, CAPTURE_ENV};
pub use channel::{DisconnectReason, NetHandle, ToGame, ToNet, INBOUND_CAPACITY};
pub use ping::{parse_favicon, parse_status, ping, Favicon, PlayerSample, ServerPing};
pub use replay::{ReplaySource, ReplaySummary, DEFAULT_STEP_MICROS};
pub use task::{ConnectOptions, Credentials, READ_TIMEOUT};

use crate::error::ClientError;

/// Select the first address supported by the IPv4-only Minecraft transport.
///
/// The vanilla launcher applies `java.net.preferIPv4Stack=true` before
/// `InetAddress.getByName` chooses an address. Apply that filter before choosing here too:
/// Windows commonly resolves `localhost` to `::1` followed by `127.0.0.1`. Choosing `::1`
/// first makes `Connection::connect` reject it without ever seeing the valid IPv4 answer.
/// Keep DNS order among IPv4 answers; this is not a retry or fallback policy.
fn first_ipv4_address(
    addresses: impl IntoIterator<Item = std::net::SocketAddr>,
) -> Option<std::net::SocketAddr> {
    addresses.into_iter().find(std::net::SocketAddr::is_ipv4)
}

/// The tokio runtime the network lives on.
///
/// "`tokio` for network only. The game loop is synchronous; the network task talks to
/// it over channels." This type is that rule made structural — it is the *only* runtime in the
/// client, the game loop never enters it, and every method the game loop calls on a
/// [`NetHandle`] is non-blocking.
///
/// A multi-threaded runtime with worker threads of its own is deliberate: a current-thread
/// runtime would need the game loop to call `block_on`, which is precisely the frame stall this
/// design exists to prevent.
///
/// # Example
///
/// ```no_run
/// # fn demo() -> Result<(), mc_client::ClientError> {
/// use mc_client::net::{ConnectOptions, NetRuntime};
///
/// let runtime = NetRuntime::new()?;
/// let mut handle = runtime.connect(ConnectOptions::offline("127.0.0.1", 25565, "Dev"));
/// // ... once per frame, never blocking:
/// while let Some(message) = handle.try_recv() {
///     let _ = message;
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct NetRuntime {
    handle: tokio::runtime::Handle,
    /// `Some` until [`Drop`], which shuts it down with a bound ([`RUNTIME_SHUTDOWN_WAIT`]).
    runtime: Option<tokio::runtime::Runtime>,
}

/// How long dropping the runtime waits for work that is still running.
///
/// A plain drop waits for every blocking job to return, and DNS work runs
/// as blocking jobs: a host lookup (`getaddrinfo`) or the SRV lookup's `nslookup` for a server
/// list ping. On a network that just dropped (a sleep and resume, a lost Wi-Fi) those take ten
/// seconds and more to give up, and quitting then hung the exit for as long. Work still running
/// after the bound is left to the process exit.
pub const RUNTIME_SHUTDOWN_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

impl Drop for NetRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(RUNTIME_SHUTDOWN_WAIT);
        }
    }
}

impl NetRuntime {
    /// Build the runtime.
    ///
    /// # Errors
    ///
    /// [`ClientError::Io`] if the OS refuses to start the worker threads.
    pub fn new() -> Result<NetRuntime, ClientError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("mc-net")
            .build()?;
        Ok(NetRuntime {
            handle: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }

    /// A handle to the runtime, for callers that need to spawn their own work on it (S03's
    /// device-code login, for instance).
    #[must_use]
    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }

    /// Spawn a connection task and return the game loop's end of it.
    ///
    /// Returns immediately: the TCP connect, the handshake and the login all happen on the
    /// runtime, and their outcome arrives as the first [`ToGame`] message.
    #[must_use]
    pub fn connect(&self, options: ConnectOptions) -> NetHandle {
        let (handle, to_game, immediate_to_game, from_game) = NetHandle::new_live_pair();
        // The replay recorder's tap flag, shared with the task.
        let raw_tap = handle.raw_tap_flag();
        // The task owns its own lifetime: it always finishes by sending a
        // [`ToGame::Disconnected`], so there is nothing for a caller to join on.
        let report = to_game.clone();
        spawn_connection(
            &self.handle,
            task::run(options, to_game, immediate_to_game, from_game, raw_tap),
            report,
        );
        handle
    }
}

/// Spawn a connection task so that a panic inside it ends the connection like a transport
/// failure instead of freezing the game.
///
/// The task always ends by sending [`ToGame::Disconnected`], except when it
/// panics: then its senders were simply dropped, the session (which only reacts to that message)
/// stayed in a world that no longer updated, and the read timeout that would have noticed lived
/// in the dead task. The player was frozen until they quit, and the quit then came with a crash
/// report. java: `NetworkManager.exceptionCaught` closes the channel with "Internal Exception"
/// and the game shows the disconnect screen. Here the panic arrives as
/// [`DisconnectReason::ConnectionLost`] (its detail is logged, not shown, like every transport
/// failure), and the task is polled inside [`mc_core::panic_scope::Contained`], so the panic is
/// not recorded as the game's crash.
fn spawn_connection<F>(
    runtime: &tokio::runtime::Handle,
    task: F,
    report: tokio::sync::mpsc::Sender<ToGame>,
) where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let task = runtime.spawn(mc_core::panic_scope::Contained::new(task));
    drop(runtime.spawn(async move {
        let Err(error) = task.await else {
            return;
        };
        if !error.is_panic() {
            return;
        }
        let payload = error.into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("a non-text panic");
        let detail = format!("Internal Exception: the connection task panicked: {message}");
        tracing::error!(%detail, "connection lost");
        let _ = report
            .send(ToGame::Disconnected(DisconnectReason::ConnectionLost {
                detail,
            }))
            .await;
    }));
}

#[cfg(test)]
mod tests {
    use super::first_ipv4_address;
    use std::net::SocketAddr;

    #[test]
    fn ipv4_address_selection_skips_ipv6_before_choosing() {
        let ipv6: SocketAddr = "[::1]:25565".parse().unwrap();
        let ipv4: SocketAddr = "127.0.0.1:25565".parse().unwrap();
        assert_eq!(first_ipv4_address([ipv6, ipv4]), Some(ipv4));
        assert_eq!(first_ipv4_address([ipv4, ipv6]), Some(ipv4));
    }

    #[test]
    fn ipv4_address_selection_preserves_first_supported_dns_answer() {
        let ipv6: SocketAddr = "[2001:db8::1]:25566".parse().unwrap();
        let first: SocketAddr = "192.0.2.7:25566".parse().unwrap();
        let second: SocketAddr = "192.0.2.3:25566".parse().unwrap();
        assert_eq!(first_ipv4_address([ipv6, first, second]), Some(first));
        assert_eq!(first_ipv4_address([second, ipv6, first]), Some(second));
    }

    #[test]
    fn ipv4_address_selection_rejects_unsupported_or_empty_results() {
        let ipv6: SocketAddr = "[::1]:25565".parse().unwrap();
        let other_ipv6: SocketAddr = "[2001:db8::1]:25565".parse().unwrap();
        assert_eq!(first_ipv4_address([ipv6, other_ipv6]), None);
        assert_eq!(first_ipv4_address([]), None);
    }
}

/// A panicking connection task ends the connection instead of freezing it.
#[cfg(test)]
mod connection_panic_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// The first message, or `None` once the handle closed without one.
    fn first_message_or_close(handle: &mut NetHandle) -> Option<ToGame> {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match handle.try_recv() {
                Some(message) => return Some(message),
                None if handle.is_closed() => return None,
                None => std::thread::sleep(Duration::from_millis(5)),
            }
        }
        panic!("the connection neither reported nor closed within 10 s");
    }

    #[test]
    fn a_panicking_connection_task_ends_in_a_disconnect() {
        let runtime = NetRuntime::new().expect("runtime");
        let (mut handle, to_game, _immediate, _from_game) = NetHandle::new_live_pair();
        let report = to_game.clone();
        let task = async move {
            let _sender = to_game;
            panic!("a decoder bug (test)");
        };
        spawn_connection(runtime.handle(), task, report);
        match first_message_or_close(&mut handle) {
            Some(ToGame::Disconnected(DisconnectReason::ConnectionLost { detail })) => {
                assert!(detail.contains("a decoder bug (test)"), "{detail}");
            }
            other => panic!("expected the disconnect, got {other:?}"),
        }
    }

    /// A blocking job that will not finish (a DNS lookup on a dead network)
    /// no longer holds the exit: the drop returns after its bound. Negative control: a plain
    /// `Runtime` drop waits for the job.
    #[test]
    fn dropping_the_runtime_does_not_wait_for_a_stuck_blocking_job() {
        let stuck = Duration::from_secs(8);
        let runtime = NetRuntime::new().expect("runtime");
        drop(
            runtime
                .handle()
                .spawn_blocking(move || std::thread::sleep(stuck)),
        );
        std::thread::sleep(Duration::from_millis(50)); // let the job start
        let started = Instant::now();
        drop(runtime);
        let took = started.elapsed();
        assert!(
            took < stuck - Duration::from_secs(3),
            "the exit waited {took:?}"
        );

        let plain = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .expect("runtime");
        let control = Duration::from_millis(1500);
        drop(plain.spawn_blocking(move || std::thread::sleep(control)));
        std::thread::sleep(Duration::from_millis(50));
        let started = Instant::now();
        drop(plain);
        assert!(started.elapsed() >= control - Duration::from_millis(200));
    }

    /// Negative control: spawned the old way, the dead task's channel just closes, and the
    /// session, which waits for `Disconnected`, hears nothing.
    #[test]
    fn a_plainly_spawned_panicking_task_sends_no_disconnect() {
        let runtime = NetRuntime::new().expect("runtime");
        let (mut handle, to_game, _immediate, _from_game) = NetHandle::new_live_pair();
        drop(runtime.handle().spawn(async move {
            let _sender = to_game;
            panic!("a decoder bug (control)");
        }));
        assert!(first_message_or_close(&mut handle).is_none());
    }
}
