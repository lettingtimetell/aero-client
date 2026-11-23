//! The Windows back end of [`super::DesktopCapture`]: WASAPI process loopback.
//!
//! The only `unsafe` code in this crate: COM calls into the audio stack. Each block says what it
//! relies on. The flow is Microsoft's "ApplicationLoopback" sample:
//! `ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK)` with
//! `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` (a process id and include/exclude its tree), wait for the
//! completion handler, then an ordinary shared-mode, event-driven loopback capture client in the
//! format we ask for (the virtual device converts). Windows before 10 version 2004 fails the
//! activation; that is reported, never a panic.
#![allow(unsafe_code)]

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use windows::core::{implement, Interface, Ref, HRESULT};
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
    AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    PROCESS_LOOPBACK_MODE, PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE,
    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
    WAVEFORMATEX, WAVE_FORMAT_PCM,
};
use windows::Win32::System::Com::StructuredStorage::{
    PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0,
};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, BLOB, COINIT_MULTITHREADED};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{
    CreateEventW, OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
};
use windows::Win32::System::Variant::VT_BLOB;

use super::{Delivery, DesktopSource, DesktopStatus, Shared, SAMPLE_RATE};

/// How long the activation may take before it counts as failed.
const ACTIVATE_TIMEOUT: Duration = Duration::from_secs(5);
/// How often a missing Discord is looked for, and a captured one checked.
const DISCORD_CHECK: Duration = Duration::from_secs(2);
/// The longest wait for the device's event, so a stop request is seen quickly.
const WAIT_MS: u32 = 50;

/// The capture thread: COM for this thread, then capture until stopped. With
/// [`DesktopSource::Discord`], Discord is looked for (and re-found after a restart) until the
/// capture is stopped.
pub(super) fn run(source: DesktopSource, shared: &Arc<Shared>) {
    // SAFETY: initialises COM for this thread only; balanced by `CoUninitialize` below when it
    // succeeded (S_OK or S_FALSE both count).
    let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if com.is_err() {
        shared.set_status(DesktopStatus::Unavailable(format!(
            "COM could not start ({com:?})"
        )));
        return;
    }
    capture_loop(source, shared);
    // SAFETY: matches the successful `CoInitializeEx` above; every COM object of this thread has
    // been released (they live inside `capture_loop`).
    unsafe { CoUninitialize() };
}

/// Consecutive failures (activation, or a stream that broke within [`STABLE`]) before desktop
/// audio counts as unavailable.
const MAX_FAILURES: u32 = 3;
/// A stream that ran this long resets the failure count.
const STABLE: Duration = Duration::from_secs(10);

/// How a capture ended without an error.
enum Ended {
    /// The capture was stopped.
    Stopped,
    /// The watched process (Discord) exited: look for it again.
    TargetExited,
}

fn capture_loop(source: DesktopSource, shared: &Arc<Shared>) {
    // One timeline for the capture's whole life, across Discord restarts and reopened streams.
    let mut delivery = Delivery::new(Instant::now(), SAMPLE_RATE, 1);
    let mut failures = 0u32;
    while !shared.stopping() {
        let (pid, mode) = match source {
            DesktopSource::AllExceptGame => (
                std::process::id(),
                PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE,
            ),
            DesktopSource::Discord => match find_discord() {
                Some(pid) => (pid, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE),
                None => {
                    shared.set_status(DesktopStatus::WaitingForDiscord);
                    // The timeline runs on (silence) while Discord is away.
                    wait_and_fill(shared, &mut delivery, DISCORD_CHECK);
                    continue;
                }
            },
        };
        let error = match open(pid, mode) {
            Ok(stream) => {
                tracing::info!(
                    source = source.label(),
                    rate = stream.rate,
                    channels = stream.channels,
                    "replay desktop audio capturing"
                );
                shared.set_status(DesktopStatus::Capturing);
                let opened = Instant::now();
                let watched = (source == DesktopSource::Discord)
                    .then(|| Watched::open(pid))
                    .flatten();
                match stream.capture(shared, &mut delivery, watched) {
                    Ok(Ended::Stopped) => return,
                    Ok(Ended::TargetExited) => {
                        tracing::info!("replay desktop audio: Discord closed, waiting for it");
                        failures = 0;
                        continue;
                    }
                    Err(error) => {
                        if opened.elapsed() >= STABLE {
                            failures = 0;
                        }
                        error
                    }
                }
            }
            Err(error) => error,
        };
        failures += 1;
        tracing::warn!(%error, failures, "replay desktop audio stream failed");
        if failures >= MAX_FAILURES {
            shared.set_status(DesktopStatus::Unavailable(error));
            return;
        }
        wait_and_fill(shared, &mut delivery, Duration::from_secs(1));
    }
}

/// Sleep up to `span` in short steps (seeing a stop request), keeping the timeline topped up
/// with silence.
fn wait_and_fill(shared: &Shared, delivery: &mut Delivery, span: Duration) {
    let until = Instant::now() + span;
    while !shared.stopping() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(u64::from(WAIT_MS)));
        delivery.idle(Instant::now(), &mut |s| shared.push(s.iter().copied()));
    }
}

/// Whether the process heap is intact (`HeapValidate` over the whole heap): the stress check
/// in `examples/desktop_check.rs` runs it after many activations.
pub(super) fn heap_is_valid() -> bool {
    use windows::Win32::System::Memory::{GetProcessHeap, HeapValidate, HEAP_FLAGS};
    // SAFETY: validates the default heap as a whole; no pointer of ours is passed.
    unsafe { GetProcessHeap().is_ok_and(|heap| HeapValidate(heap, HEAP_FLAGS(0), None).as_bool()) }
}

/// The root of Discord's process tree, if Discord is running.
fn find_discord() -> Option<u32> {
    let processes = process_list()?;
    super::discord_root(&processes)
}

/// Every process as `(pid, parent pid, exe name)`.
fn process_list() -> Option<Vec<(u32, u32, String)>> {
    // SAFETY: a snapshot handle, closed below; the entry is a plain struct with its size set as
    // the API requires.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).ok()?;
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..PROCESSENTRY32W::default()
        };
        let mut out = Vec::new();
        if Process32FirstW(snapshot, &mut entry).is_ok() {
            loop {
                let len = entry
                    .szExeFile
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(entry.szExeFile.len());
                out.push((
                    entry.th32ProcessID,
                    entry.th32ParentProcessID,
                    String::from_utf16_lossy(&entry.szExeFile[..len]),
                ));
                if Process32NextW(snapshot, &mut entry).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snapshot);
        Some(out)
    }
}

/// A process handle that says when the process exits (Discord quitting or restarting).
struct Watched(HANDLE);

impl Watched {
    fn open(pid: u32) -> Option<Watched> {
        // SAFETY: a synchronise-only handle, closed in `Drop`.
        unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }
            .ok()
            .map(Watched)
    }

    fn exited(&self) -> bool {
        // SAFETY: the handle is valid until `Drop`; a zero timeout only polls.
        unsafe { WaitForSingleObject(self.0, 0) == WAIT_OBJECT_0 }
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        // SAFETY: the handle came from `OpenProcess` and is closed once.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// An event handle, closed on drop.
struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: the handle came from `CreateEventW` and is closed once, after the audio client
        // that signals it was stopped and released (field order in `Stream`).
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// An initialised process-loopback stream.
struct Stream {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    rate: u32,
    channels: u16,
    // Dropped last: the client above may signal it until released.
    event: Event,
}

/// The completion handler `ActivateAudioInterfaceAsync` calls (on a system thread). Agile (the
/// `implement` macro's default: the free-threaded marshaler), as the API requires.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Completion {
    done: Arc<(Mutex<bool>, Condvar)>,
}

impl IActivateAudioInterfaceCompletionHandler_Impl for Completion_Impl {
    fn ActivateCompleted(
        &self,
        _operation: Ref<IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        let (lock, condvar) = &*self.done;
        *lock.lock().unwrap_or_else(PoisonError::into_inner) = true;
        condvar.notify_all();
        Ok(())
    }
}

/// The activation parameters and the `PROPVARIANT` that points at them, kept at one address
/// until the activation completes.
struct Activation {
    params: AUDIOCLIENT_ACTIVATION_PARAMS,
    /// Never dropped: the `windows` crate's `Drop` for `PROPVARIANT` calls `PropVariantClear`,
    /// which hands a `VT_BLOB`'s data to `CoTaskMemFree`, and the data here is `params` above,
    /// not COM memory. Dropping it corrupted the heap (measured: `0xC0000374` right at the free).
    prop: std::mem::ManuallyDrop<PROPVARIANT>,
}

/// Activate a process-loopback audio client for `pid` and initialise it for capture.
fn open(pid: u32, mode: PROCESS_LOOPBACK_MODE) -> Result<Stream, String> {
    // The 24 kHz mono stream the replay stores, converted by Windows' own resampler; then the
    // format of Microsoft's sample (44.1 kHz stereo, no conversion flags), converted here.
    let mut last_error = String::new();
    for (rate, channels, convert) in [(SAMPLE_RATE, 1u16, true), (44_100, 2, false)] {
        let client = activate(pid, mode)?;
        match initialize(client, rate, channels, convert) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

fn activate(pid: u32, mode: PROCESS_LOOPBACK_MODE) -> Result<IAudioClient, String> {
    let mut activation = Box::new(Activation {
        params: AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: pid,
                    ProcessLoopbackMode: mode,
                },
            },
        },
        prop: std::mem::ManuallyDrop::new(PROPVARIANT::default()),
    });
    let params_ptr: *mut AUDIOCLIENT_ACTIVATION_PARAMS = &mut activation.params;
    *activation.prop = PROPVARIANT {
        Anonymous: PROPVARIANT_0 {
            Anonymous: std::mem::ManuallyDrop::new(PROPVARIANT_0_0 {
                vt: VT_BLOB,
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: PROPVARIANT_0_0_0 {
                    blob: BLOB {
                        cbSize: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
                        pBlobData: params_ptr.cast(),
                    },
                },
            }),
        },
    };
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    let handler: IActivateAudioInterfaceCompletionHandler = Completion {
        done: Arc::clone(&done),
    }
    .into();
    // SAFETY: the PROPVARIANT and the parameters it points at live in `activation`, a box that
    // is freed only after the completion handler ran (or leaked if it never does, below). The
    // handler is a valid agile COM object.
    let operation = unsafe {
        ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(&*activation.prop),
            &handler,
        )
    }
    .map_err(|e| unsupported(&e.to_string()))?;
    let (lock, condvar) = &*done;
    let finished = {
        let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        let (guard, _) = condvar
            .wait_timeout_while(guard, ACTIVATE_TIMEOUT, |done| !*done)
            .unwrap_or_else(PoisonError::into_inner);
        *guard
    };
    if !finished {
        // The system may still read the parameters later: never free them.
        std::mem::forget(activation);
        return Err("Windows did not answer the desktop audio request in time".into());
    }
    let mut result = HRESULT(0);
    let mut unknown = None;
    // SAFETY: the operation completed; both out-pointers are valid locals.
    unsafe { operation.GetActivateResult(&mut result, &mut unknown) }
        .map_err(|e| unsupported(&e.to_string()))?;
    result.ok().map_err(|e| unsupported(&e.to_string()))?;
    drop(activation);
    unknown
        .ok_or_else(|| "Windows returned no audio client".to_owned())?
        .cast::<IAudioClient>()
        .map_err(|e| format!("desktop audio client: {e}"))
}

/// The activation failed: most likely Windows is older than 10 version 2004.
fn unsupported(error: &str) -> String {
    format!(
        "Windows could not record other programs' sound (needs Windows 10 version 2004 or \
         newer): {error}"
    )
}

fn initialize(
    client: IAudioClient,
    rate: u32,
    channels: u16,
    convert: bool,
) -> Result<Stream, String> {
    let block_align = channels * 2;
    let format = WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_PCM as u16,
        nChannels: channels,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * u32::from(block_align),
        nBlockAlign: block_align,
        wBitsPerSample: 16,
        cbSize: 0,
    };
    let mut flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    if convert {
        flags |= AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    }
    // SAFETY: an unnamed auto-reset event, closed by `Event`'s drop.
    let event = Event(
        unsafe { CreateEventW(None, false, false, None) }
            .map_err(|e| format!("desktop audio event: {e}"))?,
    );
    // SAFETY: `format` is a valid PCM WAVEFORMATEX that outlives the call; a 200 ms shared-mode
    // buffer, periodicity 0 as shared mode requires.
    unsafe { client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 2_000_000, 0, &format, None) }
        .map_err(|e| format!("desktop audio format {rate} Hz x{channels}: {e}"))?;
    // SAFETY: the event handle is valid for the client's lifetime (see `Stream`'s field order).
    unsafe { client.SetEventHandle(event.0) }.map_err(|e| format!("desktop audio event: {e}"))?;
    // SAFETY: the client is initialised; the service is a COM interface it owns.
    let capture: IAudioCaptureClient =
        unsafe { client.GetService() }.map_err(|e| format!("desktop audio capture: {e}"))?;
    Ok(Stream {
        client,
        capture,
        rate,
        channels,
        event,
    })
}

impl Stream {
    /// Capture until stopped, the stream fails, or the watched process (Discord) exits.
    fn capture(
        &self,
        shared: &Shared,
        delivery: &mut Delivery,
        watched: Option<Watched>,
    ) -> Result<Ended, String> {
        delivery.set_format(self.rate, self.channels);
        let mut out = |samples: &[i16]| shared.push(samples.iter().copied());
        // SAFETY: an initialised client.
        unsafe { self.client.Start() }.map_err(|e| format!("desktop audio start: {e}"))?;
        let mut checked = Instant::now();
        let result = loop {
            if shared.stopping() {
                break Ok(Ended::Stopped);
            }
            // SAFETY: a valid event handle; the timeout bounds the wait.
            let _ = unsafe { WaitForSingleObject(self.event.0, WAIT_MS) };
            let mut got = false;
            loop {
                // SAFETY: an initialised, started capture client.
                let pending = match unsafe { self.capture.GetNextPacketSize() } {
                    Ok(n) => n,
                    Err(e) => return self.stop_with(Err(format!("desktop audio: {e}"))),
                };
                if pending == 0 {
                    break;
                }
                let mut data: *mut u8 = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut flags = 0u32;
                // SAFETY: the out-pointers are valid locals; the buffer is released below
                // before the next `GetBuffer`.
                let buffer = unsafe {
                    self.capture
                        .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                };
                if let Err(e) = buffer {
                    return self.stop_with(Err(format!("desktop audio: {e}")));
                }
                let now = Instant::now();
                let count = frames as usize * usize::from(self.channels);
                let silent = flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
                if silent || data.is_null() || count == 0 {
                    delivery.packet(now, None, frames as usize, &mut out);
                } else {
                    // SAFETY: WASAPI hands out `frames` frames of the 16-bit format we asked for,
                    // aligned for i16 and valid until `ReleaseBuffer`.
                    let samples = unsafe { std::slice::from_raw_parts(data.cast::<i16>(), count) };
                    delivery.packet(now, Some(samples), frames as usize, &mut out);
                }
                // SAFETY: releases exactly the packet `GetBuffer` gave.
                if let Err(e) = unsafe { self.capture.ReleaseBuffer(frames) } {
                    return self.stop_with(Err(format!("desktop audio: {e}")));
                }
                got = true;
            }
            if !got {
                delivery.idle(Instant::now(), &mut out);
            }
            if let Some(watched) = &watched {
                if checked.elapsed() >= DISCORD_CHECK {
                    checked = Instant::now();
                    if watched.exited() {
                        break Ok(Ended::TargetExited);
                    }
                }
            }
        };
        self.stop_with(result)
    }

    fn stop_with(&self, result: Result<Ended, String>) -> Result<Ended, String> {
        // SAFETY: stopping a started (or failed) client is always allowed.
        let _ = unsafe { self.client.Stop() };
        result
    }
}
