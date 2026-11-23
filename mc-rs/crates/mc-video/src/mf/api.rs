//! Media Foundation's free functions, loaded at run time.
//!
//! Linking them (what `windows::Win32::Media::MediaFoundation::MFStartup` and friends do, through
//! `raw-dylib`) would put mfplat.dll and mfreadwrite.dll in `mc-client.exe`'s import table, and
//! Windows refuses to start a program whose imports are missing. Windows "N" editions ship without
//! Media Foundation unless the Media Feature Pack is installed, so the client would not start at
//! all there. Loading them here instead turns "no Media Foundation" into an ordinary
//! [`VideoError::Unavailable`] when an export is attempted.
//!
//! Every COM method call goes through an interface's vtable and needs no import, so the functions
//! below are the whole surface.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{Interface, GUID, HRESULT, PCSTR, PCWSTR};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFAttributes, IMFMediaBuffer, IMFMediaType, IMFSample, IMFSinkWriter,
    MFT_REGISTER_TYPE_INFO,
};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32,
};

use crate::VideoError;

type StartupFn = unsafe extern "system" fn(u32, u32) -> HRESULT;
type ShutdownFn = unsafe extern "system" fn() -> HRESULT;
type CreateAttributesFn = unsafe extern "system" fn(*mut *mut c_void, u32) -> HRESULT;
type CreateObjectFn = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
type CreateMemoryBufferFn = unsafe extern "system" fn(u32, *mut *mut c_void) -> HRESULT;
type TEnumExFn = unsafe extern "system" fn(
    GUID,
    u32,
    *const MFT_REGISTER_TYPE_INFO,
    *const MFT_REGISTER_TYPE_INFO,
    *mut *mut *mut c_void,
    *mut u32,
) -> HRESULT;
type CreateSinkWriterFn =
    unsafe extern "system" fn(PCWSTR, *mut c_void, *mut c_void, *mut *mut c_void) -> HRESULT;
type CreateFileFn = unsafe extern "system" fn(i32, i32, i32, PCWSTR, *mut *mut c_void) -> HRESULT;
type CreateSourceReaderFn =
    unsafe extern "system" fn(PCWSTR, *mut c_void, *mut *mut c_void) -> HRESULT;

/// The resolved entry points. Function pointers into system DLLs that stay loaded for the life of
/// the process (never freed), so they are valid for `'static`.
pub(crate) struct MfApi {
    startup: StartupFn,
    shutdown: ShutdownFn,
    create_attributes: CreateAttributesFn,
    create_media_type: CreateObjectFn,
    create_memory_buffer: CreateMemoryBufferFn,
    create_sample: CreateObjectFn,
    tenum_ex: TEnumExFn,
    create_sink_writer: CreateSinkWriterFn,
    create_file: CreateFileFn,
    create_source_reader: CreateSourceReaderFn,
}

static API: OnceLock<Result<MfApi, String>> = OnceLock::new();

/// The entry points, loaded on first use.
///
/// # Errors
/// [`VideoError::Unavailable`] when a DLL or function is missing.
pub(crate) fn api() -> Result<&'static MfApi, VideoError> {
    API.get_or_init(load)
        .as_ref()
        .map_err(|why| VideoError::Unavailable(why.clone()))
}

/// Wide, NUL-terminated.
pub(crate) fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn load() -> Result<MfApi, String> {
    let module = |name: &str| {
        let wide_name = wide(name);
        // SAFETY: a valid NUL-terminated wide string; System32 only, so no planted DLL next to
        // the exe can be picked up. The handle is never freed.
        unsafe { LoadLibraryExW(PCWSTR(wide_name.as_ptr()), None, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .map_err(|error| format!("{name} could not be loaded ({error}); Windows N editions need the Media Feature Pack"))
    };
    let mfplat = module("mfplat.dll")?;
    let mfreadwrite = module("mfreadwrite.dll")?;
    macro_rules! proc {
        ($module:expr, $name:literal, $ty:ty) => {{
            // SAFETY: a valid module handle and a NUL-terminated ANSI name.
            let found = unsafe { GetProcAddress($module, PCSTR(concat!($name, "\0").as_ptr())) }
                .ok_or_else(|| format!("{} is missing from Media Foundation", $name))?;
            // SAFETY: the documented signature of this export (mfapi.h / mfreadwrite.h), written
            // out in the type aliases above; a FARPROC and an `extern "system"` fn pointer have
            // the same size and representation.
            unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(found) }
        }};
    }
    Ok(MfApi {
        startup: proc!(mfplat, "MFStartup", StartupFn),
        shutdown: proc!(mfplat, "MFShutdown", ShutdownFn),
        create_attributes: proc!(mfplat, "MFCreateAttributes", CreateAttributesFn),
        create_media_type: proc!(mfplat, "MFCreateMediaType", CreateObjectFn),
        create_memory_buffer: proc!(mfplat, "MFCreateMemoryBuffer", CreateMemoryBufferFn),
        create_sample: proc!(mfplat, "MFCreateSample", CreateObjectFn),
        tenum_ex: proc!(mfplat, "MFTEnumEx", TEnumExFn),
        create_sink_writer: proc!(mfreadwrite, "MFCreateSinkWriterFromURL", CreateSinkWriterFn),
        create_file: proc!(mfplat, "MFCreateFile", CreateFileFn),
        create_source_reader: proc!(
            mfreadwrite,
            "MFCreateSourceReaderFromURL",
            CreateSourceReaderFn
        ),
    })
}

/// Take ownership of a COM pointer an out-parameter returned.
///
/// # Safety
/// `raw` must be null or an owned (AddRef'd) pointer to an object implementing `T`.
unsafe fn owned<T: Interface>(raw: *mut c_void, what: &'static str) -> windows::core::Result<T> {
    if raw.is_null() {
        return Err(windows::core::Error::new(
            windows::Win32::Foundation::E_POINTER,
            what,
        ));
    }
    // SAFETY: the caller guarantees an owned pointer to a `T`.
    Ok(unsafe { T::from_raw(raw) })
}

impl MfApi {
    /// `MFStartup(MF_VERSION, MFSTARTUP_LITE)`.
    pub(crate) fn startup(&self) -> windows::core::Result<()> {
        use windows::Win32::Media::MediaFoundation::{MFSTARTUP_LITE, MF_VERSION};
        // SAFETY: plain values in, HRESULT out.
        unsafe { (self.startup)(MF_VERSION, MFSTARTUP_LITE) }.ok()
    }

    /// `MFShutdown()`, balancing one [`Self::startup`].
    pub(crate) fn shutdown(&self) {
        // SAFETY: no arguments; called once per successful startup, after every object made
        // under it has been released (the session guard is dropped last).
        let _ = unsafe { (self.shutdown)() };
    }

    /// `MFCreateAttributes`.
    pub(crate) fn attributes(&self, initial: u32) -> windows::core::Result<IMFAttributes> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: a valid out-pointer; on success it holds an owned IMFAttributes.
        unsafe {
            (self.create_attributes)(&mut raw, initial).ok()?;
            owned(raw, "MFCreateAttributes")
        }
    }

    /// `MFCreateMediaType`.
    pub(crate) fn media_type(&self) -> windows::core::Result<IMFMediaType> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: as above.
        unsafe {
            (self.create_media_type)(&mut raw).ok()?;
            owned(raw, "MFCreateMediaType")
        }
    }

    /// `MFCreateMemoryBuffer`.
    pub(crate) fn memory_buffer(&self, bytes: u32) -> windows::core::Result<IMFMediaBuffer> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: as above.
        unsafe {
            (self.create_memory_buffer)(bytes, &mut raw).ok()?;
            owned(raw, "MFCreateMemoryBuffer")
        }
    }

    /// `MFCreateSample`.
    pub(crate) fn sample(&self) -> windows::core::Result<IMFSample> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: as above.
        unsafe {
            (self.create_sample)(&mut raw).ok()?;
            owned(raw, "MFCreateSample")
        }
    }

    /// `MFCreateFile(MF_ACCESSMODE_READWRITE, MF_OPENMODE_DELETE_IF_EXIST, MF_FILEFLAGS_NONE)`: a
    /// new file the MP4 sink can also read back, which the fast-start rewrite (`moov` moved in
    /// front of the data at `Finalize`) needs. The sink writer's own file opening is write-only,
    /// and `Finalize` then fails with "Access is denied" (measured).
    pub(crate) fn read_write_file(
        &self,
        path: &str,
    ) -> windows::core::Result<windows::Win32::Media::MediaFoundation::IMFByteStream> {
        use windows::Win32::Media::MediaFoundation::{
            MF_ACCESSMODE_READWRITE, MF_FILEFLAGS_NONE, MF_OPENMODE_DELETE_IF_EXIST,
        };
        let path = wide(path);
        let mut raw = std::ptr::null_mut();
        // SAFETY: plain enum values, a NUL-terminated path that outlives the call, a valid
        // out-pointer.
        unsafe {
            (self.create_file)(
                MF_ACCESSMODE_READWRITE.0,
                MF_OPENMODE_DELETE_IF_EXIST.0,
                MF_FILEFLAGS_NONE.0,
                PCWSTR(path.as_ptr()),
                &mut raw,
            )
            .ok()?;
            owned(raw, "MFCreateFile")
        }
    }

    /// `MFCreateSinkWriterFromURL(NULL, stream, attributes)`: the container type comes from
    /// `MF_TRANSCODE_CONTAINERTYPE` in `attributes`.
    pub(crate) fn sink_writer(
        &self,
        stream: &windows::Win32::Media::MediaFoundation::IMFByteStream,
        attributes: &IMFAttributes,
    ) -> windows::core::Result<IMFSinkWriter> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: a borrowed byte stream and attribute store (the callee AddRefs what it keeps)
        // and a valid out-pointer.
        unsafe {
            (self.create_sink_writer)(
                PCWSTR::null(),
                stream.as_raw(),
                attributes.as_raw(),
                &mut raw,
            )
            .ok()?;
            owned(raw, "MFCreateSinkWriterFromURL")
        }
    }

    /// `MFCreateSourceReaderFromURL(path, attributes)`: reads (and decodes) a media file.
    pub(crate) fn source_reader(
        &self,
        path: &str,
        attributes: &IMFAttributes,
    ) -> windows::core::Result<windows::Win32::Media::MediaFoundation::IMFSourceReader> {
        let path = wide(path);
        let mut raw = std::ptr::null_mut();
        // SAFETY: a NUL-terminated path that outlives the call, a borrowed attribute store and a
        // valid out-pointer.
        unsafe {
            (self.create_source_reader)(PCWSTR(path.as_ptr()), attributes.as_raw(), &mut raw)
                .ok()?;
            owned(raw, "MFCreateSourceReaderFromURL")
        }
    }

    /// `MFTEnumEx`: the activation objects of every MFT in `category` that turns `input` into
    /// `output`, in Media Foundation's preferred order.
    pub(crate) fn enum_transforms(
        &self,
        category: GUID,
        flags: u32,
        input: Option<MFT_REGISTER_TYPE_INFO>,
        output: Option<MFT_REGISTER_TYPE_INFO>,
    ) -> windows::core::Result<Vec<IMFActivate>> {
        let mut array: *mut *mut c_void = std::ptr::null_mut();
        let mut count = 0_u32;
        let input_ptr = input.as_ref().map_or(std::ptr::null(), |i| i as *const _);
        let output_ptr = output.as_ref().map_or(std::ptr::null(), |o| o as *const _);
        // SAFETY: the type-info pointers are null or point at locals that outlive the call; the
        // out-parameters are valid.
        unsafe {
            (self.tenum_ex)(
                category, flags, input_ptr, output_ptr, &mut array, &mut count,
            )
            .ok()?;
        }
        let mut found = Vec::with_capacity(count as usize);
        if !array.is_null() {
            for i in 0..count as usize {
                // SAFETY: MFTEnumEx returns `count` owned IMFActivate pointers in a
                // CoTaskMemAlloc'd array; each is taken over exactly once here.
                let raw = unsafe { *array.add(i) };
                if !raw.is_null() {
                    // SAFETY: an owned IMFActivate pointer (see above).
                    found.push(unsafe { IMFActivate::from_raw(raw) });
                }
            }
            // SAFETY: the array itself is CoTaskMemAlloc'd and owned by the caller.
            unsafe { windows::Win32::System::Com::CoTaskMemFree(Some(array as *const c_void)) };
        }
        Ok(found)
    }
}

/// One Media Foundation session on the current thread: COM (multithreaded apartment) plus
/// `MFStartup`, undone on drop. Keep it alive longer than every Media Foundation object made under
/// it (declare it as the *last* field of a struct).
pub(crate) struct MfSession {
    api: &'static MfApi,
    uninitialize_com: bool,
    /// COM initialisation is per thread: the session must end on the thread that started it.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl MfSession {
    /// Start COM and Media Foundation on this thread.
    ///
    /// # Errors
    /// [`VideoError::Unavailable`] without Media Foundation; [`VideoError::MediaFoundation`] when
    /// `MFStartup` fails.
    pub(crate) fn start() -> Result<Self, VideoError> {
        use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
        let api = api()?;
        // SAFETY: no reserved pointer. S_OK and S_FALSE both need a matching CoUninitialize;
        // RPC_E_CHANGED_MODE (a single-threaded apartment already) needs none, and Media
        // Foundation still works there.
        let uninitialize_com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok();
        if let Err(error) = api.startup() {
            if uninitialize_com {
                // SAFETY: balances the successful CoInitializeEx above.
                unsafe { windows::Win32::System::Com::CoUninitialize() };
            }
            return Err(VideoError::mf("MFStartup", &error));
        }
        Ok(Self {
            api,
            uninitialize_com,
            _not_send: std::marker::PhantomData,
        })
    }

    /// The entry points.
    pub(crate) fn api(&self) -> &'static MfApi {
        self.api
    }
}

impl Drop for MfSession {
    fn drop(&mut self) {
        self.api.shutdown();
        if self.uninitialize_com {
            // SAFETY: balances the successful CoInitializeEx in `start`, on the same thread
            // (`_not_send` keeps the session on the thread that started it).
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
    }
}
