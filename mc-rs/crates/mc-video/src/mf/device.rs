//! The hardware video encoder on the graphics adapter the game renders
//! with.
//!
//! Left to itself, the sink writer takes a hardware encoder from Media Foundation's list, and that
//! encoder makes a Direct3D 11 device of its own on the adapter it belongs to. On a PC with two
//! graphics adapters (a graphics card and the processor's own graphics, say) that need not be the
//! chip that renders the export: on the PC this was found on, the list starts with the integrated
//! AMD chip's encoder, and exports rendered on that chip still ended up on the other card's
//! encoder. Handing the sink writer a DXGI device manager made on the render adapter
//! (`MF_SINK_WRITER_D3D_MANAGER`) makes it use an encoder that runs there, on a device the export
//! owns.
//!
//! The adapter is matched by its PCI vendor and device IDs (what the renderer reports for Vulkan
//! and Direct3D 12 alike), and by name when two adapters share them. `dxgi.dll`, `d3d11.dll` and
//! `MFCreateDXGIDeviceManager` are loaded at run time, like the rest of Media Foundation
//! (`api.rs`), so the executable gains no import.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{Interface, GUID, HRESULT, PCSTR, PCWSTR};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION,
};
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE};
use windows::Win32::Media::MediaFoundation::IMFDXGIDeviceManager;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32,
};

use crate::GpuAdapter;

type CreateFactoryFn = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;
type CreateDeviceFn = unsafe extern "system" fn(
    *mut c_void,
    i32,
    *mut c_void,
    u32,
    *const D3D_FEATURE_LEVEL,
    u32,
    u32,
    *mut *mut c_void,
    *mut D3D_FEATURE_LEVEL,
    *mut *mut c_void,
) -> HRESULT;
type CreateManagerFn = unsafe extern "system" fn(*mut u32, *mut *mut c_void) -> HRESULT;

/// The three entry points. Function pointers into system DLLs that stay loaded for the life of
/// the process (never freed), so they are valid for `'static`.
struct Entry {
    create_factory: CreateFactoryFn,
    create_device: CreateDeviceFn,
    create_manager: CreateManagerFn,
}

static ENTRY: OnceLock<Result<Entry, String>> = OnceLock::new();

fn entry() -> Result<&'static Entry, String> {
    ENTRY.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn load() -> Result<Entry, String> {
    let module = |name: &str| {
        let wide_name = super::api::wide(name);
        // SAFETY: a valid NUL-terminated wide string; System32 only, so no planted DLL next to
        // the exe can be picked up. The handle is never freed.
        unsafe {
            LoadLibraryExW(
                PCWSTR(wide_name.as_ptr()),
                None,
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        }
        .map_err(|error| format!("{name} could not be loaded ({error})"))
    };
    let dxgi = module("dxgi.dll")?;
    let d3d11 = module("d3d11.dll")?;
    let mfplat = module("mfplat.dll")?;
    macro_rules! proc {
        ($module:expr, $name:literal, $ty:ty) => {{
            // SAFETY: a valid module handle and a NUL-terminated ANSI name.
            let found = unsafe { GetProcAddress($module, PCSTR(concat!($name, "\0").as_ptr())) }
                .ok_or_else(|| format!("{} is missing", $name))?;
            // SAFETY: the documented signature of this export (dxgi.h, d3d11.h, mfapi.h), written
            // out in the type aliases above; a FARPROC and an `extern "system"` fn pointer have
            // the same size and representation.
            unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(found) }
        }};
    }
    Ok(Entry {
        create_factory: proc!(dxgi, "CreateDXGIFactory1", CreateFactoryFn),
        create_device: proc!(d3d11, "D3D11CreateDevice", CreateDeviceFn),
        create_manager: proc!(mfplat, "MFCreateDXGIDeviceManager", CreateManagerFn),
    })
}

/// A Direct3D 11 device on the render adapter, wrapped in the DXGI device manager the sink
/// writer hands to its hardware encoder. Keep it alive as long as the sink writer.
pub(crate) struct EncoderDevice {
    /// What `MF_SINK_WRITER_D3D_MANAGER` takes.
    pub(crate) manager: IMFDXGIDeviceManager,
    /// The adapter's name as DXGI reports it.
    pub(crate) adapter: String,
    /// The adapter's locally unique identifier (`MFT_ENUM_ADAPTER_LUID` compares with it).
    pub(crate) luid: u64,
    /// Held so the device outlives the manager's own reference, whatever the encoder does.
    _device: ID3D11Device,
}

/// Take ownership of a COM pointer an out-parameter returned.
///
/// # Safety
/// `raw` must be null or an owned (AddRef'd) pointer to an object implementing `T`.
unsafe fn owned<T: Interface>(raw: *mut c_void, what: &str) -> Result<T, String> {
    if raw.is_null() {
        return Err(format!("{what} returned nothing"));
    }
    // SAFETY: the caller guarantees an owned pointer to a `T`.
    Ok(unsafe { T::from_raw(raw) })
}

/// A LUID as one number (high part above the low part), as `MFT_ENUM_ADAPTER_LUID` stores it.
fn luid_u64(luid: windows::Win32::Foundation::LUID) -> u64 {
    (u64::from(luid.HighPart as u32) << 32) | u64::from(luid.LowPart)
}

/// `MC_W19_R2_ENCODER_ADAPTER=<part of a name>` (tests only, read once): put the encoder on that
/// adapter instead of the render adapter, to reproduce an encoder on another chip than the one
/// rendering. Unset for every player.
fn adapter_override() -> Option<&'static str> {
    static NAME: OnceLock<Option<String>> = OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("MC_W19_R2_ENCODER_ADAPTER")
            .ok()
            .map(|name| name.trim().to_ascii_lowercase())
            .filter(|name| !name.is_empty())
    })
    .as_deref()
}

/// The DXGI adapter that is `wanted`: same PCI vendor and device; by name among several.
fn find_adapter(
    entry: &Entry,
    wanted: &GpuAdapter,
) -> Result<(IDXGIAdapter1, String, u64), String> {
    let by_name = adapter_override();
    let mut raw = std::ptr::null_mut();
    // SAFETY: a valid interface ID and out-pointer.
    unsafe { (entry.create_factory)(&IDXGIFactory1::IID, &mut raw) }
        .ok()
        .map_err(|error| format!("CreateDXGIFactory1 failed ({error})"))?;
    // SAFETY: on success an owned IDXGIFactory1.
    let factory: IDXGIFactory1 = unsafe { owned(raw, "CreateDXGIFactory1") }?;
    let mut found = Vec::new();
    // For the log (once per export): which adapters this PC has, in DXGI's order.
    let mut all = Vec::new();
    for index in 0..64 {
        // SAFETY: a live factory; the call fails (DXGI_ERROR_NOT_FOUND) past the last adapter.
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        // SAFETY: a live adapter.
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        let end = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..end]);
        all.push(format!(
            "{name} {:04X}:{:04X}",
            desc.VendorId, desc.DeviceId
        ));
        let wanted_here = match by_name {
            Some(part) => name.to_ascii_lowercase().contains(part),
            None => desc.VendorId == wanted.vendor && desc.DeviceId == wanted.device,
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || !wanted_here {
            continue;
        }
        found.push((adapter, name, luid_u64(desc.AdapterLuid)));
    }
    tracing::info!(adapters = ?all, "graphics adapters for the video encoder");
    if found.len() > 1 {
        if let Some(at) = found
            .iter()
            .position(|(_, name, _)| name.trim().eq_ignore_ascii_case(wanted.name.trim()))
        {
            return Ok(found.swap_remove(at));
        }
    }
    found.into_iter().next().ok_or_else(|| {
        format!(
            "no graphics adapter {:04X}:{:04X} ({})",
            wanted.vendor, wanted.device, wanted.name
        )
    })
}

/// A Direct3D 11 device with video support on `adapter`, protected for use from Media
/// Foundation's threads.
fn create_device(entry: &Entry, adapter: &IDXGIAdapter1) -> Result<ID3D11Device, String> {
    let levels = [
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    ];
    let flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT.0 | D3D11_CREATE_DEVICE_BGRA_SUPPORT.0;
    let mut device = std::ptr::null_mut();
    let mut context = std::ptr::null_mut();
    let mut level = D3D_FEATURE_LEVEL::default();
    // SAFETY: a live adapter (an explicit adapter needs D3D_DRIVER_TYPE_UNKNOWN and no software
    // module), a feature-level array that outlives the call and valid out-pointers.
    unsafe {
        (entry.create_device)(
            adapter.as_raw(),
            D3D_DRIVER_TYPE_UNKNOWN.0,
            std::ptr::null_mut(),
            flags,
            levels.as_ptr(),
            levels.len() as u32,
            D3D11_SDK_VERSION,
            &mut device,
            &mut level,
            &mut context,
        )
    }
    .ok()
    .map_err(|error| format!("D3D11CreateDevice failed ({error})"))?;
    // SAFETY: on success both out-pointers hold owned references.
    let device: ID3D11Device = unsafe { owned(device, "D3D11CreateDevice") }?;
    // SAFETY: as above.
    let _context: ID3D11DeviceContext = unsafe { owned(context, "D3D11CreateDevice") }?;
    // Media Foundation's encoder works on the device from threads of its own.
    let multithread = device
        .cast::<ID3D11Multithread>()
        .map_err(|error| format!("the device has no multithread protection ({error})"))?;
    // SAFETY: a live interface; the returned previous state is not needed.
    let _ = unsafe { multithread.SetMultithreadProtected(true) };
    Ok(device)
}

/// A device manager on the render adapter `wanted`, for the sink writer.
///
/// # Errors
/// What failed, as text (no adapter with those IDs, no video support, a missing DLL...). The
/// caller then lets Media Foundation choose as it did before.
pub(crate) fn encoder_device(wanted: &GpuAdapter) -> Result<EncoderDevice, String> {
    let entry = entry()?;
    let (adapter, name, luid) = find_adapter(entry, wanted)?;
    let device = create_device(entry, &adapter)?;
    let mut token = 0_u32;
    let mut raw = std::ptr::null_mut();
    // SAFETY: valid out-pointers.
    unsafe { (entry.create_manager)(&mut token, &mut raw) }
        .ok()
        .map_err(|error| format!("MFCreateDXGIDeviceManager failed ({error})"))?;
    // SAFETY: on success an owned IMFDXGIDeviceManager.
    let manager: IMFDXGIDeviceManager = unsafe { owned(raw, "MFCreateDXGIDeviceManager") }?;
    // SAFETY: a live manager and device; the token is the one the manager was created with.
    unsafe { manager.ResetDevice(&device, token) }
        .map_err(|error| format!("the device manager refused the device ({error})"))?;
    Ok(EncoderDevice {
        manager,
        adapter: name,
        luid,
        _device: device,
    })
}
