//! Windows DPAPI: `CryptProtectData` and `CryptUnprotectData`, which seal bytes so that only the
//! same Windows user on the same PC can open them again.
//!
//! This is the only `unsafe` code in `mc-auth`, and it is this small on purpose: three Win32
//! functions declared by hand (two from `crypt32.dll`, `LocalFree` from `kernel32.dll`) and the
//! `DATA_BLOB` they exchange. Declaring them here rather than through a bindings crate keeps the
//! workspace's Windows bindings (and every lane's build) unchanged.
//!
//! What it protects against: a copied or uploaded `accounts.json` (a backup, a synced folder, a
//! support zip, another Windows account on the same PC) cannot be opened. What it does not: code
//! running as the same Windows user can ask DPAPI to open the blob just as this client does.
//!
//! Both calls pass `CRYPTPROTECT_UI_FORBIDDEN` (never show a prompt), no description, no
//! reserved pointer and no prompt struct, plus a fixed, non-secret "entropy" value chosen by the
//! caller: it binds a blob to its purpose, so a blob sealed for one use is not opened by a call
//! made for another.

use std::ffi::c_void;
use std::io;

/// `DATA_BLOB` / `CRYPT_INTEGER_BLOB`.
#[repr(C)]
struct DataBlob {
    cb_data: u32,
    pb_data: *mut u8,
}

impl DataBlob {
    /// A blob that points at `bytes` (read-only use: DPAPI never writes to its inputs).
    fn borrowing(bytes: &[u8]) -> io::Result<DataBlob> {
        let cb_data = u32::try_from(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too large to seal"))?;
        Ok(DataBlob {
            cb_data,
            pb_data: bytes.as_ptr().cast_mut(),
        })
    }

    /// An empty output blob for DPAPI to fill.
    fn empty() -> DataBlob {
        DataBlob {
            cb_data: 0,
            pb_data: std::ptr::null_mut(),
        }
    }
}

/// `CRYPTPROTECT_UI_FORBIDDEN`: fail rather than show any UI.
const CRYPTPROTECT_UI_FORBIDDEN: u32 = 0x1;

#[link(name = "crypt32")]
extern "system" {
    /// `BOOL CryptProtectData(DATA_BLOB *pDataIn, LPCWSTR szDataDescr, DATA_BLOB
    /// *pOptionalEntropy, PVOID pvReserved, CRYPTPROTECT_PROMPTSTRUCT *pPromptStruct, DWORD
    /// dwFlags, DATA_BLOB *pDataOut)`.
    fn CryptProtectData(
        data_in: *const DataBlob,
        description: *const u16,
        optional_entropy: *const DataBlob,
        reserved: *const c_void,
        prompt: *const c_void,
        flags: u32,
        data_out: *mut DataBlob,
    ) -> i32;

    /// `BOOL CryptUnprotectData(DATA_BLOB *pDataIn, LPWSTR *ppszDataDescr, DATA_BLOB
    /// *pOptionalEntropy, PVOID pvReserved, CRYPTPROTECT_PROMPTSTRUCT *pPromptStruct, DWORD
    /// dwFlags, DATA_BLOB *pDataOut)`.
    fn CryptUnprotectData(
        data_in: *const DataBlob,
        description: *mut *mut u16,
        optional_entropy: *const DataBlob,
        reserved: *const c_void,
        prompt: *const c_void,
        flags: u32,
        data_out: *mut DataBlob,
    ) -> i32;
}

#[link(name = "kernel32")]
extern "system" {
    /// `HLOCAL LocalFree(HLOCAL hMem)`: DPAPI allocates its outputs with `LocalAlloc`.
    fn LocalFree(mem: *mut c_void) -> *mut c_void;
}

/// Copy a DPAPI output blob into a `Vec`, wipe it when it held plaintext, and free it.
///
/// # Safety
///
/// `blob` must be an output blob a successful DPAPI call filled: `pb_data` points at `cb_data`
/// bytes allocated with `LocalAlloc`, owned by the caller, not yet freed.
unsafe fn take_output(blob: &DataBlob, wipe: bool) -> Vec<u8> {
    if blob.pb_data.is_null() {
        return Vec::new();
    }
    let len = blob.cb_data as usize;
    // SAFETY: per this function's contract the pointer is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(blob.pb_data, len) }.to_vec();
    if wipe {
        // SAFETY: the same `len` bytes, still owned by us; zeroing them before the free keeps
        // a plaintext copy out of the freed heap block.
        unsafe { std::ptr::write_bytes(blob.pb_data, 0, len) };
    }
    // SAFETY: the pointer came from DPAPI's `LocalAlloc` and is freed exactly once, here.
    unsafe { LocalFree(blob.pb_data.cast()) };
    bytes
}

/// Seal `plain` for the current Windows user. `entropy` must be the same when unsealing.
pub(crate) fn protect(plain: &[u8], entropy: &[u8]) -> io::Result<Vec<u8>> {
    let data_in = DataBlob::borrowing(plain)?;
    let entropy = DataBlob::borrowing(entropy)?;
    let mut data_out = DataBlob::empty();
    // SAFETY: both input blobs point at live slices for the duration of the call; the
    // description, reserved and prompt pointers are null, which the API documents as allowed;
    // `data_out` is a valid, writable blob that receives a `LocalAlloc` buffer on success.
    let ok = unsafe {
        CryptProtectData(
            &data_in,
            std::ptr::null(),
            &entropy,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut data_out,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call succeeded, so `data_out` is DPAPI's own allocation, freed once in there.
    Ok(unsafe { take_output(&data_out, false) })
}

/// Open a blob [`protect`] sealed. Fails for a blob sealed by another Windows user or on another
/// PC, for a damaged blob, and for a different `entropy`.
pub(crate) fn unprotect(sealed: &[u8], entropy: &[u8]) -> io::Result<Vec<u8>> {
    let data_in = DataBlob::borrowing(sealed)?;
    let entropy = DataBlob::borrowing(entropy)?;
    let mut data_out = DataBlob::empty();
    // SAFETY: as in `protect`; a null `description` asks DPAPI not to return the description,
    // so there is nothing else to free.
    let ok = unsafe {
        CryptUnprotectData(
            &data_in,
            std::ptr::null_mut(),
            &entropy,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut data_out,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call succeeded; the plaintext buffer is wiped and freed once in there.
    Ok(unsafe { take_output(&data_out, true) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTROPY: &[u8] = b"mc-auth dpapi test";

    #[test]
    fn a_sealed_blob_opens_again_and_hides_its_contents() {
        let plain = b"fake-refresh-token-for-a-test-0123456789";
        let sealed = protect(plain, ENTROPY).expect("protect");
        assert!(sealed.len() > plain.len(), "DPAPI adds its header and MAC");
        assert!(
            !sealed.windows(plain.len()).any(|window| window == plain),
            "the plaintext must not appear in the sealed blob"
        );
        assert_eq!(unprotect(&sealed, ENTROPY).expect("unprotect"), plain);
    }

    #[test]
    fn sealing_twice_gives_different_blobs() {
        let a = protect(b"same input", ENTROPY).expect("protect");
        let b = protect(b"same input", ENTROPY).expect("protect");
        assert_ne!(a, b, "each seal uses a fresh random salt");
    }

    #[test]
    fn another_purpose_or_a_damaged_blob_does_not_open() {
        let sealed = protect(b"secret bytes", ENTROPY).expect("protect");
        assert!(unprotect(&sealed, b"some other purpose").is_err());
        let mut damaged = sealed.clone();
        let last = damaged.len() - 1;
        damaged[last] ^= 0x55;
        assert!(unprotect(&damaged, ENTROPY).is_err());
        assert!(unprotect(&[], ENTROPY).is_err());
        assert!(unprotect(b"not a dpapi blob at all", ENTROPY).is_err());
    }

    #[test]
    fn an_empty_input_round_trips() {
        let sealed = protect(&[], ENTROPY).expect("protect");
        assert!(unprotect(&sealed, ENTROPY).expect("unprotect").is_empty());
    }
}
