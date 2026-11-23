//! Leak-hunting instruments for the integration tests.
//!
//! Everything here is in-process and needs no extra crate:
//!
//! * [`TracingAlloc`], a global allocator that forwards every call to `System` and keeps the
//!   live heap's bytes and blocks (allocated minus freed, every thread). While allocation-site
//!   tracing is on ([`start_tracing`]) it also records the call stack of each new block
//!   (`RtlCaptureStackBackTrace`, which allocates nothing) in a fixed table, and forgets the
//!   entry when the block is freed. [`live_sites`] then groups the blocks still alive by call
//!   stack and names the frames (dbghelp, public symbols). A structure that grows once per
//!   cycle shows up as one site with N, 2N, ... blocks after N traced cycles; something that
//!   is merely freed late shows up with a block count unrelated to N.
//! * [`ProcessCounters`]: this process's private bytes, working set, open handles and threads.
//!
//! Tracing is Windows-only (the stack walker and symbol reader are Win32); elsewhere the
//! counters read zero and [`live_sites`] is empty, so the tests still compile and run.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// The counting (and, on demand, tracing) global allocator. Install it in a test binary with
/// `#[global_allocator] static GLOBAL: TracingAlloc = TracingAlloc;`.
pub struct TracingAlloc;

static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static LIVE_BLOCKS: AtomicI64 = AtomicI64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// New blocks get their call stack recorded while this is set.
static TRACING: AtomicBool = AtomicBool::new(false);
/// Set once the tables exist; from then on every free looks its block up.
static TABLES_READY: AtomicBool = AtomicBool::new(false);
/// Guards both tables.
static LOCK: AtomicBool = AtomicBool::new(false);
static LIVE_TABLE: AtomicPtr<LiveEntry> = AtomicPtr::new(std::ptr::null_mut());
static STACK_TABLE: AtomicPtr<StackEntry> = AtomicPtr::new(std::ptr::null_mut());
/// Entries in the live table.
static TRACKED: AtomicUsize = AtomicUsize::new(0);
/// Distinct stacks interned.
static STACKS: AtomicUsize = AtomicUsize::new(0);
/// Blocks that could not be tracked because a table was full.
static UNTRACKED: AtomicU64 = AtomicU64::new(0);

/// Frames kept per stack: deep enough to get past the allocator and collection frames.
const DEPTH: usize = 40;
/// Live table slots (open addressing). 4M x 24 bytes, committed only where touched.
const LIVE_CAP: usize = 1 << 22;
/// Stack table slots.
const STACK_CAP: usize = 1 << 17;

#[derive(Clone, Copy)]
#[repr(C)]
struct LiveEntry {
    /// 0 = empty slot.
    ptr: usize,
    size: usize,
    /// Index into the stack table plus one; 0 = unknown stack.
    stack: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StackEntry {
    hash: u32,
    /// 0 = empty slot.
    len: u32,
    frames: [usize; DEPTH],
}

/// Live heap bytes, all threads.
pub fn live_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Live heap blocks, all threads.
pub fn live_blocks() -> i64 {
    LIVE_BLOCKS.load(Ordering::Relaxed)
}

/// Allocation calls so far (alloc, alloc_zeroed and realloc).
pub fn allocations() -> u64 {
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn lock() {
    let mut spins = 0u32;
    while LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        spins += 1;
        if spins > 64 {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

fn unlock() {
    LOCK.store(false, Ordering::Release);
}

fn ptr_slot(ptr: usize) -> usize {
    ((ptr >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 22)) & (LIVE_CAP - 1)
}

/// Make the tables (once). Called from the test thread, never from inside the allocator.
fn ensure_tables() -> bool {
    if TABLES_READY.load(Ordering::Acquire) {
        return true;
    }
    let live = Layout::array::<LiveEntry>(LIVE_CAP).expect("live table layout");
    let stacks = Layout::array::<StackEntry>(STACK_CAP).expect("stack table layout");
    // SAFETY: non-zero sizes; the memory is only ever reached through these two pointers, under
    // `LOCK`, and is never freed (it lives as long as the test binary).
    let (live_ptr, stack_ptr) = unsafe { (System.alloc_zeroed(live), System.alloc_zeroed(stacks)) };
    if live_ptr.is_null() || stack_ptr.is_null() {
        return false;
    }
    LIVE_TABLE.store(live_ptr.cast(), Ordering::Release);
    STACK_TABLE.store(stack_ptr.cast(), Ordering::Release);
    TABLES_READY.store(true, Ordering::Release);
    true
}

/// Record the call stack of every block allocated from now on (Windows only). Blocks that were
/// already alive are never reported.
pub fn start_tracing() -> bool {
    if !cfg!(windows) || !ensure_tables() {
        return false;
    }
    TRACING.store(true, Ordering::SeqCst);
    true
}

/// Stop recording new blocks. Recorded blocks are still forgotten when freed.
pub fn stop_tracing() {
    TRACING.store(false, Ordering::SeqCst);
}

/// Forget every recorded block (so the next traced window starts empty).
pub fn clear_tracing() {
    if !TABLES_READY.load(Ordering::Acquire) {
        return;
    }
    lock();
    let table = LIVE_TABLE.load(Ordering::Relaxed);
    for slot in 0..LIVE_CAP {
        // SAFETY: `slot < LIVE_CAP`, inside the table; exclusive under `LOCK`.
        unsafe {
            table.add(slot).write(LiveEntry {
                ptr: 0,
                size: 0,
                stack: 0,
            })
        };
    }
    TRACKED.store(0, Ordering::Relaxed);
    unlock();
}

#[cfg(windows)]
fn capture() -> ([usize; DEPTH], u32, u32) {
    let mut frames = [0usize; DEPTH];
    let mut hash = 0u32;
    // SAFETY: `frames` holds `DEPTH` pointer-sized slots, the count passed; the walker only
    // reads the current thread's stack and unwind tables and allocates nothing.
    let captured = unsafe {
        win::RtlCaptureStackBackTrace(0, DEPTH as u32, frames.as_mut_ptr().cast(), &mut hash)
    };
    (frames, u32::from(captured), hash)
}

#[cfg(not(windows))]
fn capture() -> ([usize; DEPTH], u32, u32) {
    ([0usize; DEPTH], 0, 0)
}

/// Intern a stack under `LOCK`; returns its id (index + 1), or 0 when the table is full.
fn intern(frames: &[usize; DEPTH], len: u32, hash: u32) -> u32 {
    if len == 0 {
        return 0;
    }
    let table = STACK_TABLE.load(Ordering::Relaxed);
    let mut slot = (hash as usize).wrapping_mul(0x9E37_79B9) & (STACK_CAP - 1);
    for _ in 0..STACK_CAP {
        // SAFETY: `slot < STACK_CAP`; exclusive under `LOCK`.
        let entry = unsafe { &mut *table.add(slot) };
        if entry.len == 0 {
            if STACKS.load(Ordering::Relaxed) >= STACK_CAP / 4 * 3 {
                return 0;
            }
            entry.hash = hash;
            entry.len = len;
            entry.frames = *frames;
            STACKS.fetch_add(1, Ordering::Relaxed);
            return slot as u32 + 1;
        }
        if entry.hash == hash
            && entry.len == len
            && entry.frames[..len as usize] == frames[..len as usize]
        {
            return slot as u32 + 1;
        }
        slot = (slot + 1) & (STACK_CAP - 1);
    }
    0
}

/// Insert or overwrite `ptr` under `LOCK`.
fn insert(ptr: usize, size: usize, stack: u32) {
    let table = LIVE_TABLE.load(Ordering::Relaxed);
    let mut slot = ptr_slot(ptr);
    loop {
        // SAFETY: `slot < LIVE_CAP`; exclusive under `LOCK`.
        let entry = unsafe { &mut *table.add(slot) };
        if entry.ptr == ptr {
            entry.size = size;
            entry.stack = stack;
            return;
        }
        if entry.ptr == 0 {
            if TRACKED.load(Ordering::Relaxed) >= LIVE_CAP / 4 * 3 {
                UNTRACKED.fetch_add(1, Ordering::Relaxed);
                return;
            }
            *entry = LiveEntry { ptr, size, stack };
            TRACKED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        slot = (slot + 1) & (LIVE_CAP - 1);
    }
}

/// Remove `ptr` under `LOCK` (backward-shift deletion keeps linear probing gap-free).
fn remove(ptr: usize) {
    let table = LIVE_TABLE.load(Ordering::Relaxed);
    let mut slot = ptr_slot(ptr);
    loop {
        // SAFETY: `slot < LIVE_CAP`; exclusive under `LOCK`.
        let found = unsafe { (*table.add(slot)).ptr };
        if found == 0 {
            return;
        }
        if found == ptr {
            break;
        }
        slot = (slot + 1) & (LIVE_CAP - 1);
    }
    TRACKED.fetch_sub(1, Ordering::Relaxed);
    let mut hole = slot;
    let mut next = slot;
    loop {
        next = (next + 1) & (LIVE_CAP - 1);
        // SAFETY: indices are masked into the table; exclusive under `LOCK`.
        let entry = unsafe { *table.add(next) };
        if entry.ptr == 0 {
            break;
        }
        let home = ptr_slot(entry.ptr);
        // Leave `entry` where it is when its home lies cyclically in (hole, next].
        let stays = if hole <= next {
            hole < home && home <= next
        } else {
            hole < home || home <= next
        };
        if !stays {
            // SAFETY: as above.
            unsafe { table.add(hole).write(entry) };
            hole = next;
        }
    }
    // SAFETY: as above.
    unsafe {
        table.add(hole).write(LiveEntry {
            ptr: 0,
            size: 0,
            stack: 0,
        })
    };
}

fn on_alloc(ptr: *mut u8, size: usize) {
    LIVE_BYTES.fetch_add(size as i64, Ordering::Relaxed);
    LIVE_BLOCKS.fetch_add(1, Ordering::Relaxed);
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    if TRACING.load(Ordering::Relaxed) {
        let (frames, len, hash) = capture();
        lock();
        let stack = intern(&frames, len, hash);
        insert(ptr as usize, size, stack);
        unlock();
    }
}

/// Called **before** the block goes back to `System`, so its address cannot be handed to
/// another thread (and recorded there) before this entry is gone.
fn on_free(ptr: *mut u8, size: usize) {
    LIVE_BYTES.fetch_sub(size as i64, Ordering::Relaxed);
    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
    if TABLES_READY.load(Ordering::Acquire) {
        lock();
        remove(ptr as usize);
        unlock();
    }
}

// SAFETY: every call is forwarded to `System` unchanged; the bookkeeping only touches atomics
// and the two tables (allocated from `System` directly, so it never re-enters this allocator).
unsafe impl GlobalAlloc for TracingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            on_alloc(ptr, layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            on_alloc(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        on_free(ptr, layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Forget the old block first (it may be freed inside `realloc`), and book it back in if
        // `realloc` fails and the old block stays valid.
        on_free(ptr, layout.size());
        // SAFETY: same contract as the caller's.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if new.is_null() {
            on_alloc(ptr, layout.size());
        } else {
            on_alloc(new, new_size);
        }
        new
    }
}

/// One call stack with blocks still alive.
#[derive(Debug, Clone)]
pub struct Site {
    pub bytes: u64,
    pub blocks: u64,
    /// Symbolised frames, innermost first, with this module's and the allocator's frames
    /// dropped.
    pub frames: Vec<String>,
}

/// Group the recorded blocks that are still alive by call stack, biggest first (by bytes),
/// keeping at most `limit` sites. Pauses recording while it runs.
pub fn live_sites(limit: usize) -> Vec<Site> {
    if !TABLES_READY.load(Ordering::Acquire) {
        return Vec::new();
    }
    let was = TRACING.swap(false, Ordering::SeqCst);
    // Copy (stack, size) out under the lock into memory that does not come from this
    // allocator (taking the lock and then allocating through it would deadlock on a free).
    lock();
    let count = TRACKED.load(Ordering::Relaxed);
    let layout = Layout::array::<(u32, usize)>(count.max(1)).expect("copy layout");
    // SAFETY: non-zero size; freed below with the same layout.
    let copy = unsafe { System.alloc(layout) }.cast::<(u32, usize)>();
    let mut copied = 0usize;
    if !copy.is_null() {
        let table = LIVE_TABLE.load(Ordering::Relaxed);
        for slot in 0..LIVE_CAP {
            // SAFETY: `slot < LIVE_CAP`; exclusive under `LOCK`.
            let entry = unsafe { *table.add(slot) };
            if entry.ptr != 0 && copied < count {
                // SAFETY: `copied < count`, inside the copy.
                unsafe { copy.add(copied).write((entry.stack, entry.size)) };
                copied += 1;
            }
        }
    }
    unlock();

    let mut by_stack: HashMap<u32, (u64, u64)> = HashMap::new();
    for index in 0..copied {
        // SAFETY: written above.
        let (stack, size) = unsafe { copy.add(index).read() };
        let totals = by_stack.entry(stack).or_default();
        totals.0 += size as u64;
        totals.1 += 1;
    }
    if !copy.is_null() {
        // SAFETY: allocated above with `layout`.
        unsafe { System.dealloc(copy.cast(), layout) };
    }

    let mut sites: Vec<(u32, u64, u64)> = by_stack
        .into_iter()
        .map(|(stack, (bytes, blocks))| (stack, bytes, blocks))
        .collect();
    sites.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
    sites.truncate(limit);

    let symbols = Symbols::load();
    let stack_table = STACK_TABLE.load(Ordering::Acquire);
    let out = sites
        .into_iter()
        .map(|(stack, bytes, blocks)| {
            let frames = if stack == 0 {
                vec!["<stack not recorded>".to_owned()]
            } else {
                // SAFETY: ids are slot + 1 of written slots; written entries never change.
                let entry = unsafe { *stack_table.add(stack as usize - 1) };
                entry.frames[..entry.len as usize]
                    .iter()
                    .map(|&address| symbols.name(address))
                    .filter(|name| !is_plumbing(name))
                    .collect()
            };
            Site {
                bytes,
                blocks,
                frames,
            }
        })
        .collect();
    TRACING.store(was, Ordering::SeqCst);
    out
}

/// Frames that say nothing about who owns a block: this module, the allocator shims, raw
/// collection growth.
fn is_plumbing(name: &str) -> bool {
    const ANYWHERE: &[&str] = &[
        "ml_leak_support::",
        "TracingAlloc",
        "__rust_alloc",
        "__rust_dealloc",
        "__rust_realloc",
        "__rg_",
        "__rdl_",
        "RtlCaptureStackBackTrace",
    ];
    const PREFIXES: &[&str] = &[
        "alloc::alloc::",
        "alloc::raw_vec::",
        "<alloc::raw_vec::",
        "std::alloc::",
        "core::alloc::",
    ];
    ANYWHERE.iter().any(|part| name.contains(part))
        || PREFIXES.iter().any(|prefix| name.starts_with(prefix))
}

/// Print the top sites (`ML_TRACE_FRAMES` frames each, default 14).
pub fn print_sites(title: &str, sites: &[Site]) {
    let frames: usize = std::env::var("ML_TRACE_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(14);
    println!("---- {title}: {} sites ----", sites.len());
    for (rank, site) in sites.iter().enumerate() {
        println!(
            "#{rank}: {:.1} KB in {} blocks",
            site.bytes as f64 / 1024.0,
            site.blocks
        );
        for frame in site.frames.iter().take(frames) {
            println!("      {frame}");
        }
    }
    println!(
        "---- tracked {} blocks, {} stacks, {} untracked (table full) ----",
        TRACKED.load(Ordering::Relaxed),
        STACKS.load(Ordering::Relaxed),
        UNTRACKED.load(Ordering::Relaxed)
    );
}

/// A symbol's readable name (v0 `_R...` or legacy `_ZN...`), without the hash.
pub fn demangle(symbol: &str) -> String {
    format!("{:#}", rustc_demangle::demangle(symbol))
}

/// dbghelp, loaded at run time (no import library needed). Public symbols only: test
/// binaries are built without debug info, which is enough for function names.
struct Symbols {
    #[cfg(windows)]
    from_addr: Option<win::SymFromAddrW>,
    #[cfg(windows)]
    line_from_addr: Option<win::SymGetLineFromAddrW64>,
}

impl Symbols {
    #[cfg(windows)]
    fn load() -> Symbols {
        use std::sync::OnceLock;
        static LOADED: OnceLock<Option<(usize, usize)>> = OnceLock::new();
        let loaded = *LOADED.get_or_init(|| {
            let name: Vec<u16> = "dbghelp.dll\0".encode_utf16().collect();
            // SAFETY: a NUL-terminated wide string; the module stays loaded for the process.
            let module = unsafe { win::LoadLibraryW(name.as_ptr()) };
            if module.is_null() {
                return None;
            }
            // SAFETY: NUL-terminated export names; null is checked below.
            let (set_options, initialize, from_addr, line_from_addr) = unsafe {
                (
                    win::GetProcAddress(module, c"SymSetOptions".as_ptr().cast()),
                    win::GetProcAddress(module, c"SymInitializeW".as_ptr().cast()),
                    win::GetProcAddress(module, c"SymFromAddrW".as_ptr().cast()),
                    win::GetProcAddress(module, c"SymGetLineFromAddrW64".as_ptr().cast()),
                )
            };
            if set_options.is_null() || initialize.is_null() || from_addr.is_null() {
                return None;
            }
            // SAFETY: the exports have exactly these signatures (dbghelp.h).
            unsafe {
                let set_options =
                    std::mem::transmute::<*mut std::ffi::c_void, win::SymSetOptions>(set_options);
                let initialize =
                    std::mem::transmute::<*mut std::ffi::c_void, win::SymInitializeW>(initialize);
                // SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES; not
                // SYMOPT_PUBLICS_ONLY, so private symbols are used where a PDB has them.
                set_options(0x2 | 0x4 | 0x10);
                // The linker records only the PDB's file name (`/PDBALTPATH:%_PDB%`), so dbghelp
                // must be told to look next to the executable. Already initialised (by std's
                // backtrace printer) is fine: lookups still work.
                let search: Vec<u16> = std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|dir| dir.display().to_string()))
                    .unwrap_or_default()
                    .encode_utf16()
                    .chain(std::iter::once(0))
                    .collect();
                let initialized = initialize(win::GetCurrentProcess(), search.as_ptr(), 1);
                if std::env::var_os("ML_TRACE_DEBUG").is_some() {
                    eprintln!(
                        "dbghelp: SymInitializeW -> {initialized} (last error {})",
                        win::GetLastError()
                    );
                }
            }
            Some((from_addr as usize, line_from_addr as usize))
        });
        Symbols {
            // SAFETY: the address came from `GetProcAddress("SymFromAddrW")`.
            from_addr: loaded.map(|(address, _)| unsafe {
                std::mem::transmute::<usize, win::SymFromAddrW>(address)
            }),
            // SAFETY: from `GetProcAddress("SymGetLineFromAddrW64")`, when it exists.
            line_from_addr: loaded.and_then(|(_, address)| {
                (address != 0).then(|| unsafe {
                    std::mem::transmute::<usize, win::SymGetLineFromAddrW64>(address)
                })
            }),
        }
    }

    /// `file:line` for `address` when the binary has line tables.
    #[cfg(windows)]
    fn line(&self, address: usize) -> Option<String> {
        let line_from_addr = self.line_from_addr?;
        let mut line = win::ImagehlpLineW64 {
            size_of_struct: std::mem::size_of::<win::ImagehlpLineW64>() as u32,
            key: std::ptr::null_mut(),
            line_number: 0,
            file_name: std::ptr::null_mut(),
            address: 0,
        };
        let mut displacement = 0u32;
        // SAFETY: `line` is a correctly sized IMAGEHLP_LINEW64; on success `file_name` points at
        // a NUL-terminated string owned by dbghelp, read before the next dbghelp call.
        unsafe {
            if line_from_addr(
                win::GetCurrentProcess(),
                address as u64,
                &mut displacement,
                &mut line,
            ) == 0
                || line.file_name.is_null()
            {
                return None;
            }
            let mut length = 0;
            while *line.file_name.add(length) != 0 {
                length += 1;
            }
            let file = String::from_utf16_lossy(std::slice::from_raw_parts(line.file_name, length));
            // Keep it short and free of local paths: from the crate folder on.
            let file = file.rfind("crates\\").map_or_else(
                || file.rsplit('\\').next().unwrap_or("").to_owned(),
                |at| file[at..].to_owned(),
            );
            Some(format!("{file}:{}", line.line_number))
        }
    }

    #[cfg(not(windows))]
    fn load() -> Symbols {
        Symbols {}
    }

    #[cfg(windows)]
    fn name(&self, address: usize) -> String {
        const MAX_NAME: usize = 1024;
        let Some(from_addr) = self.from_addr else {
            return format!("{address:#x}");
        };
        // SYMBOL_INFOW (88 bytes) followed by room for the name, 8-byte aligned.
        let mut buffer = vec![0u64; (88 + MAX_NAME * 2) / 8 + 1];
        let info = buffer.as_mut_ptr().cast::<win::SymbolInfoW>();
        let mut displacement = 0u64;
        // SAFETY: `buffer` is large enough for the header and `MAX_NAME` UTF-16 units and is
        // 8-byte aligned; dbghelp writes at most `max_name_len` units.
        let found = unsafe {
            (*info).size_of_struct = std::mem::size_of::<win::SymbolInfoW>() as u32;
            (*info).max_name_len = MAX_NAME as u32;
            from_addr(
                win::GetCurrentProcess(),
                address as u64,
                &mut displacement,
                info,
            )
        };
        if found == 0 {
            if std::env::var_os("ML_TRACE_DEBUG").is_some() {
                // SAFETY: reads this thread's last-error value.
                eprintln!(
                    "dbghelp: SymFromAddrW({address:#x}) failed, error {}",
                    unsafe { win::GetLastError() }
                );
            }
            return format!("{address:#x}");
        }
        // SAFETY: on success `name_len` UTF-16 units were written at `name`.
        let name = unsafe {
            let length = ((*info).name_len as usize).min(MAX_NAME);
            let start = std::ptr::addr_of!((*info).name).cast::<u16>();
            String::from_utf16_lossy(std::slice::from_raw_parts(start, length))
        };
        let name = demangle(&name);
        match self.line(address) {
            Some(line) => format!("{name}  ({line})"),
            // No line tables: a function without a symbol of its own is reported as the
            // nearest public symbol before it. Build with line tables (see the harness) to be
            // sure; a large displacement is flagged.
            None if displacement > 0x4000 => format!("{name}+{displacement:#x}?"),
            None => name,
        }
    }

    #[cfg(not(windows))]
    fn name(&self, address: usize) -> String {
        format!("{address:#x}")
    }
}

/// What the operating system says about this process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessCounters {
    /// Committed private memory (Task Manager's "Commit size").
    pub private_bytes: u64,
    pub working_set: u64,
    pub handles: u32,
    pub threads: u32,
}

impl ProcessCounters {
    #[cfg(windows)]
    pub fn now() -> ProcessCounters {
        let mut counters = ProcessCounters::default();
        // SAFETY: plain Win32 queries on the current process with correctly sized structs;
        // the snapshot handle is closed.
        unsafe {
            let process = win::GetCurrentProcess();
            let mut memory = win::ProcessMemoryCountersEx {
                cb: std::mem::size_of::<win::ProcessMemoryCountersEx>() as u32,
                ..Default::default()
            };
            if win::K32GetProcessMemoryInfo(process, &mut memory, memory.cb) != 0 {
                counters.private_bytes = memory.private_usage as u64;
                counters.working_set = memory.working_set_size as u64;
            }
            let mut handles = 0u32;
            if win::GetProcessHandleCount(process, &mut handles) != 0 {
                counters.handles = handles;
            }
            let pid = win::GetCurrentProcessId();
            let snapshot = win::CreateToolhelp32Snapshot(0x4, 0);
            if !snapshot.is_null() && snapshot as isize != -1 {
                let mut entry = win::ThreadEntry32 {
                    size: std::mem::size_of::<win::ThreadEntry32>() as u32,
                    ..Default::default()
                };
                let mut more = win::Thread32First(snapshot, &mut entry) != 0;
                while more {
                    if entry.owner_process_id == pid {
                        counters.threads += 1;
                    }
                    more = win::Thread32Next(snapshot, &mut entry) != 0;
                }
                win::CloseHandle(snapshot);
            }
        }
        counters
    }

    #[cfg(not(windows))]
    pub fn now() -> ProcessCounters {
        ProcessCounters::default()
    }
}

#[cfg(windows)]
#[allow(non_snake_case)]
mod win {
    use std::ffi::c_void;

    pub type SymSetOptions = unsafe extern "system" fn(u32) -> u32;
    pub type SymInitializeW = unsafe extern "system" fn(*mut c_void, *const u16, i32) -> i32;
    pub type SymFromAddrW =
        unsafe extern "system" fn(*mut c_void, u64, *mut u64, *mut SymbolInfoW) -> i32;
    pub type SymGetLineFromAddrW64 =
        unsafe extern "system" fn(*mut c_void, u64, *mut u32, *mut ImagehlpLineW64) -> i32;

    /// `IMAGEHLP_LINEW64` (dbghelp.h).
    #[repr(C)]
    pub struct ImagehlpLineW64 {
        pub size_of_struct: u32,
        pub key: *mut c_void,
        pub line_number: u32,
        pub file_name: *mut u16,
        pub address: u64,
    }

    /// `SYMBOL_INFOW` (dbghelp.h); 88 bytes with the one-unit name.
    #[repr(C)]
    pub struct SymbolInfoW {
        pub size_of_struct: u32,
        pub type_index: u32,
        pub reserved: [u64; 2],
        pub index: u32,
        pub size: u32,
        pub mod_base: u64,
        pub flags: u32,
        pub value: u64,
        pub address: u64,
        pub register: u32,
        pub scope: u32,
        pub tag: u32,
        pub name_len: u32,
        pub max_name_len: u32,
        pub name: [u16; 1],
    }

    /// `PROCESS_MEMORY_COUNTERS_EX` (psapi.h).
    #[repr(C)]
    #[derive(Default)]
    pub struct ProcessMemoryCountersEx {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
        pub private_usage: usize,
    }

    /// `THREADENTRY32` (tlhelp32.h).
    #[repr(C)]
    #[derive(Default)]
    pub struct ThreadEntry32 {
        pub size: u32,
        pub usage: u32,
        pub thread_id: u32,
        pub owner_process_id: u32,
        pub base_priority: i32,
        pub delta_priority: i32,
        pub flags: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn RtlCaptureStackBackTrace(
            frames_to_skip: u32,
            frames_to_capture: u32,
            back_trace: *mut *mut c_void,
            back_trace_hash: *mut u32,
        ) -> u16;
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn GetCurrentProcessId() -> u32;
        pub fn GetProcessHandleCount(process: *mut c_void, count: *mut u32) -> i32;
        pub fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            cb: u32,
        ) -> i32;
        pub fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut c_void;
        pub fn Thread32First(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        pub fn Thread32Next(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
        pub fn LoadLibraryW(name: *const u16) -> *mut c_void;
        pub fn GetLastError() -> u32;
        pub fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
}
