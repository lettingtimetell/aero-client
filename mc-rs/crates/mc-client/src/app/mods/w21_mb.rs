//! Tests for the F3 overlay's memory figures, which since this wave
//! come from the OS's counters (`mc_render::w21_mb`) instead of a `powershell.exe` started once a
//! second while the overlay is open.

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use crate::app::Client;

    /// The overlay's `(used, allocated, max)` are this process's working set, its private bytes
    /// and the machine's physical memory, in that order, with the floors the overlay applies.
    #[test]
    fn the_f3_memory_figures_are_the_os_counters() {
        let (used, allocated, max) = Client::debug_memory_stats_uncached(0);
        let os = mc_render::w21_mb::process_memory().expect("the OS reports this process");
        let total = mc_render::w21_mb::total_physical_memory().expect("and the machine's RAM");
        assert!(
            used > 1 << 20,
            "a running test process uses more than 1 MB: {used}"
        );
        assert!(allocated >= used, "allocated {allocated} < used {used}");
        assert_eq!(max, total.max(allocated), "max is the physical memory");
        // Sampled moments apart while other test threads allocate: close, not equal.
        let near = |a: u64, b: u64| a.abs_diff(b) < 256 << 20;
        assert!(near(used, os.working_set), "{used} vs {os:?}");
        assert!(
            near(allocated, os.private.max(used)),
            "{allocated} vs {os:?}"
        );
    }

    /// The resident figure the overlay already has is a floor for "used" (and so for
    /// "allocated" and "max"), as before.
    #[test]
    fn the_resident_floor_still_applies() {
        let huge = 1_u64 << 50;
        let (used, allocated, max) = Client::debug_memory_stats_uncached(huge);
        assert_eq!((used, allocated, max), (huge, huge, huge));
    }
}
