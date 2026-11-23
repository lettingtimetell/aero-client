//! State machine for vanilla's deliberately-unlabelled "Super Secret Settings" shader cycler.
//!
//! Java: `EntityRenderer.shaderResourceLocations`, `activateNextShader`, `switchUseShader`,
//! and `stopUseShader`.  Rendering and UI code own loading a selected JSON into a
//! [`super::pipeline::PostPhase`]; this module owns only the exact, side-effect-free selection
//! rules so those callers cannot accidentally lose the 25th (off) state.

/// The exact Java order of `EntityRenderer.shaderResourceLocations`.
///
/// `entity_outline` is deliberately absent: it belongs to `RenderGlobal`, rather than to the
/// player-accessible Super Secret Settings list.
pub const SUPER_SECRET_SHADER_GROUPS: [&str; 24] = [
    "notch",
    "fxaa",
    "art",
    "bumpy",
    "blobs2",
    "pencil",
    "color_convolve",
    "deconverge",
    "flip",
    "invert",
    "ntsc",
    "outline",
    "phosphor",
    "scan_pincushion",
    "sobel",
    "bits",
    "desaturate",
    "green",
    "blur",
    "wobble",
    "blobs",
    "antialias",
    "creeper",
    "spider",
];

/// Java's sentinel value for no selected Super Secret shader.
pub const SUPER_SECRET_OFF_INDEX: usize = SUPER_SECRET_SHADER_GROUPS.len();

/// Work a live integration must perform after a successful cycle action.
///
/// A caller that receives [`Load`](Self::Load) builds a [`super::pipeline::PostChain`] from the
/// named `shaders/post/<name>.json`; [`Clear`](Self::Clear) drops it.  [`Unchanged`](Self::Unchanged)
/// means Java's support/player guard rejected the button press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperSecretShaderAction {
    /// Construct and install `shaders/post/<name>.json`.
    Load(&'static str),
    /// Remove the installed chain.
    Clear,
    /// Leave the currently installed chain alone.
    Unchanged,
}

/// Exact non-GPU state behind the Super Secret Settings button and F4 toggle.
///
/// The initial state is Java's `shaderIndex = shaderCount; useShader = false`.  The selected
/// JSON and the F4 visibility bit are intentionally separate: F4 mutes a loaded shader without
/// forgetting which one it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperSecretShaderCycle {
    index: usize,
    use_shader: bool,
}

impl Default for SuperSecretShaderCycle {
    fn default() -> Self {
        Self {
            index: SUPER_SECRET_OFF_INDEX,
            use_shader: false,
        }
    }
}

impl SuperSecretShaderCycle {
    /// The Java `shaderIndex`, where [`SUPER_SECRET_OFF_INDEX`] denotes off.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// The selected group stem, if the cycler is not at its off sentinel.
    #[must_use]
    pub const fn selected(self) -> Option<&'static str> {
        if self.index == SUPER_SECRET_OFF_INDEX {
            None
        } else {
            Some(SUPER_SECRET_SHADER_GROUPS[self.index])
        }
    }

    /// Java's `useShader` bit, which F4 flips without unloading the selected group.
    #[must_use]
    pub const fn use_shader(self) -> bool {
        self.use_shader
    }

    /// Apply `EntityRenderer.activateNextShader`'s guards and modulo-25 advance.
    ///
    /// `view_is_player` is the `getRenderViewEntity() instanceof EntityPlayer` guard.  The
    /// caller must actually perform the returned load/clear action before using this state for a
    /// live [`super::pipeline::PostPhase`].
    pub fn activate_next(
        &mut self,
        shaders_supported: bool,
        view_is_player: bool,
    ) -> SuperSecretShaderAction {
        if !shaders_supported || !view_is_player {
            return SuperSecretShaderAction::Unchanged;
        }

        self.index = (self.index + 1) % (SUPER_SECRET_SHADER_GROUPS.len() + 1);
        match self.selected() {
            // Java delegates this to `loadShader`, which sets `useShader = true` after the
            // group constructs. An integration reports a construction failure through
            // `load_failed`, restoring Java's failure state.
            Some(name) => {
                self.use_shader = true;
                SuperSecretShaderAction::Load(name)
            }
            None => SuperSecretShaderAction::Clear,
        }
    }

    /// Apply `EntityRenderer.switchUseShader` (the F4 toggle) and return the new value.
    ///
    /// This deliberately has no support or selected-group guard: Java toggles its boolean
    /// unconditionally and does not clear `theShaderGroup`.
    pub fn switch_use_shader(&mut self) -> bool {
        self.use_shader = !self.use_shader;
        self.use_shader
    }

    /// Synchronize the visibility bit after Java's entity-camera path calls `loadShader`
    /// directly.  Entity shaders do not change `shaderIndex`, but a successful load always sets
    /// `useShader = true`; keeping that bit in this state machine makes a subsequent F4 edge act
    /// on the same live renderer state as vanilla.
    pub fn set_use_shader(&mut self, use_shader: bool) {
        self.use_shader = use_shader;
    }

    /// Apply `EntityRenderer.stopUseShader`, returning the required live-chain teardown.
    ///
    /// Just like Java, this resets the selection but leaves the F4 bit untouched.
    pub fn stop(&mut self) -> SuperSecretShaderAction {
        self.index = SUPER_SECRET_OFF_INDEX;
        SuperSecretShaderAction::Clear
    }

    /// Apply `loadShader`'s failure path after an integration cannot parse/build a group.
    ///
    /// Java logs the error, resets the selected index to off, and disables shader visibility.
    pub fn load_failed(&mut self) {
        self.index = SUPER_SECRET_OFF_INDEX;
        self.use_shader = false;
    }

    /// Select a group directly, by its index into
    /// [`SUPER_SECRET_SHADER_GROUPS`], or `None` for off (the
    /// client's Super Secret Settings mod picks from a list instead of cycling).
    ///
    /// A group behaves like `activateNextShader` landing on it: the index moves there and
    /// `useShader` is re-armed, because `loadShader` sets it on success. Off is
    /// [`Self::stop`]. An index past the list counts as off. The caller performs the returned
    /// action exactly as for a button press.
    pub fn select(&mut self, index: Option<usize>) -> SuperSecretShaderAction {
        match index.filter(|&i| i < SUPER_SECRET_SHADER_GROUPS.len()) {
            Some(i) => {
                self.index = i;
                self.use_shader = true;
                SuperSecretShaderAction::Load(SUPER_SECRET_SHADER_GROUPS[i])
            }
            None => self.stop(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_list_order_and_off_sentinel_are_exact() {
        // Mutation oracle: changing the order, using `% 24`, or dropping the off sentinel makes
        // this probe fail.  These are independently observable through repeated button presses.
        assert_eq!(SUPER_SECRET_SHADER_GROUPS.len(), 24);
        assert_eq!(SUPER_SECRET_OFF_INDEX, 24);
        assert_eq!(SUPER_SECRET_SHADER_GROUPS[0], "notch");
        assert_eq!(SUPER_SECRET_SHADER_GROUPS[1], "fxaa");
        assert_eq!(SUPER_SECRET_SHADER_GROUPS[22], "creeper");
        assert_eq!(SUPER_SECRET_SHADER_GROUPS[23], "spider");

        let mut cycle = SuperSecretShaderCycle::default();
        for name in SUPER_SECRET_SHADER_GROUPS {
            assert_eq!(
                cycle.activate_next(true, true),
                SuperSecretShaderAction::Load(name)
            );
        }
        assert_eq!(cycle.index(), 23);
        assert_eq!(cycle.selected(), Some("spider"));
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Clear
        );
        assert_eq!(cycle.index(), SUPER_SECRET_OFF_INDEX);
        assert_eq!(cycle.selected(), None);
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Load("notch")
        );
    }

    #[test]
    fn cycle_is_guarded_by_support_and_player_camera() {
        let mut cycle = SuperSecretShaderCycle::default();
        assert_eq!(
            cycle.activate_next(false, true),
            SuperSecretShaderAction::Unchanged
        );
        assert_eq!(
            cycle.activate_next(true, false),
            SuperSecretShaderAction::Unchanged
        );
        assert_eq!(cycle.index(), SUPER_SECRET_OFF_INDEX);
        assert!(!cycle.use_shader());
    }

    #[test]
    fn f4_mutes_without_forgetting_and_stop_does_not_flip_the_bit() {
        let mut cycle = SuperSecretShaderCycle::default();
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Load("notch")
        );
        assert!(cycle.use_shader());
        // F4 changes only useShader; the selected JSON remains loaded.
        assert!(!cycle.switch_use_shader());
        assert_eq!(cycle.selected(), Some("notch"));
        assert!(cycle.switch_use_shader());
        assert_eq!(cycle.selected(), Some("notch"));
        assert_eq!(cycle.stop(), SuperSecretShaderAction::Clear);
        assert!(cycle.use_shader());
    }

    #[test]
    fn entity_shader_load_rearms_f4_without_changing_super_secret_selection() {
        let mut cycle = SuperSecretShaderCycle::default();
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Load("notch")
        );
        assert!(!cycle.switch_use_shader());
        cycle.set_use_shader(true);
        assert_eq!(cycle.selected(), Some("notch"));
        assert!(!cycle.switch_use_shader());
    }

    #[test]
    fn select_jumps_straight_to_a_group_and_rearms_use_shader() {
        let mut cycle = SuperSecretShaderCycle::default();
        assert!(!cycle.use_shader());
        assert_eq!(
            cycle.select(Some(22)),
            SuperSecretShaderAction::Load("creeper")
        );
        assert_eq!(cycle.index(), 22);
        assert_eq!(cycle.selected(), Some("creeper"));
        assert!(cycle.use_shader(), "a successful load re-arms useShader");
        // The next button press continues from the selection, as in Java.
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Load("spider")
        );
        assert_eq!(cycle.select(None), SuperSecretShaderAction::Clear);
        assert_eq!(cycle.index(), SUPER_SECRET_OFF_INDEX);
        assert_eq!(cycle.select(Some(24)), SuperSecretShaderAction::Clear);
        assert_eq!(cycle.selected(), None);
    }

    #[test]
    fn failed_load_returns_to_java_off_and_hidden_state() {
        let mut cycle = SuperSecretShaderCycle::default();
        assert_eq!(
            cycle.activate_next(true, true),
            SuperSecretShaderAction::Load("notch")
        );
        assert!(cycle.use_shader());
        cycle.load_failed();
        assert_eq!(cycle.index(), SUPER_SECRET_OFF_INDEX);
        assert_eq!(cycle.selected(), None);
        assert!(!cycle.use_shader());
    }
}
