//! The scoreboard: objectives, scores, display slots and teams.
//!
//! java: `scoreboard/Scoreboard.java`, `ScoreObjective.java`, `Score.java`,
//! `ScorePlayerTeam.java`, driven from `client/network/NetHandlerPlayClient.java:1869`
//! (`handleScoreboardObjective`), `:1899` (`handleUpdateScore`), `:1927`
//! (`handleDisplayScoreboard`) and `:1947` (`handleTeams`).
//!
//! # Shape
//!
//! Vanilla's `Scoreboard` keys scores by `Map<String playerName, Map<ScoreObjective, Score>>`.
//! Here a score is an `i32`, so the same information is `HashMap<player, HashMap<objective,
//! i32>>` with no empty rows kept — and the two operations that map shape actually decides are
//! reproduced deliberately:
//!
//! * `removeObjectiveFromEntity(name, null)` drops **every** objective's score for that player;
//! * `removeObjective` drops that objective's score for every player *and* clears any display
//!   slot pointing at it.
//!
//! The server side of `Scoreboard` — criteria lists, `func_96516_a`'s broadcast callbacks — is
//! not ported.
//!
//! # Exceptions are part of the client behaviour
//!
//! The four handlers run as scheduled tasks, and `Util.runTask` **logs** whatever they throw and
//! carries on. So a Java exception in the middle of a handler is not a crash and not a
//! disconnect: it is "stop here, keep everything already done". That is observable whenever a
//! server sends a duplicate create or touches something unknown, and it is reproduced:
//!
//! * S3B mode 0 for an existing objective: `addScoreObjective` throws before either setter, so
//!   the old display name and render type stay.
//! * S3B mode 2 for an unknown objective: a null dereference, so nothing is created.
//! * S3E mode 0 for an existing team: `createTeam` throws first, so neither the new
//!   prefix/suffix/colour nor the player list is applied.
//! * S3E mode 4: `removePlayerFromTeam` throws at the first listed player who is not on that
//!   team, so the players before them are removed and the rest keep their membership.

use std::collections::{HashMap, HashSet};

use mc_protocol::play::s2c::{
    DisplayScoreboard, ScoreboardObjective, ScoreboardObjectiveMode, ScoreboardRenderType, Teams,
    TeamsAction, UpdateScore, UpdateScoreAction,
};

use crate::sync::event::ScoreboardEvent;

/// How many display slots vanilla's array has: list, sidebar, belowName, then the 16
/// `sidebar.team.<colour>` slots. java: `objectiveDisplaySlots = new ScoreObjective[19]`
pub const DISPLAY_SLOTS: usize = 19;

/// The tab-list slot. java: `getObjectiveDisplaySlot(0) == "list"`
pub const SLOT_LIST: usize = 0;
/// The sidebar slot. java: `"sidebar"`
pub const SLOT_SIDEBAR: usize = 1;
/// The below-name slot. java: `"belowName"`
pub const SLOT_BELOW_NAME: usize = 2;

/// The 16 colour names `EnumChatFormatting.getFriendlyName()` produces for indices `0..=15`, in
/// index order. java: `EnumChatFormatting`'s declaration order, which *is* the colour index.
pub const CHAT_COLOR_NAMES: [&str; 16] = [
    "black",
    "dark_blue",
    "dark_green",
    "dark_aqua",
    "dark_red",
    "dark_purple",
    "gold",
    "gray",
    "dark_gray",
    "blue",
    "green",
    "aqua",
    "red",
    "light_purple",
    "yellow",
    "white",
];

/// The observable iteration order of Java 8's outer
/// `entitiesScoreObjectives: HashMap<String, ...>` for ordinary list bins.
///
/// `Score.scoreComparator` returns equality for distinct case-equivalent names. Java's stable
/// sort consequently preserves the order in which `HashMap.values()` supplied those scores:
/// bucket index first, then insertion order within that bucket. Rust's randomized `HashMap`
/// cannot supply that final tie. Keeping the outer keys separately also preserves Java's
/// lifecycle distinction between an empty per-player inner map and no outer player entry.
#[derive(Debug, Default, Clone)]
struct JavaBucketChainRank {
    capacity: usize,
    next_sequence: u64,
    sequences: HashMap<String, u64>,
}

impl JavaBucketChainRank {
    const DEFAULT_CAPACITY: usize = 16;

    /// Register `player` if new; returns whether it was inserted.
    fn ensure(&mut self, player: &str) -> bool {
        if self.sequences.contains_key(player) {
            return false;
        }
        if self.capacity == 0 {
            self.capacity = Self::DEFAULT_CAPACITY;
        }
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.sequences.insert(player.to_owned(), sequence);

        // java.util.HashMap's default load factor is 0.75 and resize happens after insertion
        // when `++size > threshold`. Resizing only splits a list bin into stable low/high chains,
        // so the original insertion sequence remains the secondary order in each new bucket.
        if self.sequences.len() > self.capacity * 3 / 4 {
            self.capacity = self.capacity.saturating_mul(2);
        }
        true
    }

    fn remove(&mut self, player: &str) {
        self.sequences.remove(player);
    }

    fn compare(&self, a: &str, b: &str) -> std::cmp::Ordering {
        self.bucket(a)
            .cmp(&self.bucket(b))
            .then_with(|| self.sequences.get(a).cmp(&self.sequences.get(b)))
    }

    fn bucket(&self, player: &str) -> usize {
        debug_assert!(self.capacity.is_power_of_two());
        (java_hash_spread(player) as usize) & (self.capacity - 1)
    }
}

/// `HashMap.hash(Object)` over Java `String.hashCode()`: wrapping `31*h + UTF16-char`, followed
/// by the JDK 8 high-bit spread `h ^ (h >>> 16)`.
fn java_hash_spread(value: &str) -> u32 {
    let hash = value.encode_utf16().fold(0_u32, |hash, unit| {
        hash.wrapping_mul(31).wrapping_add(u32::from(unit))
    });
    hash ^ (hash >> 16)
}

/// java: `String.compareToIgnoreCase`, i.e. `String.CASE_INSENSITIVE_ORDER`.
///
/// Not `a.to_lowercase().cmp(&b.to_lowercase())`: Java folds **per character** and compares the
/// folded code units, which is allocation-free and is the shape a sort comparator wants. Java's
/// loop, transcribed:
///
/// ```text
/// for i in 0..min(n1, n2):
///     c1, c2 = s1[i], s2[i]
///     if c1 != c2:
///         c1, c2 = upper(c1), upper(c2)
///         if c1 != c2:
///             c1, c2 = lower(c1), lower(c2)
///             if c1 != c2: return c1 - c2
/// return n1 - n2
/// ```
fn compare_ignore_case(a: &str, b: &str) -> std::cmp::Ordering {
    // `String.compareToIgnoreCase` walks UTF-16 `char`s, not Unicode scalar values. Keeping the
    // iterator in code-unit space matters even when neither side needs case folding: a surrogate
    // pair sorts before U+E000 in Java, while the corresponding Rust `char` sorts after it.
    fn upper(unit: u16) -> u16 {
        let Some(c) = char::from_u32(u32::from(unit)) else {
            // An individual surrogate is a valid Java `char` and has no case mapping.
            return unit;
        };
        let mut it = c.to_uppercase();
        match (it.next(), it.next()) {
            (Some(one), None) if u32::from(one) <= u32::from(u16::MAX) => one as u16,
            // Java's `Character.toUpperCase(char)` is a simple 1:1 mapping. A full mapping such
            // as sharp-s -> "SS" therefore leaves this code unit unchanged.
            _ => unit,
        }
    }

    fn lower(unit: u16) -> u16 {
        // Unicode's simple lowercase mapping is U+0069, while Rust correctly exposes the full
        // lowercase mapping "i" + U+0307. `Character.toLowerCase(char)` can only return one
        // code unit, so this one mapping must be stated explicitly.
        if unit == 0x0130 {
            return u16::from(b'i');
        }
        let Some(c) = char::from_u32(u32::from(unit)) else {
            return unit;
        };
        let mut it = c.to_lowercase();
        match (it.next(), it.next()) {
            (Some(one), None) if u32::from(one) <= u32::from(u16::MAX) => one as u16,
            _ => unit,
        }
    }

    let (mut ai, mut bi) = (a.encode_utf16(), b.encode_utf16());
    loop {
        match (ai.next(), bi.next()) {
            (Some(c1), Some(c2)) => {
                if c1 == c2 {
                    continue;
                }
                let (u1, u2) = (upper(c1), upper(c2));
                if u1 == u2 {
                    continue;
                }
                let (l1, l2) = (lower(u1), lower(u2));
                if l1 != l2 {
                    return l1.cmp(&l2);
                }
            }
            // java: `n1 - n2` once the shared prefix folds equal.
            (rest_a, rest_b) => return rest_a.is_some().cmp(&rest_b.is_some()),
        }
    }
}

/// `Team.EnumVisible` — who a team's name tags are drawn for.
///
/// java: `scoreboard/Team.java`'s inner enum, whose `func_178824_a` is a **map lookup that
/// returns `null`** for anything not in the table. `NetHandlerPlayClient.handleTeams` (`:1969`)
/// only calls `setNameTagVisibility` when that lookup is non-null, so an unrecognised string
/// leaves the team's current value alone — it does not reset it and it does not store the raw
/// text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NameTagVisibility {
    /// java: `ALWAYS("always", 0)` — and `ScorePlayerTeam`'s field default.
    #[default]
    Always,
    /// java: `NEVER("never", 1)`
    Never,
    /// java: `HIDE_FOR_OTHER_TEAMS("hideForOtherTeams", 2)`
    HideForOtherTeams,
    /// java: `HIDE_FOR_OWN_TEAM("hideForOwnTeam", 3)`
    HideForOwnTeam,
}

impl NameTagVisibility {
    /// java: `Team.EnumVisible.func_178824_a` — `null` for an unknown name, which the caller
    /// treats as "leave the team unchanged".
    #[must_use]
    pub fn from_internal_name(name: &str) -> Option<Self> {
        match name {
            "always" => Some(Self::Always),
            "never" => Some(Self::Never),
            "hideForOtherTeams" => Some(Self::HideForOtherTeams),
            "hideForOwnTeam" => Some(Self::HideForOwnTeam),
            _ => None,
        }
    }

    /// java: `Team.EnumVisible.internalName`
    #[must_use]
    pub fn internal_name(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Never => "never",
            Self::HideForOtherTeams => "hideForOtherTeams",
            Self::HideForOwnTeam => "hideForOwnTeam",
        }
    }
}

/// One objective. java: `ScoreObjective`, minus the criteria (the client always gets `DUMMY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreObjective {
    /// The internal name. java: `getName`
    pub name: String,
    /// What the sidebar header shows. java: `getDisplayName`
    pub display_name: String,
    /// java: `getRenderType`
    pub render_type: ScoreboardRenderType,
}

/// One team. java: `ScorePlayerTeam`, narrowed to what a client reads.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Team {
    /// The internal name. java: `getRegisteredName`
    pub name: String,
    /// java: `getTeamName`
    pub display_name: String,
    /// java: `getColorPrefix`
    pub prefix: String,
    /// java: `getColorSuffix`
    pub suffix: String,
    /// java: `func_98299_i` — bit 0 friendly fire, bit 1 see friendly invisibles.
    pub friendly_flags: i8,
    /// java: `getNameTagVisibility`. Resolved on arrival, the way `handleTeams` does it:
    /// an unrecognised wire string is **discarded**, not stored.
    pub name_tag_visibility: NameTagVisibility,
    /// java: `getChatFormat().getColorIndex()`; `-1` for reset/none.
    pub color: i8,
    /// java: `getMembershipCollection`
    pub members: Vec<String>,
}

impl Team {
    /// java: `ScorePlayerTeam.formatString` — `getColorPrefix() + input + getColorSuffix()`.
    #[must_use]
    pub fn format(&self, input: &str) -> String {
        format!("{}{input}{}", self.prefix, self.suffix)
    }

    /// java: `ScorePlayerTeam.func_98298_a` unpacks bit 0 into `setAllowFriendlyFire`.
    #[must_use]
    pub fn allow_friendly_fire(&self) -> bool {
        self.friendly_flags & 1 > 0
    }

    /// java: `ScorePlayerTeam.func_98298_a` unpacks bit 1 into `setSeeFriendlyInvisiblesEnabled`.
    ///
    /// This is the flag `EntityPlayer.isInvisibleToPlayer` (`:2269`) consults, which is what
    /// decides between `BodyDraw::Skip` and the 15%-alpha `BodyDraw::Ghost`.
    #[must_use]
    pub fn see_friendly_invisibles(&self) -> bool {
        self.friendly_flags & 2 > 0
    }
}

/// The next [`Scoreboard::teams_revision`]: process-wide and never `0`.
fn next_teams_revision() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Everything the four scoreboard packets carry. java: `Scoreboard`.
#[derive(Debug, Default, Clone)]
pub struct Scoreboard {
    /// java: `scoreObjectives`, keyed by internal name.
    objectives: HashMap<String, ScoreObjective>,
    /// java: `entitiesScoreObjectives`: player name, then objective name — see the module docs.
    /// One row per player, as in Java, so an S3C touches one player's
    /// row. The flattened `(player, objective)` map this replaced scanned every score on each
    /// REMOVE, so a server's churn of N rows cost N^2 (30,000 rows: 10.6 s in a debug build).
    /// A row is dropped as soon as it is empty; [`Self::player_order`] keeps Java's outer-map
    /// lifecycle.
    scores: HashMap<String, HashMap<String, i32>>,
    /// Outer Java-HashMap identity/order for player names. This deliberately survives bulk
    /// objective removal even when that leaves a player's inner map empty.
    player_order: JavaBucketChainRank,
    /// A CHANGE packet received before its objective exists is stored by Java under the inner
    /// map's `null` key. It is invisible to score queries but keeps the outer player entry alive.
    orphaned_null_scores: HashSet<String>,
    /// java: `objectiveDisplaySlots`, holding objective *names*.
    display_slots: [Option<String>; DISPLAY_SLOTS],
    /// java: `teams`, keyed by registered name.
    teams: HashMap<String, Team>,
    /// java: `teamMemberships`, player name to registered team name.
    memberships: HashMap<String, String>,
    /// Not vanilla state: changes whenever anything observable through this type changes, so
    /// per-frame consumers (the sidebar) can reuse what they built from it (#8 / C2-1). Values
    /// come from one process-wide counter, so two boards never share a non-zero revision.
    revision: u64,
    /// Changes whenever a team, its display fields or its member list may have changed;
    /// process-wide unique, like [`crate::sync::PlayerList::revision`]. Score
    /// and objective packets do not move it, unlike [`Self::revision`].
    teams_revision: u64,
}

/// Source of [`Scoreboard::revision`] values. Process-wide so a replaced board (world load,
/// new session) can never repeat a revision a cache saw on its predecessor.
static NEXT_SCOREBOARD_REVISION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

impl Scoreboard {
    /// An empty scoreboard.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop everything. java: `WorldClient`'s scoreboard is replaced wholesale on a world load.
    pub fn clear(&mut self) {
        *self = Self::default();
        self.touch();
        self.teams_revision = next_teams_revision();
    }

    /// A value that changes whenever [`Self::teams`] may have changed (S3E, or a clear). Lets
    /// per-frame consumers such as the spectator menu rebuild only when teams did. Score and
    /// objective packets do not move it.
    #[must_use]
    pub fn teams_revision(&self) -> u64 {
        self.teams_revision
    }

    /// An opaque value that changes on every observable mutation: packets, `clear`, and the
    /// zero rows `value_from_objective`/`below_name_line` create. Equal revisions mean equal
    /// sidebar contents. Reads never change it, so polling it every frame is free.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn touch(&mut self) {
        self.revision = NEXT_SCOREBOARD_REVISION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Look one objective up by internal name. java: `getObjective`
    #[must_use]
    pub fn objective(&self, name: &str) -> Option<&ScoreObjective> {
        self.objectives.get(name)
    }

    /// How many objectives are registered.
    #[must_use]
    pub fn objective_count(&self) -> usize {
        self.objectives.len()
    }

    /// Whether nothing at all has arrived — the common case, and the cheap early-out the HUD
    /// takes before doing any sidebar work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objectives.is_empty() && self.teams.is_empty()
    }

    /// The objective bound to a display slot. java: `getObjectiveInDisplaySlot`
    #[must_use]
    pub fn objective_in_display_slot(&self, slot: usize) -> Option<&ScoreObjective> {
        self.display_slots
            .get(slot)?
            .as_ref()
            .and_then(|name| self.objectives.get(name))
    }

    /// One player's score under one objective. java: `getValueFromObjective(...).getScorePoints()`
    #[must_use]
    pub fn score(&self, player: &str, objective: &str) -> Option<i32> {
        self.scores.get(player)?.get(objective).copied()
    }

    /// Vanilla's mutating `getValueFromObjective`: return the registered objective's score,
    /// creating its zero-valued row when the player has none yet.
    ///
    /// Keeping this distinct from [`Self::score`] is intentional. Most client queries are pure,
    /// but the tab list and below-name renderer call Java's object-returning accessor, whose
    /// construction side effect is shared with later sidebar/sorted-score reads.
    pub fn value_from_objective(&mut self, player: &str, objective: &str) -> Option<i32> {
        if !self.objectives.contains_key(objective) {
            return None;
        }
        let mut changed = self.player_order.ensure(player);
        let points = self.score_or_insert_zero(player, objective, &mut changed);
        // Called per Tab row per frame: only an actual insertion may change the revision.
        if changed {
            self.touch();
        }
        Some(points)
    }

    /// The points of `player` under `objective`, creating a zero row when there is none (and
    /// then setting `inserted`). Allocates only when it inserts: the Tab list and nameplates
    /// call this per row per frame.
    fn score_or_insert_zero(&mut self, player: &str, objective: &str, inserted: &mut bool) -> i32 {
        if let Some(points) = self.score(player, objective) {
            return points;
        }
        *inserted = true;
        self.scores
            .entry(player.to_owned())
            .or_default()
            .insert(objective.to_owned(), 0);
        0
    }

    /// A player's team. java: `getPlayersTeam`
    #[must_use]
    pub fn player_team(&self, player: &str) -> Option<&Team> {
        self.memberships
            .get(player)
            .and_then(|name| self.teams.get(name))
    }

    /// A team by registered name. java: `getTeam`
    #[must_use]
    pub fn team(&self, name: &str) -> Option<&Team> {
        self.teams.get(name)
    }

    /// Every registered team, in the backing map's iteration order.
    ///
    /// java: `Scoreboard.getTeams()` returns the values collection of its team map. Consumers
    /// such as `TeleportToTeam` need the complete collection rather than a name lookup.
    pub fn teams(&self) -> impl Iterator<Item = &Team> {
        self.teams.values()
    }

    /// The player's name with their team's prefix and suffix applied.
    /// java: `ScorePlayerTeam.formatPlayerName(scoreboard.getPlayersTeam(name), name)`
    #[must_use]
    pub fn format_player_name(&self, player: &str) -> String {
        match self.player_team(player) {
            Some(team) => team.format(player),
            None => player.to_owned(),
        }
    }

    /// Whether a **remote player**'s name tag is drawn, by their team's `EnumVisible` rule.
    ///
    /// java: `RendererLivingEntity.canRenderName` (`:547-579`). The switch runs only for
    /// `entity instanceof EntityPlayer && entity != thePlayer` **and only when the target has a
    /// team**; a teamless player, and every non-player entity, falls through to the generic
    /// gate (`isGuiEnabled && !isInvisibleToPlayer && riddenByEntity == null`), which is not
    /// scoreboard state and belongs to the caller.
    ///
    /// `viewer` is the local player's name; `Team.isSameTeam` is reference equality, which here
    /// is equality of the registered name.
    #[must_use]
    pub fn name_tag_visible(&self, viewer: &str, target: &str) -> bool {
        let Some(team) = self.player_team(target) else {
            return true;
        };
        let shares_team = self
            .player_team(viewer)
            .is_some_and(|mine| mine.name == team.name);
        let viewer_teamless = self.player_team(viewer).is_none();
        match team.name_tag_visibility {
            NameTagVisibility::Always => true,
            NameTagVisibility::Never => false,
            // java: `return team1 == null || team.isSameTeam(team1);`
            NameTagVisibility::HideForOtherTeams => viewer_teamless || shares_team,
            // java: `return team1 == null || !team.isSameTeam(team1);`
            NameTagVisibility::HideForOwnTeam => viewer_teamless || !shares_team,
        }
    }

    /// Whether player `attacker` may hurt player `target` under the team rules.
    ///
    /// java: `EntityPlayer.canAttackPlayer` (`:1113-1117`) —
    /// `team == null ? true : (!team.isSameTeam(team1) ? true : team.getAllowFriendlyFire())`.
    /// Both teams come from `getPlayersTeam(getName())`, and `Team.isSameTeam` is reference
    /// equality that is never true for a null `team1`, so a teamless target is always fair game.
    /// `EntityArrow.onUpdate` (`:278-285`) drops a player hit when a player shooter fails this.
    #[must_use]
    pub fn can_attack_player(&self, attacker: &str, target: &str) -> bool {
        let Some(team) = self.player_team(attacker) else {
            return true;
        };
        let same_team = self
            .player_team(target)
            .is_some_and(|other| other.name == team.name);
        !same_team || team.allow_friendly_fire()
    }

    /// Whether `viewer` can still see an **invisible player** `target`, i.e. whether the body is
    /// the 15%-alpha ghost rather than skipped.
    ///
    /// java: `EntityPlayer.isInvisibleToPlayer` (`:2269`)'s last clause —
    /// `team != null && player.getTeam() == team && team.getSeeFriendlyInvisiblesEnabled()`.
    /// Note this override is `EntityPlayer`'s: `Entity.isInvisibleToPlayer` (`:2183`) has no team
    /// clause at all, so an invisible **mob** is only ghosted for a spectator.
    #[must_use]
    pub fn sees_friendly_invisible(&self, viewer: &str, target: &str) -> bool {
        let Some(team) = self.player_team(target) else {
            return false;
        };
        team.see_friendly_invisibles()
            && self
                .player_team(viewer)
                .is_some_and(|mine| mine.name == team.name)
    }

    /// The `belowName` line drawn under a player's nameplate, e.g. `"7 Deaths"`.
    ///
    /// java: `RenderPlayer.renderOffsetLivingLabel` (`:139-152`) —
    /// `getObjectiveInDisplaySlot(2)`, then `score.getScorePoints() + " " + getDisplayName()`.
    /// It takes `&mut self` on purpose: vanilla reaches the number through
    /// `getValueFromObjective` (`Scoreboard:96`), which **creates** a zero row when none exists,
    /// so drawing the plate can add the player to that objective's sorted scores.
    ///
    /// The 100-block cut (`if (p_177069_10_ < 100.0D)`) is the caller's, because it is squared
    /// eye distance rather than scoreboard state.
    pub fn below_name_line(&mut self, player: &str) -> Option<String> {
        let objective = self.objective_in_display_slot(SLOT_BELOW_NAME)?;
        let name = objective.name.clone();
        let display_name = objective.display_name.clone();
        let mut changed = self.player_order.ensure(player);
        let points = self.score_or_insert_zero(player, &name, &mut changed);
        // Called per nameplate per frame: only an actual insertion may change the revision.
        if changed {
            self.touch();
        }
        Some(format!("{points} {display_name}"))
    }

    /// Every score under `objective`, **ascending**, with vanilla's tiebreak.
    ///
    /// java: `getSortedScores` sorts with `Score.scoreComparator`, whose equal-points branch is
    /// `p2.getPlayerName().compareToIgnoreCase(p1.getPlayerName())` — note the arguments are
    /// **reversed**, so equal scores come back in *descending*, case-insensitive name order. The
    /// sidebar then walks the list backwards, which is what puts the alphabetically-first name at
    /// the top of a block of tied rows. Getting that backwards is invisible until two players
    /// tie, which is exactly when a scoreboard is being read.
    #[must_use]
    pub fn sorted_scores(&self, objective: &str) -> Vec<(&str, i32)> {
        // The order the rows are collected in is irrelevant: the comparator below is total
        // (every player with a row is in `player_order`, whose sequences are unique).
        let mut out: Vec<(&str, i32)> = self
            .scores
            .iter()
            .filter_map(|(player, row)| row.get(objective).map(|points| (player.as_str(), *points)))
            .collect();
        out.sort_by(|(a_name, a), (b_name, b)| {
            a.cmp(b).then_with(|| {
                let folded = compare_ignore_case(b_name, a_name);
                if folded == std::cmp::Ordering::Equal {
                    self.player_order.compare(a_name, b_name)
                } else {
                    folded
                }
            })
        });
        out
    }

    /// `"list"`, `"sidebar"`, `"belowName"`, or `"sidebar.team.<colour>"`.
    /// java: `Scoreboard.getObjectiveDisplaySlot` (`:432`)
    #[must_use]
    pub fn display_slot_name(slot: usize) -> Option<String> {
        match slot {
            SLOT_LIST => Some("list".to_owned()),
            SLOT_SIDEBAR => Some("sidebar".to_owned()),
            SLOT_BELOW_NAME => Some("belowName".to_owned()),
            _ => CHAT_COLOR_NAMES
                .get(slot.wrapping_sub(3))
                .map(|colour| format!("sidebar.team.{colour}")),
        }
    }

    /// The inverse. java: `Scoreboard.getObjectiveDisplaySlotNumber` (`:463`), including its
    /// case-insensitive comparison for the three named slots and
    /// `EnumChatFormatting.getValueByName`'s colour-name normalisation. The
    /// `sidebar.team.` prefix itself remains case-sensitive, as it is in Java.
    #[must_use]
    pub fn display_slot_number(name: &str) -> Option<usize> {
        if name.eq_ignore_ascii_case("list") {
            return Some(SLOT_LIST);
        }
        if name.eq_ignore_ascii_case("sidebar") {
            return Some(SLOT_SIDEBAR);
        }
        if name.eq_ignore_ascii_case("belowName") {
            return Some(SLOT_BELOW_NAME);
        }
        let colour = name.strip_prefix("sidebar.team.")?;
        // java: EnumChatFormatting.func_175745_c(s) =
        // s.toLowerCase().replaceAll("[^a-z]", ""). The regular expression is ASCII-only,
        // so discard every non-ASCII byte after folding ASCII capitals. All sixteen enum names
        // are already canonical ASCII, but run both operands through the same transform to keep
        // the lookup's Java shape explicit.
        fn normalise_colour_name(name: &str) -> String {
            name.bytes()
                .filter_map(|byte| match byte {
                    b'A'..=b'Z' => Some((byte + (b'a' - b'A')) as char),
                    b'a'..=b'z' => Some(byte as char),
                    _ => None,
                })
                .collect()
        }
        let colour = normalise_colour_name(colour);
        CHAT_COLOR_NAMES
            .iter()
            .position(|candidate| normalise_colour_name(candidate) == colour)
            .map(|index| index + 3)
    }

    /// Apply one decoded scoreboard packet.
    // java: the four `NetHandlerPlayClient` handlers listed in this module's header.
    pub fn apply(&mut self, event: &ScoreboardEvent) {
        match event {
            ScoreboardEvent::Objective(packet) => self.apply_objective(packet),
            ScoreboardEvent::Score(packet) => self.apply_score(packet),
            ScoreboardEvent::Display(packet) => self.apply_display(packet),
            ScoreboardEvent::Teams(packet) => self.apply_teams(packet),
        }
        // Conservatively every packet, including no-ops: a spurious rebuild is harmless.
        self.touch();
    }

    // java: handleScoreboardObjective (:1869)
    fn apply_objective(&mut self, packet: &ScoreboardObjective) {
        match &packet.mode {
            // java: mode 0 is `addScoreObjective` + `setDisplayName` + `setRenderType`.
            // `addScoreObjective` throws "An objective with the name '...' already exists!"
            // before either setter runs, and `Util.runTask` only logs it, so a duplicate create
            // leaves the existing objective's display name and render type untouched.
            ScoreboardObjectiveMode::Create {
                display_name,
                render_type,
            } => {
                if self.objectives.contains_key(&packet.objective_name) {
                    return;
                }
                self.objectives.insert(
                    packet.objective_name.clone(),
                    ScoreObjective {
                        name: packet.objective_name.clone(),
                        display_name: display_name.clone(),
                        render_type: *render_type,
                    },
                );
            }
            // java: mode 2 is the same two setters on `getObjective(name)`, which is null for an
            // unknown name: the NullPointerException is logged and nothing is created.
            ScoreboardObjectiveMode::Update {
                display_name,
                render_type,
            } => {
                if let Some(objective) = self.objectives.get_mut(&packet.objective_name) {
                    objective.display_name = display_name.clone();
                    objective.render_type = *render_type;
                }
            }
            // java: `removeObjective` — drop it, clear every display slot that named it, and
            // drop every score recorded under it.
            ScoreboardObjectiveMode::Remove => {
                self.objectives.remove(&packet.objective_name);
                for slot in &mut self.display_slots {
                    if slot.as_deref() == Some(packet.objective_name.as_str()) {
                        *slot = None;
                    }
                }
                self.scores.retain(|_, row| {
                    row.remove(&packet.objective_name);
                    !row.is_empty()
                });
                // Java leaves the outer per-player maps registered even when removing this
                // objective made them empty. Their bucket position can be observed later.
            }
            // Java's handler has independent branches for modes 0, 1, and 2; every other raw
            // byte reaches it and falls through without touching the scoreboard.
            ScoreboardObjectiveMode::Unknown(_) => {}
        }
    }

    // java: handleUpdateScore (:1899)
    fn apply_score(&mut self, packet: &UpdateScore) {
        match packet.action {
            // java: `:1904-1907` — `getValueFromObjective(name, scoreobjective)` keys the map by
            // the **`ScoreObjective` object**, and `scoreobjective` is null when the objective
            // has not arrived. The row is created under the null key, where no
            // `getSortedScores(objective)` and no display slot can ever reach it; a later
            // objective of the same name is a different object and does not adopt it. So an
            // early score is orphaned in vanilla, and observationally that is a discard.
            UpdateScoreAction::Change { value } => {
                self.player_order.ensure(&packet.player_name);
                if self.objectives.contains_key(&packet.objective_name) {
                    self.scores
                        .entry(packet.player_name.clone())
                        .or_default()
                        .insert(packet.objective_name.clone(), value);
                } else {
                    self.orphaned_null_scores.insert(packet.player_name.clone());
                }
            }
            UpdateScoreAction::Remove => {
                if packet.objective_name.is_empty() {
                    // java: `removeObjectiveFromEntity(name, null)` — every objective at once.
                    self.scores.remove(&packet.player_name);
                    self.orphaned_null_scores.remove(&packet.player_name);
                    self.player_order.remove(&packet.player_name);
                } else if self.objectives.contains_key(&packet.objective_name) {
                    let has_visible_score = match self.scores.get_mut(&packet.player_name) {
                        Some(row) => {
                            row.remove(&packet.objective_name);
                            !row.is_empty()
                        }
                        None => false,
                    };
                    if !has_visible_score {
                        self.scores.remove(&packet.player_name);
                    }
                    if !has_visible_score
                        && !self.orphaned_null_scores.contains(&packet.player_name)
                    {
                        self.player_order.remove(&packet.player_name);
                    }
                }
            }
        }
    }

    // java: handleDisplayScoreboard (:1927)
    fn apply_display(&mut self, packet: &DisplayScoreboard) {
        let Ok(slot) = usize::try_from(packet.position) else {
            return;
        };
        let Some(entry) = self.display_slots.get_mut(slot) else {
            return;
        };
        // java: `:1932-1941` resolves **eagerly**: an empty name stores null, and anything else
        // stores `getObjective(name)`, which is *also* null when the objective has not arrived.
        // Vanilla holds the resolved `ScoreObjective` reference, so a later
        // `handleScoreboardObjective` for that name builds a *different* object and the slot
        // stays null forever. Binding lazily here made a display packet that arrives before its
        // objective light the sidebar up a moment later, which vanilla never does.
        *entry = if self.objectives.contains_key(&packet.objective_name) {
            Some(packet.objective_name.clone())
        } else {
            None
        };
    }

    // java: handleTeams (:1947)
    fn apply_teams(&mut self, packet: &Teams) {
        self.teams_revision = next_teams_revision();
        match &packet.action {
            TeamsAction::Create { info, players } => {
                // java: `scoreboard.createTeam(name)` is the handler's first statement and throws
                // "A team with the name '...' already exists!" for a duplicate, so neither the
                // setters nor the player list below run (the exception is only logged).
                if self.teams.contains_key(&packet.team_name) {
                    return;
                }
                let team = self.teams.entry(packet.team_name.clone()).or_default();
                team.name = packet.team_name.clone();
                team.display_name = info.display_name.clone();
                team.prefix = info.prefix.clone();
                team.suffix = info.suffix.clone();
                team.friendly_flags = info.friendly_flags;
                // java: `:1969-1974` — the enum lookup is a map `get`, and `setNameTagVisibility`
                // is skipped entirely when it misses. Overwriting with the raw string here used
                // to let a typo'd or newer-protocol value silently unhide a `never` team.
                if let Some(vis) = NameTagVisibility::from_internal_name(&info.name_tag_visibility)
                {
                    team.name_tag_visibility = vis;
                }
                team.color = info.color;
                self.add_players_to_team(players, &packet.team_name);
            }
            // java: mode 2 is the setters only — membership is untouched.
            TeamsAction::UpdateInfo { info } => {
                if let Some(team) = self.teams.get_mut(&packet.team_name) {
                    team.display_name = info.display_name.clone();
                    team.prefix = info.prefix.clone();
                    team.suffix = info.suffix.clone();
                    team.friendly_flags = info.friendly_flags;
                    if let Some(vis) =
                        NameTagVisibility::from_internal_name(&info.name_tag_visibility)
                    {
                        team.name_tag_visibility = vis;
                    }
                    team.color = info.color;
                }
            }
            // java: `removeTeam` drops the team and every membership pointing at it.
            TeamsAction::Remove => {
                if let Some(team) = self.teams.remove(&packet.team_name) {
                    for member in &team.members {
                        self.memberships.remove(member);
                    }
                }
            }
            TeamsAction::AddPlayers { players } => {
                self.add_players_to_team(players, &packet.team_name);
            }
            // java: `removePlayerFromTeam(name, scoreplayerteam)` for each listed player. It
            // throws IllegalStateException when `getPlayersTeam(name) != team`, which ends the
            // loop: players before the first miss are removed, every later one keeps its team.
            // An unknown team is null, and every path through the first player then throws
            // (a mismatch, or a null `getMembershipCollection()`), so nothing changes.
            TeamsAction::RemovePlayers { players } => {
                let Some(team) = self.teams.get_mut(&packet.team_name) else {
                    return;
                };
                // The roster is filtered once for the whole prefix of
                // removed players: one `retain` per player made a long list quadratic.
                let mut leaving = HashSet::new();
                for player in players {
                    if self.memberships.get(player) != Some(&packet.team_name) {
                        break;
                    }
                    self.memberships.remove(player);
                    leaving.insert(player.as_str());
                }
                if !leaving.is_empty() {
                    team.members.retain(|m| !leaving.contains(m.as_str()));
                }
            }
            // Java handles actions 0..=4 with independent conditionals. Unknown signed bytes
            // carry no extra fields and fall through without mutating team state.
            TeamsAction::Unknown(_) => {}
        }
    }

    /// java: `addPlayerToTeam` for each listed player, in order — a player already on another
    /// team is removed from it first, and an unknown team is a no-op
    /// (`!teams.containsKey(newTeam) -> return false`).
    ///
    /// java: `ScorePlayerTeam.membershipSet` is a `HashSet`, so each
    /// player costs constant time. Here `memberships` answers "already on this team" (a player
    /// is in a roster exactly when `memberships` names that team), and the players leaving other
    /// teams are filtered out of those rosters once per team at the end. That is the same result
    /// as removing each one as it moves (`retain` keeps everyone else's order, and nothing in
    /// this packet adds to those teams), without a roster scan per player: one 30,000-name S3E
    /// took 11 s in a debug build before.
    fn add_players_to_team(&mut self, players: &[String], team_name: &str) {
        let Some(team) = self.teams.get_mut(team_name) else {
            return;
        };
        let mut leaving: HashMap<String, HashSet<&str>> = HashMap::new();
        for player in players {
            match self
                .memberships
                .insert(player.clone(), team_name.to_owned())
            {
                Some(previous) if previous == team_name => continue,
                Some(previous) => {
                    leaving.entry(previous).or_default().insert(player.as_str());
                }
                None => {}
            }
            team.members.push(player.clone());
        }
        for (previous, gone) in leaving {
            if let Some(old) = self.teams.get_mut(&previous) {
                old.members.retain(|m| !gone.contains(m.as_str()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_protocol::play::s2c::TeamInfo;

    fn create(name: &str, display: &str) -> ScoreboardEvent {
        ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: name.to_owned(),
            mode: ScoreboardObjectiveMode::Create {
                display_name: display.to_owned(),
                render_type: ScoreboardRenderType::Integer,
            },
        })
    }

    fn set_score(player: &str, objective: &str, value: i32) -> ScoreboardEvent {
        ScoreboardEvent::Score(UpdateScore {
            player_name: player.to_owned(),
            objective_name: objective.to_owned(),
            action: UpdateScoreAction::Change { value },
        })
    }

    fn display(position: i8, objective: &str) -> ScoreboardEvent {
        ScoreboardEvent::Display(DisplayScoreboard {
            position,
            objective_name: objective.to_owned(),
        })
    }

    fn team_info(prefix: &str, suffix: &str, color: i8) -> TeamInfo {
        TeamInfo {
            display_name: "Team".to_owned(),
            prefix: prefix.to_owned(),
            suffix: suffix.to_owned(),
            friendly_flags: 0,
            name_tag_visibility: "always".to_owned(),
            color,
        }
    }

    #[test]
    fn an_objective_can_be_created_updated_and_removed() {
        let mut board = Scoreboard::new();
        assert!(board.is_empty());
        board.apply(&create("kills", "Kills"));
        assert_eq!(board.objective("kills").unwrap().display_name, "Kills");

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Update {
                display_name: "Frags".to_owned(),
                render_type: ScoreboardRenderType::Hearts,
            },
        }));
        let objective = board.objective("kills").unwrap();
        assert_eq!(objective.display_name, "Frags");
        assert_eq!(objective.render_type, ScoreboardRenderType::Hearts);

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Remove,
        }));
        assert!(board.objective("kills").is_none());
    }

    /// `teams_revision` moves on every team packet and on `clear`, never on score
    /// or objective traffic (which arrives every second in BedWars and must not rebuild the
    /// spectator menu).
    #[test]
    fn teams_revision_tracks_team_packets_and_clear_only() {
        let mut board = Scoreboard::new();
        let start = board.teams_revision();
        board.apply(&create("kills", "Kills"));
        board.apply(&set_score("bob", "kills", 3));
        board.apply(&display(1, "kills"));
        assert_eq!(board.teams_revision(), start);
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("[R] ", "", 12),
                players: vec!["bob".to_owned()],
            },
        })));
        let after_team = board.teams_revision();
        assert_ne!(after_team, start);
        board.apply(&set_score("bob", "kills", 4));
        assert_eq!(board.teams_revision(), after_team);
        board.clear();
        assert_ne!(board.teams_revision(), after_team);
        assert_ne!(board.teams_revision(), start);
    }

    #[test]
    fn unknown_objective_and_team_actions_are_live_no_ops() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("[R] ", "!", 12),
                players: vec!["bob".to_owned()],
            },
        })));
        let objective_before = board.objective("kills").unwrap().clone();
        let team_before = board.team("red").unwrap().clone();

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Unknown(-1),
        }));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Unknown(-56),
        })));

        assert_eq!(board.objective("kills"), Some(&objective_before));
        assert_eq!(board.team("red"), Some(&team_before));
    }

    /// java: `handleScoreboardObjective` mode 0 calls `addScoreObjective`, which throws for an
    /// existing name before `setDisplayName`/`setRenderType`; `Util.runTask` logs the exception.
    /// Mode 2 dereferences `getObjective(name)`, which is null for an unknown name.
    #[test]
    fn a_duplicate_objective_create_keeps_the_old_one_and_an_unknown_update_creates_nothing() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        board.apply(&set_score("bob", "kills", 3));
        board.apply(&display(1, "kills"));
        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Create {
                display_name: "Re-created".to_owned(),
                render_type: ScoreboardRenderType::Hearts,
            },
        }));
        let objective = board.objective("kills").unwrap();
        assert_eq!(
            objective.display_name, "Kills",
            "addScoreObjective threw first"
        );
        assert_eq!(objective.render_type, ScoreboardRenderType::Integer);
        assert_eq!(board.score("bob", "kills"), Some(3));
        assert_eq!(
            board
                .objective_in_display_slot(SLOT_SIDEBAR)
                .map(|o| o.display_name.as_str()),
            Some("Kills")
        );

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "ghost".to_owned(),
            mode: ScoreboardObjectiveMode::Update {
                display_name: "Ghost".to_owned(),
                render_type: ScoreboardRenderType::Integer,
            },
        }));
        assert!(board.objective("ghost").is_none(), "mode 2 on null throws");
        // ...so a display packet naming it afterwards still resolves to nothing.
        board.apply(&display(1, "ghost"));
        assert!(board.objective_in_display_slot(SLOT_SIDEBAR).is_none());
    }

    /// java: `handleTeams` mode 0 starts with `createTeam`, which throws for an existing name, so
    /// none of the setters and none of the listed players are applied.
    #[test]
    fn a_duplicate_team_create_applies_neither_its_fields_nor_its_players() {
        let mut board = Scoreboard::new();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}c[R] ", "", 12),
                players: vec!["bob".to_owned()],
            },
        })));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "blue".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}9[B] ", "", 9),
                players: vec!["ann".to_owned()],
            },
        })));
        let red_before = board.team("red").unwrap().clone();

        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}4<RED> ", "!", 4),
                players: vec!["ann".to_owned(), "carl".to_owned()],
            },
        })));
        assert_eq!(board.team("red"), Some(&red_before));
        assert_eq!(board.format_player_name("bob"), "\u{a7}c[R] bob");
        assert_eq!(
            board.player_team("ann").map(|t| t.name.as_str()),
            Some("blue"),
            "the duplicate's player list never ran"
        );
        assert!(board.player_team("carl").is_none());
    }

    /// java: `removePlayerFromTeam` throws IllegalStateException for a player who is not on the
    /// team, ending `handleTeams`' mode-4 loop at that player.
    #[test]
    fn removing_players_stops_at_the_first_one_not_on_the_team() {
        let mut board = Scoreboard::new();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("[R] ", "", 12),
                players: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            },
        })));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::RemovePlayers {
                players: vec!["a".to_owned(), "stranger".to_owned(), "c".to_owned()],
            },
        })));
        assert!(board.player_team("a").is_none(), "removed before the throw");
        assert_eq!(
            board.player_team("c").map(|t| t.name.as_str()),
            Some("red"),
            "the loop ended at 'stranger'"
        );
        assert_eq!(board.team("red").unwrap().members, ["b", "c"]);

        // An unknown team is null in Java: the first player throws, nothing changes.
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "nope".to_owned(),
            action: TeamsAction::RemovePlayers {
                players: vec!["b".to_owned()],
            },
        })));
        assert_eq!(board.player_team("b").map(|t| t.name.as_str()), Some("red"));
    }

    /// java: `removeObjective` clears every display slot naming it and drops every score under
    /// it. Leaving either behind leaves a sidebar drawing rows for an objective that is gone.
    #[test]
    fn removing_an_objective_clears_its_slot_and_all_of_its_scores() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        board.apply(&create("deaths", "Deaths"));
        board.apply(&set_score("bob", "kills", 3));
        board.apply(&set_score("bob", "deaths", 1));
        board.apply(&display(1, "kills"));
        assert!(board.objective_in_display_slot(SLOT_SIDEBAR).is_some());

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Remove,
        }));
        assert!(board.objective_in_display_slot(SLOT_SIDEBAR).is_none());
        assert_eq!(board.score("bob", "kills"), None);
        assert_eq!(
            board.score("bob", "deaths"),
            Some(1),
            "another objective's score must survive"
        );
    }

    /// java: `handleUpdateScore`'s REMOVE branch — an **empty** objective name means
    /// `removeObjectiveFromEntity(name, null)`, i.e. drop the player from every objective. This
    /// is how a server clears a disconnected player off every board with one packet.
    #[test]
    fn removing_a_score_with_no_objective_name_drops_the_player_everywhere() {
        let mut board = Scoreboard::new();
        board.apply(&create("a", "A"));
        board.apply(&create("b", "B"));
        board.apply(&set_score("bob", "a", 1));
        board.apply(&set_score("bob", "b", 2));
        board.apply(&set_score("ann", "a", 5));

        board.apply(&ScoreboardEvent::Score(UpdateScore {
            player_name: "bob".to_owned(),
            objective_name: String::new(),
            action: UpdateScoreAction::Remove,
        }));
        assert_eq!(board.score("bob", "a"), None);
        assert_eq!(board.score("bob", "b"), None);
        assert_eq!(board.score("ann", "a"), Some(5));
    }

    /// java: `Score.scoreComparator` — ascending by points, and for a tie the **reversed**
    /// case-insensitive name comparison. The sidebar draws the tail of this list bottom-up, so
    /// the order here is what puts the top scorer at the top of the board.
    #[test]
    fn scores_sort_ascending_with_vanillas_reversed_name_tiebreak() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        for (player, points) in [("Bob", 5), ("ann", 5), ("Zed", 1), ("carl", 9)] {
            board.apply(&set_score(player, "kills", points));
        }
        assert_eq!(
            board.sorted_scores("kills"),
            [("Zed", 1), ("Bob", 5), ("ann", 5), ("carl", 9)],
            "ascending by points; the 5-5 tie is descending case-insensitive by name"
        );
        assert!(board.sorted_scores("nothing").is_empty());
    }

    /// java: `getObjectiveDisplaySlot`/`getObjectiveDisplaySlotNumber` round-trip for all 19,
    /// with the `sidebar.team.<colour>` naming for 3..=18 that a team-coloured sidebar needs.
    #[test]
    fn every_display_slot_round_trips_through_its_vanilla_name() {
        assert_eq!(Scoreboard::display_slot_name(0).as_deref(), Some("list"));
        assert_eq!(Scoreboard::display_slot_name(1).as_deref(), Some("sidebar"));
        assert_eq!(
            Scoreboard::display_slot_name(2).as_deref(),
            Some("belowName")
        );
        assert_eq!(
            Scoreboard::display_slot_name(3).as_deref(),
            Some("sidebar.team.black")
        );
        assert_eq!(
            Scoreboard::display_slot_name(18).as_deref(),
            Some("sidebar.team.white")
        );
        assert_eq!(Scoreboard::display_slot_name(19), None);

        for slot in 0..DISPLAY_SLOTS {
            let name = Scoreboard::display_slot_name(slot).expect("every slot has a name");
            assert_eq!(Scoreboard::display_slot_number(&name), Some(slot), "{slot}");
        }
        // java: the three named slots are `equalsIgnoreCase`.
        assert_eq!(Scoreboard::display_slot_number("SIDEBAR"), Some(1));
        // java: EnumChatFormatting.getValueByName runs `toLowerCase` then removes every
        // non-`[a-z]` character. The colour suffix is therefore neither case- nor
        // punctuation-sensitive, unlike the `sidebar.team.` prefix.
        assert_eq!(
            Scoreboard::display_slot_number("sidebar.team.DARK_BLUE"),
            Some(4)
        );
        assert_eq!(
            Scoreboard::display_slot_number("sidebar.team.darkblue"),
            Some(4)
        );
        assert_eq!(
            Scoreboard::display_slot_number("sidebar.team.Dark Blue"),
            Some(4)
        );
        assert_eq!(
            Scoreboard::display_slot_number("sidebar.team.D-a_r.k blue"),
            Some(4)
        );
        assert_eq!(
            Scoreboard::display_slot_number("SIDEBAR.TEAM.DARK_BLUE"),
            None,
            "Scoreboard's prefix test is startsWith, not equalsIgnoreCase"
        );
        assert_eq!(Scoreboard::display_slot_number("nonsense"), None);
        assert_eq!(Scoreboard::display_slot_number("sidebar.team.mauve"), None);
    }

    /// java: `handleDisplayScoreboard` (`:1932`) resolves **eagerly** —
    /// `setObjectiveInDisplaySlot(slot, getObjective(name))` stores the *object*, and stores
    /// `null` when the objective has not arrived. A later objective of that name is a different
    /// `ScoreObjective` instance, so the dead slot never comes back to life.
    ///
    /// This test used to assert the opposite ("the binding must resolve once the objective
    /// lands") and passed against a lazy `Option<String>` lookup: the divergence was written
    /// into the assertion.
    #[test]
    fn a_display_slot_resolves_eagerly_and_a_late_objective_does_not_adopt_it() {
        let mut board = Scoreboard::new();
        board.apply(&display(1, "later"));
        assert!(
            board.objective_in_display_slot(SLOT_SIDEBAR).is_none(),
            "no objective yet, so vanilla stores null"
        );
        board.apply(&create("later", "Later"));
        assert!(
            board.objective_in_display_slot(SLOT_SIDEBAR).is_none(),
            "java: the slot holds a null reference; a same-named objective is a new object"
        );
        // Re-sending the display packet after the objective exists does bind it.
        board.apply(&display(1, "later"));
        assert_eq!(
            board
                .objective_in_display_slot(SLOT_SIDEBAR)
                .map(|o| o.name.as_str()),
            Some("later")
        );
        board.apply(&display(1, ""));
        assert!(board.objective_in_display_slot(SLOT_SIDEBAR).is_none());
    }

    /// java: `handleDisplayScoreboard` (`:1932-1941`) indexes `objectiveDisplaySlots[position]`
    /// with S3D's signed byte. Outside `0..=18` that throws ArrayIndexOutOfBoundsException inside
    /// the scheduled task, and `Util.runTask` only logs it: the client carries on and no slot
    /// changes. Ignoring the packet is that same outcome (and must not panic).
    #[test]
    fn an_out_of_range_display_slot_changes_no_slot() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        board.apply(&create("deaths", "Deaths"));
        board.apply(&display(1, "kills"));
        board.apply(&display(18, "deaths"));
        for position in [-128, -1, 19, 120, 127] {
            board.apply(&display(position, "deaths"));
            board.apply(&display(position, ""));
        }
        for slot in 0..DISPLAY_SLOTS {
            let bound = board
                .objective_in_display_slot(slot)
                .map(|o| o.name.as_str());
            let expected = match slot {
                1 => Some("kills"),
                18 => Some("deaths"),
                _ => None,
            };
            assert_eq!(bound, expected, "slot {slot}");
        }
    }

    /// java: modes 1-4 of `handleTeams` use `getTeam(name)`, null for an unknown team. Mode 1's
    /// `removeTeam(null)` and mode 2's first setter throw NullPointerException, and mode 3's
    /// `addPlayerToTeam` returns false for every player (`!teams.containsKey`).
    /// `handleScoreboardObjective` mode 1 hands null to `removeObjective`, which throws on its
    /// first line. `Util.runTask` logs each throw, so none of them changes anything.
    #[test]
    fn actions_on_an_unknown_team_or_objective_change_nothing() {
        let mut board = Scoreboard::new();
        board.apply(&create("kills", "Kills"));
        board.apply(&set_score("bob", "kills", 3));
        board.apply(&display(1, "kills"));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}c[R] ", "", 12),
                players: vec!["bob".to_owned()],
            },
        })));
        let red_before = board.team("red").unwrap().clone();

        for action in [
            TeamsAction::Remove,
            TeamsAction::UpdateInfo {
                info: team_info("\u{a7}9[G] ", "?", 9),
            },
            TeamsAction::AddPlayers {
                players: vec!["bob".to_owned(), "ann".to_owned()],
            },
            TeamsAction::RemovePlayers {
                players: vec!["bob".to_owned()],
            },
        ] {
            board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
                team_name: "ghost".to_owned(),
                action,
            })));
        }
        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "ghost".to_owned(),
            mode: ScoreboardObjectiveMode::Remove,
        }));

        assert!(board.team("ghost").is_none(), "nothing was created");
        assert_eq!(board.team("red"), Some(&red_before));
        assert_eq!(board.format_player_name("bob"), "\u{a7}c[R] bob");
        assert!(board.player_team("ann").is_none());
        assert_eq!(board.score("bob", "kills"), Some(3));
        assert_eq!(
            board
                .objective_in_display_slot(SLOT_SIDEBAR)
                .map(|o| o.name.as_str()),
            Some("kills")
        );
    }

    /// java: `EntityPlayer.canAttackPlayer` (`:1113-1117`), the predicate `EntityArrow.onUpdate`
    /// (`:278-285`) applies when a player's arrow hits a player.
    #[test]
    fn can_attack_player_follows_entity_players_team_rule() {
        fn set_red_flags(board: &mut Scoreboard, friendly_flags: i8) {
            let mut info = team_info("", "", 12);
            info.friendly_flags = friendly_flags;
            board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
                team_name: "red".to_owned(),
                action: TeamsAction::UpdateInfo { info },
            })));
        }

        let mut board = Scoreboard::new();
        assert!(board.can_attack_player("bob", "ann"), "no teams at all");
        for (team, players) in [("red", ["bob", "ann"]), ("blue", ["cid", "eve"])] {
            board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
                team_name: team.to_owned(),
                action: TeamsAction::Create {
                    info: team_info("", "", 12),
                    players: players.map(str::to_owned).to_vec(),
                },
            })));
        }
        // `team_info` leaves friendly_flags at 0: friendly fire off.
        assert!(
            !board.can_attack_player("bob", "ann"),
            "same team, friendly fire off"
        );
        assert!(
            !board.can_attack_player("bob", "bob"),
            "a player is on their own team"
        );
        assert!(board.can_attack_player("bob", "cid"), "other team");
        assert!(
            board.can_attack_player("bob", "dan"),
            "java: isSameTeam(null) is false, so a teamless target can be hit"
        );
        assert!(
            board.can_attack_player("dan", "bob"),
            "java: a teamless attacker short-circuits on team == null"
        );

        // Bit 1 (see friendly invisibles) alone does not allow it; bit 0 does.
        set_red_flags(&mut board, 2);
        assert!(!board.can_attack_player("bob", "ann"));
        set_red_flags(&mut board, 1);
        assert!(board.can_attack_player("bob", "ann"));
        assert!(
            !board.can_attack_player("cid", "eve"),
            "the attacker's own team decides: blue still forbids it"
        );
    }

    /// java: `handleUpdateScore` (`:1904`) calls `getValueFromObjective(name, scoreobjective)`
    /// with a **null** objective when it has not arrived, which files the row under the null key
    /// where no `getSortedScores(objective)` can reach it. Observationally the score is dropped.
    #[test]
    fn a_score_for_an_unarrived_objective_is_orphaned() {
        let mut board = Scoreboard::new();
        board.apply(&set_score("bob", "kills", 9));
        board.apply(&create("kills", "Kills"));
        assert_eq!(
            board.score("bob", "kills"),
            None,
            "java: the early row is keyed by the null objective and never adopted"
        );
        assert!(board.sorted_scores("kills").is_empty());
        // The ordinary order still works.
        board.apply(&set_score("bob", "kills", 9));
        assert_eq!(board.score("bob", "kills"), Some(9));
    }

    /// java: `Team.EnumVisible.func_178824_a` is a map `get`; `handleTeams` (`:1971`) only calls
    /// `setNameTagVisibility` when it is non-null, so an unknown string leaves the team alone.
    #[test]
    fn an_unrecognised_name_tag_visibility_is_discarded() {
        let mut board = Scoreboard::new();
        let mut info = team_info("", "", -1);
        info.name_tag_visibility = "never".to_owned();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: info.clone(),
                players: vec!["bob".to_owned()],
            },
        })));
        assert_eq!(
            board.team("red").unwrap().name_tag_visibility,
            NameTagVisibility::Never
        );

        info.name_tag_visibility = "hideForNobody".to_owned();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::UpdateInfo { info },
        })));
        assert_eq!(
            board.team("red").unwrap().name_tag_visibility,
            NameTagVisibility::Never,
            "an unknown value must not silently unhide the team"
        );
        assert!(!board.name_tag_visible("ann", "bob"));
    }

    /// java: `RendererLivingEntity.canRenderName` (`:547`) — the four-way switch, including
    /// both `team1 == null` escapes.
    #[test]
    fn name_tag_visibility_reproduces_can_render_names_switch() {
        fn board_with(vis: &str) -> Scoreboard {
            let mut board = Scoreboard::new();
            let mut info = team_info("", "", -1);
            info.name_tag_visibility = vis.to_owned();
            board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
                team_name: "red".to_owned(),
                action: TeamsAction::Create {
                    info: info.clone(),
                    players: vec!["bob".to_owned(), "ann".to_owned()],
                },
            })));
            board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
                team_name: "blue".to_owned(),
                action: TeamsAction::Create {
                    info,
                    players: vec!["cid".to_owned()],
                },
            })));
            board
        }

        // A teamless target always shows: the switch runs only `if (team != null)`.
        assert!(Scoreboard::new().name_tag_visible("ann", "bob"));

        assert!(board_with("always").name_tag_visible("cid", "bob"));
        assert!(!board_with("never").name_tag_visible("ann", "bob"));

        let hide_other = board_with("hideForOtherTeams");
        assert!(hide_other.name_tag_visible("ann", "bob"), "same team");
        assert!(!hide_other.name_tag_visible("cid", "bob"), "other team");
        assert!(
            hide_other.name_tag_visible("dan", "bob"),
            "java: `team1 == null ||` — a teamless viewer sees it"
        );

        let hide_own = board_with("hideForOwnTeam");
        assert!(!hide_own.name_tag_visible("ann", "bob"), "same team");
        assert!(hide_own.name_tag_visible("cid", "bob"), "other team");
        assert!(
            hide_own.name_tag_visible("dan", "bob"),
            "java: `team1 == null ||` — a teamless viewer sees it"
        );
    }

    /// java: `ScorePlayerTeam.func_98298_a` — bit 0 friendly fire, bit 1 see friendly
    /// invisibles, which is the clause `EntityPlayer.isInvisibleToPlayer` (`:2269`) reads.
    #[test]
    fn friendly_flags_unpack_the_way_func_98298_a_does() {
        let mut board = Scoreboard::new();
        let mut info = team_info("", "", -1);
        info.friendly_flags = 2;
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info,
                players: vec!["bob".to_owned(), "ann".to_owned()],
            },
        })));
        let team = board.team("red").unwrap();
        assert!(!team.allow_friendly_fire());
        assert!(team.see_friendly_invisibles());
        assert!(board.sees_friendly_invisible("ann", "bob"), "same team");
        assert!(
            !board.sees_friendly_invisible("cid", "bob"),
            "viewer has no team"
        );
        assert!(
            !board.sees_friendly_invisible("ann", "cid"),
            "a teamless target has no exemption"
        );
    }

    /// java: `RenderPlayer.renderOffsetLivingLabel` (`:139`) — slot 2, and the number comes
    /// through `getValueFromObjective`, which **creates** a zero row.
    #[test]
    fn the_below_name_line_is_points_space_display_name_and_creates_the_row() {
        let mut board = Scoreboard::new();
        assert_eq!(board.below_name_line("bob"), None, "no belowName objective");
        board.apply(&create("deaths", "Deaths"));
        board.apply(&display(2, "deaths"));
        assert_eq!(board.below_name_line("bob").as_deref(), Some("0 Deaths"));
        assert_eq!(
            board.score("bob", "deaths"),
            Some(0),
            "java: getValueFromObjective inserts the row it did not find"
        );
        board.apply(&set_score("bob", "deaths", 7));
        assert_eq!(board.below_name_line("bob").as_deref(), Some("7 Deaths"));

        // Removing the objective is the other live boundary: vanilla clears its display-slot
        // reference and every score keyed by that ScoreObjective, so the renderer must stop
        // producing a line immediately rather than retaining the last value.
        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "deaths".to_owned(),
            mode: ScoreboardObjectiveMode::Remove,
        }));
        assert_eq!(board.below_name_line("bob"), None, "removed slot is silent");
        assert_eq!(
            board.score("bob", "deaths"),
            None,
            "removed objective drops its rows"
        );
    }

    #[test]
    fn value_from_objective_creates_only_registered_objective_rows() {
        let mut board = Scoreboard::new();
        assert_eq!(board.value_from_objective("bob", "missing"), None);
        assert_eq!(board.score("bob", "missing"), None);

        board.apply(&create("list", "List"));
        assert_eq!(board.value_from_objective("bob", "list"), Some(0));
        assert_eq!(board.score("bob", "list"), Some(0));
        board.apply(&set_score("bob", "list", 9));
        assert_eq!(board.value_from_objective("bob", "list"), Some(9));
    }

    /// java: `String.CASE_INSENSITIVE_ORDER`. The tie-break argument order is `p2` against `p1`,
    /// so equal scores come back **descending** by folded name — which is what the sidebar's
    /// backwards walk turns into A-before-B on screen.
    #[test]
    fn the_tiebreak_folds_case_the_way_java_does() {
        use std::cmp::Ordering;
        assert_eq!(compare_ignore_case("Bob", "bob"), Ordering::Equal);
        assert_eq!(compare_ignore_case("ann", "Bob"), Ordering::Less);
        assert_eq!(compare_ignore_case("bob", "bobby"), Ordering::Less);
        // '_' (0x5F) sits between 'Z' and 'a'; folded to lowercase it sorts before every letter.
        assert_eq!(compare_ignore_case("_x", "ax"), Ordering::Less);
        assert_eq!(compare_ignore_case("_x", "Ax"), Ordering::Less);
        // Negative control that bites: swap the arguments and every strict case flips, so a
        // sign error in `sorted_scores`' tie-break could not pass this test.
        assert_eq!(compare_ignore_case("Bob", "ann"), Ordering::Greater);
        assert_eq!(compare_ignore_case("bobby", "bob"), Ordering::Greater);
        assert_eq!(compare_ignore_case("ax", "_x"), Ordering::Greater);

        // And the tie-break's *direction*, which is the half of F-P1-scoreboard-01-1 that
        // claimed a sign flip: java compares `p2` against `p1`, so equal scores come back
        // descending by name.
        let mut board = Scoreboard::new();
        board.apply(&create("k", "K"));
        board.apply(&set_score("ann", "k", 5));
        board.apply(&set_score("Bob", "k", 5));
        assert_eq!(
            board.sorted_scores("k"),
            [("Bob", 5), ("ann", 5)],
            "java: compare(p1, p2) returns p2.compareToIgnoreCase(p1) on a tie"
        );

        // Java's two-stage simple fold equates Turkish dotted/dotless I with ASCII i. These
        // assertions fail if this regresses to Rust's full `char::to_lowercase` mapping.
        assert_eq!(compare_ignore_case("\u{130}", "i"), Ordering::Equal);
        assert_eq!(compare_ignore_case("\u{131}", "i"), Ordering::Equal);
        assert_eq!(compare_ignore_case("\u{17f}", "s"), Ordering::Equal);
        assert_eq!(compare_ignore_case("\u{130}vy", "jane"), Ordering::Less);

        // Java compares UTF-16 code units. U+10000 starts with D800 and therefore sorts before
        // the BMP private-use character E000; scalar-value comparison gives the opposite sign.
        assert_eq!(compare_ignore_case("\u{10000}", "\u{e000}"), Ordering::Less);
        assert_eq!(
            compare_ignore_case("\u{e000}", "\u{10000}"),
            Ordering::Greater
        );

        // Exercise the public sorting path too. Equal scores use the comparator with reversed
        // arguments, so the greater folded name precedes the lesser one in this returned list.
        let mut unicode_board = Scoreboard::new();
        unicode_board.apply(&create("unicode", "Unicode"));
        for player in ["\u{130}vy", "jane", "\u{10000}", "\u{e000}"] {
            unicode_board.apply(&set_score(player, "unicode", 1));
        }
        assert_eq!(
            unicode_board.sorted_scores("unicode"),
            [
                ("\u{e000}", 1),
                ("\u{10000}", 1),
                ("jane", 1),
                ("\u{130}vy", 1),
            ]
        );
    }

    /// Java's stable score sort preserves the outer HashMap's encounter order when
    /// `compareToIgnoreCase` returns zero for distinct names. That is bucket/chain order, not an
    /// arbitrary lexical fallback and not Rust HashMap iteration.
    #[test]
    fn case_equivalent_score_names_follow_java_outer_bucket_chain_lifecycle() {
        let mut board = Scoreboard::new();
        board.apply(&create("k", "K"));
        board.apply(&set_score("bob", "k", 5));
        board.apply(&set_score("Bob", "k", 5));
        assert_eq!(java_hash_spread("bob") & 15, java_hash_spread("Bob") & 15);
        assert_eq!(board.sorted_scores("k"), [("bob", 5), ("Bob", 5)]);

        board.apply(&ScoreboardEvent::Score(UpdateScore {
            player_name: "bob".to_owned(),
            objective_name: "k".to_owned(),
            action: UpdateScoreAction::Remove,
        }));
        board.apply(&set_score("bob", "k", 5));
        assert_eq!(
            board.sorted_scores("k"),
            [("Bob", 5), ("bob", 5)],
            "removing the last inner score removes the outer key; reinsertion appends its chain node"
        );

        // Different buckets override insertion order even though the score comparator equates
        // the names. Lowercase is inserted second but occupies the earlier Java bucket.
        let mut buckets = Scoreboard::new();
        buckets.apply(&create("k", "K"));
        buckets.apply(&set_score("ABCDEFGH", "k", 5));
        buckets.apply(&set_score("abcdefgh", "k", 5));
        assert_eq!(
            buckets.sorted_scores("k"),
            [("abcdefgh", 5), ("ABCDEFGH", 5)]
        );
    }

    /// At the thirteenth outer key Java's default HashMap grows from 16 to 32 buckets and
    /// stably splits each old chain into low/high halves. The two case variants share bucket 9
    /// before the resize; afterward the later low-half node precedes the earlier high-half node.
    #[test]
    fn case_equivalent_score_order_tracks_java_hashmap_resize_splitting() {
        let mut small = Scoreboard::new();
        small.apply(&create("k", "K"));
        small.apply(&set_score("aaaaaaaa", "k", 5));
        small.apply(&set_score("AAaAaaaa", "k", 5));
        assert_eq!(small.sorted_scores("k"), [("aaaaaaaa", 5), ("AAaAaaaa", 5)]);

        let mut grown = Scoreboard::new();
        grown.apply(&create("k", "K"));
        grown.apply(&set_score("aaaaaaaa", "k", 5));
        grown.apply(&set_score("AAaAaaaa", "k", 5));
        for index in 0..11 {
            grown.apply(&set_score(&format!("filler{index}"), "k", 100 + index));
        }
        assert_eq!(grown.player_order.capacity, 32);
        let tied: Vec<_> = grown
            .sorted_scores("k")
            .into_iter()
            .filter(|(_, points)| *points == 5)
            .collect();
        assert_eq!(tied, [("AAaAaaaa", 5), ("aaaaaaaa", 5)]);
    }

    /// Java preserves an empty outer map after bulk objective removal, and an early score under
    /// the inner null-objective key also creates the outer player entry. Both affect a later
    /// case-equivalent tie even though neither currently contributes a visible score.
    #[test]
    fn invisible_java_outer_rows_keep_their_later_tie_position() {
        let mut board = Scoreboard::new();
        board.apply(&set_score("bob", "missing", 1));
        board.apply(&create("k", "K"));
        board.apply(&set_score("Bob", "k", 5));
        board.apply(&set_score("bob", "k", 5));
        assert_eq!(board.sorted_scores("k"), [("bob", 5), ("Bob", 5)]);

        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "k".to_owned(),
            mode: ScoreboardObjectiveMode::Remove,
        }));
        board.apply(&create("k", "K"));
        board.apply(&set_score("Bob", "k", 5));
        board.apply(&set_score("bob", "k", 5));
        assert_eq!(board.sorted_scores("k"), [("bob", 5), ("Bob", 5)]);
    }

    /// java: `ScorePlayerTeam.formatPlayerName` = prefix + name + suffix, and `addPlayerToTeam`
    /// moves a player off their previous team first.
    #[test]
    fn team_membership_formats_names_and_a_player_belongs_to_one_team_at_a_time() {
        let mut board = Scoreboard::new();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}c[R] ", "\u{a7}r!", 12),
                players: vec!["bob".to_owned()],
            },
        })));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "blue".to_owned(),
            action: TeamsAction::Create {
                info: team_info("\u{a7}9[B] ", "", 9),
                players: Vec::new(),
            },
        })));

        assert_eq!(board.format_player_name("bob"), "\u{a7}c[R] bob\u{a7}r!");
        assert_eq!(
            board.format_player_name("nobody"),
            "nobody",
            "java: formatPlayerName(null, name) == name"
        );

        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "blue".to_owned(),
            action: TeamsAction::AddPlayers {
                players: vec!["bob".to_owned()],
            },
        })));
        assert_eq!(
            board.player_team("bob").map(|t| t.name.as_str()),
            Some("blue")
        );
        assert!(
            board.team("red").unwrap().members.is_empty(),
            "java: addPlayerToTeam calls removePlayerFromTeams first"
        );
        assert_eq!(board.format_player_name("bob"), "\u{a7}9[B] bob");

        // An unknown team is a no-op, not an insert. java: `!teams.containsKey -> return false`.
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "green".to_owned(),
            action: TeamsAction::AddPlayers {
                players: vec!["ann".to_owned()],
            },
        })));
        assert!(board.player_team("ann").is_none());
    }

    /// java: `removeTeam` drops every membership pointing at it; `UpdateInfo` (mode 2) leaves
    /// membership alone.
    #[test]
    fn removing_a_team_unbinds_its_members_and_mode_two_keeps_them() {
        let mut board = Scoreboard::new();
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Create {
                info: team_info("[R] ", "", 12),
                players: vec!["bob".to_owned(), "ann".to_owned()],
            },
        })));
        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::UpdateInfo {
                info: team_info("<R> ", "", 12),
            },
        })));
        assert_eq!(board.format_player_name("bob"), "<R> bob");
        assert_eq!(board.team("red").unwrap().members.len(), 2);

        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::RemovePlayers {
                players: vec!["ann".to_owned()],
            },
        })));
        assert!(board.player_team("ann").is_none());
        assert_eq!(board.team("red").unwrap().members, ["bob"]);

        board.apply(&ScoreboardEvent::Teams(Box::new(Teams {
            team_name: "red".to_owned(),
            action: TeamsAction::Remove,
        })));
        assert!(board.team("red").is_none());
        assert!(board.player_team("bob").is_none());
        assert_eq!(board.format_player_name("bob"), "bob");
    }

    /// The colour index is what `GuiIngame`'s `3 + colorIndex` slot selection depends on, so the
    /// table has to line up with `EnumChatFormatting`'s declaration order exactly.
    #[test]
    fn the_colour_table_matches_enum_chat_formatting() {
        assert_eq!(CHAT_COLOR_NAMES[12], "red");
        assert_eq!(CHAT_COLOR_NAMES[15], "white");
    }
}
