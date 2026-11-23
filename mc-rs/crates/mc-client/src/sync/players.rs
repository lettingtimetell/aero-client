//! The tab list: who is online, their skin properties, game mode and ping.
//!
//! java: `client/network/NetHandlerPlayClient.playerInfoMap` + `handlePlayerListItem`
//! (`:1618`), `client/network/NetworkPlayerInfo.java`, and
//! `handlePlayerListHeaderFooter` (`:1600`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use mc_protocol::play::s2c::{
    AddPlayerEntry, GameMode, PlayerListItem, ProfileProperty, UpdateDisplayNameEntry,
    UpdateGameModeEntry, UpdateLatencyEntry,
};
use mc_ui::font::component::ChatComponent;
use uuid::Uuid;

use crate::app::appearance::{decode_textures_property, ProfileTextures};
use crate::error::ClientError;

/// The signature verdict and decoded payload of one profile's `textures` property, worked out at
/// most once per profile generation.
///
/// java: `SkinManager.loadProfileTextures` runs `getTextures(profile, true)` once on its thread
/// pool, then `NetworkPlayerInfo` keeps the result. Resolving the property on every use costs an
/// RSA verification per skin, cape and arm-model lookup, on the game thread, every frame.
///
/// Clones share the answer, which is what lets a spawn snapshot reuse its tab-list row. A profile
/// replacement builds a new row and therefore starts empty, so the cache cannot outlive the
/// property list it describes. It carries no identity of its own, so equality ignores it.
#[derive(Clone, Default)]
pub struct TrustedTextures(Arc<TrustedState>);

#[derive(Default)]
struct TrustedState {
    verdict: OnceLock<TexturesVerdict>,
    /// A background verification of this row has been queued, so a frame that still finds no
    /// verdict does not queue another one.
    queued: AtomicBool,
}

struct TexturesVerdict {
    signature_valid: bool,
    /// `None` when the signature failed or the payload is unusable.
    decoded: Option<Arc<ProfileTextures>>,
}

impl TexturesVerdict {
    /// The verdict for a `textures` value whose signature check returned `signature_valid`.
    /// Shared by the synchronous path and the background verifier, so both build the same thing.
    fn checked(signature_valid: bool, value: &str) -> Self {
        Self {
            signature_valid,
            decoded: signature_valid
                .then(|| decode_textures_property(value))
                .flatten()
                .map(Arc::new),
        }
    }
}

/// Rows waiting for the background verifier. A full queue makes a row fall back to verifying on
/// first use, as before, so a server flooding tab-list rows cannot queue unbounded work; real tab
/// lists are at most a few hundred rows.
const MAX_QUEUED_VERIFICATIONS: usize = 4096;

/// Net-side pre-checks waiting for the background verifier. They only warm `mc_auth`'s memo, so a
/// full queue just drops them (the row then verifies when first drawn); the small bound keeps a
/// server that floods signed rows from holding a core busy.
const MAX_QUEUED_PRECHECKS: usize = 256;

/// One RSA `textures` check for the background verifier. `row` is `None` for a net-side
/// pre-check, which only warms `mc_auth`'s memo.
struct VerifyJob {
    row: Option<Arc<TrustedState>>,
    value: String,
    signature: String,
}

/// The verifier's two queues. Row jobs (a row a frame is waiting to draw) are served before
/// pre-checks, so a row never waits behind a join burst of pre-checks for other players.
#[derive(Default)]
struct VerifyQueues {
    rows: std::collections::VecDeque<VerifyJob>,
    prechecks: std::collections::VecDeque<VerifyJob>,
}

/// The verifier thread's shared state: its queues and the condition variable it sleeps on.
type VerifyShared = std::sync::Arc<(std::sync::Mutex<VerifyQueues>, std::sync::Condvar)>;

/// Hand one check to the one long-lived thread that runs texture-signature checks off the game
/// thread (HITCH-1). Never blocks; `false` when that job's queue is full or the thread could not
/// be started, so the caller can fall back to what it did before.
///
/// java: `SkinManager.loadProfileTextures` runs `getTextures(profile, true)` — the RSA check — on
/// its `THREAD_POOL` and only then schedules the download on the game thread, so a row's skin
/// request is never synchronous with the frame that first needs it there either. Row jobs run
/// before pre-checks; each queue is FIFO.
fn submit_verification(job: VerifyJob) -> bool {
    static SHARED: OnceLock<Option<VerifyShared>> = OnceLock::new();
    let shared = SHARED.get_or_init(|| {
        let shared: VerifyShared = std::sync::Arc::default();
        let worker = std::sync::Arc::clone(&shared);
        std::thread::Builder::new()
            .name("mc-skin-verify".to_owned())
            .spawn(move || verify_worker(&worker))
            .ok()?;
        Some(shared)
    });
    let Some(shared) = shared.as_ref() else {
        return false;
    };
    let (queues, ready) = &**shared;
    let mut queues = queues
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (queue, bound) = if job.row.is_some() {
        (&mut queues.rows, MAX_QUEUED_VERIFICATIONS)
    } else {
        (&mut queues.prechecks, MAX_QUEUED_PRECHECKS)
    };
    if queue.len() >= bound {
        return false;
    }
    queue.push_back(job);
    ready.notify_one();
    true
}

/// The `mc-skin-verify` thread: row jobs first, then pre-checks, forever.
fn verify_worker(shared: &VerifyShared) {
    let (queues, ready) = &**shared;
    loop {
        let job = {
            let mut guard = queues
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if let Some(job) = guard
                    .rows
                    .pop_front()
                    .or_else(|| guard.prechecks.pop_front())
                {
                    break job;
                }
                guard = ready
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        if job
            .row
            .as_ref()
            .is_some_and(|row| row.verdict.get().is_some())
        {
            continue;
        }
        // A panicking check must not strand its row as "pending" forever: treat it as an
        // invalid signature (the row keeps the default skin) and keep serving.
        // Contained, so this recovered panic is not the game's crash report.
        let verdict = mc_core::panic_scope::contain(|| {
            let valid = mc_auth::verify_textures_signature(&job.value, &job.signature);
            TexturesVerdict::checked(valid, &job.value)
        })
        .unwrap_or(TexturesVerdict {
            signature_valid: false,
            decoded: None,
        });
        if let Some(row) = job.row {
            let _ = row.verdict.get_or_init(|| verdict);
        }
    }
}

/// Queue a net-side pre-check of one signed `textures` pair, so the game thread usually finds the
/// verdict already remembered. Never blocks; a full queue drops the pre-check, and the row then
/// queues its own check when first drawn.
pub(crate) fn precheck_texture_signature(value: String, signature: String) {
    let _ = submit_verification(VerifyJob {
        row: None,
        value,
        signature,
    });
}

/// java: `Iterables.getFirst(profile.properties.get("textures"), null)` in authlib's
/// `YggdrasilMinecraftSessionService.getTextures`: only the first value is ever considered.
fn first_textures_property(properties: &[ProfileProperty]) -> Option<&ProfileProperty> {
    properties
        .iter()
        .find(|property| property.name == "textures")
}

impl TrustedTextures {
    /// Whether this row's signature verdict exists yet. The background verifier
    /// can deliver it frames after the packet that added the row, so a consumer caching on
    /// [`PlayerList::revision`] must notice that too (see [`PlayerList::resolved_textures`]).
    pub(crate) fn is_resolved(&self) -> bool {
        self.0.verdict.get().is_some()
    }

    fn verdict(&self, properties: &[ProfileProperty]) -> &TexturesVerdict {
        self.0.verdict.get_or_init(|| {
            let signed = first_textures_property(properties).filter(|property| {
                property.signature.as_deref().is_some_and(|signature| {
                    mc_auth::verify_textures_signature(&property.value, signature)
                })
            });
            TexturesVerdict {
                signature_valid: signed.is_some(),
                decoded: signed
                    .and_then(|property| decode_textures_property(&property.value))
                    .map(Arc::new),
            }
        })
    }

    /// [`Self::verdict`] for frame-thread callers: never runs the RSA check itself.
    ///
    /// Answers from the row's own verdict or from `mc_auth`'s memo (filled by the net-side
    /// pre-check) when either has one. Otherwise it queues the row on the background verifier and
    /// returns `None` — "not verified yet" — for the frame or two that takes; callers treat that
    /// exactly like a skin whose download has not landed (UUID-default Steve/Alex), which is what
    /// vanilla shows while `SkinManager`'s pool verifies. Rows with no signed `textures` property
    /// need no RSA and resolve immediately; a full queue falls back to verifying inline.
    fn try_verdict(&self, properties: &[ProfileProperty]) -> Option<&TexturesVerdict> {
        if let Some(verdict) = self.0.verdict.get() {
            return Some(verdict);
        }
        let Some((property, signature)) = first_textures_property(properties)
            .and_then(|property| Some((property, property.signature.as_deref()?)))
        else {
            return Some(self.verdict(properties));
        };
        // Consulted even while this row's own job is queued: a pre-check of the same pair that
        // finishes first answers the row at once instead of after its queued job.
        if let Some(valid) = mc_auth::remembered_textures_verdict(&property.value, signature) {
            return Some(
                self.0
                    .verdict
                    .get_or_init(|| TexturesVerdict::checked(valid, &property.value)),
            );
        }
        if self.0.queued.swap(true, Ordering::AcqRel) {
            // The verifier fills this row's verdict itself; nothing to look up meanwhile.
            return None;
        }
        let queued = submit_verification(VerifyJob {
            row: Some(Arc::clone(&self.0)),
            value: property.value.clone(),
            signature: signature.to_owned(),
        });
        if queued {
            None
        } else {
            self.0.queued.store(false, Ordering::Release);
            Some(self.verdict(properties))
        }
    }

    /// [`Self::profile`] for render and UI callers: `None` also while the signature is still
    /// being verified off-thread (see [`Self::try_verdict`]). Same answer as `profile` once known.
    pub(crate) fn try_profile<'a>(
        &self,
        properties: &'a [ProfileProperty],
    ) -> Option<(&'a str, Arc<ProfileTextures>)> {
        let verdict = self.try_verdict(properties)?;
        let decoded = verdict.decoded.clone()?;
        let value = verdict
            .signature_valid
            .then(|| first_textures_property(properties))
            .flatten()?;
        Some((value.value.as_str(), decoded))
    }

    /// The signed payload, if the signature holds.
    pub(crate) fn signed_value<'a>(&self, properties: &'a [ProfileProperty]) -> Option<&'a str> {
        self.verdict(properties)
            .signature_valid
            .then(|| first_textures_property(properties))
            .flatten()
            .map(|property| property.value.as_str())
    }

    /// The signed payload and its decoded texture map, if both are usable.
    ///
    /// Synchronous: verifies on the calling thread when nothing has yet. Frame-path callers use
    /// [`Self::try_profile`]; this stays for callers that must have the answer now.
    #[allow(dead_code)]
    pub(crate) fn profile<'a>(
        &self,
        properties: &'a [ProfileProperty],
    ) -> Option<(&'a str, Arc<ProfileTextures>)> {
        let decoded = self.verdict(properties).decoded.clone()?;
        Some((self.signed_value(properties)?, decoded))
    }
}

impl PartialEq for TrustedTextures {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for TrustedTextures {}

impl std::fmt::Debug for TrustedTextures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TrustedTextures")
    }
}

/// One row of the tab list. java: `NetworkPlayerInfo`
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerInfo {
    /// The account UUID — the map key, repeated here so an iterator sees it.
    pub uuid: Uuid,
    /// The account name.
    pub username: String,
    /// Signed profile properties: `textures` (skin/cape) and anything else the server attached.
    /// java: `GameProfile.getProperties`
    pub properties: Vec<ProfileProperty>,
    /// Survival/creative/adventure/spectator, or [`GameMode::NotSet`].
    pub game_mode: GameMode,
    /// Round-trip time in milliseconds, as the **server** measured it. This is the only latency
    /// figure vanilla ever shows — see [`crate::sync::ClientState::latency_ms`].
    pub ping_ms: i32,
    /// A server-supplied override for the displayed name, already parsed.
    /// java: `NetworkPlayerInfo.displayName`
    pub display_name: Option<ChatComponent>,
    /// The signature verdict for [`Self::properties`], worked out on first use.
    pub trusted_textures: TrustedTextures,
}

impl PlayerInfo {
    /// The signed `textures` property's base64 payload, if the server sent one. S12 needs it to
    /// fetch a skin.
    ///
    /// java: `NetworkPlayerInfo.loadPlayerTextures` calls
    /// `SkinManager.loadProfileTextures(..., true)`, so a missing, empty, or invalid signature
    /// is insecure and cannot supply a skin.  This is the secure session-service seam: its
    /// all-or-nothing result reaches the same domain-gated parser as a valid profile.
    #[must_use]
    pub fn textures(&self) -> Option<&str> {
        self.trusted_textures.signed_value(&self.properties)
    }
}

/// Everyone the server says is online, plus the tab list's header and footer.
#[derive(Debug, Default, Clone)]
pub struct PlayerList {
    /// [`mc_core::FastMap`]: probed per rendered player per frame (skins,
    /// nameplates, Tab, spectator menu). UUIDs are server-chosen, so the hasher stays seeded;
    /// iteration order stays per-process random, as with `std`'s hasher.
    entries: mc_core::FastMap<Uuid, PlayerInfo>,
    /// java: `GuiPlayerTabOverlay.header`. `None` before the server has ever sent one — see
    /// [`PlayerList::set_header_footer`] for why an empty component does **not** clear it.
    pub header: Option<ChatComponent>,
    /// java: `GuiPlayerTabOverlay.footer`
    pub footer: Option<ChatComponent>,
    /// Changes on every `apply`/`clear`. Drawn from a process-wide counter, so two
    /// lists never share a value and a cache keyed on it cannot confuse a reconnect's fresh list
    /// with the old one. `0` only for a list nothing was ever applied to.
    revision: u64,
}

impl PlayerList {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look one player up.
    #[must_use]
    pub fn get(&self, uuid: &Uuid) -> Option<&PlayerInfo> {
        self.entries.get(uuid)
    }

    /// A value that changes whenever any row may have changed — added, removed, or its game
    /// mode, ping, display name or profile replaced. Lets per-frame consumers (the spectator
    /// menu) rebuild only when the list did. Not part of any Java type.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// How many rows already have a `textures` signature verdict. A verdict can appear without
    /// any packet (it is resolved on first use, or by a background verifier), and it decides
    /// whether a row has a skin URL at all, so per-frame caches key on this next to
    /// [`Self::revision`]. One pointer load per row.
    #[must_use]
    pub fn resolved_textures(&self) -> usize {
        self.entries
            .values()
            .filter(|info| info.trusted_textures.is_resolved())
            .count()
    }

    /// Look up the exact case-sensitive profile name.
    ///
    /// java: `NetHandlerPlayClient.getPlayerInfo(String)` walks the map values and compares
    /// `GameProfile.getName().equals(name)`. `TeleportToTeam` uses this route for scoreboard
    /// membership strings, which do not carry UUIDs.
    #[must_use]
    pub fn get_by_username(&self, username: &str) -> Option<&PlayerInfo> {
        self.entries.values().find(|info| info.username == username)
    }

    /// How many players are listed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nobody is listed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every row, in no particular order. java: vanilla sorts in `GuiPlayerTabOverlay`, not
    /// here, so the ordering stays the UI's decision.
    pub fn iter(&self) -> impl Iterator<Item = &PlayerInfo> {
        self.entries.values()
    }

    /// Drop every row (a disconnect, or a `loadWorld(null)`).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.revision = next_list_revision();
        self.header = None;
        self.footer = None;
    }

    /// Apply an `S38PacketPlayerListItem`.
    ///
    /// java: `handlePlayerListItem` (`:1618`). Two details worth keeping: an update action for
    /// a UUID that was never added is **ignored**, not inserted (`networkplayerinfo` is `null`
    /// and the switch is skipped), and `ADD_PLAYER` for a UUID already present **replaces** the
    /// row wholesale.
    ///
    /// # Errors
    ///
    /// [`ClientError::Chat`] if a display-name component does not parse. Vanilla would have
    /// thrown inside Gson during packet decode; failing here keeps the same "a malformed
    /// component is a protocol error" behaviour rather than silently dropping the name.
    pub fn apply(&mut self, packet: &PlayerListItem) -> Result<(), ClientError> {
        // Bumped up front: an error path leaves the rows unchanged, and a spurious bump only
        // costs a consumer one rebuild.
        self.revision = next_list_revision();
        match packet {
            PlayerListItem::AddPlayer(entries) => {
                // Java parses every AddPlayerData (including its optional chat component) in
                // S38PacketPlayerListItem.readPacketData before processPacket can mutate the
                // map. Preserve that packet-level atomicity: one malformed later display name
                // must not leave earlier rows installed.
                let parsed = entries
                    .iter()
                    .map(player_info)
                    .collect::<Result<Vec<_>, _>>()?;
                for info in parsed {
                    self.entries.insert(info.uuid, info);
                }
            }
            PlayerListItem::UpdateGameMode(entries) => {
                for UpdateGameModeEntry { uuid, game_mode } in entries {
                    if let Some(info) = self.entries.get_mut(uuid) {
                        info.game_mode = *game_mode;
                    }
                }
            }
            PlayerListItem::UpdateLatency(entries) => {
                for UpdateLatencyEntry { uuid, ping } in entries {
                    if let Some(info) = self.entries.get_mut(uuid) {
                        info.ping_ms = *ping;
                    }
                }
            }
            PlayerListItem::UpdateDisplayName(entries) => {
                // As for ADD_PLAYER, decode all components before applying any entry.
                let parsed = entries
                    .iter()
                    .map(|UpdateDisplayNameEntry { uuid, display_name }| {
                        Ok((*uuid, parse_optional_component(display_name.as_deref())?))
                    })
                    .collect::<Result<Vec<_>, ClientError>>()?;
                for (uuid, display_name) in parsed {
                    if let Some(info) = self.entries.get_mut(&uuid) {
                        info.display_name = display_name;
                    }
                }
            }
            PlayerListItem::RemovePlayer(uuids) => {
                for uuid in uuids {
                    self.entries.remove(uuid);
                }
            }
        }
        Ok(())
    }

    /// Apply `S47PacketPlayerListHeaderFooter`.
    ///
    /// java: `handlePlayerListHeaderFooter` (`:1600`) stores `null` when
    /// `getFormattedText().length() == 0`.
    ///
    /// That check is **effectively vestigial in 1.8.9**, and the port keeps it that way rather
    /// than "fixing" it. `ChatComponentStyle.getFormattedText` (`:87`) iterates the component
    /// and its siblings and unconditionally appends `EnumChatFormatting.RESET` after each — and
    /// `iterator()` always yields at least the component itself. So even `{"text":""}` produces
    /// `"§r"`, length 2, and vanilla stores it. A client that treated an empty *text* as a clear
    /// would show no header where vanilla shows an empty line, and would differ from every
    /// server that clears the header by sending an empty component.
    ///
    /// # Errors
    ///
    /// [`ClientError::Chat`] if either component fails to parse.
    pub fn set_header_footer(&mut self, header: &str, footer: &str) -> Result<(), ClientError> {
        // S47 reads both components before dispatch, so a malformed footer cannot install the
        // already-decoded header. Parse first and commit together.
        let header = non_empty_component(header)?;
        let footer = non_empty_component(footer)?;
        self.header = header;
        self.footer = footer;
        Ok(())
    }
}

fn player_info(entry: &AddPlayerEntry) -> Result<PlayerInfo, ClientError> {
    Ok(PlayerInfo {
        uuid: entry.uuid,
        username: entry.username.clone(),
        properties: entry.properties.clone(),
        game_mode: entry.game_mode,
        ping_ms: entry.ping,
        display_name: parse_optional_component(entry.display_name.as_deref())?,
        trusted_textures: TrustedTextures::default(),
    })
}

fn parse_optional_component(json: Option<&str>) -> Result<Option<ChatComponent>, ClientError> {
    match json {
        // java: S38 reads `readChatComponent` after a true `hasDisplayName`; Gson can still
        // resolve `null`/empty/`[]` JSON to a null component, and NetworkPlayerInfo stores that
        // null exactly like an absent display name.
        Some(json) => Ok(ChatComponent::from_json_nullable(json)?),
        None => Ok(None),
    }
}

fn non_empty_component(json: &str) -> Result<Option<ChatComponent>, ClientError> {
    let component = ChatComponent::from_json(json)?;
    let lang = mc_assets::Language::new("en_US");
    if component.formatted_text(&lang).is_empty() {
        Ok(None)
    } else {
        Ok(Some(component))
    }
}

/// The next [`PlayerList::revision`]: process-wide and never `0`.
fn next_list_revision() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed official `?unsigned=false` Notch profile property.  The generic `add` helper
    // models a secure tab-list entry, so unrelated player-list tests do not accidentally depend
    // on an invalid placeholder signature.
    const NOTCH_TEXTURE_VALUE: &str = "ewogICJ0aW1lc3RhbXAiIDogMTc4NzY5MTA3NTE3MiwKICAicHJvZmlsZUlkIiA6ICIwNjlhNzlmNDQ0ZTk0NzI2YTViZWZjYTkwZTM4YWFmNSIsCiAgInByb2ZpbGVOYW1lIiA6ICJOb3RjaCIsCiAgInNpZ25hdHVyZVJlcXVpcmVkIiA6IHRydWUsCiAgInRleHR1cmVzIiA6IHsKICAgICJTS0lOIiA6IHsKICAgICAgInVybCIgOiAiaHR0cDovL3RleHR1cmVzLm1pbmVjcmFmdC5uZXQvdGV4dHVyZS8yOTIwMDlhNDkyNWI1OGYwMmM3N2RhZGMzZWNlZjA3ZWE0Yzc0NzJmNjRlMGZkYzMyY2U1NTIyNDg5MzYyNjgwIgogICAgfQogIH0KfQ==";
    const NOTCH_TEXTURE_SIGNATURE: &str = "oqgTiH1ZWCjCcU9a5bOJpskVzloygYgbdNZ16Xtn1RgKrRsPLutWL9Uok/rkg4re1O+TbsKL0lKYzQtZH0+OfWdfWIoWE4bBanHuhejGWu69EWVvcxlpEiNhZptzER9KfPGP5Q7SPsT+4MaSj/AbJP+yKWNjriYZZFV00bWSW/aexY6PJSoXMD/R57gn0TucdaZp1nrbfgL9DXf+6R6T8ou2Lz4qsPAoB2CJI2I8F9HPsLJ/dUinrp8gmjfKnvvJUf+a6XeG+BESWqqChuiD+XJjvnZSyM3Fs1zdb+RkqvFNqT4W14ohyh3mxyX3Q2S/8S+Q2axT47nnjvy9WMVpaDay1o8pquzy+zRqPps1tRFTScrp4w4jVdSzm8xL4FhNqm2H/QUoAYczVUyAfBx4Q8w9daelERlNOcQAIEeauBzan/728P/F4Rxc24Wl9iV5Qb+XeE/858QHNplwLIcSv4Wfp6G29++0XVljN7K2Vtbilp9LipVrDG028XMFY6eY/lb1n9C4hiMWF/N4YTXz8Tm+IBIBqKpgeUCzb3ZmB7bvVubPUgWdL3K7piiClRQ8ssIAN67DEOm/sTgyv0eFitWiZopPhBvgizOBPDv2/GSa3WdFRiNAtaxrqOZVO3W4HUn62XNz+leI640BMjAYMvIcpNU2x0R+S8ILf7v0zcM=";

    fn uuid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn add(uuid: Uuid, name: &str, ping: i32) -> PlayerListItem {
        PlayerListItem::AddPlayer(vec![AddPlayerEntry {
            uuid,
            username: name.to_owned(),
            properties: vec![ProfileProperty {
                name: "textures".into(),
                value: NOTCH_TEXTURE_VALUE.into(),
                signature: Some(NOTCH_TEXTURE_SIGNATURE.into()),
            }],
            game_mode: GameMode::Survival,
            ping,
            display_name: None,
        }])
    }

    /// Poll a fresh row's frame-path lookup the way successive frames would, until the background
    /// verifier answers or a generous deadline passes.
    fn try_profile_until_ready(
        properties: &[ProfileProperty],
    ) -> (TrustedTextures, Option<(String, Arc<ProfileTextures>)>) {
        let row = TrustedTextures::default();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            if let Some((value, decoded)) = row.try_profile(properties) {
                let answer = Some((value.to_owned(), decoded));
                return (row, answer);
            }
            if row.0.verdict.get().is_some() || std::time::Instant::now() > deadline {
                return (row, None);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// HITCH-1: the frame-path lookup reaches the same verdict as the synchronous one — same
    /// payload, same decoded URLs and model — it only may take a frame or two to get there.
    #[test]
    fn frame_path_profile_matches_the_synchronous_verdict_once_verified() {
        let properties = vec![ProfileProperty {
            name: "textures".into(),
            value: NOTCH_TEXTURE_VALUE.into(),
            signature: Some(NOTCH_TEXTURE_SIGNATURE.into()),
        }];
        let (row, answer) = try_profile_until_ready(&properties);
        let synchronous = TrustedTextures::default();
        let expected = synchronous
            .profile(&properties)
            .expect("the official Notch property verifies");
        let (value, decoded) = answer.expect("the background verifier answers");
        assert_eq!(value, expected.0);
        assert_eq!(decoded.skin_url, expected.1.skin_url);
        assert_eq!(decoded.cape_url, expected.1.cape_url);
        assert_eq!(decoded.model, expected.1.model);
        // Once known, the row answers synchronously and identically on every later frame.
        assert_eq!(
            row.try_profile(&properties).map(|(value, _)| value),
            Some(NOTCH_TEXTURE_VALUE)
        );
        assert_eq!(row.signed_value(&properties), Some(NOTCH_TEXTURE_VALUE));
    }

    /// A row already queued behind other checks still answers from `mc_auth`'s memo as soon as
    /// a check of the same pair (e.g. the net-side pre-check) has landed there, instead of waiting
    /// for its own queued job (Phase 3a review S2).
    #[test]
    fn a_queued_row_answers_from_the_memo_without_waiting_for_its_job() {
        let properties = vec![ProfileProperty {
            name: "textures".into(),
            value: NOTCH_TEXTURE_VALUE.into(),
            signature: Some(NOTCH_TEXTURE_SIGNATURE.into()),
        }];
        let row = TrustedTextures::default();
        // As if its own job were queued and not yet served.
        row.0.queued.store(true, Ordering::Release);
        assert!(mc_auth::verify_textures_signature(
            NOTCH_TEXTURE_VALUE,
            NOTCH_TEXTURE_SIGNATURE
        ));
        assert_eq!(
            row.try_profile(&properties).map(|(value, _)| value),
            Some(NOTCH_TEXTURE_VALUE),
            "the memo answer must not wait for the queued job"
        );
    }

    /// A bad signature stays insecure on the frame path: it never yields a profile, and the
    /// row's verdict ends up exactly the synchronous one (`signature_valid == false`).
    #[test]
    fn frame_path_rejects_a_bad_signature_like_the_synchronous_path() {
        let properties = vec![ProfileProperty {
            name: "textures".into(),
            value: NOTCH_TEXTURE_VALUE.into(),
            signature: Some("QUJDRA==".into()),
        }];
        let (row, answer) = try_profile_until_ready(&properties);
        assert!(answer.is_none());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while row.0.verdict.get().is_none() && std::time::Instant::now() < deadline {
            let _ = row.try_profile(&properties);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            row.0.verdict.get().map(|verdict| verdict.signature_valid),
            Some(false)
        );
        assert_eq!(row.signed_value(&properties), None);
    }

    /// Rows without a signed `textures` property need no RSA, so the frame path answers them on
    /// the first call instead of queueing them.
    #[test]
    fn frame_path_resolves_unsigned_rows_without_queueing() {
        let unsigned = vec![ProfileProperty {
            name: "textures".into(),
            value: NOTCH_TEXTURE_VALUE.into(),
            signature: None,
        }];
        let row = TrustedTextures::default();
        assert!(row.try_profile(&unsigned).is_none());
        assert!(
            row.0.verdict.get().is_some(),
            "resolved without the verifier"
        );
        assert!(!row.0.queued.load(Ordering::Acquire));
        let none = TrustedTextures::default();
        assert!(none.try_profile(&[]).is_none());
        assert!(none.0.verdict.get().is_some());
    }

    #[test]
    fn add_then_update_latency_and_game_mode() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(1), "Notch", 42)).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list.get(&uuid(1)).unwrap().ping_ms, 42);
        assert_eq!(
            list.get(&uuid(1)).unwrap().textures(),
            Some(NOTCH_TEXTURE_VALUE)
        );

        list.apply(&PlayerListItem::UpdateLatency(vec![UpdateLatencyEntry {
            uuid: uuid(1),
            ping: 137,
        }]))
        .unwrap();
        list.apply(&PlayerListItem::UpdateGameMode(vec![UpdateGameModeEntry {
            uuid: uuid(1),
            game_mode: GameMode::Creative,
        }]))
        .unwrap();

        let info = list.get(&uuid(1)).unwrap();
        assert_eq!(info.ping_ms, 137);
        assert_eq!(info.game_mode, GameMode::Creative);
        assert_eq!(
            info.username, "Notch",
            "an update must not clobber the name"
        );
    }

    /// The spectator menu caches on `revision`, so every row mutation and every
    /// clear must move it, and two lists must never share a value (a reconnect's fresh list
    /// cannot alias the old one's cache key).
    #[test]
    fn revision_moves_on_every_apply_and_clear_and_is_unique_per_list() {
        let mut list = PlayerList::new();
        let fresh = PlayerList::new();
        assert_eq!(
            list.revision(),
            fresh.revision(),
            "untouched lists are both empty"
        );
        let mut seen = vec![list.revision()];
        list.apply(&add(uuid(1), "Notch", 42)).unwrap();
        seen.push(list.revision());
        // The verdict appears on first use without any packet; caches key on this count too.
        assert_eq!(list.resolved_textures(), 0);
        assert!(list.get(&uuid(1)).unwrap().textures().is_some());
        assert_eq!(list.resolved_textures(), 1);
        list.apply(&PlayerListItem::UpdateGameMode(vec![UpdateGameModeEntry {
            uuid: uuid(1),
            game_mode: GameMode::Spectator,
        }]))
        .unwrap();
        seen.push(list.revision());
        list.clear();
        seen.push(list.revision());
        let mut other = PlayerList::new();
        other.apply(&add(uuid(1), "Notch", 42)).unwrap();
        seen.push(other.revision());
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seen.len(), "{seen:?}");
    }

    /// java: `networkplayerinfo` is `null` for an unknown UUID and the switch is skipped
    /// entirely — an update never creates a row.
    #[test]
    fn an_update_for_an_unknown_uuid_creates_nothing() {
        let mut list = PlayerList::new();
        list.apply(&PlayerListItem::UpdateLatency(vec![UpdateLatencyEntry {
            uuid: uuid(9),
            ping: 1,
        }]))
        .unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn remove_player_drops_the_row() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(1), "Notch", 0)).unwrap();
        list.apply(&add(uuid(2), "jeb_", 0)).unwrap();
        list.apply(&PlayerListItem::RemovePlayer(vec![uuid(1)]))
            .unwrap();
        assert_eq!(list.len(), 1);
        assert!(list.get(&uuid(1)).is_none());
        assert!(list.get(&uuid(2)).is_some());
    }

    #[test]
    fn display_name_is_parsed_and_clearable() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(1), "Notch", 0)).unwrap();
        list.apply(&PlayerListItem::UpdateDisplayName(vec![
            UpdateDisplayNameEntry {
                uuid: uuid(1),
                display_name: Some(r#"{"text":"[Admin] Notch"}"#.into()),
            },
        ]))
        .unwrap();
        let lang = mc_assets::Language::new("en_US");
        assert_eq!(
            list.get(&uuid(1))
                .unwrap()
                .display_name
                .as_ref()
                .unwrap()
                .unformatted_text(&lang),
            "[Admin] Notch"
        );

        list.apply(&PlayerListItem::UpdateDisplayName(vec![
            UpdateDisplayNameEntry {
                uuid: uuid(1),
                display_name: None,
            },
        ]))
        .unwrap();
        assert!(list.get(&uuid(1)).unwrap().display_name.is_none());
    }

    /// java: S38 reads `readChatComponent` after a true `hasDisplayName`,
    /// and Gson resolves `null`, an empty document and `[]` to a null component. `handlePlayer
    /// ListItem` stores that null like an absent name: the row is still added (ADD_PLAYER) or its
    /// name cleared (UPDATE_DISPLAY_NAME). The former strict parse rejected the whole packet.
    #[test]
    fn a_gson_null_display_name_is_stored_as_no_display_name() {
        for null_json in ["null", "", "[]", "  null  "] {
            assert!(
                ChatComponent::from_json(null_json).is_err(),
                "control: the former strict parse rejected {null_json:?} and the whole packet"
            );
            let mut list = PlayerList::new();
            let mut packet = add(uuid(1), "Notch", 0);
            let PlayerListItem::AddPlayer(entries) = &mut packet else {
                unreachable!()
            };
            entries[0].display_name = Some(null_json.to_owned());
            list.apply(&packet)
                .unwrap_or_else(|error| panic!("{null_json:?} must add the row: {error}"));
            let row = list.get(&uuid(1)).expect("row added");
            assert!(row.display_name.is_none(), "{null_json:?}");

            list.apply(&PlayerListItem::UpdateDisplayName(vec![
                UpdateDisplayNameEntry {
                    uuid: uuid(1),
                    display_name: Some(r#"{"text":"[VIP] Notch"}"#.into()),
                },
            ]))
            .unwrap();
            assert!(list.get(&uuid(1)).unwrap().display_name.is_some());
            list.apply(&PlayerListItem::UpdateDisplayName(vec![
                UpdateDisplayNameEntry {
                    uuid: uuid(1),
                    display_name: Some(null_json.to_owned()),
                },
            ]))
            .unwrap_or_else(|error| panic!("{null_json:?} must clear the name: {error}"));
            assert!(
                list.get(&uuid(1)).unwrap().display_name.is_none(),
                "{null_json:?}"
            );
        }
    }

    /// The `getFormattedText().length() == 0` check, ported literally.
    ///
    /// `ChatComponentStyle.getFormattedText` always appends `§r`, so an empty-text component is
    /// **not** empty and vanilla keeps it. This test pins that, because "empty text clears the
    /// header" is the intuitive-but-wrong reading and the one a reviewer will reach for.
    #[test]
    fn an_empty_text_component_is_still_stored_because_vanilla_appends_a_reset() {
        let mut list = PlayerList::new();
        list.set_header_footer(r#"{"text":"Welcome"}"#, r#"{"text":""}"#)
            .unwrap();
        assert!(list.header.is_some());
        assert!(
            list.footer.is_some(),
            "getFormattedText() is \"\\u{{a7}}r\", not \"\""
        );
        let lang = mc_assets::Language::new("en_US");
        assert_eq!(
            list.footer.as_ref().unwrap().formatted_text(&lang),
            "\u{a7}r"
        );
    }

    #[test]
    fn a_malformed_display_name_is_an_error_not_a_panic() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(1), "Notch", 0)).unwrap();
        assert!(matches!(
            list.apply(&PlayerListItem::UpdateDisplayName(vec![
                UpdateDisplayNameEntry {
                    uuid: uuid(1),
                    display_name: Some("{not json".into()),
                },
            ])),
            Err(ClientError::Chat(_))
        ));
    }

    #[test]
    fn an_unsigned_or_empty_signature_cannot_supply_a_skin_texture() {
        for signature in [None, Some(String::new())] {
            let mut packet = add(uuid(1), "Notch", 0);
            let PlayerListItem::AddPlayer(entries) = &mut packet else {
                unreachable!();
            };
            entries[0].properties[0].signature = signature;

            let mut list = PlayerList::new();
            list.apply(&packet).unwrap();
            let info = list.get(&uuid(1)).unwrap();
            assert_eq!(info.textures(), None);
            assert_eq!(
                info.properties.len(),
                1,
                "the GameProfile property remains stored; only skin use is gated"
            );
        }
    }

    #[test]
    fn an_invalid_nonempty_signature_cannot_supply_a_skin_texture() {
        let mut packet = add(uuid(1), "Notch", 0);
        let PlayerListItem::AddPlayer(entries) = &mut packet else {
            unreachable!();
        };
        entries[0].properties[0].signature = Some("not-base64".into());

        let mut list = PlayerList::new();
        list.apply(&packet).unwrap();
        assert_eq!(list.get(&uuid(1)).unwrap().textures(), None);
    }

    /// authlib 1.5.21 `YggdrasilMinecraftSessionService.getTextures` calls
    /// `Iterables.getFirst(profile.properties.get("textures"), null)` and validates that one
    /// property. It does not skip an insecure first value to find a later signed duplicate.
    #[test]
    fn an_unsigned_first_textures_property_is_not_skipped_for_a_later_signed_one() {
        let mut packet = add(uuid(1), "Notch", 0);
        let PlayerListItem::AddPlayer(entries) = &mut packet else {
            unreachable!();
        };
        entries[0].properties = vec![
            ProfileProperty {
                name: "textures".into(),
                value: "unsigned-first".into(),
                signature: None,
            },
            ProfileProperty {
                name: "textures".into(),
                value: "signed-second".into(),
                signature: Some("wire-signature".into()),
            },
        ];

        let mut list = PlayerList::new();
        list.apply(&packet).unwrap();
        assert_eq!(list.get(&uuid(1)).unwrap().textures(), None);
    }

    #[test]
    fn add_player_is_atomic_when_a_later_display_name_is_malformed() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(9), "existing", 9)).unwrap();

        let mut first = match add(uuid(1), "first", 1) {
            PlayerListItem::AddPlayer(mut entries) => entries.remove(0),
            _ => unreachable!(),
        };
        first.display_name = Some(r#"{"text":"valid"}"#.into());
        let mut second = match add(uuid(2), "second", 2) {
            PlayerListItem::AddPlayer(mut entries) => entries.remove(0),
            _ => unreachable!(),
        };
        second.display_name = Some("{not json".into());

        assert!(matches!(
            list.apply(&PlayerListItem::AddPlayer(vec![first, second])),
            Err(ClientError::Chat(_))
        ));
        assert_eq!(list.len(), 1);
        assert!(list.get(&uuid(9)).is_some());
        assert!(list.get(&uuid(1)).is_none());
        assert!(list.get(&uuid(2)).is_none());
    }

    /// The paired control for the malformed-late batch: validation does not turn a valid
    /// multi-entry ADD_PLAYER into a partial/no-op update.  Java has already decoded every
    /// entry by `processPacket`, then installs each `AddPlayerData` in packet order.
    #[test]
    fn a_valid_multi_entry_add_player_batch_installs_every_row() {
        let first = match add(uuid(1), "first", 1) {
            PlayerListItem::AddPlayer(mut entries) => entries.remove(0),
            _ => unreachable!(),
        };
        let mut second = match add(uuid(2), "second", 2) {
            PlayerListItem::AddPlayer(mut entries) => entries.remove(0),
            _ => unreachable!(),
        };
        second.display_name = Some(r#"{"text":"second"}"#.into());

        let mut list = PlayerList::new();
        list.apply(&PlayerListItem::AddPlayer(vec![first, second]))
            .unwrap();

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(&uuid(1)).unwrap().username, "first");
        assert_eq!(list.get(&uuid(2)).unwrap().ping_ms, 2);
        let lang = mc_assets::Language::new("en_US");
        assert_eq!(
            list.get(&uuid(2))
                .unwrap()
                .display_name
                .as_ref()
                .unwrap()
                .unformatted_text(&lang),
            "second"
        );
    }

    #[test]
    fn update_display_name_is_atomic_when_a_later_component_is_malformed() {
        let mut list = PlayerList::new();
        list.apply(&add(uuid(1), "first", 1)).unwrap();
        list.apply(&add(uuid(2), "second", 2)).unwrap();

        let packet = PlayerListItem::UpdateDisplayName(vec![
            UpdateDisplayNameEntry {
                uuid: uuid(1),
                display_name: Some(r#"{"text":"valid"}"#.into()),
            },
            UpdateDisplayNameEntry {
                uuid: uuid(2),
                display_name: Some("{not json".into()),
            },
        ]);
        assert!(matches!(list.apply(&packet), Err(ClientError::Chat(_))));
        assert!(list.get(&uuid(1)).unwrap().display_name.is_none());
        assert!(list.get(&uuid(2)).unwrap().display_name.is_none());
    }

    #[test]
    fn header_footer_update_is_atomic_when_the_footer_is_malformed() {
        let mut list = PlayerList::new();
        list.set_header_footer(r#"{"text":"old header"}"#, r#"{"text":"old footer"}"#)
            .unwrap();

        assert!(matches!(
            list.set_header_footer(r#"{"text":"new header"}"#, "{not json"),
            Err(ClientError::Chat(_))
        ));
        let lang = mc_assets::Language::new("en_US");
        assert_eq!(
            list.header.as_ref().unwrap().unformatted_text(&lang),
            "old header"
        );
        assert_eq!(
            list.footer.as_ref().unwrap().unformatted_text(&lang),
            "old footer"
        );
    }
}
