//! `servers.dat`: the saved server list, its NBT shape, and the ping state each entry carries.
//!
//! java: `client/multiplayer/ServerData.java`, `client/multiplayer/ServerList.java`
//!
//! The file is an **uncompressed** NBT compound with one key, `servers`, holding a list of
//! compounds with `name`, `ip`, and optionally `icon` (base64 PNG) and `acceptTextures` (byte).
//!
//! The "uncompressed" is easy to get wrong and the real file proves it: `ServerList` calls
//! `CompressedStreamTools.read(File)` / `safeWrite`, and *those two overloads do not compress* —
//! only `readCompressed`/`writeCompressed` do. `MCP-919-main/jars/servers.dat` starts `0A 00 00`,
//! a bare `TAG_Compound` with an empty name, not a gzip magic. Writing gzip here would produce
//! a file vanilla cannot read.
//!
//! Everything in the file is **untrusted** — it is user-editable and the icon arrives over the
//! network — so nothing here panics on bad input: a malformed entry is skipped, a bad icon is
//! dropped, and a truncated file is an error value.

use std::io::Write;

use mc_nbt::{Nbt, NbtCompound, NbtLimits};

use crate::UiError;

/// The maximum number of servers `servers.dat` may contain.
///
/// **Not a vanilla limit** — vanilla reads the whole list. It exists because `servers.dat` is
/// attacker-influenced in the "someone handed you a modpack" sense and an unbounded list is an
/// unbounded allocation. 4096 is far past any real list.
pub const MAX_SERVERS: usize = 4096;

/// Whether the client accepts a server's resource pack.
// java: ServerData.ServerResourceMode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResourceMode {
    /// Always download.
    Enabled,
    /// Never download.
    Disabled,
    /// Ask each time. Vanilla's default, and what a missing `acceptTextures` key means.
    #[default]
    Prompt,
}

impl ResourceMode {
    /// The translation key for this mode's label on the add-server screen.
    #[must_use]
    pub const fn translation_key(self) -> &'static str {
        match self {
            ResourceMode::Enabled => "addServer.resourcePack.enabled",
            ResourceMode::Disabled => "addServer.resourcePack.disabled",
            ResourceMode::Prompt => "addServer.resourcePack.prompt",
        }
    }

    /// The next mode in vanilla's cycle order (enabled → disabled → prompt → enabled).
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            ResourceMode::Enabled => ResourceMode::Disabled,
            ResourceMode::Disabled => ResourceMode::Prompt,
            ResourceMode::Prompt => ResourceMode::Enabled,
        }
    }
}

/// How far a server entry has got through its status ping.
///
/// Vanilla smears this across three fields (`field_78841_f`, `pingToServer == -2`,
/// `pingToServer < 0`); naming the states makes the row renderer a `match` instead of a
/// four-way boolean puzzle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PingState {
    /// No ping has been started. java: `field_78841_f == false`
    #[default]
    Unqueried,
    /// A ping is in flight. java: `field_78841_f == true && pingToServer == -2`
    Pinging,
    /// The socket has been handed to the status pinger, which has installed its handler but has
    /// not received `S00` yet. Java sets `serverMOTD = "Pinging..."` and `pingToServer = -1`
    /// at this point; the row intentionally displays the no-connection sprite until its reply.
    // java: OldServerPinger.java:57-59
    AwaitingStatus,
    /// The ping came back. java: `pingToServer >= 0`
    Ok,
    /// The ping failed. java: `pingToServer == -1`
    Failed,
}

/// One saved server, plus whatever the last ping learned about it.
// java: ServerData.java:7
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerEntry {
    /// The player-visible name. java: `serverName`
    pub name: String,
    /// `host` or `host:port`. java: `serverIP`
    pub address: String,
    /// The "3/20" population line. java: `populationInfo`
    pub population: String,
    /// The MOTD, with formatting codes. java: `serverMOTD`
    pub motd: String,
    /// Round-trip time in milliseconds, once [`PingState::Ok`].
    pub ping_ms: i64,
    /// The server's protocol version. 47 is 1.8.9. java: `version`
    pub protocol: i32,
    /// The server's human-readable version name. java: `gameVersion`
    pub game_version: String,
    /// The sample player list shown on hover, newline separated. java: `playerList`
    pub player_list: String,
    /// Where the ping has got to.
    pub state: PingState,
    /// Resource-pack policy.
    pub resource_mode: ResourceMode,
    /// The favicon as the raw base64 payload (no `data:` prefix), if the server sent one.
    pub icon_base64: Option<String>,
    /// Whether this entry came from LAN discovery rather than the file.
    pub lan: bool,
}

impl ServerEntry {
    /// A fresh entry with vanilla's defaults.
    #[must_use]
    pub fn new(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
            population: String::new(),
            motd: String::new(),
            ping_ms: 0,
            protocol: crate::screens::multiplayer::CLIENT_PROTOCOL,
            game_version: "1.8.9".to_owned(),
            player_list: String::new(),
            state: PingState::Unqueried,
            resource_mode: ResourceMode::Prompt,
            icon_base64: None,
            lan: false,
        }
    }

    /// Split `address` into a host and a port, defaulting to 25565.
    ///
    /// Vanilla's `ServerAddress.fromString` splits on every colon and treats more than two
    /// pieces as one bare host. An invalid port falls back to 25565.
    // java: client/multiplayer/ServerAddress.java:39
    #[must_use]
    pub fn host_and_port(&self) -> (&str, u16) {
        parse_address(&self.address)
    }

    /// Vanilla's five-bar ping icon row, 0 (best) to 5 (no connection).
    ///
    /// The thresholds are 150 / 300 / 600 / 1000 ms.
    // java: ServerListEntryNormal.java:105
    #[must_use]
    pub fn ping_bars(&self) -> i32 {
        match self.ping_ms {
            p if p < 0 => 5,
            p if p < 150 => 0,
            p if p < 300 => 1,
            p if p < 600 => 2,
            p if p < 1000 => 3,
            _ => 4,
        }
    }

    /// Whether this server's protocol differs from ours, which vanilla renders in dark red.
    #[must_use]
    pub fn version_mismatch(&self) -> bool {
        self.protocol != crate::screens::multiplayer::CLIENT_PROTOCOL
    }

    /// The NBT compound vanilla writes for this entry.
    ///
    /// Only the four persisted keys are written: everything else is ping state that is
    /// re-derived on every visit to the screen.
    // java: ServerData.java:48 (getNBTCompound)
    #[must_use]
    pub fn to_nbt(&self) -> Nbt {
        let mut c = NbtCompound::new();
        c.insert("name", Nbt::String(self.name.clone()));
        c.insert("ip", Nbt::String(self.address.clone()));
        if let Some(icon) = &self.icon_base64 {
            c.insert("icon", Nbt::String(icon.clone()));
        }
        match self.resource_mode {
            ResourceMode::Enabled => {
                c.insert("acceptTextures", Nbt::Byte(1));
            }
            ResourceMode::Disabled => {
                c.insert("acceptTextures", Nbt::Byte(0));
            }
            // java: PROMPT writes nothing at all, which is how it round-trips as "ask me".
            ResourceMode::Prompt => {}
        }
        Nbt::Compound(c)
    }

    /// Read one entry back. A non-compound, or one missing `name`/`ip`, yields `None`.
    // java: ServerData.java:84 (getServerDataFromNBTCompound)
    #[must_use]
    pub fn from_nbt(value: &Nbt) -> Option<Self> {
        let Nbt::Compound(c) = value else {
            return None;
        };
        // java: NBTTagCompound.getString returns "" for a missing key, so vanilla happily
        // creates a nameless, address-less entry. Reproduced: dropping it would silently
        // renumber the list and break the "edit entry 3" path.
        let name = match c.get("name") {
            Some(Nbt::String(s)) => s.clone(),
            _ => String::new(),
        };
        let address = match c.get("ip") {
            Some(Nbt::String(s)) => s.clone(),
            _ => String::new(),
        };
        let mut e = ServerEntry::new(name, address);
        if let Some(Nbt::String(icon)) = c.get("icon") {
            e.icon_base64 = Some(icon.clone());
        }
        // java: hasKey("acceptTextures", 1) — id 1 is TAG_Byte, so a boolean written as any
        // other type is ignored and the mode stays PROMPT.
        e.resource_mode = match c.get("acceptTextures") {
            Some(Nbt::Byte(0)) => ResourceMode::Disabled,
            Some(Nbt::Byte(_)) => ResourceMode::Enabled,
            _ => ResourceMode::Prompt,
        };
        Some(e)
    }
}

/// Split `host[:port]`, defaulting the port to 25565.
///
/// Bracketed IPv6 literals are handled first; unbracketed input is split on every colon.
#[must_use]
pub fn parse_address(address: &str) -> (&str, u16) {
    if let Some(rest) = address.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            let host = &rest[..end];
            let port = rest[end + 1..]
                .trim()
                .strip_prefix(':')
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_PORT);
            return (host, port);
        }
    }
    let mut parts: Vec<&str> = address.split(':').collect();
    // java: String.split drops trailing empty strings.
    while parts.len() > 1 && parts.last() == Some(&"") {
        parts.pop();
    }
    match parts.as_slice() {
        [host, port] => (host, port.trim().parse().unwrap_or(DEFAULT_PORT)),
        [host] => (host, DEFAULT_PORT),
        _ => (address, DEFAULT_PORT),
    }
}

/// Convert a parsed or SRV-selected host like Java 8's default `java.net.IDN.toASCII`.
///
/// Minecraft calls `IDN.toASCII` *after* `ServerAddress.fromString` has optionally replaced the
/// host with an SRV target. Java's no-flags overload uses the IDNA 2003-compatible transitional
/// mappings, does not enable STD3 ASCII restrictions, preserves a final root-label dot, and
/// rejects empty/interior or over-63-byte labels. The deprecated `idna::Config` API is intentional
/// here: its transitional switch is the compatibility behavior this old protocol needs.
///
/// # Errors
///
/// Returns [`ServerAddressIdnError`] when Java would throw `IllegalArgumentException`.
// java: client/multiplayer/ServerAddress.java:20-23
#[allow(deprecated)]
pub fn java_idn_to_ascii(host: &str) -> Result<String, ServerAddressIdnError> {
    // Java treats the empty whole name and the root name as special successes. UTS 46's generic
    // domain processor rejects them as empty labels, so preserve the Java entry-point behavior.
    if host.is_empty() || host == "." {
        return Ok(host.to_owned());
    }

    // `java.net.IDN` splits the domain first and only runs Nameprep on labels that contain a
    // non-ASCII code unit (`IDN.toASCIIInternal`, step 1). That means an ASCII-only label keeps
    // its original case (`FOO.Bar` stays `FOO.Bar`), while UTS 46's whole-domain helper lowercases
    // every label. Besides being observable in the C00Handshake bytes, this matters for proxies
    // that route on the literal host text, so process one Java label at a time here.
    let mut ascii = String::new();
    let mut start = 0;
    for (index, character) in host.char_indices() {
        if matches!(character, '.' | '\u{3002}' | '\u{ff0e}' | '\u{ff61}') {
            append_java_idn_label(&host[start..index], &mut ascii)?;
            ascii.push('.');
            start = index + character.len_utf8();
        }
    }
    // A single trailing separator denotes the root label and is emitted without asking
    // `toASCIIInternal` to process an empty label. A second separator still reaches the empty-label
    // check in the loop above and fails.
    if start < host.len() {
        append_java_idn_label(&host[start..], &mut ascii)?;
    }
    Ok(ascii)
}

#[allow(deprecated)]
fn append_java_idn_label(label: &str, output: &mut String) -> Result<(), ServerAddressIdnError> {
    // Java's root-label special case is handled by the caller; every empty label reached here is
    // an interior/trailing separator and must fail exactly like IDN.toASCIIInternal.
    if label.is_empty() {
        return Err(ServerAddressIdnError);
    }
    let ascii = if label.is_ascii() {
        // Java skips Nameprep for an all-ASCII label, including its case and non-LDH characters
        // (the no-flags overload does not enable USE_STD3_ASCII_RULES).
        label.to_owned()
    } else {
        idna::Config::default()
            .use_std3_ascii_rules(false)
            .check_hyphens(false)
            .transitional_processing(true)
            .verify_dns_length(false)
            .to_ascii(label)
            .map_err(|_| ServerAddressIdnError)?
    };

    // `java.net.IDN` validates each DNS label but, unlike UTS 46 VerifyDnsLength, does not reject
    // a long *whole* name. This distinction is observable for four legal 63-byte labels.
    if ascii.len() > 63 {
        return Err(ServerAddressIdnError);
    }
    output.push_str(&ascii);
    Ok(())
}

/// Java `IDN.toASCII` rejected a saved server host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerAddressIdnError;

impl std::fmt::Display for ServerAddressIdnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("server host is not valid under Java IDN.toASCII rules")
    }
}

impl std::error::Error for ServerAddressIdnError {}

/// Minecraft's default server port.
pub const DEFAULT_PORT: u16 = 25565;

/// Vanilla's fallback text when a LAN beacon has no valid MOTD section.
pub const MISSING_LAN_MOTD: &str = "missing no";

/// Extract the display MOTD from a 1.8.9 LAN beacon.
// java: ThreadLanServerPing.getMotdFromPingResponse
#[must_use]
pub fn lan_motd(response: &str) -> &str {
    let Some(start) = response.find("[MOTD]") else {
        return MISSING_LAN_MOTD;
    };
    let value_start = start + "[MOTD]".len();
    response[value_start..]
        .find("[/MOTD]")
        .map(|end| &response[value_start..value_start + end])
        .unwrap_or(MISSING_LAN_MOTD)
}

/// Extract the advertised port from a 1.8.9 LAN beacon.
///
/// The second-`[/MOTD]` rejection looks strange, but is intentional vanilla behaviour.
// java: ThreadLanServerPing.getAdFromPingResponse
#[must_use]
pub fn lan_ad(response: &str) -> Option<&str> {
    let motd_end = response.find("[/MOTD]")?;
    let after_motd = motd_end + "[/MOTD]".len();
    if response[after_motd..].contains("[/MOTD]") {
        return None;
    }
    let ad_start = response[after_motd..].find("[AD]")? + after_motd + "[AD]".len();
    let ad_end = response[ad_start..].find("[/AD]")? + ad_start;
    Some(&response[ad_start..ad_end])
}

/// Convert one received LAN beacon into vanilla's display MOTD and address key.
///
/// `LanServerList.func_77551_a` concatenates the datagram sender's host address, a colon, and
/// the advertisement text without validating or normalising the port. The resulting string is
/// also the exact duplicate key. A missing/invalid `[AD]` section is the only rejected case.
// java: LanServerDetector.LanServerList.func_77551_a
#[must_use]
pub fn lan_announcement(response: &str, sender_host: &str) -> Option<(String, String)> {
    let ad = lan_ad(response)?;
    Some((lan_motd(response).to_owned(), format!("{sender_host}:{ad}")))
}

/// The whole saved list.
// java: ServerList.java:13
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerList {
    servers: Vec<ServerEntry>,
}

impl ServerList {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many servers are saved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.servers.len()
    }

    /// Whether nothing is saved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// The entries, in list order.
    #[must_use]
    pub fn entries(&self) -> &[ServerEntry] {
        &self.servers
    }

    /// The entries, mutably — the ping worker writes results back through this.
    pub fn entries_mut(&mut self) -> &mut [ServerEntry] {
        &mut self.servers
    }

    /// One entry by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&ServerEntry> {
        self.servers.get(index)
    }

    /// One entry by index, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ServerEntry> {
        self.servers.get_mut(index)
    }

    /// Append an entry. java: `addServerData`
    pub fn add(&mut self, entry: ServerEntry) {
        self.servers.push(entry);
    }

    /// Remove an entry, returning it. Out-of-range indices are ignored, where vanilla throws.
    // java: ServerList.java:92 (removeServerData)
    pub fn remove(&mut self, index: usize) -> Option<ServerEntry> {
        (index < self.servers.len()).then(|| self.servers.remove(index))
    }

    /// Replace an entry in place. java: `func_147413_a`
    pub fn set(&mut self, index: usize, entry: ServerEntry) -> bool {
        match self.servers.get_mut(index) {
            Some(slot) => {
                *slot = entry;
                true
            }
            None => false,
        }
    }

    /// Swap two entries. Vanilla's up/down arrows are two of these.
    // java: ServerList.java:116 (swapServers)
    pub fn swap(&mut self, a: usize, b: usize) -> bool {
        if a < self.servers.len() && b < self.servers.len() {
            self.servers.swap(a, b);
            true
        } else {
            false
        }
    }

    /// Parse a `servers.dat` payload.
    ///
    /// Vanilla reads this file as raw NBT.  In particular, it does not sniff gzip or zlib
    /// wrappers: a third-party launcher that writes either framing leaves vanilla unable to
    /// load the saved list.
    ///
    /// # Errors
    ///
    /// [`UiError::ServerList`] if the bytes are not readable NBT, or the root is not a compound.
    /// A *missing* `servers` list is not an error — vanilla treats it as an empty list, and a
    /// fresh install has exactly that.
    pub fn read(bytes: &[u8]) -> Result<Self, UiError> {
        // Local file, so no protocol size limits: vanilla passes `NBTSizeTracker.INFINITE`.
        let (_, root) = mc_nbt::from_slice_with_limits(bytes, NbtLimits::UNLIMITED)
            .map_err(|e| UiError::ServerList(e.to_string()))?;
        Self::from_nbt(&root)
    }

    /// Parse an already-decompressed NBT root.
    ///
    /// # Errors
    ///
    /// [`UiError::ServerList`] if the root is not a compound.
    pub fn from_nbt(root: &Nbt) -> Result<Self, UiError> {
        let Nbt::Compound(c) = root else {
            return Err(UiError::ServerList("root tag is not a compound".to_owned()));
        };
        let mut list = ServerList::new();
        // java: getTagList("servers", 10) returns an *empty* list when the key is missing or
        // holds the wrong type, so neither is an error here either.
        if let Some(Nbt::List(items)) = c.get("servers") {
            for item in items.iter().take(MAX_SERVERS) {
                if let Some(entry) = ServerEntry::from_nbt(item) {
                    list.add(entry);
                }
            }
        }
        Ok(list)
    }

    /// The NBT root vanilla writes.
    // java: ServerList.java:60 (saveServerList)
    #[must_use]
    pub fn to_nbt(&self) -> Nbt {
        let mut root = NbtCompound::new();
        root.insert(
            "servers",
            Nbt::list(self.servers.iter().map(ServerEntry::to_nbt).collect()),
        );
        Nbt::Compound(root)
    }

    /// Write a `servers.dat` payload: uncompressed NBT with an empty root name.
    ///
    /// # Errors
    ///
    /// [`UiError::ServerList`] on any write failure.
    pub fn write(&self, w: &mut impl Write) -> Result<(), UiError> {
        mc_nbt::write(w, "", &self.to_nbt()).map_err(|e| UiError::ServerList(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ServerList {
        let mut list = ServerList::new();
        let mut a = ServerEntry::new("Example", "mc.example.net");
        a.resource_mode = ResourceMode::Enabled;
        a.icon_base64 = Some("aGVsbG8=".to_owned());
        list.add(a);
        list.add(ServerEntry::new("Local", "127.0.0.1:25566"));
        let mut c = ServerEntry::new("NoPacks", "example.com");
        c.resource_mode = ResourceMode::Disabled;
        list.add(c);
        list
    }

    #[test]
    fn a_server_list_round_trips_through_nbt() {
        let list = sample();
        let back = ServerList::from_nbt(&list.to_nbt()).expect("parse");
        assert_eq!(back.len(), 3);
        assert_eq!(back.get(0).unwrap().name, "Example");
        assert_eq!(back.get(0).unwrap().resource_mode, ResourceMode::Enabled);
        assert_eq!(
            back.get(0).unwrap().icon_base64.as_deref(),
            Some("aGVsbG8=")
        );
        assert_eq!(back.get(1).unwrap().resource_mode, ResourceMode::Prompt);
        assert_eq!(back.get(2).unwrap().resource_mode, ResourceMode::Disabled);
        assert_eq!(back, list);
    }

    #[test]
    fn a_server_list_round_trips_through_the_on_disk_format() {
        let list = sample();
        let mut buf = Vec::new();
        list.write(&mut buf).expect("write");
        // The file vanilla writes is *not* compressed: a bare TAG_Compound with an empty name.
        assert_eq!(&buf[..3], &[0x0A, 0x00, 0x00], "root tag, empty name");
        let back = ServerList::read(&buf).expect("read");
        assert_eq!(back, list);
    }

    #[test]
    fn compressed_server_lists_are_rejected_like_vanillas_raw_reader() {
        let list = sample();
        let mut gzip = Vec::new();
        mc_nbt::write_gzip(&mut gzip, "", &list.to_nbt()).expect("gzip");
        assert_eq!(&gzip[..2], &[0x1F, 0x8B]);
        assert!(ServerList::read(&gzip).is_err(), "gzip is not raw NBT");

        let mut zlib = Vec::new();
        mc_nbt::write_zlib(&mut zlib, "", &list.to_nbt()).expect("zlib");
        assert_eq!(zlib[0] & 0x0F, 8, "a zlib header starts with deflate");
        assert!(ServerList::read(&zlib).is_err(), "zlib is not raw NBT");
    }

    #[test]
    fn prompt_mode_writes_no_accept_textures_key_at_all() {
        // This is the one that makes "ask me every time" survive a save: vanilla distinguishes
        // PROMPT from ENABLED/DISABLED by the *absence* of the key, not by a third value.
        let entry = ServerEntry::new("x", "y");
        let Nbt::Compound(c) = entry.to_nbt() else {
            panic!("not a compound")
        };
        assert!(c.get("acceptTextures").is_none());
        assert!(c.get("icon").is_none(), "no icon key when there is no icon");
        assert_eq!(c.get("name"), Some(&Nbt::String("x".into())));
    }

    #[test]
    fn a_non_byte_accept_textures_falls_back_to_prompt() {
        // java: hasKey("acceptTextures", 1) type-checks for TAG_Byte.
        let mut c = NbtCompound::new();
        c.insert("name", Nbt::String("n".into()));
        c.insert("ip", Nbt::String("i".into()));
        c.insert("acceptTextures", Nbt::Int(1));
        let e = ServerEntry::from_nbt(&Nbt::Compound(c)).expect("entry");
        assert_eq!(e.resource_mode, ResourceMode::Prompt);
    }

    #[test]
    fn a_missing_servers_key_is_an_empty_list_not_an_error() {
        let root = Nbt::Compound(NbtCompound::new());
        assert!(ServerList::from_nbt(&root).expect("ok").is_empty());
    }

    #[test]
    fn a_non_compound_root_is_an_error_not_a_panic() {
        assert!(ServerList::from_nbt(&Nbt::Int(1)).is_err());
    }

    #[test]
    fn garbage_bytes_are_an_error_not_a_panic() {
        assert!(ServerList::read(&[0xDE, 0xAD, 0xBE, 0xEF]).is_err());
        assert!(ServerList::read(&[]).is_err());
        // A truncated but well-started file: TAG_Compound, empty name, then nothing.
        assert!(ServerList::read(&[0x0A, 0x00, 0x00, 0x09]).is_err());
    }

    #[test]
    fn a_missing_name_reads_as_the_empty_string_like_vanilla() {
        let mut c = NbtCompound::new();
        c.insert("ip", Nbt::String("host".into()));
        let e = ServerEntry::from_nbt(&Nbt::Compound(c)).expect("entry");
        assert_eq!(e.name, "");
        assert_eq!(e.address, "host");
    }

    #[test]
    fn addresses_follow_vanillas_all_colons_then_fallback_rule() {
        assert_eq!(parse_address("example.com"), ("example.com", 25565));
        assert_eq!(parse_address("example.com:25566"), ("example.com", 25566));
        assert_eq!(parse_address("  example.com:1  "), ("  example.com", 1));
        // Vanilla's own fallback: a non-numeric port is the default, not an error.
        assert_eq!(parse_address("example.com:abc"), ("example.com", 25565));
        assert_eq!(parse_address("[::1]:25566"), ("::1", 25566));
        assert_eq!(parse_address("[::1]"), ("::1", 25565));
        assert_eq!(parse_address("::1"), ("::1", 25565));
        assert_eq!(parse_address("a:b:c"), ("a:b:c", 25565));
        assert_eq!(parse_address("host:"), ("host", 25565));
        // 65536 does not fit a u16, so it falls back rather than wrapping to 0.
        assert_eq!(parse_address("h:65536"), ("h", 25565));
    }

    #[test]
    fn server_hosts_use_java_8_idn_transitional_and_label_rules() {
        // Oracle values from java.net.IDN.toASCII(String), the no-flags overload used by 1.8.9.
        for (unicode, ascii) in [
            ("b\u{fc}cher.example", "xn--bcher-kva.example"),
            ("fa\u{df}.de", "fass.de"),
            ("ma\u{f1}ana.com.", "xn--maana-pta.com."),
            // Java skips Nameprep for ASCII-only labels, so case is retained per label even when
            // a neighbouring label needs punycode or a Unicode dot is normalized.
            ("B\u{fc}CHER.Example", "xn--bcher-kva.Example"),
            ("FOO\u{3002}Bar", "FOO.Bar"),
            ("FOO\u{ff0e}Bar", "FOO.Bar"),
            ("FOO\u{ff61}Bar", "FOO.Bar"),
            (
                "\u{4f8b}\u{3048}\u{3002}\u{30c6}\u{30b9}\u{30c8}",
                "xn--r8jz45g.xn--zckzah",
            ),
            ("foo_bar.example", "foo_bar.example"),
            ("-foo.example", "-foo.example"),
            ("::1", "::1"),
            ("", ""),
            (".", "."),
        ] {
            assert_eq!(
                java_idn_to_ascii(unicode).as_deref(),
                Ok(ascii),
                "{unicode:?}"
            );
        }

        // A separator followed by another separator reaches Java's empty-label check. The root
        // name is the only whole-string separator special case.
        assert_eq!(java_idn_to_ascii("FOO\u{3002}").as_deref(), Ok("FOO."));
        assert_eq!(java_idn_to_ascii("FOO..Bar"), Err(ServerAddressIdnError));
        assert_eq!(java_idn_to_ascii("a..b"), Err(ServerAddressIdnError));
        assert_eq!(
            java_idn_to_ascii(&format!("{}.com", "a".repeat(64))),
            Err(ServerAddressIdnError)
        );

        let four_max_labels = format!("{}.", vec!["a".repeat(63); 4].join("."));
        assert_eq!(
            java_idn_to_ascii(&four_max_labels).as_deref(),
            Ok(four_max_labels.as_str()),
            "Java validates labels individually rather than imposing UTS 46's 253-byte total"
        );
    }

    #[test]
    fn lan_beacon_parsers_reproduce_vanillas_odd_delimiter_rules() {
        let valid = "[MOTD]A local world[/MOTD][AD]25566[/AD]";
        assert_eq!(lan_motd(valid), "A local world");
        assert_eq!(lan_ad(valid), Some("25566"));
        assert_eq!(
            lan_announcement(valid, "192.0.2.7"),
            Some(("A local world".to_owned(), "192.0.2.7:25566".to_owned()))
        );
        assert_eq!(
            lan_announcement("[MOTD]x[/MOTD][AD]not-a-port[/AD]", "2001:db8::1"),
            Some(("x".to_owned(), "2001:db8::1:not-a-port".to_owned())),
            "vanilla concatenates the raw AD text instead of normalising it"
        );

        assert_eq!(lan_motd("garbage"), MISSING_LAN_MOTD);
        assert_eq!(lan_motd("[MOTD]unfinished"), MISSING_LAN_MOTD);
        assert_eq!(lan_ad("[MOTD]x[/MOTD]"), None);
        assert_eq!(lan_ad("[MOTD]x[/MOTD][AD]unfinished"), None);
        assert_eq!(lan_announcement("[MOTD]x[/MOTD]", "192.0.2.7"), None);
        assert_eq!(
            lan_ad("[MOTD]x[/MOTD][/MOTD][AD]25565[/AD]"),
            None,
            "a second MOTD close invalidates an otherwise usable advertisement"
        );
    }

    #[test]
    fn ping_bars_use_vanillas_four_thresholds() {
        let mut e = ServerEntry::new("n", "a");
        for (ms, bars) in [
            (0, 0),
            (149, 0),
            (150, 1),
            (299, 1),
            (300, 2),
            (599, 2),
            (600, 3),
            (999, 3),
            (1000, 4),
            (10_000, 4),
            (-1, 5),
        ] {
            e.ping_ms = ms;
            assert_eq!(e.ping_bars(), bars, "{ms} ms");
        }
    }

    #[test]
    fn list_mutation_is_bounds_checked_instead_of_panicking() {
        let mut list = sample();
        assert!(list.remove(99).is_none());
        assert!(!list.swap(0, 99));
        assert!(!list.set(99, ServerEntry::new("a", "b")));
        assert!(list.swap(0, 2));
        assert_eq!(list.get(0).unwrap().name, "NoPacks");
        assert_eq!(list.remove(0).unwrap().name, "NoPacks");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn the_entry_count_is_capped() {
        let items: Vec<Nbt> = (0..MAX_SERVERS + 10)
            .map(|i| {
                let mut c = NbtCompound::new();
                c.insert("name", Nbt::String(format!("s{i}")));
                c.insert("ip", Nbt::String("h".into()));
                Nbt::Compound(c)
            })
            .collect();
        let mut root = NbtCompound::new();
        root.insert("servers", Nbt::list(items));
        let list = ServerList::from_nbt(&Nbt::Compound(root)).expect("ok");
        assert_eq!(list.len(), MAX_SERVERS);
    }

    #[test]
    fn resource_mode_cycles_the_way_the_add_server_button_does() {
        assert_eq!(ResourceMode::Enabled.next(), ResourceMode::Disabled);
        assert_eq!(ResourceMode::Disabled.next(), ResourceMode::Prompt);
        assert_eq!(ResourceMode::Prompt.next(), ResourceMode::Enabled);
    }
}
