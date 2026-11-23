//! `S29PacketSoundEffect` / `S48PacketResourcePackSend` handling.
//!
//! **Dependency note**: `mc-audio`
//! has no dependency on `mc-protocol` — S06 is being built concurrently and its packet enums
//! don't exist yet. Rather than block on it, this module defines plain structs that mirror the
//! *documented, decoded* shape of the two packets (protocol 47 / the Java accessors
//! below), plus the handler functions that turn them into this crate's own types
//! ([`PlayRequest`], a registry reload). Once S06 lands its `S2CPlay` variants, the caller (S14,
//! wiring the network dispatch) converts from the real packet type to
//! [`SoundEffectPacket`]/[`ResourcePackSendPacket`] at the call site — a couple of field copies —
//! rather than this crate depending on `mc-protocol` at all. No API here should need to change
//! when that happens.
//!
// java: S29PacketSoundEffect.java, S48PacketResourcePackSend.java,
// NetHandlerPlayClient.handleSoundEffect / handleResourcePack

use mc_assets::ResourceLocation;
use mc_math::DVec3;

use crate::engine::PlayRequest;
use crate::AudioError;

/// The decoded shape of `S29PacketSoundEffect` (play state, S2C, id `0x29` in protocol 47).
///
/// Wire layout (`PacketBuffer`, big-endian): `String soundName` (capped 256 chars), `i32 posX`,
/// `i32 posY`, `i32 posZ` (each the world coordinate `× 8.0`, truncated to `i32` — **not** the
/// 26/12/26-bit packed `Position` type H5 warns about; this is a separate, simpler fixed-point
/// encoding specific to this packet), `f32 volume`, `u8 pitch` (`(pitch × 63.0)` on the wire).
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEffectPacket {
    /// The sound event name, e.g. `"random.explode"` — note: **not** namespaced on the wire in
    /// 1.8.9 (no `minecraft:` prefix); [`SoundEffectPacket::into_request`] defaults it.
    pub sound_name: String,
    /// World X, already divided back down from the wire's `× 8` fixed point.
    pub x: f64,
    /// World Y.
    pub y: f64,
    /// World Z.
    pub z: f64,
    /// `S29PacketSoundEffect.getVolume()` — a multiplier, not a `[0,1]` fraction.
    pub volume: f32,
    /// `S29PacketSoundEffect.getPitch()` — already divided back down from the wire byte.
    pub pitch: f32,
}

impl SoundEffectPacket {
    /// Builds this from the *raw wire* fields, applying the same fixed-point conversion the
    /// Java getters do (`S29PacketSoundEffect.getX`/`getY`/`getZ`/`getPitch`). Use this once S06
    /// hands over a real decoded packet carrying the raw `i32`/`u8` wire fields; use the struct
    /// literal directly if a caller (e.g. a test, or a future S06 type with `f64`/`f32` fields
    /// already) has already converted them.
    pub fn from_wire_fields(
        sound_name: String,
        pos_x: i32,
        pos_y: i32,
        pos_z: i32,
        volume: f32,
        pitch_byte: u8,
    ) -> Self {
        // java: `(double)((float)posX / 8.0F)` — the intermediate f32 round-trip is
        // load-bearing, not a simplification: matching it keeps this bit-exact with vanilla for
        // every coordinate a real server can send.
        Self {
            sound_name,
            x: (pos_x as f32 / 8.0) as f64,
            y: (pos_y as f32 / 8.0) as f64,
            z: (pos_z as f32 / 8.0) as f64,
            volume,
            pitch: pitch_byte as f32 / 63.0,
        }
    }

    /// Turns this into a [`PlayRequest`], ready for [`crate::AudioEngine::play`].
    ///
    /// java: `NetHandlerPlayClient.handleSoundEffect` → `WorldClient.playSound` → ultimately a
    /// `PositionedSoundRecord(location, volume, pitch, x, y, z)` — always positional (`LINEAR`
    /// attenuation), volume/pitch taken from the packet verbatim (no server-side default here).
    pub fn into_request(self) -> Result<PlayRequest, AudioError> {
        let event =
            ResourceLocation::parse(&self.sound_name).map_err(|source| AudioError::Asset {
                location: self.sound_name.clone(),
                source,
            })?;
        Ok(PlayRequest {
            event,
            volume: self.volume,
            pitch: self.pitch,
            position: Some(DVec3::new(self.x, self.y, self.z)),
            attenuation: crate::AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        })
    }
}

/// The decoded shape of `S48PacketResourcePackSend` (play state, S2C, id `0x48`).
///
/// Wire layout: `String url` (capped 32767 chars), `String hash` (capped 40 chars — a SHA-1 hex
/// digest, though vanilla never validates that beyond the length cap).
///
/// **What this crate does and does not do with this packet.** `mc-audio` does not fetch resource
/// packs — downloading over HTTP, `level://` save-relative lookup, and the
/// `C19PacketResourcePackStatus` accept/decline handshake are connection- and asset-loading
/// concerns (S14/S21, via `mc-assets`). What *is* this crate's job is the consequence for sound
/// once a pack has been applied: vanilla's `SoundHandler` is itself an
/// `IResourceManagerReloadListener` (`SoundHandler.onResourceManagerReload`), so every pack swap
/// — this packet's ultimate effect — re-parses every `sounds.json` the manager can now see.
/// That's [`crate::AudioEngine::reload_sounds`]: once the pack this packet references has been
/// downloaded and pushed onto the caller's `ResourceManager`, call it with the updated manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePackSendPacket {
    /// The pack's download URL, or a `level://<path>` reference to a pack bundled in the save.
    pub url: String,
    /// SHA-1 hex digest of the pack, for the client's accept/decline and cache-hit logic.
    pub hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_fixed_point_round_trips_typical_coordinates() {
        // java: new S29PacketSoundEffect(name, 12.5, 64.0, -30.25, 1.0, 1.0) would encode
        // posX=100, posY=512, posZ=-242 (12.5*8=100, 64*8=512, -30.25*8=-242).
        let p = SoundEffectPacket::from_wire_fields("random.click".into(), 100, 512, -242, 1.0, 63);
        assert_eq!(p.x, 12.5);
        assert_eq!(p.y, 64.0);
        assert_eq!(p.z, -30.25);
        assert!((p.pitch - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pitch_byte_zero_is_silent_pitch_not_normal() {
        let p = SoundEffectPacket::from_wire_fields("e".into(), 0, 0, 0, 1.0, 0_u8);
        assert_eq!(p.pitch, 0.0);
    }

    #[test]
    fn into_request_is_always_positional_with_linear_attenuation() {
        let p = SoundEffectPacket {
            sound_name: "random.explode".into(),
            x: 1.0,
            y: 2.0,
            z: 3.0,
            volume: 4.0,
            pitch: 1.0,
        };
        let request = p.into_request().unwrap();
        assert_eq!(request.position, Some(DVec3::new(1.0, 2.0, 3.0)));
        assert_eq!(request.volume, 4.0);
        assert_eq!(request.event, ResourceLocation::minecraft("random.explode"));
    }

    #[test]
    fn bare_name_defaults_to_minecraft_namespace() {
        // 1.8.9 servers never send a `namespace:` prefix on this packet; confirm the default
        // ResourceLocation::parse applies still resolves it the way the registry expects.
        let p = SoundEffectPacket {
            sound_name: "mob.zombie.say".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            volume: 1.0,
            pitch: 1.0,
        };
        let request = p.into_request().unwrap();
        assert_eq!(request.event.namespace, "minecraft");
    }

    #[test]
    fn empty_sound_name_forms_the_vanilla_defaulted_location() {
        let p = SoundEffectPacket {
            sound_name: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            volume: 1.0,
            pitch: 1.0,
        };
        let request = p
            .into_request()
            .expect("vanilla ResourceLocation accepts empty names");
        assert_eq!(request.event, ResourceLocation::minecraft(""));
    }

    /// `EntityVillager.useRecipe` and `verifySellingItem` choose `mob.villager.yes` or
    /// `mob.villager.no` on the server, then `WorldManager.playSound` sends that choice as S29.
    /// The client must preserve the selected event, positional source, volume and quantized
    /// pitch; it must not synthesize a second sound from `MC|TrSel`/`MC|TrList`.
    #[test]
    fn villager_trade_sounds_keep_server_selected_s29_fields() {
        for (name, pitch_byte) in [("mob.villager.yes", 50_u8), ("mob.villager.no", 76_u8)] {
            let request = SoundEffectPacket::from_wire_fields(
                name.to_owned(),
                4 * 8,
                64 * 8,
                -2 * 8,
                1.0,
                pitch_byte,
            )
            .into_request()
            .expect("vanilla villager trade names are valid sound locations");
            assert_eq!(request.event, ResourceLocation::minecraft(name));
            assert_eq!(request.position, Some(DVec3::new(4.0, 64.0, -2.0)));
            assert_eq!(request.volume, 1.0);
            assert_eq!(request.pitch, pitch_byte as f32 / 63.0);
            assert_eq!(request.attenuation, crate::AttenuationType::Linear);
            assert!(!request.repeat);
            assert_eq!(request.repeat_delay_ticks, 0);
        }
    }
}
