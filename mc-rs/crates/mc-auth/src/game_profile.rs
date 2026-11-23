//! The small `GameProfile` NBT bridge used by skull tile entities.
//!
//! `authlib` owns `GameProfile`, but vanilla's `NBTUtil` is the code that turns the `Owner`
//! compound on a player skull back into one.  Keeping that conversion next to the authlib-shaped
//! data means consumers do not have to independently reimplement the nested property lists.

use mc_nbt::{Nbt, NbtCompound, TagType};
use uuid::Uuid;

/// One `com.mojang.authlib.properties.Property` attached to a [`GameProfile`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameProfileProperty {
    /// The property-map key, for example `"textures"`.
    pub name: String,
    /// The property's opaque value.
    pub value: String,
    /// The optional signature supplied by Mojang's session service.
    pub signature: Option<String>,
}

/// The `GameProfile` data serialized by `NBTUtil.writeGameProfile`.
///
/// Authlib's properties are a multimap: more than one property may have the same name.  A vector
/// retains each occurrence rather than collapsing the data to a map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameProfile {
    /// The profile UUID from the NBT `Id` string, when it was present and valid.
    ///
    /// java: `GameProfile` permits a null UUID.  In particular, the client-side legacy skull
    /// path constructs `new GameProfile(null, name)` before attempting the optional cache lookup;
    /// collapsing that state to `Uuid::nil()` would make a malformed/Name-only owner collide with
    /// a real profile key.
    pub id: Option<Uuid>,
    /// The profile name from the NBT `Name` string.
    pub name: String,
    /// Every `Properties.<name>[]` value, including optional signatures.
    pub properties: Vec<GameProfileProperty>,
}

/// Read a `GameProfile` from vanilla's `NBTUtil` compound format.
///
/// This matches `NBTUtil.readGameProfileFromNBT`: a profile is present when either `Name` or
/// `Id` is a non-empty string.  An absent or malformed UUID becomes Java's nullable UUID rather
/// than rejecting an otherwise useful name/profile. `Properties` is a compound whose keys are
/// property names and whose values are lists of compounds containing `Value` and an optional
/// `Signature`. Malformed list entries are ignored; a missing or non-string `Value` uses Java
/// `NBTTagCompound.getString`'s empty-string default.
#[must_use]
pub fn read_game_profile_from_nbt(nbt: &NbtCompound) -> Option<GameProfile> {
    let name = nbt.get_str("Name").unwrap_or_default();
    // java: `UUID.fromString(s1)` inside a catch-all, so a spelling Java 8 accepts (such as
    // "1-2-3-4-5") keeps its UUID and anything else becomes null.
    let id = nbt.get_str("Id").and_then(java_uuid_from_string);
    if name.is_empty() && !nbt.get_str("Id").is_some_and(|value| !value.is_empty()) {
        return None;
    }
    let mut properties = Vec::new();

    if let Some(by_name) = nbt.get_compound("Properties") {
        for (name, values) in by_name.iter() {
            let Nbt::List(values) = values else {
                continue;
            };
            // `NBTTagCompound.getTagList(name, 10)` returns an empty list when a non-empty
            // list has a different element tag type.  Inspect the retained `tagType` without
            // changing `NbtList`'s semantics.
            if !values.is_empty() && values.tag_type() != TagType::Compound {
                continue;
            }
            for value in values {
                let Some(value) = value.as_compound() else {
                    continue;
                };
                properties.push(GameProfileProperty {
                    name: name.to_owned(),
                    value: value.get_str("Value").unwrap_or_default().to_owned(),
                    signature: value.get_str("Signature").map(ToOwned::to_owned),
                });
            }
        }
    }

    Some(GameProfile {
        id,
        name: name.to_owned(),
        properties,
    })
}

/// Java 8's `UUID.fromString`: the five-component form, as the runtime 1.8.9 ships with
/// parses it.
///
/// Java 8 does not require the conventional 8-4-4-4-12 component widths, nor a 36-character
/// total. It decodes each component as a positive `long`, then assembles the UUID with wrapping
/// 16/48-bit shifts and unmasked ORs, so short and over-wide components are accepted and differ
/// from the Rust `uuid` crate's set of spellings. Both the login packet (`LoginSuccess`) and
/// `NBTUtil.readGameProfileFromNBT` parse through this.
#[must_use]
pub fn java_uuid_from_string(value: &str) -> Option<Uuid> {
    fn digit(value: char) -> Option<u64> {
        let code = value as u32;
        let latin = match code {
            0x0041..=0x0046 => Some(code - 0x0041 + 10),
            0x0061..=0x0066 => Some(code - 0x0061 + 10),
            0xff21..=0xff26 => Some(code - 0xff21 + 10),
            0xff41..=0xff46 => Some(code - 0xff41 + 10),
            _ => None,
        };
        if let Some(digit) = latin {
            return Some(u64::from(digit));
        }

        // `Long.decode` ultimately uses Java 8's `Character.digit(char, 16)`. These are the BMP
        // decimal-zero code points from that exact runtime; supplementary digits are presented
        // to its `char` loop as surrogate halves and are therefore rejected.
        const DECIMAL_ZEROES: &[u32] = &[
            0x0030, 0x0660, 0x06f0, 0x07c0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66,
            0x0ce6, 0x0d66, 0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0,
            0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xaa50,
            0xabf0, 0xff10,
        ];
        DECIMAL_ZEROES
            .iter()
            .find(|&&zero| (zero..zero + 10).contains(&code))
            .map(|&zero| u64::from(code - zero))
    }

    fn component(value: &str) -> Option<u64> {
        if value.is_empty() {
            return None;
        }
        let mut decoded = 0_u64;
        for character in value.chars() {
            decoded = decoded.checked_mul(16)?.checked_add(digit(character)?)?;
        }
        // Java 8 prepends `0x` and calls Long.decode, so the positive spelling cannot exceed
        // Long.MAX_VALUE.
        (decoded <= i64::MAX as u64).then_some(decoded)
    }

    // Java's one-argument `String.split` drops every trailing empty component.
    let mut parts = value.trim_end_matches('-').split('-');
    let first = component(parts.next()?)?;
    let second = component(parts.next()?)?;
    let third = component(parts.next()?)?;
    let fourth = component(parts.next()?)?;
    let fifth = component(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }

    let most = ((first << 16) | second) << 16 | third;
    let least = (fourth << 48) | fifth;
    Some(Uuid::from_u64_pair(most, least))
}

/// Write a [`GameProfile`] in vanilla's `NBTUtil.writeGameProfile` shape.
///
/// Empty names and a missing UUID are deliberately omitted.  Java's `GameProfile` allows both
/// values to be null/empty, and the distinction matters for the legacy `SkullOwner` promotion:
/// a name-only profile must not acquire UUID nil merely because it crossed a packet boundary.
#[must_use]
pub fn write_game_profile_to_nbt(profile: &GameProfile) -> NbtCompound {
    write_game_profile_preserving(&NbtCompound::new(), profile)
}

/// Write known profile fields into an existing owner compound while retaining unknown fields.
///
/// `NBTUtil.writeGameProfile` receives a destination compound and only replaces the fields it
/// knows.  Starting from the existing compound reproduces that useful property for custom owner
/// metadata; the property-map is rebuilt only when the decoded profile carried properties.
#[must_use]
pub fn write_game_profile_preserving(existing: &NbtCompound, profile: &GameProfile) -> NbtCompound {
    let mut owner = existing.clone();
    if profile.name.is_empty() {
        owner.remove("Name");
    } else {
        owner.insert("Name", Nbt::String(profile.name.clone()));
    }
    match profile.id {
        Some(id) => {
            owner.insert("Id", Nbt::String(id.to_string()));
        }
        None => {
            owner.remove("Id");
        }
    }
    if profile.properties.is_empty() {
        owner.remove("Properties");
    } else {
        let mut properties = NbtCompound::new();
        for property in &profile.properties {
            let mut entry = NbtCompound::new();
            entry.insert("Value", Nbt::String(property.value.clone()));
            if let Some(signature) = &property.signature {
                entry.insert("Signature", Nbt::String(signature.clone()));
            }
            let tag = Nbt::Compound(entry);
            if let Some(Nbt::List(values)) = properties.get_mut(&property.name) {
                values.push(tag);
            } else {
                properties.insert(property.name.clone(), Nbt::list(vec![tag]));
            }
        }
        owner.insert("Properties", Nbt::Compound(properties));
    }
    owner
}

/// Promote a legacy string `SkullOwner` into the compound form used by item and tile renderers.
///
/// The optional `resolved` profile is the caller's already-authoritative cache result (for
/// example a tab-list profile).  `None` intentionally keeps a Name-only profile: this helper
/// performs no network lookup and never fabricates a UUID.  A compound owner is left untouched;
/// callers can use [`write_game_profile_preserving`] when they have a refreshed profile.
#[must_use]
pub fn normalize_legacy_skull_owner(
    root: &mut NbtCompound,
    resolved: Option<&GameProfile>,
) -> Option<GameProfile> {
    let name = match root.get("SkullOwner") {
        Some(Nbt::String(name)) if !name.is_empty() => name.clone(),
        Some(Nbt::Compound(owner)) => return read_game_profile_from_nbt(owner),
        _ => return None,
    };
    let profile = resolved.cloned().unwrap_or(GameProfile {
        id: None,
        name,
        properties: Vec::new(),
    });
    let owner = write_game_profile_to_nbt(&profile);
    root.insert("SkullOwner", Nbt::Compound(owner));
    Some(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_nbt::Nbt;

    fn property(value: &str, signature: Option<&str>) -> Nbt {
        let mut property = NbtCompound::new();
        property.insert("Value", Nbt::String(value.to_owned()));
        if let Some(signature) = signature {
            property.insert("Signature", Nbt::String(signature.to_owned()));
        }
        Nbt::Compound(property)
    }

    #[test]
    fn reads_uuid_and_every_signed_or_unsigned_property() {
        let mut properties = NbtCompound::new();
        properties.insert(
            "textures",
            Nbt::list(vec![
                property("skin-one", Some("sig-one")),
                property("skin-two", None),
            ]),
        );
        properties.insert(
            "example",
            Nbt::list(vec![property("value", Some("signature"))]),
        );
        let mut nbt = NbtCompound::new();
        nbt.insert(
            "Id",
            Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
        );
        nbt.insert("Name", Nbt::String("Notch".to_owned()));
        nbt.insert("Properties", Nbt::Compound(properties));

        let profile = read_game_profile_from_nbt(&nbt).expect("valid GameProfile");
        assert_eq!(
            profile.id,
            Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap())
        );
        assert_eq!(profile.name, "Notch");
        assert_eq!(
            profile.properties,
            vec![
                GameProfileProperty {
                    name: "textures".to_owned(),
                    value: "skin-one".to_owned(),
                    signature: Some("sig-one".to_owned()),
                },
                GameProfileProperty {
                    name: "textures".to_owned(),
                    value: "skin-two".to_owned(),
                    signature: None,
                },
                GameProfileProperty {
                    name: "example".to_owned(),
                    value: "value".to_owned(),
                    signature: Some("signature".to_owned()),
                },
            ]
        );
    }

    #[test]
    fn java_accepts_name_only_id_only_and_malformed_id_when_a_field_is_nonempty() {
        let mut name_only = NbtCompound::new();
        name_only.insert("Name", Nbt::String("Notch".to_owned()));
        assert_eq!(
            read_game_profile_from_nbt(&name_only).map(|profile| (profile.name, profile.id)),
            Some(("Notch".to_owned(), None))
        );

        let mut id_only = NbtCompound::new();
        id_only.insert(
            "Id",
            Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
        );
        assert_eq!(
            read_game_profile_from_nbt(&id_only).map(|profile| (profile.name, profile.id)),
            Some((
                String::new(),
                Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap())
            ))
        );

        let mut malformed_id = name_only.clone();
        malformed_id.insert("Id", Nbt::String("definitely-not-a-uuid".to_owned()));
        assert_eq!(
            read_game_profile_from_nbt(&malformed_id).map(|profile| (profile.name, profile.id)),
            Some(("Notch".to_owned(), None))
        );

        let mut empty = NbtCompound::new();
        empty.insert("Name", Nbt::String(String::new()));
        empty.insert("Id", Nbt::String(String::new()));
        assert_eq!(read_game_profile_from_nbt(&empty), None);

        let mut wrong_types = NbtCompound::new();
        wrong_types.insert("Name", Nbt::Int(1));
        wrong_types.insert("Id", Nbt::Int(1));
        assert_eq!(read_game_profile_from_nbt(&wrong_types), None);
    }

    /// java: `NBTUtil.readGameProfileFromNBT` keeps whatever `UUID.fromString` accepts. The
    /// expectations are Java 8's own output (`UUID.fromString(...).toString()` on 1.8.0_482).
    #[test]
    fn a_skull_owner_id_parses_the_way_java_8s_uuid_from_string_does() {
        let profile_id = |id: &str| {
            let mut owner = NbtCompound::new();
            owner.insert("Name", Nbt::String("Head".to_owned()));
            owner.insert("Id", Nbt::String(id.to_owned()));
            read_game_profile_from_nbt(&owner)
                .expect("a name is present")
                .id
                .map(|uuid| uuid.hyphenated().to_string())
        };
        for (id, java) in [
            ("1-2-3-4-5", "00000001-0002-0003-0004-000000000005"),
            (
                "0000000000000001-0000000000000002-3-4-5",
                "00000001-0002-0003-0004-000000000005",
            ),
            ("1-10000-3-4-5", "00000001-0000-0003-0004-000000000005"),
            (
                "1-2-3-4-1000000000000",
                "00000001-0002-0003-0005-000000000000",
            ),
            ("1-2-3-4-5---", "00000001-0002-0003-0004-000000000005"),
            (
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000001",
            ),
        ] {
            assert_eq!(profile_id(id).as_deref(), Some(java), "Id {id:?}");
        }
        // java: NumberFormatException / IllegalArgumentException, caught: a null UUID.
        for id in [
            "+1-2-3-4-5",
            "-1-2-3-4",
            "1-2-3-4-",
            "8000000000000000-2-3-4-5",
            "x",
        ] {
            assert_eq!(profile_id(id), None, "Id {id:?}");
        }
    }

    #[test]
    fn malformed_property_entries_do_not_discard_the_profile() {
        let mut properties = NbtCompound::new();
        properties.insert("not-a-list", Nbt::String("ignored".to_owned()));
        properties.insert(
            "mixed",
            Nbt::list(vec![{
                let mut entry = NbtCompound::new();
                entry.insert("Signature", Nbt::Int(5));
                Nbt::Compound(entry)
            }]),
        );
        properties.insert(
            "wrong-list-type",
            Nbt::list(vec![Nbt::String("ignored".to_owned())]),
        );
        let mut nbt = NbtCompound::new();
        nbt.insert("Id", Nbt::String(Uuid::nil().to_string()));
        nbt.insert("Name", Nbt::String(String::new()));
        nbt.insert("Properties", Nbt::Compound(properties));

        let profile = read_game_profile_from_nbt(&nbt).expect("a valid Id remains enough");
        assert_eq!(profile.properties.len(), 1);
        assert_eq!(profile.properties[0].name, "mixed");
        assert!(profile.properties[0].value.is_empty());
        assert_eq!(profile.properties[0].signature, None);
    }

    #[test]
    fn writer_omits_null_fields_and_retains_all_property_occurrences() {
        let profile = GameProfile {
            id: None,
            name: "NameOnly".to_owned(),
            properties: vec![
                GameProfileProperty {
                    name: "textures".to_owned(),
                    value: "one".to_owned(),
                    signature: None,
                },
                GameProfileProperty {
                    name: "textures".to_owned(),
                    value: "two".to_owned(),
                    signature: Some("sig".to_owned()),
                },
            ],
        };
        let written = write_game_profile_to_nbt(&profile);
        assert_eq!(written.get_str("Name"), Some("NameOnly"));
        assert!(!written.contains_key("Id"));
        let values = written
            .get_compound("Properties")
            .and_then(|properties| properties.get("textures"))
            .and_then(Nbt::as_list)
            .expect("textures list");
        assert_eq!(values.len(), 2);
        assert_eq!(
            values[0].as_compound().unwrap().get_str("Value"),
            Some("one")
        );
        assert_eq!(
            values[1].as_compound().unwrap().get_str("Signature"),
            Some("sig")
        );

        let id_only = GameProfile {
            id: Some(Uuid::nil()),
            name: String::new(),
            properties: Vec::new(),
        };
        let written = write_game_profile_to_nbt(&id_only);
        assert!(!written.contains_key("Name"));
        assert_eq!(
            written.get_str("Id"),
            Some("00000000-0000-0000-0000-000000000000")
        );
    }

    #[test]
    fn legacy_owner_normalization_preserves_custom_owner_fields_and_can_use_cached_profile() {
        let mut root = NbtCompound::new();
        root.insert("CustomRoot", Nbt::Int(7));
        root.insert("SkullOwner", Nbt::String("LegacyName".to_owned()));
        let profile = normalize_legacy_skull_owner(&mut root, None).expect("legacy profile");
        assert_eq!(profile.id, None);
        assert_eq!(profile.name, "LegacyName");
        let owner = root.get_compound("SkullOwner").expect("promoted owner");
        assert_eq!(owner.get_str("Name"), Some("LegacyName"));
        assert!(!owner.contains_key("Id"));
        assert_eq!(root.get_i32("CustomRoot"), Some(7));

        let cached = GameProfile {
            id: Some(Uuid::from_u128(9)),
            name: "CachedName".to_owned(),
            properties: Vec::new(),
        };
        root.insert("SkullOwner", Nbt::String("LegacyName".to_owned()));
        let _ = normalize_legacy_skull_owner(&mut root, Some(&cached));
        let owner = root.get_compound("SkullOwner").expect("cached owner");
        assert_eq!(owner.get_str("Name"), Some("CachedName"));
        assert_eq!(
            owner.get_str("Id"),
            Some("00000000-0000-0000-0000-000000000009")
        );
    }

    #[test]
    fn profile_write_preserves_unknown_owner_fields_while_replacing_known_fields() {
        let mut existing = NbtCompound::new();
        existing.insert("Name", Nbt::String("old".to_owned()));
        existing.insert("Unknown", Nbt::String("keep".to_owned()));
        let profile = GameProfile {
            id: Some(Uuid::from_u128(11)),
            name: "new".to_owned(),
            properties: vec![GameProfileProperty {
                name: "custom".to_owned(),
                value: "v".to_owned(),
                signature: None,
            }],
        };
        let written = write_game_profile_preserving(&existing, &profile);
        assert_eq!(written.get_str("Unknown"), Some("keep"));
        assert_eq!(written.get_str("Name"), Some("new"));
        assert_eq!(
            written.get_str("Id"),
            Some("00000000-0000-0000-0000-00000000000b")
        );
        assert!(written
            .get_compound("Properties")
            .unwrap()
            .contains_key("custom"));
    }
}
