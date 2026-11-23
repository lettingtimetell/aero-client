//! Minecraft's server-join hash — SHA-1 rendered as *signed* two's-complement hex (hazard H3).
//!
//! Vanilla computes this in `CryptManager.getServerIdHash` and then formats it with
//! `new BigInteger(digest).toString(16)`. `BigInteger(byte[])` interprets the 20 digest bytes as
//! a **signed big-endian two's-complement integer**, so:
//!
//! * a digest whose top bit is set prints as `-` followed by the hex of its negation, and
//! * leading zeros are stripped, so the string is often shorter than 40 characters.
//!
//! Nothing else in the ecosystem formats hashes this way, and getting it wrong fails only at
//! join time against online-mode servers, which is exactly when it is expensive to debug.
//!
//! // java: CryptManager.java (getServerIdHash / digestOperation)

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use base64::Engine;
use rsa::pkcs8::DecodePublicKey;
use rsa::{Pkcs1v15Sign, RsaPublicKey};
use sha1::{Digest, Sha1};

/// authlib 1.5.21's `/yggdrasil_session_pubkey.der`, copied byte-for-byte from the bundled
/// authlib JAR.  The source asset is base64 only so it remains reviewable in the text tree; the
/// decoded value is the original 550-byte X.509 SubjectPublicKeyInfo DER.
const YGGDRASIL_SESSION_PUBKEY_DER_B64: &str = include_str!("yggdrasil_session_pubkey.der.b64");

/// Verdicts retained before the memo is cleared. A profile generation asks once, so this is far
/// more rows than a server's tab list; the bound only stops a hostile server growing it forever.
const MAX_REMEMBERED_VERDICTS: usize = 4096;

/// The longest value and signature remembered. Real pairs are a kilobyte or two, but one network
/// string can be about 128 KiB and this memo outlives the connection.
const MAX_REMEMBERED_PAIR_BYTES: usize = 8 * 1024;

/// The bundled session key, parsed once. `None` only if the bundled asset is malformed.
fn session_public_key() -> Option<&'static RsaPublicKey> {
    static KEY: OnceLock<Option<RsaPublicKey>> = OnceLock::new();
    KEY.get_or_init(|| {
        let der = base64::engine::general_purpose::STANDARD
            .decode(YGGDRASIL_SESSION_PUBKEY_DER_B64.trim())
            .ok()?;
        RsaPublicKey::from_public_key_der(&der).ok()
    })
    .as_ref()
}

/// Verdicts keyed on the exact value and signature, so neither can be reused for the other's
/// partner. Separate from [`check_textures_signature`] because one RSA check costs hundreds of
/// microseconds, and repeated rows and a background pre-check should not pay for it again.
fn remembered_verdicts() -> &'static Mutex<HashMap<(String, String), bool>> {
    static VERDICTS: OnceLock<Mutex<HashMap<(String, String), bool>>> = OnceLock::new();
    VERDICTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Verify an authlib `textures` property signature.
///
/// `YggdrasilMinecraftSessionService.getTextures(profile, true)` calls
/// `Property.isSignatureValid(publicKey)`, which uses Java's `SHA1withRSA` over
/// `value.getBytes()`. Texture-property values are base64 protocol text (ASCII), so their Java
/// bytes are exactly Rust's `value.as_bytes()` on every launcher charset.
///
/// Malformed base64, a malformed bundled key, and a failed RSA check are all insecure rather
/// than protocol errors: authlib returns `false` for an invalid property and the caller gates
/// the entire textures map before parsing it.
///
/// Safe to call from any thread. Results are remembered for this exact value and signature, so a
/// worker can verify a tab-list update ahead of the game thread needing the answer.
#[must_use]
pub fn verify_textures_signature(value: &str, signature: &str) -> bool {
    if value.len() + signature.len() > MAX_REMEMBERED_PAIR_BYTES {
        return check_textures_signature(value, signature);
    }
    let key = (value.to_owned(), signature.to_owned());
    if let Some(&verdict) = remembered_verdicts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
    {
        return verdict;
    }
    // Outside the lock: two threads racing here both reach the same answer.
    let verdict = check_textures_signature(value, signature);
    let mut verdicts = remembered_verdicts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if verdicts.len() >= MAX_REMEMBERED_VERDICTS {
        verdicts.clear();
    }
    verdicts.insert(key, verdict);
    verdict
}

/// The remembered [`verify_textures_signature`] verdict for this exact pair, without ever running
/// the RSA check. `None` means nobody has verified the pair yet (or the memo was cleared).
///
/// For the game thread: it asks this first and hands a miss to a background verifier instead of
/// paying hundreds of microseconds per row inside a frame (HITCH-1).
#[must_use]
pub fn remembered_textures_verdict(value: &str, signature: &str) -> Option<bool> {
    if value.len() + signature.len() > MAX_REMEMBERED_PAIR_BYTES {
        return None;
    }
    remembered_verdicts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(value.to_owned(), signature.to_owned()))
        .copied()
}

fn check_textures_signature(value: &str, signature: &str) -> bool {
    let Some(key) = session_public_key() else {
        return false;
    };
    let Ok(signature) = base64::engine::general_purpose::STANDARD.decode(signature) else {
        return false;
    };
    let digest = Sha1::digest(value.as_bytes());
    key.verify(Pkcs1v15Sign::new::<Sha1>(), &digest, &signature)
        .is_ok()
}

/// Compute the `serverId` string sent to `sessionserver.mojang.com` and to the server itself.
///
/// The SHA-1 is fed, in this exact order:
///
/// 1. the ISO-8859-1 bytes of the server ID string from `S01PacketEncryptionRequest`,
/// 2. the 16-byte AES shared secret the client generated,
/// 3. the server's DER-encoded X.509 public key, byte for byte as it arrived on the wire.
///
/// The digest is then rendered with [`signed_hex`].
///
/// S07 calls this between generating the AES key and sending `C01PacketEncryptionResponse`.
///
/// ```
/// // The three public test vectors are just SHA-1 of a string, so they can be reproduced by
/// // putting the whole input in the server-ID slot.
/// assert_eq!(
///     mc_auth::server_hash("Notch", &[], &[]),
///     "4ed1f46bbe04bc756bcb17c0c7ce3e4632f06a48"
/// );
/// assert_eq!(
///     mc_auth::server_hash("jeb_", &[], &[]),
///     "-7c9d5b0044c130109a5d7b5fb5c317c02b4e28c1"
/// );
/// ```
#[must_use]
pub fn server_hash(server_id: &str, shared_secret: &[u8], pubkey_der: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(latin1_bytes(server_id));
    hasher.update(shared_secret);
    hasher.update(pubkey_der);
    let digest: [u8; 20] = hasher.finalize().into();
    signed_hex(&digest)
}

/// Encode a Rust string like Java's `String.getBytes("ISO_8859_1")`.
///
/// Java replaces each Unicode code point outside Latin-1 with `?`; truncating to the low byte
/// would therefore be wrong for values above U+00FF. Rust strings cannot contain Java's lone
/// UTF-16 surrogates, but supplementary code points are represented by one replacement byte on
/// both sides.
fn latin1_bytes(value: &str) -> Vec<u8> {
    value
        .chars()
        .map(|character| u8::try_from(character as u32).unwrap_or(b'?'))
        .collect()
}

/// Render a big-endian byte string as Java's `new BigInteger(bytes).toString(16)` would.
///
/// Exposed because S07's tests and any future parity harness want to check the formatting
/// independently of the hashing.
#[must_use]
pub fn signed_hex(digest: &[u8]) -> String {
    if digest.is_empty() {
        return "0".to_owned();
    }

    let negative = digest[0] & 0x80 != 0;
    let magnitude: Vec<u8> = if negative {
        twos_complement_negate(digest)
    } else {
        digest.to_vec()
    };

    let mut out = String::with_capacity(digest.len() * 2 + 1);
    if negative {
        out.push('-');
    }

    // Hex of the magnitude with leading zeros stripped. Java prints "0" for a zero value.
    let mut started = false;
    for byte in magnitude {
        if started {
            out.push_str(&format!("{byte:02x}"));
        } else if byte != 0 {
            out.push_str(&format!("{byte:x}"));
            started = true;
        }
    }
    if !started {
        out.push('0');
    }
    out
}

/// Negate a big-endian two's-complement integer: invert every bit, then add one.
///
/// The 20-byte SHA-1 case never overflows (the only value that would is `-2^159`, i.e. `0x80`
/// followed by zeros, whose negation is itself and which SHA-1 will not produce in practice);
/// the wrapping add below is therefore exact for every reachable input.
fn twos_complement_negate(bytes: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = bytes.iter().map(|b| !b).collect();
    for byte in out.iter_mut().rev() {
        let (next, carry) = byte.overflowing_add(1);
        *byte = next;
        if !carry {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured from the official `sessionserver.mojang.com` response for Notch's UUID with
    // `?unsigned=false`.  It is deliberately a real authlib signature, not one made by a test
    // private key, so this pins both the bundled DER resource and SHA1withRSA's digest prefix.
    const NOTCH_TEXTURE_VALUE: &str = "ewogICJ0aW1lc3RhbXAiIDogMTc4NzY5MTA3NTE3MiwKICAicHJvZmlsZUlkIiA6ICIwNjlhNzlmNDQ0ZTk0NzI2YTViZWZjYTkwZTM4YWFmNSIsCiAgInByb2ZpbGVOYW1lIiA6ICJOb3RjaCIsCiAgInNpZ25hdHVyZVJlcXVpcmVkIiA6IHRydWUsCiAgInRleHR1cmVzIiA6IHsKICAgICJTS0lOIiA6IHsKICAgICAgInVybCIgOiAiaHR0cDovL3RleHR1cmVzLm1pbmVjcmFmdC5uZXQvdGV4dHVyZS8yOTIwMDlhNDkyNWI1OGYwMmM3N2RhZGMzZWNlZjA3ZWE0Yzc0NzJmNjRlMGZkYzMyY2U1NTIyNDg5MzYyNjgwIgogICAgfQogIH0KfQ==";
    const NOTCH_TEXTURE_SIGNATURE: &str = "oqgTiH1ZWCjCcU9a5bOJpskVzloygYgbdNZ16Xtn1RgKrRsPLutWL9Uok/rkg4re1O+TbsKL0lKYzQtZH0+OfWdfWIoWE4bBanHuhejGWu69EWVvcxlpEiNhZptzER9KfPGP5Q7SPsT+4MaSj/AbJP+yKWNjriYZZFV00bWSW/aexY6PJSoXMD/R57gn0TucdaZp1nrbfgL9DXf+6R6T8ou2Lz4qsPAoB2CJI2I8F9HPsLJ/dUinrp8gmjfKnvvJUf+a6XeG+BESWqqChuiD+XJjvnZSyM3Fs1zdb+RkqvFNqT4W14ohyh3mxyX3Q2S/8S+Q2axT47nnjvy9WMVpaDay1o8pquzy+zRqPps1tRFTScrp4w4jVdSzm8xL4FhNqm2H/QUoAYczVUyAfBx4Q8w9daelERlNOcQAIEeauBzan/728P/F4Rxc24Wl9iV5Qb+XeE/858QHNplwLIcSv4Wfp6G29++0XVljN7K2Vtbilp9LipVrDG028XMFY6eY/lb1n9C4hiMWF/N4YTXz8Tm+IBIBqKpgeUCzb3ZmB7bvVubPUgWdL3K7piiClRQ8ssIAN67DEOm/sTgyv0eFitWiZopPhBvgizOBPDv2/GSa3WdFRiNAtaxrqOZVO3W4HUn62XNz+leI640BMjAYMvIcpNU2x0R+S8ILf7v0zcM=";

    // The canonical vectors from the protocol-encryption documentation. `simon` is the case
    // with a stripped leading zero; `jeb_` is the negative case.
    const NOTCH: &str = "4ed1f46bbe04bc756bcb17c0c7ce3e4632f06a48";
    const JEB: &str = "-7c9d5b0044c130109a5d7b5fb5c317c02b4e28c1";
    const SIMON: &str = "88e16a1019277b15d58faf0541e11910eb756f6";

    #[test]
    fn known_test_vectors() {
        assert_eq!(server_hash("Notch", &[], &[]), NOTCH);
        assert_eq!(server_hash("jeb_", &[], &[]), JEB);
        assert_eq!(server_hash("simon", &[], &[]), SIMON);
    }

    #[test]
    fn official_signed_texture_property_verifies_with_authlibs_bundled_key() {
        assert!(verify_textures_signature(
            NOTCH_TEXTURE_VALUE,
            NOTCH_TEXTURE_SIGNATURE
        ));
    }

    /// The memo must not let a remembered good pair vouch for a different value or signature,
    /// in either order of first use.
    #[test]
    fn a_remembered_verdict_covers_only_its_exact_value_and_signature() {
        for _ in 0..2 {
            assert!(verify_textures_signature(
                NOTCH_TEXTURE_VALUE,
                NOTCH_TEXTURE_SIGNATURE
            ));
            assert!(!verify_textures_signature(
                "ewogIA==",
                NOTCH_TEXTURE_SIGNATURE
            ));
            assert!(!verify_textures_signature(NOTCH_TEXTURE_VALUE, "AAAA"));
        }
    }

    /// HITCH-1: the game thread's no-RSA lookup sees exactly what a verification remembered, and
    /// nothing before it.
    #[test]
    fn a_remembered_verdict_is_visible_without_rechecking_only_after_a_verification() {
        let value = "ewogICJyZW1lbWJlcmVkIiA6IDEKfQ==";
        assert_eq!(remembered_textures_verdict(value, "QUJD"), None);
        assert!(!verify_textures_signature(value, "QUJD"));
        assert_eq!(remembered_textures_verdict(value, "QUJD"), Some(false));
        assert!(verify_textures_signature(
            NOTCH_TEXTURE_VALUE,
            NOTCH_TEXTURE_SIGNATURE
        ));
        assert_eq!(
            remembered_textures_verdict(NOTCH_TEXTURE_VALUE, NOTCH_TEXTURE_SIGNATURE),
            Some(true)
        );
        assert_eq!(
            remembered_textures_verdict(NOTCH_TEXTURE_VALUE, "QUJD"),
            None
        );
    }

    /// Measurement aid, not a gate: what one uncached RSA-4096 texture check costs on this
    /// machine. `cargo test -p mc-auth --release --lib rsa_texture_check_cost -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn rsa_texture_check_cost() {
        let _ = session_public_key();
        let runs = 50u32;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            assert!(check_textures_signature(
                NOTCH_TEXTURE_VALUE,
                NOTCH_TEXTURE_SIGNATURE
            ));
        }
        let per_check = start.elapsed() / runs;
        eprintln!("rsa_texture_check_cost: {per_check:?} per uncached check ({runs} runs)");
    }

    #[test]
    fn a_single_mutated_signature_byte_is_rejected() {
        let mut signature = base64::engine::general_purpose::STANDARD
            .decode(NOTCH_TEXTURE_SIGNATURE)
            .unwrap();
        signature[0] ^= 1;
        let signature = base64::engine::general_purpose::STANDARD.encode(signature);
        assert!(!verify_textures_signature(NOTCH_TEXTURE_VALUE, &signature));
    }

    #[test]
    fn a_mutated_texture_value_is_rejected_by_the_signature_gate() {
        // Keep the replacement ASCII/base64-valid: this isolates the RSA message input from
        // malformed-base64 handling and bites a verifier that authenticates only the signature
        // bytes (or a cached value), rather than the supplied property value.
        let mut value = NOTCH_TEXTURE_VALUE.to_owned();
        value.replace_range(0..1, "f");
        assert!(!verify_textures_signature(&value, NOTCH_TEXTURE_SIGNATURE));
    }

    #[test]
    fn simon_has_its_leading_zero_stripped() {
        // Guards against a "pad to 40 chars" regression: the raw digest starts with 0x08.
        assert_eq!(SIMON.len(), 39);
        assert!(!server_hash("simon", &[], &[]).starts_with('0'));
    }

    #[test]
    fn inputs_are_concatenated_in_order() {
        // Splitting the same bytes across the three arguments must not change the result:
        // this is what pins the server-id / secret / pubkey ordering.
        assert_eq!(server_hash("Not", b"ch", b""), NOTCH);
        assert_eq!(server_hash("", b"Not", b"ch"), NOTCH);
        assert_eq!(server_hash("N", b"otc", b"h"), NOTCH);
        // ...and reordering them must.
        assert_ne!(server_hash("ch", b"Not", b""), NOTCH);
    }

    #[test]
    fn server_id_uses_java_iso_8859_1_encoding() {
        // Oracles generated by Java 21 using exactly the vanilla expression:
        // new BigInteger(MessageDigest.getInstance("SHA-1")
        //     .digest(value.getBytes("ISO_8859_1"))).toString(16).
        // U+00E9 bites a UTF-8 mutant (C3 A9 instead of E9).
        assert_eq!(
            server_hash("\u{00e9}", &[], &[]),
            "1599e9fa41ec68c80230491902786bee889f5bcb"
        );
        // U+0100 bites a low-byte-truncation mutant (00 instead of Java's 3F replacement).
        assert_eq!(
            server_hash("\u{0100}", &[], &[]),
            "5bab61eb53176449e25c2c82f172b82cb13ffb9d"
        );
        // Java treats a valid surrogate pair as one unmappable code point, hence one `?`.
        assert_eq!(
            server_hash("\u{1f600}", &[], &[]),
            "5bab61eb53176449e25c2c82f172b82cb13ffb9d"
        );
        assert_eq!(
            server_hash("\u{00e9}\u{0100}\u{1f600}", &[], &[]),
            "4d7c820100237ab080b0d28f1ab7e4f208fc8154"
        );
    }

    #[test]
    fn java_latin1_boundaries_are_encoded_byte_for_byte() {
        assert_eq!(
            latin1_bytes("\u{0000}\u{007f}\u{0080}\u{00ff}\u{0100}\u{1f600}"),
            [0x00, 0x7f, 0x80, 0xff, b'?', b'?']
        );
    }

    #[test]
    fn realistic_shapes_produce_a_plausible_hash() {
        let secret = [0x42u8; 16];
        let pubkey = vec![0xABu8; 162];
        let h = server_hash("", &secret, &pubkey);
        assert!(h.len() <= 41);
        assert!(h
            .trim_start_matches('-')
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn signed_hex_edges() {
        assert_eq!(signed_hex(&[0x00; 20]), "0");
        assert_eq!(signed_hex(&[]), "0");
        assert_eq!(signed_hex(&[0x00, 0x01]), "1");
        assert_eq!(signed_hex(&[0x0f, 0xff]), "fff");
        // -1 in two's complement is all ones.
        assert_eq!(signed_hex(&[0xff; 20]), "-1");
        // -256.
        assert_eq!(signed_hex(&[0xff, 0x00]), "-100");
        // Most negative reachable-ish value: carry propagates through every byte.
        assert_eq!(signed_hex(&[0x80, 0x00]), "-8000");
    }

    #[test]
    fn negate_is_an_involution_for_nonextreme_values() {
        let v = vec![0x12, 0x34, 0x56, 0x78];
        assert_eq!(twos_complement_negate(&twos_complement_negate(&v)), v);
    }
}
