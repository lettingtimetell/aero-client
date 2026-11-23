//! Protocol encryption: the RSA key exchange, and the AES-CFB8 stream it establishes.
//!
//! java: `util/CryptManager.java`, `network/NettyEncryptingEncoder.java`,
//! `network/NettyEncryptingDecoder.java`, `network/NettyEncryptionTranslator.java`
//!
//! # The exchange
//!
//! 1. The server sends `S01PacketEncryptionRequest`: a (usually empty) server id, its RSA public
//!    key as X.509 `SubjectPublicKeyInfo` DER, and a random verify token.
//! 2. The client generates a fresh 16-byte AES key — [`SharedSecret::generate`], java:
//!    `CryptManager.createNewSharedKey` (`KeyGenerator.getInstance("AES")`, `init(128)`).
//! 3. The client computes the join hash over `(server_id, secret, public_key_der)` and registers
//!    it with the session server. That hash is hazard **H3** and lives in `mc-auth`; this module
//!    calls [`mc_auth::server_hash`] rather than reimplementing it.
//! 4. The client RSA-encrypts the secret and the verify token — separately, PKCS#1 v1.5, java:
//!    `CryptManager.encryptData` (`Cipher.getInstance("RSA")` defaults to
//!    `RSA/ECB/PKCS1Padding`) — and sends `C01PacketEncryptionResponse`.
//! 5. **After** that packet is on the wire in the clear, both directions switch to AES-CFB8.
//!    java: `NetHandlerLoginClient.handleEncryptionRequest`'s write-completion listener calls
//!    `NetworkManager.enableEncryption` only once the response has actually been flushed.
//!
//! # Hazard H4 — CFB8 is a *stream*, not a per-packet transform
//!
//! `NetworkManager.enableEncryption` installs exactly two `javax.crypto.Cipher` objects for the
//! life of the connection and every subsequent byte goes through `Cipher.update` on those same
//! two objects — never `doFinal`, never a fresh instance. The CFB8 shift register therefore
//! carries across packet, frame, and TCP-segment boundaries alike.
//!
//! [`EncryptStream`] and [`DecryptStream`] are the Rust equivalents: they own the cipher state
//! and are applied to whatever slice of the byte stream happens to arrive. Re-creating one
//! mid-connection resets the shift register and corrupts everything after it — the classic
//! "works right up until one packet after login" failure. `chunking_matches_one_shot` in this
//! module's tests is the regression test for exactly that.
//!
//! Note also *where* in the pipeline this sits: encryption is the outermost layer, applied to
//! the length-prefixed frame including its prefix. See [`crate::connection`] for the full
//! ordering and the `NetworkManager` pipeline it is derived from.

use aes::cipher::inout::InOutBuf;
use aes::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use aes::Aes128;
use rand::rngs::OsRng;
use rand::RngCore;
use rsa::pkcs8::DecodePublicKey;
use rsa::{Pkcs1v15Encrypt, RsaPublicKey};

use crate::login::{EncryptionRequest, EncryptionResponse};
use crate::ProtoError;

/// The AES shared secret is always 128 bits. java: `CryptManager.createNewSharedKey` →
/// `keygenerator.init(128)`.
pub const SHARED_SECRET_LEN: usize = 16;

type Aes128Cfb8Enc = cfb8::Encryptor<Aes128>;
type Aes128Cfb8Dec = cfb8::Decryptor<Aes128>;

/// The 16-byte AES key the client generates for one connection.
///
/// `Debug` is redacted: this value plus the server's public key is enough to impersonate the
/// join, and connection state gets logged liberally during development.
#[derive(Clone, PartialEq, Eq)]
pub struct SharedSecret([u8; SHARED_SECRET_LEN]);

impl SharedSecret {
    /// Generate a fresh secret from the OS CSPRNG.
    ///
    /// java: `CryptManager.createNewSharedKey`. Java's `KeyGenerator` also draws from a
    /// cryptographically secure source; [`OsRng`] is the direct equivalent and, unlike a seeded
    /// userspace generator, cannot silently degrade.
    #[must_use]
    pub fn generate() -> SharedSecret {
        let mut bytes = [0u8; SHARED_SECRET_LEN];
        OsRng.fill_bytes(&mut bytes);
        SharedSecret(bytes)
    }

    /// Wrap an existing 16-byte key. Only useful for tests and for the server side of a
    /// loopback harness; real clients use [`SharedSecret::generate`].
    #[must_use]
    pub fn from_bytes(bytes: [u8; SHARED_SECRET_LEN]) -> SharedSecret {
        SharedSecret(bytes)
    }

    /// The raw key bytes, as fed to SHA-1 for the join hash and to RSA for the response packet.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; SHARED_SECRET_LEN] {
        &self.0
    }
}

impl std::fmt::Debug for SharedSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SharedSecret(<redacted>)")
    }
}

/// The client→server half of the AES-CFB8 stream.
///
/// One instance per connection, for the whole life of the connection (hazard H4). Note the IV:
/// vanilla passes the key itself as the IV (`new IvParameterSpec(key.getEncoded())` in
/// `CryptManager.createNetCipherInstance`), so the same 16 bytes serve as both.
pub struct EncryptStream(Aes128Cfb8Enc);

impl EncryptStream {
    /// Start the outbound stream. java: `CryptManager.createNetCipherInstance(ENCRYPT_MODE, key)`
    #[must_use]
    pub fn new(secret: &SharedSecret) -> EncryptStream {
        let key = secret.as_bytes();
        EncryptStream(Aes128Cfb8Enc::new(key.into(), key.into()))
    }

    /// Encrypt `buf` in place, advancing the shift register by `buf.len()` bytes.
    ///
    /// Any slicing of the stream is legal — one call per frame, one call per socket write, or
    /// one call per byte all produce identical output, because the state lives in `self`.
    pub fn apply(&mut self, buf: &mut [u8]) {
        let (blocks, tail) = InOutBuf::from(buf).into_chunks();
        debug_assert_eq!(
            tail.len(),
            0,
            "CFB8's block size is one byte: there is no tail"
        );
        self.0.encrypt_blocks_inout_mut(blocks);
    }
}

impl std::fmt::Debug for EncryptStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EncryptStream(<cfb8 state>)")
    }
}

/// The server→client half of the AES-CFB8 stream. See [`EncryptStream`]; the same continuity
/// rule applies, and the same key is used in both directions.
pub struct DecryptStream(Aes128Cfb8Dec);

impl DecryptStream {
    /// Start the inbound stream. java: `CryptManager.createNetCipherInstance(DECRYPT_MODE, key)`
    #[must_use]
    pub fn new(secret: &SharedSecret) -> DecryptStream {
        let key = secret.as_bytes();
        DecryptStream(Aes128Cfb8Dec::new(key.into(), key.into()))
    }

    /// Decrypt `buf` in place, advancing the shift register by `buf.len()` bytes.
    pub fn apply(&mut self, buf: &mut [u8]) {
        let (blocks, tail) = InOutBuf::from(buf).into_chunks();
        debug_assert_eq!(
            tail.len(),
            0,
            "CFB8's block size is one byte: there is no tail"
        );
        self.0.decrypt_blocks_inout_mut(blocks);
    }
}

impl std::fmt::Debug for DecryptStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DecryptStream(<cfb8 state>)")
    }
}

/// Parse the server's public key from the X.509 `SubjectPublicKeyInfo` DER carried by
/// [`EncryptionRequest::public_key_der`].
///
/// java: `CryptManager.decodePublicKey` (`new X509EncodedKeySpec(encodedKey)` +
/// `KeyFactory.getInstance("RSA")`) — Java's `PublicKey.getEncoded()` produces SPKI, which is
/// what the server writes, so SPKI is what we must read.
///
/// # Errors
///
/// [`ProtoError::PublicKey`] if the bytes are not a well-formed RSA SPKI key. Vanilla logs
/// "Public key reconstitute failed!" and returns `null` here, which then NPEs; returning a typed
/// error instead means a hostile server cannot crash the client.
pub fn parse_public_key_der(der: &[u8]) -> Result<RsaPublicKey, ProtoError> {
    RsaPublicKey::from_public_key_der(der).map_err(|e| ProtoError::PublicKey(e.to_string()))
}

/// RSA-encrypt one payload under the server's public key, PKCS#1 v1.5.
///
/// java: `CryptManager.encryptData` → `Cipher.getInstance(key.getAlgorithm())`. That
/// transformation string is bare `"RSA"`, which the SunJCE provider expands to
/// `RSA/ECB/PKCS1Padding` — i.e. PKCS#1 v1.5 encryption padding, **not** OAEP. Using OAEP here
/// is a silent, total failure: the server decrypts garbage and kicks with a generic error.
///
/// # Errors
///
/// [`ProtoError::RsaEncrypt`] if `plaintext` is too long for the modulus, or the RNG fails.
pub fn encrypt_pkcs1v15(key: &RsaPublicKey, plaintext: &[u8]) -> Result<Vec<u8>, ProtoError> {
    key.encrypt(&mut OsRng, Pkcs1v15Encrypt, plaintext)
        .map_err(|e| ProtoError::RsaEncrypt(e.to_string()))
}

/// Everything the client derives from one [`EncryptionRequest`].
#[derive(Debug, Clone)]
pub struct EncryptionExchange {
    /// The freshly generated AES key. Feed it to [`EncryptStream`]/[`DecryptStream`] *after*
    /// [`response`](Self::response) has been flushed.
    pub secret: SharedSecret,
    /// The join hash to hand to [`mc_auth::join_server`] (hazard H3: signed two's-complement
    /// hex, so it may start with `-`).
    pub server_hash: String,
    /// The packet to send back.
    pub response: EncryptionResponse,
}

/// Do the whole client side of the key exchange for one [`EncryptionRequest`].
///
/// This is deliberately synchronous and side-effect-free: it does not touch the network and it
/// does not call the session server, so it is fully testable. The caller is responsible for
/// (a) registering [`EncryptionExchange::server_hash`] with the session server *before* sending
/// the response, and (b) enabling the ciphers *after*. [`crate::connection::login`] does both in
/// the right order.
///
/// # Errors
///
/// [`ProtoError::PublicKey`] for an unparseable key, [`ProtoError::RsaEncrypt`] if either
/// encryption fails.
pub fn begin_encryption(
    request: &EncryptionRequest,
    secret: SharedSecret,
) -> Result<EncryptionExchange, ProtoError> {
    let key = parse_public_key_der(&request.public_key_der)?;
    // java: NetHandlerLoginClient.handleEncryptionRequest —
    //   new BigInteger(CryptManager.getServerIdHash(s, publickey, secretkey)).toString(16)
    // Note the argument order the digest is fed in is (server id, secret, key), which is *not*
    // the order they appear in that Java call; `CryptManager.getServerIdHash` reorders them.
    let server_hash = mc_auth::server_hash(
        &request.server_id,
        secret.as_bytes(),
        &request.public_key_der,
    );
    let response = EncryptionResponse {
        shared_secret_encrypted: encrypt_pkcs1v15(&key, secret.as_bytes())?,
        verify_token_encrypted: encrypt_pkcs1v15(&key, &request.verify_token)?,
    };
    Ok(EncryptionExchange {
        secret,
        server_hash,
        response,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cipher_streams_cannot_be_cloned() {
        // Inference selects the blanket `()` implementation only while the type is not Clone.
        // Adding Clone creates a second applicable implementation and makes this expression
        // ambiguous at compile time, without requiring another test-only dependency.
        trait AmbiguousIfClone<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        struct Invalid;
        impl<T: Clone> AmbiguousIfClone<Invalid> for T {}

        let _ = <EncryptStream as AmbiguousIfClone<_>>::marker;
        let _ = <DecryptStream as AmbiguousIfClone<_>>::marker;
    }

    fn hex(s: &str) -> Vec<u8> {
        assert!(s.len().is_multiple_of(2));
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex digit"))
            .collect()
    }

    /// NIST SP 800-38A, appendix F.3.7 (`CFB8-AES128.Encrypt`) — the canonical public vector for
    /// this mode. Only the first 18 plaintext bytes are covered by the published segments.
    #[test]
    fn nist_sp800_38a_cfb8_aes128_vector() {
        let key: [u8; 16] = hex("2b7e151628aed2a6abf7158809cf4f3c").try_into().unwrap();
        let iv: [u8; 16] = hex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let plaintext = hex("6bc1bee22e409f96e93d7e117393172aae2d");
        let ciphertext = hex("3b79424c9c0dd436bace9e0ed4586a4f32b9");

        let mut enc = Aes128Cfb8Enc::new(&key.into(), &iv.into());
        let mut buf = plaintext.clone();
        let (blocks, _) = InOutBuf::from(&mut buf[..]).into_chunks();
        enc.encrypt_blocks_inout_mut(blocks);
        assert_eq!(buf, ciphertext, "CFB8 encryption diverges from NIST F.3.7");

        let mut dec = Aes128Cfb8Dec::new(&key.into(), &iv.into());
        let (blocks, _) = InOutBuf::from(&mut buf[..]).into_chunks();
        dec.decrypt_blocks_inout_mut(blocks);
        assert_eq!(buf, plaintext);
    }

    /// Minecraft's own twist: the key doubles as the IV
    /// (`new IvParameterSpec(key.getEncoded())`). Encrypting with key==IV must agree with a
    /// plain CFB8 run configured that way, and `EncryptStream` must not quietly use a zero IV.
    #[test]
    fn key_doubles_as_iv() {
        let secret = SharedSecret::from_bytes([0x42; 16]);
        let mut ours = *b"hello world! this is my plaintext.";
        EncryptStream::new(&secret).apply(&mut ours);

        let mut reference = *b"hello world! this is my plaintext.";
        let mut cipher = Aes128Cfb8Enc::new(&[0x42u8; 16].into(), &[0x42u8; 16].into());
        let (blocks, _) = InOutBuf::from(&mut reference[..]).into_chunks();
        cipher.encrypt_blocks_inout_mut(blocks);

        assert_eq!(ours, reference);

        // ...and demonstrably *not* what a zero IV would produce.
        let mut zero_iv = *b"hello world! this is my plaintext.";
        let mut cipher = Aes128Cfb8Enc::new(&[0x42u8; 16].into(), &[0u8; 16].into());
        let (blocks, _) = InOutBuf::from(&mut zero_iv[..]).into_chunks();
        cipher.encrypt_blocks_inout_mut(blocks);
        assert_ne!(ours, zero_iv);
    }

    /// **Hazard H4.** Splitting the stream at arbitrary points — which is what a TCP socket
    /// does to you whether you like it or not — must not change a single output byte.
    #[test]
    fn chunking_matches_one_shot() {
        let secret = SharedSecret::generate();
        let plaintext: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();

        let mut one_shot = plaintext.clone();
        EncryptStream::new(&secret).apply(&mut one_shot);

        for chunk in [1usize, 3, 7, 16, 17, 64, 999] {
            let mut chunked = plaintext.clone();
            let mut stream = EncryptStream::new(&secret);
            for piece in chunked.chunks_mut(chunk) {
                stream.apply(piece);
            }
            assert_eq!(
                chunked, one_shot,
                "CFB8 stream was not continuous across {chunk}-byte writes"
            );
        }
    }

    /// The failure mode H4 describes, made explicit: a cipher re-created per "packet" produces
    /// the right bytes for the first packet and garbage for every one after it.
    #[test]
    fn reinitialising_per_packet_is_wrong_after_the_first() {
        let secret = SharedSecret::generate();
        let packets: [&[u8]; 3] = [b"first packet", b"second packet", b"third packet"];

        let mut continuous = EncryptStream::new(&secret);
        let correct: Vec<Vec<u8>> = packets
            .iter()
            .map(|p| {
                let mut b = p.to_vec();
                continuous.apply(&mut b);
                b
            })
            .collect();

        let restarted: Vec<Vec<u8>> = packets
            .iter()
            .map(|p| {
                let mut b = p.to_vec();
                EncryptStream::new(&secret).apply(&mut b);
                b
            })
            .collect();

        assert_eq!(
            correct[0], restarted[0],
            "the first packet always looks fine"
        );
        assert_ne!(correct[1], restarted[1], "the second is where it breaks");
        assert_ne!(correct[2], restarted[2]);
    }

    #[test]
    fn encrypt_then_decrypt_round_trips_across_chunks() {
        let secret = SharedSecret::generate();
        let mut enc = EncryptStream::new(&secret);
        let mut dec = DecryptStream::new(&secret);

        // Deliberately misaligned: encrypt in one shape, decrypt in another.
        let mut wire = Vec::new();
        for n in [5usize, 1, 40, 3] {
            let mut piece: Vec<u8> = (0..n).map(|i| i as u8).collect();
            enc.apply(&mut piece);
            wire.extend_from_slice(&piece);
        }
        let mut plain = wire.clone();
        for piece in plain.chunks_mut(7) {
            dec.apply(piece);
        }

        let expected: Vec<u8> = [5usize, 1, 40, 3]
            .iter()
            .flat_map(|&n| (0..n).map(|i| i as u8))
            .collect();
        assert_eq!(plain, expected);
    }

    #[test]
    fn empty_slice_does_not_advance_the_stream() {
        let secret = SharedSecret::generate();
        let mut a = EncryptStream::new(&secret);
        a.apply(&mut []);
        let mut buf_a = *b"abc";
        a.apply(&mut buf_a);

        let mut buf_b = *b"abc";
        EncryptStream::new(&secret).apply(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn shared_secret_debug_is_redacted() {
        let secret = SharedSecret::from_bytes([0xAB; 16]);
        let shown = format!("{secret:?}");
        assert!(!shown.contains("ab"), "{shown}");
        assert!(!shown.contains("171"), "{shown}");
    }

    #[test]
    fn generated_secrets_differ() {
        let a = SharedSecret::generate();
        let b = SharedSecret::generate();
        assert_ne!(a.as_bytes(), b.as_bytes());
        assert_ne!(a.as_bytes(), &[0u8; 16]);
    }

    #[test]
    fn garbage_public_key_is_an_error_not_a_panic() {
        assert!(matches!(
            parse_public_key_der(&[]),
            Err(ProtoError::PublicKey(_))
        ));
        assert!(matches!(
            parse_public_key_der(&[0xAA; 162]),
            Err(ProtoError::PublicKey(_))
        ));
    }

    /// A real key round-trip: SPKI DER out of the `rsa` crate, back in through our parser, and
    /// a PKCS#1 v1.5 ciphertext the matching private key can decrypt. This is the same shape
    /// the server exercises, minus the network.
    #[test]
    fn rsa_exchange_round_trips_through_der() {
        use rsa::pkcs8::EncodePublicKey;
        use rsa::RsaPrivateKey;

        let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit keygen");
        let der = private
            .to_public_key()
            .to_public_key_der()
            .expect("SPKI encode")
            .as_bytes()
            .to_vec();

        let request = EncryptionRequest {
            server_id: String::new(),
            public_key_der: der,
            verify_token: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let secret = SharedSecret::generate();
        let exchange = begin_encryption(&request, secret.clone()).expect("exchange");

        let got_secret = private
            .decrypt(Pkcs1v15Encrypt, &exchange.response.shared_secret_encrypted)
            .expect("decrypt secret");
        assert_eq!(got_secret, secret.as_bytes());

        let got_token = private
            .decrypt(Pkcs1v15Encrypt, &exchange.response.verify_token_encrypted)
            .expect("decrypt token");
        assert_eq!(got_token, request.verify_token);

        // 1024-bit modulus → 128-byte ciphertexts, and PKCS#1 v1.5 is randomised, so two
        // encryptions of the same plaintext must differ.
        assert_eq!(exchange.response.shared_secret_encrypted.len(), 128);
        let again = begin_encryption(&request, secret.clone()).expect("exchange");
        assert_ne!(
            again.response.shared_secret_encrypted,
            exchange.response.shared_secret_encrypted
        );
    }

    /// Hazard H3, checked end to end rather than trusting the wiring: `begin_encryption` must
    /// feed the digest `(server_id, secret, public_key_der)` in that order.
    #[test]
    fn server_hash_is_wired_to_mc_auth_in_the_right_order() {
        use rsa::pkcs8::EncodePublicKey;
        use rsa::RsaPrivateKey;

        let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit keygen");
        let der = private
            .to_public_key()
            .to_public_key_der()
            .expect("SPKI encode")
            .as_bytes()
            .to_vec();
        let request = EncryptionRequest {
            server_id: "MyServerId".to_owned(),
            public_key_der: der.clone(),
            verify_token: vec![1, 2, 3, 4],
        };
        let secret = SharedSecret::from_bytes([0x11; 16]);
        let exchange = begin_encryption(&request, secret.clone()).expect("exchange");

        assert_eq!(
            exchange.server_hash,
            mc_auth::server_hash("MyServerId", &[0x11; 16], &der)
        );
        // Any other ordering of the three inputs must produce a different hash, so a
        // transposition here cannot pass unnoticed.
        assert_ne!(
            exchange.server_hash,
            mc_auth::server_hash("MyServerId", &der, &[0x11; 16])
        );
    }
}
