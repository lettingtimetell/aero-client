//! Round-trip property tests.
//!
//! The invariant that matters for the wire is *byte* stability, not value equality:
//! `encode → decode → encode` must reproduce the first encoding exactly. Stating it that
//! way also survives NaN payloads, which compare unequal to themselves but must still
//! serialise back to the same bits.

use proptest::collection::vec;
use proptest::prelude::*;

use mc_nbt::{compress, Nbt, NbtCompound, NbtLimits, TagType};

/// Characters chosen to hit the modified-UTF-8 edges (hazard H2) far more often than
/// chance would: NUL, the 1/2/3-byte boundaries, and astral planes.
fn arb_char() -> impl Strategy<Value = char> {
    prop_oneof![
        3 => Just('\0'),
        2 => Just('\u{7F}'),
        2 => Just('\u{80}'),
        2 => Just('\u{7FF}'),
        2 => Just('\u{800}'),
        2 => Just('\u{FFFF}'),
        3 => Just('\u{1F600}'),
        2 => Just('\u{10FFFF}'),
        12 => any::<char>(),
    ]
}

fn arb_string() -> impl Strategy<Value = String> {
    vec(arb_char(), 0..10).prop_map(|cs| cs.into_iter().collect())
}

/// Any leaf tag. Floats are generated from raw bits so NaNs and subnormals show up.
fn arb_leaf() -> impl Strategy<Value = Nbt> {
    prop_oneof![
        any::<i8>().prop_map(Nbt::Byte),
        any::<i16>().prop_map(Nbt::Short),
        any::<i32>().prop_map(Nbt::Int),
        any::<i64>().prop_map(Nbt::Long),
        any::<u32>().prop_map(|b| Nbt::Float(f32::from_bits(b))),
        any::<u64>().prop_map(|b| Nbt::Double(f64::from_bits(b))),
        vec(any::<u8>(), 0..24).prop_map(Nbt::ByteArray),
        vec(any::<i32>(), 0..12).prop_map(Nbt::IntArray),
        arb_string().prop_map(Nbt::String),
    ]
}

/// NBT lists are homogeneous; drop anything that does not match the first element.
fn homogeneous(mut items: Vec<Nbt>) -> Vec<Nbt> {
    if let Some(first) = items.first().map(Nbt::tag_type) {
        items.retain(|i| i.tag_type() == first);
    }
    items
}

fn arb_nbt() -> impl Strategy<Value = Nbt> {
    arb_leaf().prop_recursive(4, 64, 5, |inner| {
        prop_oneof![
            vec(inner.clone(), 0..5).prop_map(|items| Nbt::list(homogeneous(items))),
            vec((arb_string(), inner), 0..5)
                .prop_map(|es| Nbt::Compound(es.into_iter().collect::<NbtCompound>())),
        ]
    })
}

/// A root is always a named tag; compounds are the only root vanilla ever writes, but the
/// codec is not restricted to them.
fn arb_root() -> impl Strategy<Value = (String, Nbt)> {
    (arb_string(), arb_nbt())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// encode → decode → encode is byte-stable.
    #[test]
    fn nbt_round_trips_byte_identically((name, value) in arb_root()) {
        let first = mc_nbt::to_vec(&name, &value).expect("encode");
        let (name2, value2) =
            mc_nbt::from_slice_with_limits(&first, NbtLimits::UNLIMITED).expect("decode");
        let second = mc_nbt::to_vec(&name2, &value2).expect("re-encode");

        prop_assert_eq!(&name2, &name);
        prop_assert_eq!(&second, &first);
    }

    /// Values without NaN payloads compare equal after a round trip, too.
    #[test]
    fn values_survive_the_round_trip((name, value) in arb_root()) {
        let bytes = mc_nbt::to_vec(&name, &value).expect("encode");
        let (_, decoded) =
            mc_nbt::from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).expect("decode");
        if !has_nan(&value) {
            prop_assert_eq!(decoded, value);
        }
    }

    /// The gzip and zlib containers are transparent.
    #[test]
    fn compressed_round_trips((name, value) in arb_root()) {
        let raw = mc_nbt::to_vec(&name, &value).expect("encode");

        let mut gz = Vec::new();
        compress::write_gzip(&mut gz, &name, &value).expect("gzip");
        let (gz_name, gz_value) =
            compress::read_gzip_with_limits(&mut &gz[..], NbtLimits::UNLIMITED).expect("gunzip");
        prop_assert_eq!(mc_nbt::to_vec(&gz_name, &gz_value).expect("re-encode"), raw.clone());

        let mut zl = Vec::new();
        compress::write_zlib(&mut zl, &name, &value).expect("zlib");
        let (zl_name, zl_value) =
            compress::read_zlib_with_limits(&mut &zl[..], NbtLimits::UNLIMITED).expect("inflate");
        prop_assert_eq!(mc_nbt::to_vec(&zl_name, &zl_value).expect("re-encode"), raw);
    }

    /// Modified UTF-8 is byte-stable for every string Rust can hold.
    #[test]
    fn modified_utf8_round_trips(s in arb_string()) {
        let bytes = mc_nbt::modified_utf8::encode(&s).expect("encode");
        prop_assert_eq!(bytes.len(), mc_nbt::modified_utf8::encoded_len(&s));
        prop_assert!(!bytes.contains(&0), "modified UTF-8 never emits a NUL byte");
        let back = mc_nbt::modified_utf8::decode(&bytes).expect("decode");
        prop_assert_eq!(&back, &s);
        prop_assert_eq!(mc_nbt::modified_utf8::encode(&back).expect("re-encode"), bytes);
    }

    /// Arbitrary bytes must produce an error, never a panic and never an OOM.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in vec(any::<u8>(), 0..512)) {
        let _ = mc_nbt::from_slice(&bytes);
        let _ = compress::read_any(&bytes, NbtLimits::PROTOCOL);
    }

    /// The same, but biased towards well-formed headers so the parser gets further in.
    #[test]
    fn plausible_garbage_never_panics(
        ty in 0u8..=11,
        len in any::<i32>(),
        tail in vec(any::<u8>(), 0..64),
    ) {
        let mut bytes = vec![ty, 0x00, 0x00];
        bytes.extend_from_slice(&len.to_be_bytes());
        bytes.extend_from_slice(&tail);
        let _ = mc_nbt::from_slice(&bytes);
        let _ = mc_nbt::from_slice_with_limits(&bytes, NbtLimits::UNLIMITED);
    }
}

fn has_nan(v: &Nbt) -> bool {
    match v {
        Nbt::Float(f) => f.is_nan(),
        Nbt::Double(d) => d.is_nan(),
        Nbt::List(items) => items.iter().any(has_nan),
        Nbt::Compound(c) => c.iter().any(|(_, v)| has_nan(v)),
        _ => false,
    }
}

#[test]
fn nan_payloads_keep_their_exact_bits() {
    // A signalling NaN with a distinctive payload: value equality would hide a mangled
    // round trip, byte equality does not.
    let value = Nbt::list(vec![
        Nbt::Float(f32::from_bits(0x7F80_0001)),
        Nbt::Float(f32::from_bits(0xFFC0_DEAD)),
    ]);
    let first = mc_nbt::to_vec("", &value).expect("encode");
    let (name, decoded) = mc_nbt::from_slice(&first).expect("decode");
    assert_eq!(mc_nbt::to_vec(&name, &decoded).expect("re-encode"), first);
    assert_eq!(decoded.list_tag_type(), Some(TagType::Float));
}
