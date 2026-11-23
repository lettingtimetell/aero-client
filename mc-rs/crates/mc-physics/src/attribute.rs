//! The iteration order of vanilla's UUID-keyed attribute-modifier sets.
//!
//! `ModifiableAttributeInstance` stores each operation's modifiers in Guava's
//! `Sets.newHashSet()`.  `computeValue` iterates that set, rather than the S20
//! packet order.  This matters for operation 2: the intermediate `f64` round
//! after each multiplication makes the otherwise commutative product observable.

/// Return the source indices in the normal-list-bin order a Java 8
/// `HashSet<UUID>` visits them.
///
/// The input is the insertion order of one operation's modifiers, and an entry
/// is a UUID represented as `Uuid::as_u128()`.  A valid vanilla attribute set
/// has unique UUIDs; for a duplicate this follows `HashSet`'s first-entry-wins
/// behaviour (the real client's `applyModifier` rejects it before evaluation).
///
/// This models `HashMap`'s default 16-bucket table, 0.75 load factor, hash
/// spreader, tail insertion, and resize split.  A tree bin needs ten
/// deliberately same-hash UUIDs.  Its first visited node is selected using
/// HotSpot object identity hashes, which are not present on the wire and are
/// therefore not reproducible in a Rust client; no vanilla 1.8.9 attribute
/// packet reaches that adversarial shape.  The ordinary list-bin behaviour is
/// the complete observable behaviour for vanilla packets.
#[must_use]
pub fn java_uuid_hash_set_order(uuids: &[u128]) -> Vec<usize> {
    // `new HashSet()` creates its backing Java 8 HashMap lazily.  Its first
    // insertion allocates the default 16 buckets, with threshold 12.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 16];
    let mut size = 0_usize;
    let mut threshold = 12_usize;
    // "Already in the set" by hash, not by scanning the bucket: the
    // same first-entry-wins answer, but a server choosing UUIDs that all share one bucket (the
    // Java hash is an XOR of the four 32-bit words) can no longer make one S20 cost m^2.
    let mut present = std::collections::HashSet::with_capacity(uuids.len());

    for (index, &uuid) in uuids.iter().enumerate() {
        if !present.insert(uuid) {
            continue;
        }
        let hash = java_hash_uuid(uuid);
        let bucket = (hash as usize) & (buckets.len() - 1);
        buckets[bucket].push(index);
        size += 1;

        // Java's `putVal`: after adding entry number 13 at default capacity,
        // resize.  Rehashing keeps each old bucket's relative linked order.
        if size > threshold {
            let old_capacity = buckets.len();
            let mut resized = vec![Vec::new(); old_capacity * 2];
            for entries in buckets {
                for entry in entries {
                    let new_bucket = (java_hash_uuid(uuids[entry]) as usize) & (resized.len() - 1);
                    resized[new_bucket].push(entry);
                }
            }
            buckets = resized;
            threshold = threshold.saturating_mul(2);
        }
    }

    buckets.into_iter().flatten().collect()
}

/// Java's `UUID.hashCode()` followed by `HashMap.hash(Object)`.
fn java_hash_uuid(uuid: u128) -> u32 {
    let most = (uuid >> 64) as u64;
    let least = uuid as u64;
    let uuid_hash = ((most >> 32) as u32) ^ (most as u32) ^ ((least >> 32) as u32) ^ (least as u32);
    uuid_hash ^ (uuid_hash >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visits_uuid_buckets_not_packet_insertion_order() {
        // UUID hashes are respectively 18, 1 and 2.  18 and 2 collide in the
        // initial 16-bucket table, while 1 is visited first.
        let wire = [18_u128, 1, 2];
        assert_eq!(java_uuid_hash_set_order(&wire), vec![1, 0, 2]);
    }

    #[test]
    fn resize_keeps_java_bucket_split_and_chain_order() {
        // Enough entries to cross Java HashMap's 12-entry threshold.  The old
        // bucket-1 chain splits into bucket 1 then bucket 17 after resize.
        let mut wire = (1_u128..=12).collect::<Vec<_>>();
        wire.extend([17, 33]);
        let ordered = java_uuid_hash_set_order(&wire);
        let actual = ordered.into_iter().map(|i| wire[i]).collect::<Vec<_>>();
        assert_eq!(actual, vec![1, 33, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]);
    }

    #[test]
    fn preserves_first_duplicate_as_hash_set_does() {
        assert_eq!(java_uuid_hash_set_order(&[2, 18, 2, 1]), vec![3, 0, 1]);
    }

    /// The order as computed before duplicates were found by hash:
    /// each insertion scanned its bucket.
    fn order_by_scanning_buckets(uuids: &[u128]) -> Vec<usize> {
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 16];
        let mut size = 0_usize;
        let mut threshold = 12_usize;
        for (index, &uuid) in uuids.iter().enumerate() {
            let bucket = (java_hash_uuid(uuid) as usize) & (buckets.len() - 1);
            if buckets[bucket]
                .iter()
                .any(|&existing| uuids[existing] == uuid)
            {
                continue;
            }
            buckets[bucket].push(index);
            size += 1;
            if size > threshold {
                let mut resized = vec![Vec::new(); buckets.len() * 2];
                for entries in buckets {
                    for entry in entries {
                        let new_bucket =
                            (java_hash_uuid(uuids[entry]) as usize) & (resized.len() - 1);
                        resized[new_bucket].push(entry);
                    }
                }
                buckets = resized;
                threshold = threshold.saturating_mul(2);
            }
        }
        buckets.into_iter().flatten().collect()
    }

    /// The same order as the bucket scan for random sets drawn from
    /// small pools (so duplicates, shared buckets and resizes all occur), and linear time for
    /// UUIDs that all hash alike.
    #[test]
    fn finding_duplicates_by_hash_keeps_the_order_and_is_linear_for_one_shared_bucket() {
        let mut state = 0x5EC0_A771_B0E5_u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for round in 0..2_000_u32 {
            let len = (next() % 60) as usize;
            let pool = 1 + next() % 40;
            let uuids: Vec<u128> = (0..len)
                .map(|_| {
                    let pick = u128::from(next() % pool);
                    // Half the rounds use same-hash UUIDs: (i << 32 | i) in the high word.
                    if round.is_multiple_of(2) {
                        ((pick << 32) | pick) << 64
                    } else {
                        pick * 17
                    }
                })
                .collect();
            assert_eq!(
                java_uuid_hash_set_order(&uuids),
                order_by_scanning_buckets(&uuids),
                "{uuids:?}"
            );
        }
        let same_hash: Vec<u128> = (1..=60_000_u128).map(|i| ((i << 32) | i) << 64).collect();
        assert!(same_hash.iter().all(|&uuid| java_hash_uuid(uuid) == 0));
        let start = std::time::Instant::now();
        assert_eq!(java_uuid_hash_set_order(&same_hash).len(), 60_000);
        let took = start.elapsed();
        assert!(
            took < std::time::Duration::from_secs(3),
            "60,000 same-hash UUIDs took {took:?}"
        );
    }
}
