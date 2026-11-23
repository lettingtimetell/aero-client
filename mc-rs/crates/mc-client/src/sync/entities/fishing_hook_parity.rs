//! The client-side `EntityFishHook` flight, end to end, against the
//! real Java classes.
//!
//! Every table below was printed by `crates/mc-client/javaref/FishHookRef.java`, which runs the
//! recompiled vanilla 1.8.9 classes:
//!
//! * **server**: a real server-world hook cast by a real `EntityPlayer` holding a rod, ticked by
//!   its real `onUpdate`, with a real `EntityTrackerEntry(hook, 64, 5, true)` (the
//!   `EntityTracker.java:72-74` registration) producing the packet stream;
//! * **client**: a client-world hook built from that `S0EPacketSpawnObject` exactly as
//!   `NetHandlerPlayClient.handleSpawnObject` builds it, then fed the same S12/S15-S18 packets as
//!   `handleEntityVelocity`/`handleEntityMovement`/`handleEntityTeleport` do, one tick at a time.
//!
//! This test replays those packets through the tracker's public `apply_*` entry points (the ones
//! `sync::dispatch` calls) and the live entity tick, then compares every client tick bit for bit:
//! position, motion, `inGround`, `ticksInAir` and `fishPosRotationIncrements`.
//!
//! Rotation is compared to within `1e-3` degrees modulo one turn rather than bit for bit: the
//! recompiled classes contain `Math.PI` where the shipped jar has `(double)(float)Math.PI`
//! (`javap -c ur.t_()` offsets 950-977), mc-rs follows the jar, and Java keeps its eased angle
//! next to a `prevRotation` it shifts by whole turns. Rotation only feeds the F3+B look line;
//! position and motion never read it, so they stay exact.
//!
//! The scenarios include the two vanilla oddities that were reported as possibly not vanilla:
//! `EntityTrackerEntry.updatePlayerList` advances `encodedPos*` on its first call without
//! sending a position (`:194`, `:197`, `:249-254`), so a short cast's first relative move is
//! measured from a position the client never received and pulls the hook back towards the face
//! (`down_at_feet`, `floor_in_front`); and the `inGround` latch lasts exactly one tick on the
//! client, because `xTile/yTile/zTile/inTile` are never assigned outside the constructors and NBT
//! (`javap -c ur`: the only `putfield g/h/i/ar` are in `<init>` and `a(dn)`).

use super::*;

const HOOK: i32 = 900;
const ANGLER: i32 = 55;

#[derive(Clone, Copy, Debug)]
enum Pkt {
    Vel([i16; 3]),
    Rel([i8; 3], Option<(i8, i8)>, bool),
    Tp([i32; 3], i8, i8, bool),
}

struct Expect {
    t: u32,
    pos: [u64; 3],
    motion: [u64; 3],
    rot: [u32; 2],
    in_ground: bool,
    ticks_in_air: u32,
    increments: i32,
}

struct HookRef {
    name: &'static str,
    angler: [u64; 3],
    stone: &'static [([i32; 3], [i32; 3])],
    spawn: [i32; 3],
    spawn_look: (i8, i8),
    spawn_velocity: [i16; 3],
    client_seed: i64,
    packets: &'static [(u32, Pkt)],
    ticks: &'static [Expect],
}

#[rustfmt::skip]
const REFS: &[HookRef] = &[
        // server t=1 pos=(8.3400, 6.5200, 8.5000) motion=(0.0183, -0.0058, 1.5008) inGround=true caught=false
        // server t=2 pos=(8.3407, 6.5197, 8.5131) motion=(0.0006, -0.0371, 0.0120) inGround=false caught=false
        // server t=3 pos=(8.3413, 6.4826, 8.5251) motion=(0.0006, -0.0709, 0.0111) inGround=false caught=false
        // server t=4 pos=(8.3419, 6.4116, 8.5361) motion=(0.0005, -0.1021, 0.0102) inGround=false caught=false
        // server t=5 pos=(8.3425, 6.3096, 8.5463) motion=(0.0005, -0.1307, 0.0094) inGround=false caught=false
        // server t=6 pos=(8.3429, 6.1789, 8.5557) motion=(0.0005, -0.1570, 0.0086) inGround=false caught=false
        // server t=7 pos=(8.3434, 6.0218, 8.5643) motion=(0.0004, -0.1813, 0.0079) inGround=false caught=false
        // server t=8 pos=(8.3438, 5.8406, 8.5722) motion=(0.0004, -0.2036, 0.0073) inGround=false caught=false
        // server t=9 pos=(8.3442, 5.6370, 8.5795) motion=(0.0004, -0.2241, 0.0067) inGround=false caught=false
        // server t=10 pos=(8.3446, 5.4129, 8.5862) motion=(0.0003, -0.2430, 0.0062) inGround=false caught=false
        // server t=11 pos=(8.3449, 5.1699, 8.5923) motion=(0.0003, -0.2603, 0.0057) inGround=false caught=false
        // server t=12 pos=(8.3449, 5.1699, 8.5923) motion=(0.0003, -0.2603, 0.0057) inGround=true caught=false
        // server t=13 pos=(8.3449, 5.1473, 8.5928) motion=(0.0000, -0.0576, 0.0005) inGround=false caught=false
        // server t=14 pos=(8.3450, 5.0897, 8.5933) motion=(0.0000, -0.0898, 0.0004) inGround=false caught=false
        // server t=15 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0002) inGround=false caught=false
        // server t=16 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0002) inGround=true caught=false
        // server t=17 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0039, 0.0000) inGround=true caught=false
        // server t=18 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0007, 0.0000) inGround=true caught=false
        // server t=19 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=false caught=false
        // server t=20 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=true caught=false
        // server t=21 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0007, 0.0000) inGround=true caught=false
        // server t=22 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=false caught=false
        // server t=23 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=true caught=false
        // server t=24 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0024, 0.0000) inGround=true caught=false
        // server t=25 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0004, 0.0000) inGround=true caught=false
        // server t=26 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=false caught=false
        // server t=27 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=true caught=false
        // server t=28 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0012, 0.0000) inGround=true caught=false
        // server t=29 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=false caught=false
        // server t=30 pos=(8.3450, 5.0000, 8.5937) motion=(0.0000, -0.0200, 0.0000) inGround=true caught=false
    HookRef {
        name: "wall_at_face",
        // angler (8.5, 5.0, 8.5) yaw 0.0 pitch 0.0; prev_pos_at_spawn=(0.0, 0.0, 0.0)
        angler: [0x4021000000000000, 0x4014000000000000, 0x4021000000000000],
        stone: &[([0, 4, 0], [15, 4, 15]), ([0, 5, 10], [15, 8, 10]), ],
        spawn: [266, 208, 272],
        spawn_look: (0, -1),
        spawn_velocity: [146, -46, 12006],
        client_seed: 12,
        packets: &[
            (7, Pkt::Vel([3, -1256, 68])),
            (7, Pkt::Rel([0, -11, 1], Some((1, -35)), false)),
            (12, Pkt::Vel([2, -2082, 45])),
            (12, Pkt::Rel([1, -32, 1], Some((1, -54)), false)),
            (17, Pkt::Vel([0, -159, 1])),
            (17, Pkt::Tp([267, 160, 274], 2, -46, true)),
            (22, Pkt::Rel([0, 0, 0], Some((2, -37)), true)),
            (27, Pkt::Rel([0, 0, 0], Some((3, -24)), true)),
        ],
        ticks: &[
            Expect { t: 1, pos: [0x4020a00000000000, 0x401a000000000000, 0x4021000000000000], motion: [0x3f92b020c49ba5e3, 0xbf778d4fdf3b645a, 0x3ff803126e978d50], rot: [0x00000000, 0xbfb40000], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.3125, 6.5000, 8.5000)
            Expect { t: 2, pos: [0x4020a15d30a6e148, 0x4019ffedf6cbc6a8, 0x40212a4590e7dd2f], motion: [0x3f641413dc0640c5, 0xbfa2dfbf019b783a, 0x3fb371eba3c6861d], rot: [0x3ebd3ec7, 0xbf913850], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3152, 6.4999, 8.5826)
            Expect { t: 3, pos: [0x4020a29e71e4a1ac, 0x4019da2e78c88fb8, 0x40215129682f6a3b], motion: [0x3f6278df17b97459, 0xbfb21a53faba9a8a, 0x3fb1e3afd98edc15], rot: [0x3f2a5229, 0xc0c2a74c], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3176, 6.4631, 8.6585)
            Expect { t: 4, pos: [0x4020a3c5ffd61d43, 0x401991c528dda54e, 0x402174f0c7e287f3], motion: [0x3f60fe8fd3519507, 0xbfba134e81ce7d8c, 0x3fb0754fe1a47698], rot: [0x3f66e209, 0xc15ee6a6], in_ground: false, ticks_in_air: 2, increments: 0 }, // (8.3199, 6.3924, 8.7284)
            Expect { t: 5, pos: [0x4020a4d5e8d3525c, 0x40192977eed66b58, 0x402195db67a5d0e0], motion: [0x3f5f4508ad102f8d, 0xbfc0b480e68e214f, 0x3fae487e89adbf44], rot: [0x3f8baa63, 0xc1b58647], in_ground: false, ticks_in_air: 3, increments: 0 }, // (8.3219, 6.2905, 8.7927)
            Expect { t: 6, pos: [0x4020a5d01118badd, 0x4018a3d3e7a1fa4e, 0x4021b423e62f7e9f], motion: [0x3f5cc4a19dbbd204, 0xbfc4143ee661be06, 0x3fabdc4b7ce75884], rot: [0x3f9f0b85, 0xc1fa3180], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3239, 6.1600, 8.8518)
            Expect { t: 7, pos: [0x4020a3e00b65d1e9, 0x4018a28d4516a6df, 0x40217d6d441fa9bf], motion: [0x3f389374bc6a7efa, 0xbfc4189374bc6a7f, 0x3f816872b020c49c], rot: [0x3fa607ae, 0xc21505d5], in_ground: false, ticks_in_air: 4, increments: 2 }, // (8.3201, 6.1587, 8.7450)
            Expect { t: 8, pos: [0x4020a1f005b2e8f4, 0x4018a146a28b5370, 0x402146b6a20fd4e0], motion: [0x3f389374bc6a7efa, 0xbfc4189374bc6a7f, 0x3f816872b020c49c], rot: [0x3fad03d7, 0xc22cf2ea], in_ground: false, ticks_in_air: 4, increments: 1 }, // (8.3163, 6.1575, 8.6381)
            Expect { t: 9, pos: [0x4020a00000000000, 0x4018a00000000000, 0x4021100000000000], motion: [0x3f389374bc6a7efa, 0xbfc4189374bc6a7f, 0x3f816872b020c49c], rot: [0x3fb40000, 0xc244e000], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3125, 6.1563, 8.5313)
            Expect { t: 10, pos: [0x4020a03126e978d5, 0x4017ff3b645a1cac, 0x4021145a1cac0831], motion: [0x3f369c23be76c8b4, 0xbfc732df55a85879, 0x3f8003eea6e978d5], rot: [0x3fd0ab28, 0xc26304b4], in_ground: false, ticks_in_air: 5, increments: 0 }, // (8.3129, 5.9993, 8.5398)
            Expect { t: 11, pos: [0x4020a05e5f30f5c3, 0x401745a469acd9e8, 0x4021185b1855c28f], motion: [0x3f34cd16ab515204, 0xbfca0da006bbf6bc, 0x3f7d77e01d5ddedb], rot: [0x3fe799d2, 0xc27ba2e4], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3132, 5.8180, 8.5476)
            Expect { t: 12, pos: [0x4020a5943f75f92d, 0x401663c2f11de69b, 0x40211ae7658e81b5], motion: [0x3f30624dd2f1a9fc, 0xbfd0a7ef9db22d0e, 0x3f770a3d70a3d70a], rot: [0x3fd6668c, 0xc28680f7], in_ground: false, ticks_in_air: 6, increments: 2 }, // (8.3234, 5.5974, 8.5525)
            Expect { t: 13, pos: [0x4020aaca1fbafc96, 0x401581e1788ef34e, 0x40211d73b2c740da], motion: [0x3f30624dd2f1a9fc, 0xbfd0a7ef9db22d0e, 0x3f770a3d70a3d70a], rot: [0x3fc53346, 0xc28f307c], in_ground: false, ticks_in_air: 6, increments: 1 }, // (8.3336, 5.3768, 8.5575)
            Expect { t: 14, pos: [0x4020b00000000000, 0x4014a00000000000, 0x4021200000000000], motion: [0x3f30624dd2f1a9fc, 0xbfd0a7ef9db22d0e, 0x3f770a3d70a3d70a], rot: [0x3fb40000, 0xc297e000], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3438, 5.1563, 8.5625)
            Expect { t: 15, pos: [0x4020b00000000000, 0x4014a00000000000, 0x4021200000000000], motion: [0x3f30624dd2f1a9fc, 0xbfd0a7ef9db22d0e, 0x3f770a3d70a3d70a], rot: [0x3fb40000, 0xc297e000], in_ground: true, ticks_in_air: 7, increments: 0 }, // (8.3438, 5.1563, 8.5625)
            Expect { t: 16, pos: [0x4020b001703a020c, 0x4014865d22498106, 0x402120053355847b], motion: [0x3ee52c4ba9b5e4fe, 0xbfaea25e36df22f7, 0x3f03234f3507eebb], rot: [0x4086f298, 0xc29d765f], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.1312, 8.5625)
            Expect { t: 17, pos: [0x4020b000f57c015d, 0x401459936c3100af, 0x40212003778e5852], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x406fee20, 0xc2941995], in_ground: false, ticks_in_air: 0, increments: 2 }, // (8.3438, 5.0875, 8.5625)
            Expect { t: 18, pos: [0x4020b0007abe00ae, 0x40142cc9b6188058, 0x40212001bbc72c29], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x4051f710, 0xc28abcca], in_ground: false, ticks_in_air: 0, increments: 1 }, // (8.3438, 5.0437, 8.5625)
            Expect { t: 19, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40340000, 0xc2816000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 20, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40340000, 0xc2816000], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 21, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf642ae170000000, 0x3eb0cec70a3d70a4], rot: [0x40340000, 0xc2816000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 22, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40340000, 0xc271e000], in_ground: true, ticks_in_air: 0, increments: 2 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 23, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40340000, 0xc2610000], in_ground: true, ticks_in_air: 0, increments: 1 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 24, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40340000, 0xc2502000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 25, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf5a9262c5604189, 0x3eb5ef4a1cac0831], rot: [0x40340000, 0xc2502000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 26, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200003e7cefd], motion: [0x0000000000000000, 0xbf947ae140000000, 0x3e6f3e77e9c7d4fe], rot: [0x40100000, 0xc2268000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 27, pos: [0x4020b00000000000, 0x4014000000000000, 0x40212000029a89fe], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x403a0000, 0xc21c0000], in_ground: false, ticks_in_air: 0, increments: 2 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 28, pos: [0x4020b00000000000, 0x4014000000000000, 0x40212000014d44ff], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40640000, 0xc2118000], in_ground: false, ticks_in_air: 0, increments: 1 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 29, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40870000, 0xc2070000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3438, 5.0000, 8.5625)
            Expect { t: 30, pos: [0x4020b00000000000, 0x4014000000000000, 0x4021200000000000], motion: [0x0000000000000000, 0xbf945a1cac083127, 0x3f20624dd2f1a9fc], rot: [0x40870000, 0xc2070000], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.3438, 5.0000, 8.5625)
        ],
    },
        // server t=1 pos=(8.2729, 5.0261, 8.5991) motion=(-0.0815, -1.4112, 0.1648) inGround=false caught=false
        // server t=2 pos=(8.2729, 5.0261, 8.5991) motion=(-0.0815, -1.4112, 0.1648) inGround=true caught=false
        // server t=3 pos=(8.2729, 5.0261, 8.5991) motion=(-0.0086, -0.1581, 0.0307) inGround=true caught=false
        // server t=4 pos=(8.2723, 5.0254, 8.6028) motion=(-0.0006, -0.0374, 0.0034) inGround=false caught=false
        // server t=5 pos=(8.2723, 5.0254, 8.6028) motion=(-0.0006, -0.0374, 0.0034) inGround=true caught=false
        // server t=6 pos=(8.2722, 5.0202, 8.6029) motion=(-0.0000, -0.0416, 0.0001) inGround=false caught=false
        // server t=7 pos=(8.2722, 5.0202, 8.6029) motion=(-0.0000, -0.0416, 0.0001) inGround=true caught=false
        // server t=8 pos=(8.2722, 5.0120, 8.6029) motion=(-0.0000, -0.0444, 0.0000) inGround=false caught=false
        // server t=9 pos=(8.2722, 5.0120, 8.6029) motion=(-0.0000, -0.0444, 0.0000) inGround=true caught=false
        // server t=10 pos=(8.2722, 5.0057, 8.6029) motion=(-0.0000, -0.0425, 0.0000) inGround=false caught=false
        // server t=11 pos=(8.2722, 5.0057, 8.6029) motion=(-0.0000, -0.0425, 0.0000) inGround=true caught=false
        // server t=12 pos=(8.2722, 5.0013, 8.6029) motion=(-0.0000, -0.0409, 0.0000) inGround=false caught=false
        // server t=13 pos=(8.2722, 5.0013, 8.6029) motion=(-0.0000, -0.0409, 0.0000) inGround=true caught=false
        // server t=14 pos=(8.2722, 5.0013, 8.6029) motion=(-0.0000, -0.0042, 0.0000) inGround=true caught=false
        // server t=15 pos=(8.2722, 5.0012, 8.6029) motion=(-0.0000, -0.0369, 0.0000) inGround=false caught=false
        // server t=16 pos=(8.2722, 5.0012, 8.6029) motion=(-0.0000, -0.0369, 0.0000) inGround=true caught=false
        // server t=17 pos=(8.2722, 5.0012, 8.6029) motion=(-0.0000, -0.0042, 0.0000) inGround=true caught=false
        // server t=18 pos=(8.2722, 5.0005, 8.6029) motion=(-0.0000, -0.0374, 0.0000) inGround=false caught=false
        // server t=19 pos=(8.2722, 5.0005, 8.6029) motion=(-0.0000, -0.0374, 0.0000) inGround=true caught=false
        // server t=20 pos=(8.2722, 5.0002, 8.6029) motion=(-0.0000, -0.0371, 0.0000) inGround=false caught=false
        // server t=21 pos=(8.2722, 5.0002, 8.6029) motion=(-0.0000, -0.0371, 0.0000) inGround=true caught=false
        // server t=22 pos=(8.2722, 5.0002, 8.6029) motion=(-0.0000, -0.0073, 0.0000) inGround=true caught=false
        // server t=23 pos=(8.2722, 5.0002, 8.6029) motion=(-0.0000, -0.0006, 0.0000) inGround=true caught=false
        // server t=24 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0369, 0.0000) inGround=false caught=false
        // server t=25 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0369, 0.0000) inGround=true caught=false
        // server t=26 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0068, 0.0000) inGround=true caught=false
        // server t=27 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0011, 0.0000) inGround=true caught=false
        // server t=28 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0368, 0.0000) inGround=false caught=false
        // server t=29 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0368, 0.0000) inGround=true caught=false
        // server t=30 pos=(8.2722, 5.0001, 8.6029) motion=(-0.0000, -0.0065, 0.0000) inGround=true caught=false
    HookRef {
        name: "down_at_feet",
        // angler (8.5, 5.0, 8.5) yaw 30.0 pitch 82.0; prev_pos_at_spawn=(0.0, 0.0, 0.0)
        angler: [0x4021000000000000, 0x4014000000000000, 0x4021000000000000],
        stone: &[([0, 4, 0], [15, 4, 15]), ],
        spawn: [267, 208, 269],
        spawn_look: (-19, -59),
        spawn_velocity: [-708, -11951, 1432],
        client_seed: 22,
        packets: &[
            (2, Pkt::Vel([-651, -11289, 1318])),
            (7, Pkt::Vel([0, -332, 0])),
            (7, Pkt::Rel([0, 0, 0], Some((-18, -52)), false)),
            (12, Pkt::Rel([0, 0, 0], Some((-15, -56)), false)),
            (22, Pkt::Rel([0, 0, 0], Some((-15, -61)), false)),
            (27, Pkt::Vel([0, -54, 0])),
            (27, Pkt::Rel([0, 0, 0], Some((-24, -62)), false)),
        ],
        ticks: &[
            Expect { t: 1, pos: [0x402082b020c49ba6, 0x40140645a1cac083, 0x40212ba5e353f7cf], motion: [0xbfb4d7f0f3958106, 0xbff69421c6eacd9f, 0x3fc5143bfd916872], rot: [0xc1d517d8, 0xc2a5b461], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.2553, 5.0061, 8.5853)
            Expect { t: 2, pos: [0x402082b020c49ba6, 0x40140645a1cac083, 0x40212ba5e353f7cf], motion: [0xbfb4d4fdf3b645a2, 0xbff693f7ced91687, 0x3fc516872b020c4a], rot: [0xc1d517d8, 0xc2a5b461], in_ground: true, ticks_in_air: 2, increments: 0 }, // (8.2553, 5.0061, 8.5853)
            Expect { t: 3, pos: [0x402082b020c49ba6, 0x40140645a1cac083, 0x40212ba5e353f7cf], motion: [0xbf8867e4c76c8b44, 0xbf92c8301d70a3d7, 0x3f9d401a6428f5c3], rot: [0xc1d517d8, 0xc2a5b461], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.2553, 5.0061, 8.5853)
            Expect { t: 4, pos: [0x40208235f99fdfdb, 0x401403d1705aa048, 0x40212c061520e480], motion: [0xbf4c185cf64d7c21, 0xbfa3f86b1d7ef06a, 0x3f461fef28b149b5], rot: [0xc1fd528c, 0xc29e0781], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.2543, 5.0037, 8.5860)
            Expect { t: 5, pos: [0x40208235f99fdfdb, 0x401403d1705aa048, 0x40212c061520e480], motion: [0xbf4c185cf64d7c21, 0xbfa3f86b1d7ef06a, 0x3f461fef28b149b5], rot: [0xc1fd528c, 0xc29e0781], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.2543, 5.0037, 8.5860)
            Expect { t: 6, pos: [0x40208235f99fdfdb, 0x401403d1705aa048, 0x40212c061520e480], motion: [0xbf1ae798ade1ad99, 0xbf70f72a4bcb2b83, 0x3f0d5a642e26851e], rot: [0xc1fd528c, 0xc29e0781], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.2543, 5.0037, 8.5860)
            Expect { t: 7, pos: [0x4020917951153fe7, 0x4016028ba03c6adb, 0x40210d59636b4300], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc1ec61b3, 0xc29a1a56], in_ground: true, ticks_in_air: 0, increments: 2 }, // (8.2841, 5.5025, 8.5261)
            Expect { t: 8, pos: [0x4020a0bca88a9ff4, 0x40180145d01e356e, 0x4020eeacb1b5a180], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc1db70da, 0xc2962d2b], in_ground: true, ticks_in_air: 0, increments: 1 }, // (8.3139, 6.0012, 8.4662)
            Expect { t: 9, pos: [0x4020b00000000000, 0x401a000000000000, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc1ca8000, 0xc2924000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.3438, 6.5000, 8.4063)
            Expect { t: 10, pos: [0x4020b00000000000, 0x4019f83e4ad28f5c, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa668e009cd7ff7, 0x0000000000000000], rot: [0xc1a20000, 0xc2990000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3438, 6.4924, 8.4063)
            Expect { t: 11, pos: [0x4020b00000000000, 0x4019cb6c8abef45c, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfb3baaff9dfb216, 0x0000000000000000], rot: [0xc181999a, 0xc29e6666], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3438, 6.4487, 8.4063)
            Expect { t: 12, pos: [0x4020b00000000000, 0x4019dcf30729f83d, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc18ea667, 0xc29e1999], in_ground: false, ticks_in_air: 1, increments: 2 }, // (8.3438, 6.4658, 8.4063)
            Expect { t: 13, pos: [0x4020b00000000000, 0x4019ee798394fc1e, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc19bb334, 0xc29dcccc], in_ground: false, ticks_in_air: 1, increments: 1 }, // (8.3438, 6.4829, 8.4063)
            Expect { t: 14, pos: [0x4020b00000000000, 0x401a000000000000, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc1a8c000, 0xc29d8000], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3438, 6.5000, 8.4063)
            Expect { t: 15, pos: [0x4020b00000000000, 0x4019d5810624dd2f, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfb331e3aa29c779, 0x0000000000000000], rot: [0xc1870000, 0xc2a20000], in_ground: false, ticks_in_air: 2, increments: 0 }, // (8.3438, 6.4585, 8.4063)
            Expect { t: 16, pos: [0x4020b00000000000, 0x401988b9777c3611, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfbb1480c75a16cc, 0x0000000000000000], rot: [0xc1580000, 0xc2a5999a], in_ground: false, ticks_in_air: 3, increments: 0 }, // (8.3438, 6.3835, 8.4063)
            Expect { t: 17, pos: [0x4020b00000000000, 0x40191c67745ecdb6, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfc12ad0589b5020, 0x0000000000000000], rot: [0xc12ccccd, 0xc2a87ae2], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3438, 6.2777, 8.4063)
            Expect { t: 18, pos: [0x4020b00000000000, 0x40189310f199f335, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfc4811759adb7a9, 0x0000000000000000], rot: [0xc10a3d71, 0xc2aac8b5], in_ground: false, ticks_in_air: 5, increments: 0 }, // (8.3438, 6.1436, 8.4063)
            Expect { t: 19, pos: [0x4020b00000000000, 0x4017ef0836cc8578, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfc79306c1ffc465, 0x0000000000000000], rot: [0xc0dd2f1b, 0xc2aca091], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3438, 5.9834, 8.4063)
            Expect { t: 20, pos: [0x4020b00000000000, 0x4017327000bc8755, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfca661637504658, 0x0000000000000000], rot: [0xc0b0f27c, 0xc2ae1a0e], in_ground: false, ticks_in_air: 7, increments: 0 }, // (8.3438, 5.7993, 8.4063)
            Expect { t: 21, pos: [0x4020b00000000000, 0x40165f3f4f020522, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfccff4d6697ffaa, 0x0000000000000000], rot: [0xc08d8eca, 0xc2af480b], in_ground: false, ticks_in_air: 8, increments: 0 }, // (8.3438, 5.5930, 8.4063)
            Expect { t: 22, pos: [0x4020b00000000000, 0x401794d4df56ae17, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc11faf99, 0xc2ae0ab2], in_ground: false, ticks_in_air: 8, increments: 2 }, // (8.3438, 5.8953, 8.4063)
            Expect { t: 23, pos: [0x4020b00000000000, 0x4018ca6a6fab570c, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc17897cc, 0xc2accd59], in_ground: false, ticks_in_air: 8, increments: 1 }, // (8.3438, 6.1977, 8.4063)
            Expect { t: 24, pos: [0x4020b00000000000, 0x401a000000000000, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa53f7ced916873, 0x0000000000000000], rot: [0xc1a8c000, 0xc2ab9000], in_ground: false, ticks_in_air: 8, increments: 0 }, // (8.3438, 6.5000, 8.4063)
            Expect { t: 25, pos: [0x4020b00000000000, 0x4019d5810624dd2f, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfb331e3aa29c779, 0x0000000000000000], rot: [0xc1870000, 0xc2ad4000], in_ground: false, ticks_in_air: 9, increments: 0 }, // (8.3438, 6.4585, 8.4063)
            Expect { t: 26, pos: [0x4020b00000000000, 0x401988b9777c3611, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfbb1480c75a16cc, 0x0000000000000000], rot: [0xc1580000, 0xc2ae999a], in_ground: false, ticks_in_air: 10, increments: 0 }, // (8.3438, 6.3835, 8.4063)
            Expect { t: 27, pos: [0x4020b00000000000, 0x4019b07ba4fd7961, 0x4020d00000000000], motion: [0x0000000000000000, 0xbf7ba5e353f7ced9, 0x0000000000000000], rot: [0xc1a20000, 0xc2ae8667], in_ground: false, ticks_in_air: 10, increments: 2 }, // (8.3438, 6.4223, 8.4063)
            Expect { t: 28, pos: [0x4020b00000000000, 0x4019d83dd27ebcb0, 0x4020d00000000000], motion: [0x0000000000000000, 0xbf7ba5e353f7ced9, 0x0000000000000000], rot: [0xc1d80000, 0xc2ae7334], in_ground: false, ticks_in_air: 10, increments: 1 }, // (8.3438, 6.4612, 8.4063)
            Expect { t: 29, pos: [0x4020b00000000000, 0x401a000000000000, 0x4020d00000000000], motion: [0x0000000000000000, 0xbf7ba5e353f7ced9, 0x0000000000000000], rot: [0xc2070000, 0xc2ae6000], in_ground: false, ticks_in_air: 10, increments: 0 }, // (8.3438, 6.5000, 8.4063)
            Expect { t: 30, pos: [0x4020b00000000000, 0x4019f916872b020c, 0x4020d00000000000], motion: [0x0000000000000000, 0xbfa605681e703af9, 0x0000000000000000], rot: [0xc1d80000, 0xc2af8000], in_ground: false, ticks_in_air: 11, increments: 0 }, // (8.3438, 6.4933, 8.4063)
        ],
    },
        // server t=1 pos=(8.8619, 6.9004, 3.9299) motion=(0.4712, 0.3132, 1.2652) inGround=false caught=false
        // server t=2 pos=(9.3331, 7.2136, 5.1951) motion=(0.4335, 0.2513, 1.1640) inGround=false caught=false
        // server t=3 pos=(9.7666, 7.4649, 6.3590) motion=(0.3988, 0.1944, 1.0708) inGround=false caught=false
        // server t=4 pos=(10.1655, 7.6593, 7.4298) motion=(0.3669, 0.1421, 0.9852) inGround=false caught=false
        // server t=5 pos=(10.5324, 7.8013, 8.4150) motion=(0.3376, 0.0939, 0.9064) inGround=false caught=false
        // server t=6 pos=(10.8700, 7.8952, 9.3214) motion=(0.3106, 0.0496, 0.8338) inGround=false caught=false
        // server t=7 pos=(11.1806, 7.9448, 10.1552) motion=(0.2857, 0.0088, 0.7671) inGround=false caught=false
        // server t=8 pos=(11.4663, 7.9536, 10.9224) motion=(0.2629, -0.0287, 0.7058) inGround=false caught=false
        // server t=9 pos=(11.7292, 7.9249, 11.6281) motion=(0.2418, -0.0632, 0.6493) inGround=false caught=false
        // server t=10 pos=(11.9710, 7.8617, 12.2774) motion=(0.2225, -0.0949, 0.5974) inGround=false caught=false
        // server t=11 pos=(12.1935, 7.7668, 12.8748) motion=(0.2047, -0.1241, 0.5496) inGround=false caught=false
        // server t=12 pos=(12.3982, 7.6426, 13.4244) motion=(0.1883, -0.1510, 0.5056) inGround=false caught=false
        // server t=13 pos=(12.5865, 7.4916, 13.9300) motion=(0.1733, -0.1757, 0.4652) inGround=false caught=false
        // server t=14 pos=(12.7598, 7.3159, 14.3951) motion=(0.1594, -0.1985, 0.4279) inGround=false caught=false
        // server t=15 pos=(12.9192, 7.1174, 14.8231) motion=(0.1466, -0.2194, 0.3937) inGround=false caught=false
        // server t=16 pos=(13.0658, 6.8980, 15.2168) motion=(0.1349, -0.2386, 0.3622) inGround=false caught=false
        // server t=17 pos=(13.2007, 6.6594, 15.5790) motion=(0.1241, -0.2564, 0.3332) inGround=false caught=false
        // server t=18 pos=(13.3249, 6.4030, 15.9122) motion=(0.1142, -0.2726, 0.3066) inGround=false caught=false
        // server t=19 pos=(13.4391, 6.1304, 16.2188) motion=(0.1051, -0.2876, 0.2821) inGround=false caught=false
        // server t=20 pos=(13.5441, 5.8428, 16.5009) motion=(0.0966, -0.3014, 0.2595) inGround=false caught=false
        // server t=21 pos=(13.6408, 5.5413, 16.7604) motion=(0.0889, -0.3141, 0.2387) inGround=false caught=false
        // server t=22 pos=(13.7297, 5.2272, 16.9991) motion=(0.0818, -0.3258, 0.2196) inGround=false caught=false
        // server t=23 pos=(13.7297, 5.2272, 16.9991) motion=(0.0818, -0.3258, 0.2196) inGround=true caught=false
        // server t=24 pos=(13.7419, 5.2184, 17.0416) motion=(0.0112, -0.0449, 0.0391) inGround=false caught=false
        // server t=25 pos=(13.7531, 5.1735, 17.0808) motion=(0.0103, -0.0781, 0.0360) inGround=false caught=false
        // server t=26 pos=(13.7635, 5.0953, 17.1168) motion=(0.0095, -0.1087, 0.0331) inGround=false caught=false
        // server t=27 pos=(13.7635, 5.0953, 17.1168) motion=(0.0095, -0.1087, 0.0331) inGround=true caught=false
        // server t=28 pos=(13.7649, 5.0846, 17.1226) motion=(0.0013, -0.0467, 0.0054) inGround=false caught=false
        // server t=29 pos=(13.7662, 5.0380, 17.1280) motion=(0.0012, -0.0797, 0.0049) inGround=false caught=false
        // server t=30 pos=(13.7662, 5.0380, 17.1280) motion=(0.0012, -0.0797, 0.0049) inGround=true caught=false
        // server t=31 pos=(13.7663, 5.0360, 17.1285) motion=(0.0000, -0.0387, 0.0005) inGround=false caught=false
        // server t=32 pos=(13.7663, 5.0360, 17.1285) motion=(0.0000, -0.0387, 0.0005) inGround=true caught=false
        // server t=33 pos=(13.7663, 5.0285, 17.1286) motion=(0.0000, -0.0436, 0.0001) inGround=false caught=false
        // server t=34 pos=(13.7663, 5.0285, 17.1286) motion=(0.0000, -0.0436, 0.0001) inGround=true caught=false
        // server t=35 pos=(13.7663, 5.0258, 17.1286) motion=(0.0000, -0.0393, 0.0000) inGround=false caught=false
        // server t=36 pos=(13.7663, 5.0258, 17.1286) motion=(0.0000, -0.0393, 0.0000) inGround=true caught=false
        // server t=37 pos=(13.7663, 5.0226, 17.1286) motion=(0.0000, -0.0398, 0.0000) inGround=false caught=false
        // server t=38 pos=(13.7663, 5.0226, 17.1286) motion=(0.0000, -0.0398, 0.0000) inGround=true caught=false
        // server t=39 pos=(13.7663, 5.0172, 17.1286) motion=(0.0000, -0.0418, 0.0000) inGround=false caught=false
        // server t=40 pos=(13.7663, 5.0172, 17.1286) motion=(0.0000, -0.0418, 0.0000) inGround=true caught=false
    HookRef {
        name: "forward_cast",
        // angler (8.5, 5.0, 2.5) yaw -20.0 pitch -15.0; prev_pos_at_spawn=(0.0, 0.0, 0.0)
        angler: [0x4021000000000000, 0x4014000000000000, 0x4004000000000000],
        stone: &[([-16, 4, -16], [31, 4, 47]), ],
        spawn: [267, 208, 81],
        spawn_look: (14, 10),
        spawn_velocity: [4097, 3043, 11001],
        client_seed: 32,
        packets: &[
            (2, Pkt::Vel([3769, 2505, 10121])),
            (7, Pkt::Vel([2484, 396, 6670])),
            (7, Pkt::Tp([347, 252, 298], 14, 7, false)),
            (12, Pkt::Vel([1637, -993, 4396])),
            (12, Pkt::Rel([43, -4, 113], Some((14, 0)), false)),
            (17, Pkt::Vel([1079, -1909, 2897])),
            (17, Pkt::Rel([28, -28, 75], Some((14, -10)), false)),
            (22, Pkt::Vel([711, -2512, 1909])),
            (22, Pkt::Rel([18, -43, 50], Some((14, -24)), false)),
            (27, Pkt::Vel([76, -869, 265])),
            (27, Pkt::Rel([4, -14, 11], Some((12, -29)), false)),
            (32, Pkt::Vel([0, -309, 3])),
            (32, Pkt::Rel([0, 0, 0], Some((10, -41)), false)),
            (37, Pkt::Rel([0, 0, 0], Some((15, -49)), false)),
        ],
        ticks: &[
            Expect { t: 1, pos: [0x4021b6353f7ced91, 0x401b85810624dd2f, 0x400f404189374bc6], motion: [0x3fde276756439582, 0x3fd40a915a758e22, 0x3ff43de94013f7cf], rot: [0x419eae9d, 0x416280c4], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.8559, 6.8804, 3.9064)
            Expect { t: 2, pos: [0x4022a76c8b439581, 0x401cc624dd2f1aa0, 0x4014af9db22d0e56], motion: [0x3fdbbd662fed9169, 0x3fd014e3c29a6b51, 0x3ff29f655b7020c5], rot: [0x419fa026, 0x415efd61], in_ground: false, ticks_in_air: 2, increments: 0 }, // (9.3270, 7.1935, 5.1715)
            Expect { t: 3, pos: [0x40238557bcc3020c, 0x401dc7731958c155, 0x4019577709091687], motion: [0x3fd9854990055ab6, 0x3fc8e136fb14fcec, 0x3ff122011be5b568], rot: [0x41a0615c, 0x4156fde5], in_ground: false, ticks_in_air: 3, increments: 0 }, // (9.7604, 7.4448, 6.3354)
            Expect { t: 4, pos: [0x4024518209432ce2, 0x401e8e7cd131693c, 0x401d9ff7500283e1], motion: [0x3fd77a9fdd90a57e, 0x3fc22dd0b7687ccc, 0x3fef863f849afc7d], rot: [0x41a0fbf7, 0x414ae2f6], in_ground: false, ticks_in_air: 4, increments: 0 }, // (10.1592, 7.6391, 7.4062)
            Expect { t: 5, pos: [0x40250d57082fb20e, 0x401f1feb56ecad22, 0x4020c85fa04af1b8], motion: [0x3fd599c64d4b3fe5, 0x3fb8074ba6dd30bc, 0x3fed00a0defbdb5b], rot: [0x41a177a8, 0x413aee3e], in_ground: false, ticks_in_air: 5, increments: 0 }, // (10.5261, 7.7812, 8.3914)
            Expect { t: 6, pos: [0x4025ba253a9a0c0d, 0x401f8008858821e5, 0x40229869ae3aaf6e], motion: [0x3fd3df648a993f1c, 0x3fa95eef331244f7, 0x3feaaea883588636], rot: [0x41a1da8f, 0x41274887], in_ground: false, ticks_in_air: 6, increments: 0 }, // (10.8636, 7.8750, 9.2977)
            Expect { t: 7, pos: [0x4025ba253a9a0c0d, 0x401f8008858821e5, 0x40229869ae3aaf6e], motion: [0x3fd3df3b645a1cac, 0x3fa95810624dd2f2, 0x3feaae147ae147ae], rot: [0x41a0670a, 0x412405af], in_ground: false, ticks_in_air: 6, increments: 2 }, // (10.8636, 7.8750, 9.2977)
            Expect { t: 8, pos: [0x4025ba253a9a0c0d, 0x401f8008858821e5, 0x40229869ae3aaf6e], motion: [0x3fd3df3b645a1cac, 0x3fa95810624dd2f2, 0x3feaae147ae147ae], rot: [0x419ef385, 0x4120c2d8], in_ground: false, ticks_in_air: 6, increments: 1 }, // (10.8636, 7.8750, 9.2977)
            Expect { t: 9, pos: [0x4025ba253a9a0c0d, 0x401f8008858821e5, 0x40229869ae3aaf6e], motion: [0x3fd3df3b645a1cac, 0x3fa95810624dd2f2, 0x3feaae147ae147ae], rot: [0x419d8000, 0x411d8000], in_ground: false, ticks_in_air: 6, increments: 0 }, // (10.8636, 7.8750, 9.2977)
            Expect { t: 10, pos: [0x4026591f15bcdcf2, 0x401fb2b8a64cbd8b, 0x4024434af5e8c3e9], motion: [0x3fd24840e7020c4a, 0x3f81e647130be0e9, 0x3fe88bac78851eb8], rot: [0x419eae7d, 0x410830bb], in_ground: false, ticks_in_air: 7, increments: 0 }, // (11.1741, 7.9245, 10.1314)
            Expect { t: 11, pos: [0x4026eb611cf4ed54, 0x401fbbabc9d6437b, 0x4025cc05bd7115d4], motion: [0x3fd0d1d55486c152, 0xbf9d730d29295988, 0x3fe694fadc26b2f4], rot: [0x419fa06f, 0x40ddd210], in_ground: false, ticks_in_air: 8, increments: 0 }, // (11.4597, 7.9333, 10.8985)
            Expect { t: 12, pos: [0x40276796134df38d, 0x401f7d1d31398252, 0x40271803d3a0b938], motion: [0x3fca3126e978d4fe, 0xbfbfc6a7ef9db22d, 0x3fe195810624dd2f], rot: [0x419eeaf5, 0x4093e160], in_ground: false, ticks_in_air: 8, increments: 2 }, // (11.7023, 7.8722, 11.5469)
            Expect { t: 13, pos: [0x4027e3cb09a6f9c6, 0x401f3e8e989cc129, 0x40286401e9d05c9c], motion: [0x3fca3126e978d4fe, 0xbfbfc6a7ef9db22d, 0x3fe195810624dd2f], rot: [0x419e357a, 0x4013e160], in_ground: false, ticks_in_air: 8, increments: 1 }, // (11.9449, 7.8111, 12.1953)
            Expect { t: 14, pos: [0x4028600000000000, 0x401f000000000000, 0x4029b00000000000], motion: [0x3fca3126e978d4fe, 0xbfbfc6a7ef9db22d, 0x3fe195810624dd2f], rot: [0x419d8000, 0x00000000], in_ground: false, ticks_in_air: 8, increments: 0 }, // (12.1875, 7.7500, 12.8438)
            Expect { t: 15, pos: [0x4028c8c49ba5e354, 0x401e80e560418937, 0x402ac95810624dd3], motion: [0x3fc818bd6d7ced91, 0xbfc353cde18bac70, 0x3fe02d623dc6a7f0], rot: [0x419eaddc, 0xc018fd39], in_ground: false, ticks_in_air: 9, increments: 0 }, // (12.3921, 7.6259, 13.3933)
            Expect { t: 16, pos: [0x40292927915bd70a, 0x401de646f1352bd4, 0x402bcc2e343eb852], motion: [0x3fc62b3da8ea803e, 0xbfc67dd7a51134db, 0x3fddc425707cb3b3], rot: [0x419f9f5f, 0xc0a145ef], in_ground: false, ticks_in_air: 10, increments: 0 }, // (12.5804, 7.4749, 13.8988)
            Expect { t: 17, pos: [0x40297b6fb63d3a07, 0x401d1984a0ce1d38, 0x402ca81ecd7f258c], motion: [0x3fc14395810624dd, 0xbfce8b4395810625, 0x3fd72d0e56041893], rot: [0x419eea3f, 0xc100c1fa], in_ground: false, ticks_in_air: 10, increments: 2 }, // (12.7411, 7.2749, 14.3284)
            Expect { t: 18, pos: [0x4029cdb7db1e9d04, 0x401c4cc250670e9c, 0x402d840f66bf92c6], motion: [0x3fc14395810624dd, 0xbfce8b4395810625, 0x3fd72d0e56041893], rot: [0x419e3520, 0xc130e0fd], in_ground: false, ticks_in_air: 10, increments: 1 }, // (12.9018, 7.0750, 14.7579)
            Expect { t: 19, pos: [0x402a200000000000, 0x401b800000000000, 0x402e600000000000], motion: [0x3fc14395810624dd, 0xbfce8b4395810625, 0x3fd72d0e56041893], rot: [0x419d8000, 0xc1610000], in_ground: false, ticks_in_air: 10, increments: 0 }, // (13.0625, 6.8750, 15.1875)
            Expect { t: 20, pos: [0x402a650e56041893, 0x401a8ba5e353f7cf, 0x402f196872b020c5], motion: [0x3fbfc408e2978d50, 0xbfd067caee9096bb, 0x3fd552695fdd2f1a], rot: [0x419eaf48, 0xc18cb6af], in_ground: false, ticks_in_air: 11, increments: 0 }, // (13.1974, 6.6364, 15.5496)
            Expect { t: 21, pos: [0x402aa49667c947ae, 0x40198529346aee63, 0x402fc3fbbdaf0a3e], motion: [0x3fbd3978d95beae5, 0xbfd172bd94f32f37, 0x3fd39dbd20b99ac2], rot: [0x419fa1f3, 0xc1a9d6da], in_ground: false, ticks_in_air: 12, increments: 0 }, // (13.3215, 6.3800, 15.8828)
            Expect { t: 22, pos: [0x402ad8644530da74, 0x40186370cd9c9eed, 0x40302bfe948fae15], motion: [0x3fb6c083126e978d, 0xbfd4189374bc6a7f, 0x3fce8b4395810625], rot: [0x419eebf7, 0xc1cb39e7], in_ground: false, ticks_in_air: 12, increments: 2 }, // (13.4226, 6.0971, 16.1719)
            Expect { t: 23, pos: [0x402b0c3222986d3a, 0x401741b866ce4f76, 0x403075ff4a47d70a], motion: [0x3fb6c083126e978d, 0xbfd4189374bc6a7f, 0x3fce8b4395810625], rot: [0x419e35fc, 0xc1ec9cf4], in_ground: false, ticks_in_air: 12, increments: 1 }, // (13.5238, 5.8142, 16.4609)
            Expect { t: 24, pos: [0x402b400000000000, 0x4016200000000000, 0x4030c00000000000], motion: [0x3fb6c083126e978d, 0xbfd4189374bc6a7f, 0x3fce8b4395810625], rot: [0x419d8000, 0xc2070000], in_ground: false, ticks_in_air: 12, increments: 0 }, // (13.6250, 5.5313, 16.7500)
            Expect { t: 25, pos: [0x402b6d810624dd2f, 0x4014de76c8b43958, 0x4030fd16872b020c], motion: [0x3fb4ee8d1753f7cf, 0xbfd4d7f0f2b367a1, 0x3fcc19b917374bc7], rot: [0x419eaf23, 0xc214c491], in_ground: false, ticks_in_air: 13, increments: 0 }, // (13.7139, 5.2173, 16.9886)
            Expect { t: 26, pos: [0x402b6d810624dd2f, 0x4014de76c8b43958, 0x4030fd16872b020c], motion: [0x3fb4ee8d1753f7cf, 0xbfd4d7f0f2b367a1, 0x3fcc19b917374bc7], rot: [0x419eaf23, 0xc214c491], in_ground: true, ticks_in_air: 14, increments: 0 }, // (13.7139, 5.2173, 16.9886)
            Expect { t: 27, pos: [0x402b73ab596de8ca, 0x4014b44f3078263b, 0x4031060f04c756b3], motion: [0x3f8374bc6a7ef9db, 0xbfbbced916872b02, 0x3fa0f5c28f5c28f6], rot: [0x4196ca17, 0xc2198db6], in_ground: true, ticks_in_air: 14, increments: 2 }, // (13.7259, 5.1761, 17.0237)
            Expect { t: 28, pos: [0x402b79d5acb6f465, 0x40148a27983c131e, 0x40310f078263ab5a], motion: [0x3f8374bc6a7ef9db, 0xbfbbced916872b02, 0x3fa0f5c28f5c28f6], rot: [0x418ee50c, 0xc21e56db], in_ground: true, ticks_in_air: 14, increments: 1 }, // (13.7380, 5.1349, 17.0587)
            Expect { t: 29, pos: [0x402b800000000000, 0x4014600000000000, 0x4031180000000000], motion: [0x3f8374bc6a7ef9db, 0xbfbbced916872b02, 0x3fa0f5c28f5c28f6], rot: [0x41870000, 0xc2232000], in_ground: true, ticks_in_air: 14, increments: 0 }, // (13.7500, 5.0938, 17.0938)
            Expect { t: 30, pos: [0x402b80b552f020c5, 0x401456337a499581, 0x4031192124810000], motion: [0x3f54da2d8f3c12a0, 0xbfa7595eaf7f633d, 0x3f70a02dd65709f0], rot: [0x4187dac9, 0xc235de5b], in_ground: false, ticks_in_air: 0, increments: 0 }, // (13.7514, 5.0842, 17.0982)
            Expect { t: 31, pos: [0x402b815c245c9aa6, 0x40142780bcea96bb, 0x40311a2b275e6571], motion: [0x3f532f1fb29313a7, 0xbfb42950a2628da4, 0x3f6e975e9def9c98], rot: [0x418889c7, 0xc2553b48], in_ground: false, ticks_in_air: 1, increments: 0 }, // (13.7527, 5.0386, 17.1022)
            Expect { t: 32, pos: [0x402b80e8183dbc6f, 0x40143a55d3470f27, 0x403119721a3eee4b], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x41808685, 0xc25b0785], in_ground: false, ticks_in_air: 1, increments: 2 }, // (13.7518, 5.0570, 17.0994)
            Expect { t: 33, pos: [0x402b80740c1ede38, 0x40144d2ae9a38794, 0x403118b90d1f7726], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x41710685, 0xc260d3c2], in_ground: false, ticks_in_air: 1, increments: 1 }, // (13.7509, 5.0754, 17.0966)
            Expect { t: 34, pos: [0x402b800000000000, 0x4014600000000000, 0x4031180000000000], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x41610000, 0xc266a000], in_ground: false, ticks_in_air: 1, increments: 0 }, // (13.7500, 5.0938, 17.0938)
            Expect { t: 35, pos: [0x402b800000000000, 0x40143872b020c49c, 0x403118189374bc6a], motion: [0x0000000000000000, 0xbfb2848bed758e20, 0x3f369c23be76c8b4], rot: [0x41340000, 0xc280070e], in_ground: false, ticks_in_air: 2, increments: 0 }, // (13.7500, 5.0551, 17.0941)
            Expect { t: 36, pos: [0x402b800000000000, 0x40143872b020c49c, 0x403118189374bc6a], motion: [0x0000000000000000, 0xbfb2848bed758e20, 0x3f369c23be76c8b4], rot: [0x41340000, 0xc280070e], in_ground: true, ticks_in_air: 3, increments: 0 }, // (13.7500, 5.0551, 17.0941)
            Expect { t: 37, pos: [0x402b800000000000, 0x401445a1cac08313, 0x40311810624dd2f1], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x41688000, 0xc2834a09], in_ground: true, ticks_in_air: 3, increments: 2 }, // (13.7500, 5.0680, 17.0940)
            Expect { t: 38, pos: [0x402b800000000000, 0x401452d0e560418a, 0x403118083126e978], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x418e8000, 0xc2868d04], in_ground: true, ticks_in_air: 3, increments: 1 }, // (13.7500, 5.0809, 17.0939)
            Expect { t: 39, pos: [0x402b800000000000, 0x4014600000000000, 0x4031180000000000], motion: [0x0000000000000000, 0xbfa3c6a7ef9db22d, 0x3f389374bc6a7efa], rot: [0x41a8c000, 0xc289d000], in_ground: true, ticks_in_air: 3, increments: 0 }, // (13.7500, 5.0938, 17.0938)
            Expect { t: 40, pos: [0x402b800000000000, 0x401458fbe6c4c49c, 0x40311802d8819168], motion: [0x0000000000000000, 0xbfa611a7a9a4b353, 0x3f04f1ce2018e937], rot: [0x41870000, 0xc2921ad0], in_ground: false, ticks_in_air: 0, increments: 0 }, // (13.7500, 5.0869, 17.0938)
        ],
    },
        // server t=1 pos=(8.3469, 8.0097, 8.5504) motion=(0.0063, 1.3338, 0.0464) inGround=false caught=false
        // server t=2 pos=(8.3532, 9.3435, 8.5968) motion=(0.0058, 1.1903, 0.0427) inGround=false caught=false
        // server t=3 pos=(8.3591, 10.5337, 8.6395) motion=(0.0054, 1.0582, 0.0393) inGround=false caught=false
        // server t=4 pos=(8.3645, 11.5920, 8.6787) motion=(0.0049, 0.9368, 0.0361) inGround=false caught=false
        // server t=5 pos=(8.3694, 12.5288, 8.7148) motion=(0.0045, 0.8250, 0.0332) inGround=false caught=false
        // server t=6 pos=(8.3739, 13.3538, 8.7481) motion=(0.0042, 0.7222, 0.0306) inGround=false caught=false
        // server t=7 pos=(8.3781, 14.0760, 8.7786) motion=(0.0038, 0.6277, 0.0281) inGround=false caught=false
        // server t=8 pos=(8.3820, 14.7037, 8.8067) motion=(0.0035, 0.5406, 0.0259) inGround=false caught=false
        // server t=9 pos=(8.3855, 15.2443, 8.8326) motion=(0.0033, 0.4606, 0.0238) inGround=false caught=false
        // server t=10 pos=(8.3888, 15.7049, 8.8564) motion=(0.0030, 0.3869, 0.0219) inGround=false caught=false
        // server t=11 pos=(8.3918, 16.0918, 8.8783) motion=(0.0028, 0.3192, 0.0201) inGround=false caught=false
        // server t=12 pos=(8.3945, 16.4110, 8.8985) motion=(0.0025, 0.2569, 0.0185) inGround=false caught=false
        // server t=13 pos=(8.3971, 16.6679, 8.9170) motion=(0.0023, 0.1995, 0.0171) inGround=false caught=false
        // server t=14 pos=(8.3994, 16.8674, 8.9341) motion=(0.0021, 0.1467, 0.0157) inGround=false caught=false
        // server t=15 pos=(8.4015, 17.0141, 8.9497) motion=(0.0020, 0.0982, 0.0144) inGround=false caught=false
        // server t=16 pos=(8.4035, 17.1123, 8.9642) motion=(0.0018, 0.0535, 0.0133) inGround=false caught=false
        // server t=17 pos=(8.4053, 17.1659, 8.9775) motion=(0.0017, 0.0125, 0.0122) inGround=false caught=false
        // server t=18 pos=(8.4070, 17.1783, 8.9897) motion=(0.0015, -0.0253, 0.0112) inGround=false caught=false
        // server t=19 pos=(8.4086, 17.1530, 9.0009) motion=(0.0014, -0.0601, 0.0103) inGround=false caught=false
        // server t=20 pos=(8.4100, 17.0929, 9.0112) motion=(0.0013, -0.0921, 0.0095) inGround=false caught=false
        // server t=21 pos=(8.4113, 17.0008, 9.0208) motion=(0.0012, -0.1215, 0.0088) inGround=false caught=false
        // server t=22 pos=(8.4125, 16.8793, 9.0295) motion=(0.0011, -0.1486, 0.0081) inGround=false caught=false
        // server t=23 pos=(8.4136, 16.7307, 9.0376) motion=(0.0010, -0.1735, 0.0074) inGround=false caught=false
        // server t=24 pos=(8.4146, 16.5571, 9.0450) motion=(0.0009, -0.1964, 0.0068) inGround=false caught=false
        // server t=25 pos=(8.4155, 16.3607, 9.0518) motion=(0.0009, -0.2175, 0.0063) inGround=false caught=false
        // server t=26 pos=(8.4164, 16.1432, 9.0581) motion=(0.0008, -0.2369, 0.0058) inGround=false caught=false
        // server t=27 pos=(8.4172, 15.9063, 9.0638) motion=(0.0007, -0.2548, 0.0053) inGround=false caught=false
        // server t=28 pos=(8.4179, 15.6515, 9.0691) motion=(0.0007, -0.2712, 0.0049) inGround=false caught=false
        // server t=29 pos=(8.4186, 15.3803, 9.0740) motion=(0.0006, -0.2863, 0.0045) inGround=false caught=false
        // server t=30 pos=(8.4192, 15.0940, 9.0785) motion=(0.0006, -0.3002, 0.0041) inGround=false caught=false
        // server t=31 pos=(8.4197, 14.7938, 9.0826) motion=(0.0005, -0.3130, 0.0038) inGround=false caught=false
        // server t=32 pos=(8.4203, 14.4809, 9.0864) motion=(0.0005, -0.3247, 0.0035) inGround=false caught=false
        // server t=33 pos=(8.4207, 14.1561, 9.0899) motion=(0.0004, -0.3356, 0.0032) inGround=false caught=false
        // server t=34 pos=(8.4212, 13.8206, 9.0931) motion=(0.0004, -0.3455, 0.0030) inGround=false caught=false
        // server t=35 pos=(8.4216, 13.4751, 9.0961) motion=(0.0004, -0.3547, 0.0027) inGround=false caught=false
        // server t=36 pos=(8.4220, 13.1204, 9.0988) motion=(0.0003, -0.3631, 0.0025) inGround=false caught=false
        // server t=37 pos=(8.4223, 12.7573, 9.1013) motion=(0.0003, -0.3708, 0.0023) inGround=false caught=false
        // server t=38 pos=(8.4226, 12.3864, 9.1036) motion=(0.0003, -0.3780, 0.0021) inGround=false caught=false
        // server t=39 pos=(8.4229, 12.0085, 9.1058) motion=(0.0003, -0.3845, 0.0020) inGround=false caught=false
        // server t=40 pos=(8.4232, 11.6239, 9.1077) motion=(0.0002, -0.3906, 0.0018) inGround=false caught=false
        // server t=41 pos=(8.4234, 11.2333, 9.1095) motion=(0.0002, -0.3961, 0.0017) inGround=false caught=false
        // server t=42 pos=(8.4236, 10.8372, 9.1112) motion=(0.0002, -0.4012, 0.0015) inGround=false caught=false
        // server t=43 pos=(8.4238, 10.4360, 9.1127) motion=(0.0002, -0.4059, 0.0014) inGround=false caught=false
        // server t=44 pos=(8.4240, 10.0300, 9.1141) motion=(0.0002, -0.4103, 0.0013) inGround=false caught=false
        // server t=45 pos=(8.4242, 9.6197, 9.1154) motion=(0.0002, -0.4142, 0.0012) inGround=false caught=false
        // server t=46 pos=(8.4244, 9.2055, 9.1165) motion=(0.0001, -0.4179, 0.0011) inGround=false caught=false
        // server t=47 pos=(8.4245, 8.7876, 9.1176) motion=(0.0001, -0.4213, 0.0010) inGround=false caught=false
        // server t=48 pos=(8.4247, 8.3663, 9.1186) motion=(0.0001, -0.4244, 0.0009) inGround=false caught=false
        // server t=49 pos=(8.4248, 7.9419, 9.1196) motion=(0.0001, -0.4272, 0.0008) inGround=false caught=false
        // server t=50 pos=(8.4249, 7.5147, 9.1204) motion=(0.0001, -0.4298, 0.0008) inGround=false caught=false
        // server t=51 pos=(8.4250, 7.0849, 9.1212) motion=(0.0001, -0.4323, 0.0007) inGround=false caught=false
        // server t=52 pos=(8.4251, 6.6526, 9.1219) motion=(0.0001, -0.4345, 0.0007) inGround=false caught=false
        // server t=53 pos=(8.4252, 6.2181, 9.1226) motion=(0.0001, -0.4365, 0.0006) inGround=false caught=false
        // server t=54 pos=(8.4253, 5.7816, 9.1232) motion=(0.0001, -0.4384, 0.0006) inGround=false caught=false
        // server t=55 pos=(8.4254, 5.3432, 9.1237) motion=(0.0001, -0.4401, 0.0005) inGround=false caught=false
        // server t=56 pos=(8.4254, 5.3432, 9.1237) motion=(0.0001, -0.4401, 0.0005) inGround=true caught=false
        // server t=57 pos=(8.4254, 5.2659, 9.1238) motion=(0.0000, -0.1080, 0.0001) inGround=false caught=false
        // server t=58 pos=(8.4254, 5.1579, 9.1239) motion=(0.0000, -0.1361, 0.0001) inGround=false caught=false
        // server t=59 pos=(8.4254, 5.0217, 9.1240) motion=(0.0000, -0.1620, 0.0001) inGround=false caught=false
        // server t=60 pos=(8.4254, 5.0217, 9.1240) motion=(0.0000, -0.1620, 0.0001) inGround=true caught=false
        // server t=61 pos=(8.4254, 5.0184, 9.1240) motion=(0.0000, -0.0399, 0.0000) inGround=false caught=false
        // server t=62 pos=(8.4254, 5.0184, 9.1240) motion=(0.0000, -0.0399, 0.0000) inGround=true caught=false
        // server t=63 pos=(8.4254, 5.0151, 9.1240) motion=(0.0000, -0.0398, 0.0000) inGround=false caught=false
        // server t=64 pos=(8.4254, 5.0151, 9.1240) motion=(0.0000, -0.0398, 0.0000) inGround=true caught=false
        // server t=65 pos=(8.4254, 5.0090, 9.1240) motion=(0.0000, -0.0424, 0.0000) inGround=false caught=false
        // server t=66 pos=(8.4254, 5.0090, 9.1240) motion=(0.0000, -0.0424, 0.0000) inGround=true caught=false
        // server t=67 pos=(8.4254, 5.0009, 9.1240) motion=(0.0000, -0.0443, 0.0000) inGround=false caught=false
        // server t=68 pos=(8.4254, 5.0009, 9.1240) motion=(0.0000, -0.0443, 0.0000) inGround=true caught=false
        // server t=69 pos=(8.4254, 5.0009, 9.1240) motion=(0.0000, -0.0017, 0.0000) inGround=true caught=false
        // server t=70 pos=(8.4254, 5.0008, 9.1240) motion=(0.0000, -0.0369, 0.0000) inGround=false caught=false
        // server t=71 pos=(8.4254, 5.0008, 9.1240) motion=(0.0000, -0.0369, 0.0000) inGround=true caught=false
        // server t=72 pos=(8.4254, 5.0007, 9.1240) motion=(0.0000, -0.0369, 0.0000) inGround=false caught=false
        // server t=73 pos=(8.4254, 5.0007, 9.1240) motion=(0.0000, -0.0369, 0.0000) inGround=true caught=false
        // server t=74 pos=(8.4254, 5.0007, 9.1240) motion=(0.0000, -0.0046, 0.0000) inGround=true caught=false
        // server t=75 pos=(8.4254, 5.0006, 9.1240) motion=(0.0000, -0.0369, 0.0000) inGround=false caught=false
    HookRef {
        name: "straight_up",
        // angler (8.5, 5.0, 8.5) yaw 0.0 pitch -88.0; prev_pos_at_spawn=(0.0, 0.0, 0.0)
        angler: [0x4021000000000000, 0x4014000000000000, 0x4021000000000000],
        stone: &[([0, 4, 0], [15, 4, 15]), ],
        spawn: [266, 208, 272],
        spawn_look: (5, 62),
        spawn_velocity: [55, 11917, 403],
        client_seed: 42,
        packets: &[
            (2, Pkt::Vel([50, 10670, 371])),
            (7, Pkt::Vel([33, 5777, 244])),
            (7, Pkt::Tp([267, 427, 279], 5, 62, false)),
            (12, Pkt::Vel([22, 2553, 161])),
            (12, Pkt::Rel([1, 87, 5], None, false)),
            (17, Pkt::Vel([14, 428, 106])),
            (17, Pkt::Rel([0, 33, 2], None, false)),
            (22, Pkt::Vel([9, -972, 70])),
            (22, Pkt::Rel([0, 0, 0], Some((5, 0)), false)),
            (27, Pkt::Vel([6, -1895, 46])),
            (27, Pkt::Rel([1, -31, 3], Some((5, -42)), false)),
            (32, Pkt::Vel([4, -2503, 30])),
            (32, Pkt::Rel([0, -43, 1], Some((5, -57)), false)),
            (37, Pkt::Vel([2, -2904, 20])),
            (37, Pkt::Rel([0, -54, 1], Some((5, -62)), false)),
            (42, Pkt::Vel([1, -3169, 13])),
            (42, Pkt::Rel([0, -60, 0], None, false)),
            (47, Pkt::Vel([1, -3343, 8])),
            (47, Pkt::Rel([0, -65, 0], None, false)),
            (52, Pkt::Rel([0, -68, 0], None, false)),
            (57, Pkt::Vel([0, -3521, 4])),
            (57, Pkt::Rel([0, -56, 0], None, false)),
            (62, Pkt::Vel([0, -319, 0])),
            (62, Pkt::Rel([0, -10, 0], Some((9, -64)), false)),
        ],
        ticks: &[
            Expect { t: 1, pos: [0x4020a3851eb851ec, 0x401ff5604189374c, 0x402119cac083126f], motion: [0x3f79e83e4a3d70a4, 0x3ff556a6a6c96bba, 0x3fa7ba882de353f8], rot: [0x40e5bd16, 0x42aeb7d3], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3194, 7.9896, 8.5504)
            Expect { t: 2, pos: [0x4020a6b851eb851f, 0x4022a5916872b021, 0x40213189374bc6a8], motion: [0x3f778d4fe6666667, 0x3ff30b439b855326, 0x3fa5d82fde041893], rot: [0x40e8ea05, 0x42aef87f], in_ground: false, ticks_in_air: 2, increments: 0 }, // (8.3256, 9.3234, 8.5968)
            Expect { t: 3, pos: [0x4020a9a9fbe851ec, 0x402506f9dbe35a86, 0x402147616729cac1], motion: [0x3f75aaf79d1f601a, 0x3ff0ee829f832337, 0x3fa418cfe6d9e692], rot: [0x40eb7465, 0x42af25e3], in_ground: false, ticks_in_air: 3, increments: 0 }, // (8.3314, 10.5136, 8.6394)
            Expect { t: 4, pos: [0x4020ac5f5adbf5d8, 0x402724ca2fd3beed, 0x40215b7a3710a4a8], motion: [0x3f73ef35bf946a8c, 0x3fedfa088cddab1f, 0x3fa27d3a2bee22d1], rot: [0x40ed7ca2, 0x42af42d2], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3367, 11.5719, 8.6787)
            Expect { t: 5, pos: [0x4020aedd4193e865, 0x4029046ab8a1999f, 0x40216df7713c92cb], motion: [0x3f7256f407c1481d, 0x3fea66a52f8d3be2, 0x3fa10291b2b67800], rot: [0x40ef1ce3, 0x42af515b], in_ground: false, ticks_in_air: 5, increments: 0 }, // (8.3415, 12.5086, 8.7148)
            Expect { t: 6, pos: [0x4020b1282014e08e, 0x402aaad50b9a6d5d, 0x40217efa02ef4943], motion: [0x3f70df5b686e018d, 0x3fe71c7cf76be2af, 0x3f9f4c68481dfb31], rot: [0x40f069f0, 0x42af52d6], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3460, 13.3337, 8.7480)
            Expect { t: 7, pos: [0x4020b1282014e08e, 0x402aaad50b9a6d5d, 0x40217efa02ef4943], motion: [0x3f70e5604189374c, 0x3fe71ba5e353f7cf, 0x3f9f3b645a1cac08], rot: [0x40eb46a0, 0x42af01e4], in_ground: false, ticks_in_air: 6, increments: 2 }, // (8.3460, 13.3337, 8.7480)
            Expect { t: 8, pos: [0x4020b1282014e08e, 0x402aaad50b9a6d5d, 0x40217efa02ef4943], motion: [0x3f70e5604189374c, 0x3fe71ba5e353f7cf, 0x3f9f3b645a1cac08], rot: [0x40e62350, 0x42aeb0f2], in_ground: false, ticks_in_air: 6, increments: 1 }, // (8.3460, 13.3337, 8.7480)
            Expect { t: 9, pos: [0x4020b1282014e08e, 0x402aaad50b9a6d5d, 0x40217efa02ef4943], motion: [0x3f70e5604189374c, 0x3fe71ba5e353f7cf, 0x3f9f3b645a1cac08], rot: [0x40e10000, 0x42ae6000], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3460, 13.3337, 8.7480)
            Expect { t: 10, pos: [0x4020b344cc1d11b5, 0x402c1c8f69cfacda, 0x40218e97b51c5799], motion: [0x3f6f16b125e353f8, 0x3fe414ee3fbbcd36, 0x3f9cbbc2c20c49ba], rot: [0x40e54b3b, 0x42ae8617], in_ground: false, ticks_in_air: 7, increments: 0 }, // (8.3501, 14.0558, 8.7785)
            Expect { t: 11, pos: [0x4020b536372f6fea, 0x402d5dde4dcb69ad, 0x40219cf5967d5dbe], motion: [0x3f6c99ff2b8fd0c6, 0x3fe14c30d156e33e, 0x3f9a6f4ccf0cad90], rot: [0x40e8bacc, 0x42ae95e9], in_ground: false, ticks_in_air: 8, increments: 0 }, // (8.3539, 14.6833, 8.8066)
            Expect { t: 12, pos: [0x4020b8cecf74f547, 0x402e493ede879bc9, 0x4021a8a3b9a8e929], motion: [0x3f66872b020c49ba, 0x3fd46c8b43958106, 0x3f949ba5e353f7cf], rot: [0x40e8bacc, 0x42ae95e9], in_ground: false, ticks_in_air: 8, increments: 2 }, // (8.3610, 15.1431, 8.8294)
            Expect { t: 13, pos: [0x4020bc6767ba7aa4, 0x402f349f6f43cde4, 0x4021b451dcd47494], motion: [0x3f66872b020c49ba, 0x3fd46c8b43958106, 0x3f949ba5e353f7cf], rot: [0x40e8bacc, 0x42ae95e9], in_ground: false, ticks_in_air: 8, increments: 1 }, // (8.3680, 15.6028, 8.8522)
            Expect { t: 14, pos: [0x4020c00000000000, 0x4030100000000000, 0x4021c00000000000], motion: [0x3f66872b020c49ba, 0x3fd46c8b43958106, 0x3f949ba5e353f7cf], rot: [0x40e8bacc, 0x42ae95e9], in_ground: false, ticks_in_air: 8, increments: 0 }, // (8.3750, 16.0625, 8.8750)
            Expect { t: 15, pos: [0x4020c16872b020c5, 0x403061b22d0e5604, 0x4021ca4dd2f1a9fc], motion: [0x3f64b9cb6e978d50, 0x3fd06f5451944674, 0x3f92f598a3b645a2], rot: [0x40ebfbb3, 0x42ae3635], in_ground: false, ticks_in_air: 9, increments: 0 }, // (8.3778, 16.3816, 8.8951)
            Expect { t: 16, pos: [0x4020c2b40f670a3e, 0x4030a36f7e54a71e, 0x4021d3c89f43851f], motion: [0x3f631154c7b535d9, 0x3fc9879fb06f877c, 0x3f91714f0252db71], rot: [0x40ee95f7, 0x42adb45c], in_ground: false, ticks_in_air: 10, increments: 0 }, // (8.3803, 16.6384, 8.9136)
            Expect { t: 17, pos: [0x4020c1cd5f9a06d4, 0x4030ca4a54386f69, 0x4021d7db14d7ae15], motion: [0x3f5cac083126e979, 0x3fab645a1cac0831, 0x3f8b22d0e5604189], rot: [0x40ee95f7, 0x42adb45c], in_ground: false, ticks_in_air: 10, increments: 2 }, // (8.3785, 16.7902, 8.9216)
            Expect { t: 18, pos: [0x4020c0e6afcd036a, 0x4030f1252a1c37b4, 0x4021dbed8a6bd70a], motion: [0x3f5cac083126e979, 0x3fab645a1cac0831, 0x3f8b22d0e5604189], rot: [0x40ee95f7, 0x42adb45c], in_ground: false, ticks_in_air: 10, increments: 1 }, // (8.3768, 16.9420, 8.9295)
            Expect { t: 19, pos: [0x4020c00000000000, 0x4031180000000000, 0x4021e00000000000], motion: [0x3f5cac083126e979, 0x3fab645a1cac0831, 0x3f8b22d0e5604189], rot: [0x40ee95f7, 0x42adb45c], in_ground: false, ticks_in_air: 10, increments: 0 }, // (8.3750, 17.0938, 8.9375)
            Expect { t: 20, pos: [0x4020c0e560418937, 0x403125b22d0e5604, 0x4021e6c8b4395810], motion: [0x3f5a60d45e353f7d, 0x3f896fa85288ce78, 0x3f88f712224dd2f1], rot: [0x40ef052f, 0x42a95a67], in_ground: false, ticks_in_air: 11, increments: 0 }, // (8.3768, 17.1473, 8.9508)
            Expect { t: 21, pos: [0x4020c1b866e47ae1, 0x403128e02218a71e, 0x4021ed0678c1eb85], motion: [0x3f58449a72898a5a, 0xbf99fb8961f12133, 0x3f86f7c907d47539], rot: [0x40ef5e0d, 0x429998fe], in_ground: false, ticks_in_air: 12, increments: 0 }, // (8.3784, 17.1597, 8.9629)
            Expect { t: 22, pos: [0x4020c12599eda741, 0x4031234016bb1a14, 0x4021e8aefb2bf259], motion: [0x3f526e978d4fdf3b, 0xbfbf1a9fbe76c8b4, 0x3f81eb851eb851ec], rot: [0x40ea9409, 0x424ccbfd], in_ground: false, ticks_in_air: 12, increments: 2 }, // (8.3772, 17.1377, 8.9545)
            Expect { t: 23, pos: [0x4020c092ccf6d3a0, 0x40311da00b5d8d0a, 0x4021e4577d95f92c], motion: [0x3f526e978d4fdf3b, 0xbfbf1a9fbe76c8b4, 0x3f81eb851eb851ec], rot: [0x40e5ca04, 0x41cccbfd], in_ground: false, ticks_in_air: 12, increments: 1 }, // (8.3761, 17.1157, 8.9460)
            Expect { t: 24, pos: [0x4020c00000000000, 0x4031180000000000, 0x4021e00000000000], motion: [0x3f526e978d4fdf3b, 0xbfbf1a9fbe76c8b4, 0x3f81eb851eb851ec], rot: [0x40e10000, 0x00000000], in_ground: false, ticks_in_air: 12, increments: 0 }, // (8.3750, 17.0938, 8.9375)
            Expect { t: 25, pos: [0x4020c09374bc6a7f, 0x4030f8e560418937, 0x4021e47ae147ae14], motion: [0x3f50f51aced91687, 0xbfc304ab64710cb2, 0x3f807c84bae147af], rot: [0x40e2e39f, 0xc1895ae5], in_ground: false, ticks_in_air: 13, increments: 0 }, // (8.3761, 16.9723, 8.9463)
            Expect { t: 26, pos: [0x4020c11b1d92e148, 0x4030d2dc0978a71e, 0x4021e89a02766666], motion: [0x3f4f33a200f9fb06, 0xbfc63509d5b9983d, 0x3f7e55c10f2becf2], rot: [0x40e46689, 0xc1f8e19d], in_ground: false, ticks_in_air: 14, increments: 0 }, // (8.3772, 16.8237, 8.9543)
            Expect { t: 27, pos: [0x4020c61213b740db, 0x4030973d5ba5c4bf, 0x4021f5bc01a44444], motion: [0x3f489374bc6a7efa, 0xbfce51eb851eb852, 0x3f778d4fdf3b645a], rot: [0x40e3445b, 0xc221b5df], in_ground: false, ticks_in_air: 14, increments: 2 }, // (8.3869, 16.5908, 8.9800)
            Expect { t: 28, pos: [0x4020cb0909dba06e, 0x40305b9eadd2e260, 0x402202de00d22222], motion: [0x3f489374bc6a7efa, 0xbfce51eb851eb852, 0x3f778d4fdf3b645a], rot: [0x40e2222e, 0xc246faf0], in_ground: false, ticks_in_air: 14, increments: 1 }, // (8.3966, 16.3579, 9.0056)
            Expect { t: 29, pos: [0x4020d00000000000, 0x4030200000000000, 0x4022100000000000], motion: [0x3f489374bc6a7efa, 0xbfce51eb851eb852, 0x3f778d4fdf3b645a], rot: [0x40e10000, 0xc26c4000], in_ground: false, ticks_in_air: 14, increments: 0 }, // (8.4063, 16.1250, 9.0313)
            Expect { t: 30, pos: [0x4020d0624dd2f1aa, 0x402fc6b851eb851f, 0x402212f1a9fbe76d], motion: [0x3f469c23be76c8b4, 0xbfd04d6a1a32617c, 0x3f75aaf796872b02], rot: [0x40e38f94, 0xc281f068], in_ground: false, ticks_in_air: 15, increments: 0 }, // (8.4070, 15.8881, 9.0370)
            Expect { t: 31, pos: [0x4020d0bcbe61eb85, 0x402f444d0119f213, 0x402215a708eeb852], motion: [0x3f44cd16ab515204, 0xbfd15a78fa80a5ae, 0x3f73ef35b98343ef], rot: [0x40e59c0c, 0xc28b78ad], in_ground: false, ticks_in_air: 16, increments: 0 }, // (8.4077, 15.6334, 9.0423)
            Expect { t: 32, pos: [0x4020d07dd44147ae, 0x402eb2de00bbf6b7, 0x4022191a05f47ae1], motion: [0x3f40624dd2f1a9fc, 0xbfd40624dd2f1aa0, 0x3f6eb851eb851eb8], rot: [0x40e412b3, 0xc2926b1e], in_ground: false, ticks_in_air: 16, increments: 2 }, // (8.4072, 15.3493, 9.0490)
            Expect { t: 33, pos: [0x4020d03eea20a3d7, 0x402e216f005dfb5c, 0x40221c8d02fa3d70], motion: [0x3f40624dd2f1a9fc, 0xbfd40624dd2f1aa0, 0x3f6eb851eb851eb8], rot: [0x40e2895a, 0xc2995d8f], in_ground: false, ticks_in_air: 16, increments: 1 }, // (8.4067, 15.0653, 9.0558)
            Expect { t: 34, pos: [0x4020d00000000000, 0x402d900000000000, 0x4022200000000000], motion: [0x3f40624dd2f1a9fc, 0xbfd40624dd2f1aa0, 0x3f6eb851eb851eb8], rot: [0x40e10000, 0xc2a05000], in_ground: false, ticks_in_air: 16, increments: 0 }, // (8.4063, 14.7813, 9.0625)
            Expect { t: 35, pos: [0x4020d04189374bc7, 0x402cefced916872b, 0x402221eb851eb852], motion: [0x3f3e2584fdf3b646, 0xbfd4c6fbd7e48e8a, 0x3f6c432cae147ae1], rot: [0x40e49b2c, 0xc2a3f910], in_ground: false, ticks_in_air: 17, increments: 0 }, // (8.4068, 14.4684, 9.0663)
            Expect { t: 36, pos: [0x4020d07dd44147ae, 0x402c4996fa5762b7, 0x402223afb7e9999a], motion: [0x3f3bbc1e39c1c2b0, 0xbfd5786577136893, 0x3f6a005c5625a685], rot: [0x40e77d9f, 0xc2a6eebe], in_ground: false, ticks_in_air: 18, increments: 0 }, // (8.4072, 14.1437, 9.0697)
            Expect { t: 37, pos: [0x4020d053e2d62fc9, 0x402b9664a6e4ec7a, 0x402227ca7a9bbbbc], motion: [0x3f30624dd2f1a9fc, 0xbfd73b645a1cac08, 0x3f647ae147ae147b], rot: [0x40e553bf, 0xc2a969d4], in_ground: false, ticks_in_air: 18, increments: 2 }, // (8.4069, 13.7937, 9.0777)
            Expect { t: 38, pos: [0x4020d029f16b17e4, 0x402ae3325372763d, 0x40222be53d4dddde], motion: [0x3f30624dd2f1a9fc, 0xbfd73b645a1cac08, 0x3f647ae147ae147b], rot: [0x40e329e0, 0xc2abe4ea], in_ground: false, ticks_in_air: 18, increments: 1 }, // (8.4066, 13.4437, 9.0857)
            Expect { t: 39, pos: [0x4020d00000000000, 0x402a300000000000, 0x4022300000000000], motion: [0x3f30624dd2f1a9fc, 0xbfd73b645a1cac08, 0x3f647ae147ae147b], rot: [0x40e10000, 0xc2ae6000], in_ground: false, ticks_in_air: 18, increments: 0 }, // (8.4063, 13.0938, 9.0938)
            Expect { t: 40, pos: [0x4020d020c49ba5e3, 0x40297624dd2f1aa0, 0x40223147ae147ae1], motion: [0x3f2e2584fdf3b646, 0xbfd7ba882d013a92, 0x3f62d7731eb851ec], rot: [0x40d88bf6, 0xc2af5764], in_ground: false, ticks_in_air: 19, increments: 0 }, // (8.4065, 12.7308, 9.0963)
            Expect { t: 41, pos: [0x4020d03eea20a3d7, 0x4028b8509bc710cb, 0x4022327525466666], motion: [0x3f2bbc1e39c1c2b0, 0xbfd82f802c9af0b3, 0x3f615592e41919af], rot: [0x40d1c8d7, 0xc2b02154], in_ground: false, ticks_in_air: 20, increments: 0 }, // (8.4067, 12.3600, 9.0986)
            Expect { t: 42, pos: [0x4020d029f16b17e5, 0x4027f58b1284b5dd, 0x402231a36e2eeeef], motion: [0x3f20624dd2f1a9fc, 0xbfd95a1cac083127, 0x3f5a9fbe76c8b439], rot: [0x40d1c8d7, 0xc2b02154], in_ground: false, ticks_in_air: 20, increments: 2 }, // (8.4066, 11.9796, 9.0970)
            Expect { t: 43, pos: [0x4020d014f8b58bf2, 0x402732c589425aee, 0x402230d1b7177778], motion: [0x3f20624dd2f1a9fc, 0xbfd95a1cac083127, 0x3f5a9fbe76c8b439], rot: [0x40d1c8d7, 0xc2b02154], in_ground: false, ticks_in_air: 20, increments: 1 }, // (8.4064, 11.5992, 9.0954)
            Expect { t: 44, pos: [0x4020d00000000000, 0x4026700000000000, 0x4022300000000000], motion: [0x3f20624dd2f1a9fc, 0xbfd95a1cac083127, 0x3f5a9fbe76c8b439], rot: [0x40d1c8d7, 0xc2b02154], in_ground: false, ticks_in_air: 20, increments: 0 }, // (8.4063, 11.2188, 9.0938)
            Expect { t: 45, pos: [0x4020d010624dd2f2, 0x4025a52f1a9fbe77, 0x402230d4fdf3b646], motion: [0x3f1e2584fdf3b646, 0xbfd9add597af4f0d, 0x3f587e7c0e560418], rot: [0x40c3faea, 0xc2b0cf53], in_ground: false, ticks_in_air: 21, increments: 0 }, // (8.4064, 10.8226, 9.0954)
            Expect { t: 46, pos: [0x4020d01f751051ec, 0x4024d7c06de243ff, 0x40223198f1d428f6], motion: [0x3f1bbc1e39c1c2b0, 0xbfd9fadbe13789db, 0x3f5688d88eed6e2f], rot: [0x40b8efc4, 0xc2b15cbb], in_ground: false, ticks_in_air: 22, increments: 0 }, // (8.4065, 10.4214, 9.0969)
            Expect { t: 47, pos: [0x4020d014f8b58bf3, 0x4024052af3ec2d55, 0x40223110a1381b4f], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40b8efc4, 0xc2b15cbb], in_ground: false, ticks_in_air: 22, increments: 2 }, // (8.4064, 10.0101, 9.0958)
            Expect { t: 48, pos: [0x4020d00a7c5ac5fa, 0x4023329579f616aa, 0x40223088509c0da8], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40b8efc4, 0xc2b15cbb], in_ground: false, ticks_in_air: 22, increments: 1 }, // (8.4063, 9.5988, 9.0948)
            Expect { t: 49, pos: [0x4020d00000000000, 0x4022600000000000, 0x4022300000000000], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40b8efc4, 0xc2b15cbb], in_ground: false, ticks_in_air: 22, increments: 0 }, // (8.4063, 9.1875, 9.0938)
            Expect { t: 50, pos: [0x4020d010624dd2f2, 0x40218a0c49ba5e35, 0x40223083126e978d], motion: [0x3f1e2584fdf3b646, 0xbfdaf5ad9df9096b, 0x3f4e2584fdf3b646], rot: [0x40c18cb1, 0xc2b1d59f], in_ground: false, ticks_in_air: 23, increments: 0 }, // (8.4064, 8.7696, 9.0948)
            Expect { t: 51, pos: [0x4020d01f751051ec, 0x4020b25edcca95ea, 0x402230fba8828f5c], motion: [0x3f1bbc1e39c1c2b0, 0xbfdb2879a9eba500, 0x3f4bbc1e39c1c2b0], rot: [0x40c870a4, 0xc2b23792], in_ground: false, ticks_in_air: 24, increments: 0 }, // (8.4065, 8.3484, 9.0957)
            Expect { t: 52, pos: [0x4020d014f8b58bf3, 0x401fadd3d10e1d38, 0x402230a7c5ac5f93], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40c870a4, 0xc2b23792], in_ground: false, ticks_in_air: 24, increments: 2 }, // (8.4064, 7.9198, 9.0950)
            Expect { t: 53, pos: [0x4020d00a7c5ac5fa, 0x401df6e9e8870e9c, 0x40223053e2d62fca], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40c870a4, 0xc2b23792], in_ground: false, ticks_in_air: 24, increments: 1 }, // (8.4063, 7.4911, 9.0944)
            Expect { t: 54, pos: [0x4020d00000000000, 0x401c400000000000, 0x4022300000000000], motion: [0x3f20624dd2f1a9fc, 0xbfdabe76c8b43958, 0x3f50624dd2f1a9fc], rot: [0x40c870a4, 0xc2b23792], in_ground: false, ticks_in_air: 24, increments: 0 }, // (8.4063, 7.0625, 9.0938)
            Expect { t: 55, pos: [0x4020d010624dd2f2, 0x401a94189374bc6a, 0x40223083126e978d], motion: [0x3f1e2584fdf3b646, 0xbfdaf5ad9df9096b, 0x3f4e2584fdf3b646], rot: [0x40cdf3cb, 0xc2b284b2], in_ground: false, ticks_in_air: 25, increments: 0 }, // (8.4064, 6.6446, 9.0948)
            Expect { t: 56, pos: [0x4020d01f751051ec, 0x4018e4bdb9952bd3, 0x402230fba8828f5c], motion: [0x3f1bbc1e39c1c2b0, 0xbfdb2879a9eba500, 0x3f4bbc1e39c1c2b0], rot: [0x40d25cb9, 0xc2b2c3a1], in_ground: false, ticks_in_air: 26, increments: 0 }, // (8.4065, 6.2234, 9.0957)
            Expect { t: 57, pos: [0x4020d014f8b58bf3, 0x4017add3d10e1d37, 0x402230a7c5ac5f93], motion: [0x0000000000000000, 0xbfdc2b020c49ba5e, 0x3f40624dd2f1a9fc], rot: [0x40d25cb9, 0xc2b2c3a1], in_ground: false, ticks_in_air: 26, increments: 2 }, // (8.4064, 5.9198, 9.0950)
            Expect { t: 58, pos: [0x4020d00a7c5ac5fa, 0x401676e9e8870e9c, 0x40223053e2d62fca], motion: [0x0000000000000000, 0xbfdc2b020c49ba5e, 0x3f40624dd2f1a9fc], rot: [0x40d25cb9, 0xc2b2c3a1], in_ground: false, ticks_in_air: 26, increments: 1 }, // (8.4063, 5.6161, 9.0944)
            Expect { t: 59, pos: [0x4020d00000000000, 0x4015400000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbfdc2b020c49ba5e, 0x3f40624dd2f1a9fc], rot: [0x40d25cb9, 0xc2b2c3a1], in_ground: false, ticks_in_air: 26, increments: 0 }, // (8.4063, 5.3125, 9.0938)
            Expect { t: 60, pos: [0x4020d00000000000, 0x4015400000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbfdc2b020c49ba5e, 0x3f40624dd2f1a9fc], rot: [0x40d25cb9, 0xc2b2c3a1], in_ground: true, ticks_in_air: 27, increments: 0 }, // (8.4063, 5.3125, 9.0938)
            Expect { t: 61, pos: [0x4020d00000000000, 0x40153b129686e979, 0x40223008f4847ae1], motion: [0x0000000000000000, 0xbfa51bb2f90134fa, 0x3f107a3b76795d70], rot: [0x40a84a2e, 0xc2b2af98], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.3077, 9.0938)
            Expect { t: 62, pos: [0x4020d00000000000, 0x4014d20c6459f0fb, 0x40223005f85851eb], motion: [0x0000000000000000, 0xbfa46a7ef9db22d1, 0x0000000000000000], rot: [0x40f73174, 0xc2b31fbb], in_ground: false, ticks_in_air: 0, increments: 2 }, // (8.4063, 5.2051, 9.0938)
            Expect { t: 63, pos: [0x4020d00000000000, 0x40146906322cf87e, 0x40223002fc2c28f6], motion: [0x0000000000000000, 0xbfa46a7ef9db22d1, 0x0000000000000000], rot: [0x41230c5d, 0xc2b38fde], in_ground: false, ticks_in_air: 0, increments: 1 }, // (8.4063, 5.1026, 9.0938)
            Expect { t: 64, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbfa46a7ef9db22d1, 0x0000000000000000], rot: [0x414a8000, 0xc2b40000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 65, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbfa46a7ef9db22d1, 0x0000000000000000], rot: [0x414a8000, 0xc2b40000], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 66, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf642b4432c08313, 0x0000000000000000], rot: [0x414a8000, 0xc2b40000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 67, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf36d68eb522bed8, 0x0000000000000000], rot: [0x414a8000, 0xc2b40000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 68, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf947ae140000000, 0x0000000000000000], rot: [0x41220000, 0xc2900000], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 69, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf947ae140000000, 0x0000000000000000], rot: [0x41220000, 0xc2900000], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 70, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf5902e871803080, 0x0000000000000000], rot: [0x41220000, 0xc2900000], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 71, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf947ae140000000, 0x0000000000000000], rot: [0x4101999a, 0xc2666666], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 72, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf947ae140000000, 0x0000000000000000], rot: [0x4101999a, 0xc2666666], in_ground: true, ticks_in_air: 1, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 73, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf705aaa0e58e180, 0x0000000000000000], rot: [0x4101999a, 0xc2666666], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 74, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf36d7cfc288507a, 0x0000000000000000], rot: [0x4101999a, 0xc2666666], in_ground: true, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
            Expect { t: 75, pos: [0x4020d00000000000, 0x4014000000000000, 0x4022300000000000], motion: [0x0000000000000000, 0xbf947ae140000000, 0x0000000000000000], rot: [0x40cf5c2a, 0xc23851eb], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.4063, 5.0000, 9.0938)
        ],
    },
        // server t=1 pos=(8.3479, 5.2308, 9.2402) motion=(0.0073, -1.2228, 0.6810) inGround=false caught=false
        // server t=2 pos=(8.3479, 5.2308, 9.2402) motion=(0.0073, -1.2228, 0.6810) inGround=true caught=false
        // server t=3 pos=(8.3482, 5.0548, 9.2984) motion=(0.0002, -0.1987, 0.0535) inGround=false caught=false
        // server t=4 pos=(8.3482, 5.0548, 9.2984) motion=(0.0002, -0.1987, 0.0535) inGround=true caught=false
        // server t=5 pos=(8.3482, 5.0523, 9.3070) motion=(0.0000, -0.0391, 0.0080) inGround=false caught=false
        // server t=6 pos=(8.3482, 5.0132, 9.3150) motion=(0.0000, -0.0728, 0.0073) inGround=false caught=false
        // server t=7 pos=(8.3482, 5.0132, 9.3150) motion=(0.0000, -0.0728, 0.0073) inGround=true caught=false
        // server t=8 pos=(8.3482, 5.0052, 9.3163) motion=(0.0000, -0.0442, 0.0012) inGround=false caught=false
        // server t=9 pos=(8.3482, 5.0052, 9.3163) motion=(0.0000, -0.0442, 0.0012) inGround=true caught=false
        // server t=10 pos=(8.3482, 5.0052, 9.3163) motion=(0.0000, -0.0078, 0.0002) inGround=true caught=false
        // server t=11 pos=(8.3482, 5.0047, 9.3163) motion=(0.0000, -0.0373, 0.0000) inGround=false caught=false
        // server t=12 pos=(8.3482, 5.0047, 9.3163) motion=(0.0000, -0.0373, 0.0000) inGround=true caught=false
        // server t=13 pos=(8.3482, 5.0020, 9.3163) motion=(0.0000, -0.0393, 0.0000) inGround=false caught=false
        // server t=14 pos=(8.3482, 5.0020, 9.3163) motion=(0.0000, -0.0393, 0.0000) inGround=true caught=false
        // server t=15 pos=(8.3482, 5.0020, 9.3163) motion=(0.0000, -0.0074, 0.0000) inGround=true caught=false
        // server t=16 pos=(8.3482, 5.0019, 9.3163) motion=(0.0000, -0.0369, 0.0000) inGround=false caught=false
        // server t=17 pos=(8.3482, 5.0019, 9.3163) motion=(0.0000, -0.0369, 0.0000) inGround=true caught=false
        // server t=18 pos=(8.3482, 5.0012, 9.3163) motion=(0.0000, -0.0374, 0.0000) inGround=false caught=false
        // server t=19 pos=(8.3482, 5.0012, 9.3163) motion=(0.0000, -0.0374, 0.0000) inGround=true caught=false
        // server t=20 pos=(8.3482, 5.0012, 9.3163) motion=(0.0000, -0.0069, 0.0000) inGround=true caught=false
        // server t=21 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0375, 0.0000) inGround=false caught=false
        // server t=22 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0375, 0.0000) inGround=true caught=false
        // server t=23 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0019, 0.0000) inGround=true caught=false
        // server t=24 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0368, 0.0000) inGround=false caught=false
        // server t=25 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0368, 0.0000) inGround=true caught=false
        // server t=26 pos=(8.3482, 5.0005, 9.3163) motion=(0.0000, -0.0053, 0.0000) inGround=true caught=false
        // server t=27 pos=(8.3482, 5.0001, 9.3163) motion=(0.0000, -0.0371, 0.0000) inGround=false caught=false
        // server t=28 pos=(8.3482, 5.0001, 9.3163) motion=(0.0000, -0.0371, 0.0000) inGround=true caught=false
        // server t=29 pos=(8.3482, 5.0001, 9.3163) motion=(0.0000, -0.0043, 0.0000) inGround=true caught=false
        // server t=30 pos=(8.3482, 5.0001, 9.3163) motion=(0.0000, -0.0006, 0.0000) inGround=true caught=false
    HookRef {
        name: "floor_in_front",
        // angler (8.5, 5.0, 8.5) yaw 0.0 pitch 60.0; prev_pos_at_spawn=(0.0, 0.0, 0.0)
        angler: [0x4021000000000000, 0x4014000000000000, 0x4021000000000000],
        stone: &[([0, 4, 0], [15, 4, 15]), ],
        spawn: [266, 208, 272],
        spawn_look: (0, -43),
        spawn_velocity: [63, -10313, 5921],
        client_seed: 52,
        packets: &[
            (2, Pkt::Vel([58, -9782, 5448])),
            (7, Pkt::Vel([0, -582, 58])),
            (7, Pkt::Rel([0, -7, 3], None, false)),
            (12, Pkt::Vel([0, -298, 0])),
            (12, Pkt::Rel([0, 0, 0], Some((0, -49)), false)),
            (17, Pkt::Rel([0, 0, 0], Some((0, -54)), false)),
            (22, Pkt::Rel([0, 0, 0], Some((0, -58)), false)),
            (27, Pkt::Vel([0, -42, 0])),
        ],
        ticks: &[
            Expect { t: 1, pos: [0x4020a4083126e979, 0x4014d7ef9db22d0e, 0x40227af1a9fbe76d], motion: [0x3f7daceee9fbe76d, 0xbff390918327381e, 0x3fe5ca0e47ad0e56], rot: [0x3df9b83d, 0xc2719c1f], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3204, 5.2109, 9.2401)
            Expect { t: 2, pos: [0x4020a4083126e979, 0x4014d7ef9db22d0e, 0x40227af1a9fbe76d], motion: [0x3f7db22d0e560419, 0xbff390624dd2f1aa, 0x3fe5cac083126e98], rot: [0x3df9b83d, 0xc2719c1f], in_ground: true, ticks_in_air: 2, increments: 0 }, // (8.3204, 5.2109, 9.2401)
            Expect { t: 3, pos: [0x4020a492c66eb852, 0x4014a2d36589f7cf, 0x4022a6f9daf60419], motion: [0x3f4fdfc65050dc21, 0xbfb5a2daf4f3e7d4, 0x3fb441354604188e], rot: [0x3e742372, 0xc25a2943], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3214, 5.1590, 9.3261)
            Expect { t: 4, pos: [0x4020a5124587f995, 0x40144c47f9b62830, 0x4022cf7c45820c4a], motion: [0x3f4d52fe29daf7ce, 0xbfbd5378c5296f91, 0x3fb2a2643bd89626], rot: [0x3ea9c71e, 0xc25409c9], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3224, 5.0745, 9.4052)
            Expect { t: 5, pos: [0x4020a5124587f995, 0x40144c47f9b62830, 0x4022cf7c45820c4a], motion: [0x3f4d52fe29daf7ce, 0xbfbd5378c5296f91, 0x3fb2a2643bd89626], rot: [0x3ea9c71e, 0xc25409c9], in_ground: true, ticks_in_air: 2, increments: 0 }, // (8.3224, 5.0745, 9.4052)
            Expect { t: 6, pos: [0x4020a521027776a7, 0x40144b43856c808d, 0x4022d5d77235c5ed], motion: [0x3f1b1e339238adfb, 0xbfa34f422f7024c5, 0x3f876400b0fcc841], rot: [0x3ebcf2ea, 0xc22d4a7a], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3225, 5.0735, 9.4177)
            Expect { t: 7, pos: [0x4020a36b56fa4f1a, 0x4015e78258f3005e, 0x4022493a4c23d949], motion: [0x0000000000000000, 0xbfb29fbe76c8b439, 0x3f7db22d0e560419], rot: [0x3ebcf2ea, 0xc22d4a7a], in_ground: false, ticks_in_air: 0, increments: 2 }, // (8.3192, 5.4761, 9.1430)
            Expect { t: 8, pos: [0x4020a1b5ab7d278d, 0x401783c12c79802f, 0x4021bc9d2611eca4], motion: [0x0000000000000000, 0xbfb29fbe76c8b439, 0x3f7db22d0e560419], rot: [0x3ebcf2ea, 0xc22d4a7a], in_ground: false, ticks_in_air: 0, increments: 1 }, // (8.3158, 5.8787, 8.8684)
            Expect { t: 9, pos: [0x4020a00000000000, 0x4019200000000000, 0x4021300000000000], motion: [0x0000000000000000, 0xbfb29fbe76c8b439, 0x3f7db22d0e560419], rot: [0x3ebcf2ea, 0xc22d4a7a], in_ground: false, ticks_in_air: 0, increments: 0 }, // (8.3125, 6.2813, 8.5938)
            Expect { t: 10, pos: [0x4020a00000000000, 0x4018d5810624dd2f, 0x402133b645a1cac1], motion: [0x0000000000000000, 0xbfba8e0ca229c778, 0x3f7b52008624dd2f], rot: [0x3e9728bb, 0xc24e1475], in_ground: false, ticks_in_air: 1, increments: 0 }, // (8.3125, 6.2085, 8.6010)
            Expect { t: 11, pos: [0x4020a00000000000, 0x40186b48d39c3611, 0x4021372085b28f5d], motion: [0x0000000000000000, 0xbfc0ecf71439f269, 0x3f79227b64579870], rot: [0x3e71dac5, 0xc269ebb5], in_ground: false, ticks_in_air: 2, increments: 0 }, // (8.3125, 6.1048, 8.6077)
            Expect { t: 12, pos: [0x4020a00000000000, 0x4018a785e268240b, 0x402134c05921b4e9], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x3e213c83, 0xc277d279], in_ground: false, ticks_in_air: 2, increments: 2 }, // (8.3125, 6.1636, 8.6030)
            Expect { t: 13, pos: [0x4020a00000000000, 0x4018e3c2f1341206, 0x402132602c90da74], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x3da13c83, 0xc282dc9e], in_ground: false, ticks_in_air: 2, increments: 1 }, // (8.3125, 6.2224, 8.5984)
            Expect { t: 14, pos: [0x4020a00000000000, 0x4019200000000000, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc289d000], in_ground: false, ticks_in_air: 2, increments: 0 }, // (8.3125, 6.2813, 8.5938)
            Expect { t: 15, pos: [0x4020a00000000000, 0x4018f9db22d0e560, 0x4021300000000000], motion: [0x0000000000000000, 0xbfb231a4bfbb2fec, 0x0000000000000000], rot: [0x00000000, 0xc2924000], in_ground: false, ticks_in_air: 3, increments: 0 }, // (8.3125, 6.2440, 8.5938)
            Expect { t: 16, pos: [0x4020a00000000000, 0x4018b1148fd1f8a0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfba28c1c66f27d5, 0x0000000000000000], rot: [0x00000000, 0xc2990000], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3125, 6.1729, 8.5938)
            Expect { t: 17, pos: [0x4020a00000000000, 0x4018d60db536a5c0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc298a000], in_ground: false, ticks_in_air: 4, increments: 2 }, // (8.3125, 6.2090, 8.5938)
            Expect { t: 18, pos: [0x4020a00000000000, 0x4018fb06da9b52e0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc2984000], in_ground: false, ticks_in_air: 4, increments: 1 }, // (8.3125, 6.2451, 8.5938)
            Expect { t: 19, pos: [0x4020a00000000000, 0x4019200000000000, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc297e000], in_ground: false, ticks_in_air: 4, increments: 0 }, // (8.3125, 6.2813, 8.5938)
            Expect { t: 20, pos: [0x4020a00000000000, 0x4018f9db22d0e560, 0x4021300000000000], motion: [0x0000000000000000, 0xbfb231a4bfbb2fec, 0x0000000000000000], rot: [0x00000000, 0xc29d8000], in_ground: false, ticks_in_air: 5, increments: 0 }, // (8.3125, 6.2440, 8.5938)
            Expect { t: 21, pos: [0x4020a00000000000, 0x4018b1148fd1f8a0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfba28c1c66f27d5, 0x0000000000000000], rot: [0x00000000, 0xc2a20000], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3125, 6.1729, 8.5938)
            Expect { t: 22, pos: [0x4020a00000000000, 0x4018d60db536a5c0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc2a26000], in_ground: false, ticks_in_air: 6, increments: 2 }, // (8.3125, 6.2090, 8.5938)
            Expect { t: 23, pos: [0x4020a00000000000, 0x4018fb06da9b52e0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc2a2c000], in_ground: false, ticks_in_air: 6, increments: 1 }, // (8.3125, 6.2451, 8.5938)
            Expect { t: 24, pos: [0x4020a00000000000, 0x4019200000000000, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa3126e978d4fdf, 0x0000000000000000], rot: [0x00000000, 0xc2a32000], in_ground: false, ticks_in_air: 6, increments: 0 }, // (8.3125, 6.2813, 8.5938)
            Expect { t: 25, pos: [0x4020a00000000000, 0x4018f9db22d0e560, 0x4021300000000000], motion: [0x0000000000000000, 0xbfb231a4bfbb2fec, 0x0000000000000000], rot: [0x00000000, 0xc2a68000], in_ground: false, ticks_in_air: 7, increments: 0 }, // (8.3125, 6.2440, 8.5938)
            Expect { t: 26, pos: [0x4020a00000000000, 0x4018b1148fd1f8a0, 0x4021300000000000], motion: [0x0000000000000000, 0xbfba28c1c66f27d5, 0x0000000000000000], rot: [0x00000000, 0xc2a93333], in_ground: false, ticks_in_air: 8, increments: 0 }, // (8.3125, 6.1729, 8.5938)
            Expect { t: 27, pos: [0x4020a00000000000, 0x4018abb44e48c154, 0x4021300000000000], motion: [0x0000000000000000, 0xbfa55087007c84b3, 0x0000000000000000], rot: [0x00000000, 0xc2ab5c29], in_ground: false, ticks_in_air: 9, increments: 0 }, // (8.3125, 6.1677, 8.5938)
            Expect { t: 28, pos: [0x4020a00000000000, 0x401881134047c84b, 0x4021300000000000], motion: [0x0000000000000000, 0xbfb339ba37feb729, 0x0000000000000000], rot: [0x00000000, 0xc2ad1687], in_ground: false, ticks_in_air: 10, increments: 0 }, // (8.3125, 6.1260, 8.5938)
            Expect { t: 29, pos: [0x4020a00000000000, 0x4018342c5767cd6e, 0x4021300000000000], motion: [0x0000000000000000, 0xbfbb1bb6cef739a0, 0x0000000000000000], rot: [0x00000000, 0xc2ae786c], in_ground: false, ticks_in_air: 11, increments: 0 }, // (8.3125, 6.0510, 8.5938)
            Expect { t: 30, pos: [0x4020a00000000000, 0x4017c7bd7c2bf088, 0x4021300000000000], motion: [0x0000000000000000, 0xbfc12e218512af63, 0x0000000000000000], rot: [0x00000000, 0xc2af938a], in_ground: false, ticks_in_air: 12, increments: 0 }, // (8.3125, 5.9451, 8.5938)
        ],
    },
];

/// `S0E`/`S14`-family angle bytes, decoded as `NetHandlerPlayClient` does.
fn angle(byte: i8) -> f32 {
    mc_protocol::Angle(byte as u8).to_degrees()
}

/// The client hook's `EntityFishHook.inGround`.
fn fishing_hook_in_ground(tracked: &TrackedEntity) -> bool {
    tracked.fishing_flight.in_ground
}

fn world_for(reference: &HookRef) -> World {
    let mut world = World::new();
    for x in -2..=3 {
        for z in -2..=3 {
            world.load_empty_chunk(ChunkPos { x, z });
        }
    }
    let stone = BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
    for &(from, to) in reference.stone {
        for x in from[0]..=to[0] {
            for y in from[1]..=to[1] {
                for z in from[2]..=to[2] {
                    world.set_block(BlockPos::new(x, y, z), stone);
                }
            }
        }
    }
    world
}

fn apply(tracker: &mut EntityTracker, world: &mut World, packet: Pkt) {
    let applied = match packet {
        Pkt::Vel(motion) => tracker.apply_velocity(world, HOOK, motion),
        Pkt::Rel(delta, look, on_ground) => tracker.apply_relative_move_grounded(
            world,
            HOOK,
            delta,
            look.map(|(yaw, pitch)| (angle(yaw), angle(pitch))),
            Some(on_ground),
        ),
        Pkt::Tp(position, yaw, pitch, on_ground) => tracker.apply_teleport_grounded(
            world,
            HOOK,
            position,
            angle(yaw),
            angle(pitch),
            Some(on_ground),
        ),
    };
    assert!(applied, "{packet:?} must reach the hook");
}

/// Replay one Java scenario and return every tick that diverged.
fn divergences(reference: &HookRef) -> Vec<String> {
    let mut world = world_for(reference);
    let mut tracker = EntityTracker::new();
    let angler = reference.angler.map(f64::from_bits);
    // The Java client world holds the angler as an ordinary loaded `EntityPlayer`, so it is an
    // entity-collision candidate once the five-tick grace has expired.
    tracker.spawn_packet(
        &mut world,
        ANGLER,
        TrackedKind::Player,
        Some(Uuid::from_u128(0x55)),
        [
            (angler[0] * 32.0) as i32,
            (angler[1] * 32.0) as i32,
            (angler[2] * 32.0) as i32,
        ],
        0.0,
        0.0,
        DVec3::ZERO,
        Metadata::new(),
    );
    let hook = tracker.spawn_packet(
        &mut world,
        HOOK,
        TrackedKind::FishingBobber,
        None,
        reference.spawn,
        angle(reference.spawn_look.0),
        angle(reference.spawn_look.1),
        DVec3::ZERO,
        Metadata::new(),
    );
    assert!(tracker.set_fishing_bobber_owner(HOOK, ANGLER));
    // java: the generator replaces the client hook's `Entity.rand` with `new Random(seed)` right
    // after construction; mc-rs keeps that per-entity stream in `particle_random`.
    tracker
        .by_network_id
        .get_mut(&HOOK)
        .unwrap()
        .particle_random = JavaRandom::new(reference.client_seed);
    assert!(tracker.apply_velocity(&mut world, HOOK, reference.spawn_velocity));

    let mut out = Vec::new();
    for expect in reference.ticks {
        for &(when, packet) in reference.packets {
            if when == expect.t {
                apply(&mut tracker, &mut world, packet);
            }
        }
        tracker.tick_interpolation(&mut world);
        let tracked = tracker.get(HOOK).expect("the hook stays tracked");
        let position = world.ecs.get::<&Position>(hook).unwrap().0;
        let motion = world.ecs.get::<&Velocity>(hook).unwrap().0;
        let rotation = *world.ecs.get::<&Rotation>(hook).unwrap();
        let want_rot = expect.rot.map(f32::from_bits);
        let got_pos = [position.x, position.y, position.z];
        let got_motion = [motion.x, motion.y, motion.z];
        let mut problems = Vec::new();
        if got_pos.map(f64::to_bits) != expect.pos {
            problems.push(format!(
                "pos {got_pos:?} != {:?}",
                expect.pos.map(f64::from_bits)
            ));
        }
        if got_motion.map(f64::to_bits) != expect.motion {
            problems.push(format!(
                "motion {got_motion:?} != {:?}",
                expect.motion.map(f64::from_bits)
            ));
        }
        // Java keeps `rotation` near a `prevRotation` it shifts by whole turns
        // (`EntityFishHook.java:314-332`); mc-rs wraps the difference instead. Same angle, so
        // compare modulo one turn.
        let turn_error = |got: f32, want: f32| {
            let error = (got - want).rem_euclid(360.0);
            error.min(360.0 - error)
        };
        if turn_error(rotation.yaw, want_rot[0]) > 1e-3
            || turn_error(rotation.pitch, want_rot[1]) > 1e-3
        {
            problems.push(format!(
                "rot ({}, {}) != ({}, {})",
                rotation.yaw, rotation.pitch, want_rot[0], want_rot[1]
            ));
        }
        if fishing_hook_in_ground(tracked) != expect.in_ground {
            problems.push(format!(
                "inGround {} != {}",
                fishing_hook_in_ground(tracked),
                expect.in_ground
            ));
        }
        if tracked.fishing_flight.ticks_in_air != expect.ticks_in_air {
            problems.push(format!(
                "ticksInAir {} != {}",
                tracked.fishing_flight.ticks_in_air, expect.ticks_in_air
            ));
        }
        if tracked.interpolation_steps != expect.increments {
            problems.push(format!(
                "increments {} != {}",
                tracked.interpolation_steps, expect.increments
            ));
        }
        if !problems.is_empty() {
            out.push(format!(
                "{} t={}: {}",
                reference.name,
                expect.t,
                problems.join("; ")
            ));
        }
    }
    out
}

#[test]
fn fishing_hook_client_flight_matches_the_java_classes_tick_for_tick() {
    let all: Vec<String> = REFS.iter().flat_map(divergences).collect();
    assert!(
        all.is_empty(),
        "{} diverging ticks:\n{}",
        all.len(),
        all.join("\n")
    );
}
