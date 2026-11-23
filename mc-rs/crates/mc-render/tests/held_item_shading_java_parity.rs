//! Bug report: "held items still look slightly too far to the right, check
//! for other things besides too far to the right as well".
//!
//! `held_item_java_parity.rs`proves the held items cover vanilla's pixels. This test
//! checks what those pixels look like: the colour of interior pixels of each held item, i.e. the
//! texture times `GL_LIGHTING` with `RenderHelper`'s two item lights.
//!
//! The samples come from the PNGs `crates/mc-client/javaref/HeldItemRef.java`
//! (`generate-held-items.ps1`) writes: the real recompiled 1.8.9 `ItemRenderer` at 1920x1080 on
//! this machine's GL driver. Each sample is a pixel whose 5x5 neighbourhood is a single colour, so
//! edge rasterisation cannot decide it. The tolerance is 2 of 255 per channel.
//!
//! What it caught: the drawn bow. `ItemRenderer.doBowTransformations` ends with
//! `scale(1, 1, 1 + f1 * 0.2)`, and GL lights its face normals as `GL_RESCALE_NORMAL` does
//! (`f (M^-1)^T n`, not renormalised), which mc-rs reproduces in
//! `mc_render::pipeline::gl_rescale_normal_matrix`. With normalised forward-transformed normals
//! (before the fix) 138,771 of the drawn bow's pixels were more than 2 off, by up to 76.
//!
//! Skips (loudly) without the vanilla jar (set `MC_JARS_DIR`) or a wgpu adapter.

use std::path::{Path, PathBuf};

use mc_assets::{Atlas, BakedModel, ResourceManager, ZipResourcePack};
use mc_render::pipeline::{FirstPersonPhase, FirstPersonUseAction};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 1920;
const H: u32 = 1080;
const MAX_ITEM_USE_DURATION: i32 = 72_000;

/// One `HeldItemRef` case: item, swing, use count, and interior samples `[x, y, 0xRRGGBB]`.
struct ShadeRef {
    name: &'static str,
    item: &'static str,
    swing: f32,
    use_count: i32,
    samples: &'static [[u32; 3]],
}

// Interior pixels of HeldItemRef.java's PNGs (scratch tool InteriorSamples, 24 per case).
const REFS: &[ShadeRef] = &[
    ShadeRef {
        name: "sword_idle",
        item: "diamond_sword",
        swing: 0.0,
        use_count: 0,
        samples: &[
            [1917, 340, 0x071f1b],
            [1872, 424, 0x071f1b],
            [1783, 467, 0x071f1b],
            [1907, 510, 0x197565],
            [1834, 547, 0x156355],
            [1720, 585, 0x071f1b],
            [1809, 618, 0x156355],
            [1744, 651, 0x197565],
            [1754, 681, 0x156355],
            [1839, 710, 0x197565],
            [1784, 739, 0x197565],
            [1656, 767, 0x071f1b],
            [1861, 793, 0x0d3931],
            [1856, 820, 0x071f1b],
            [1827, 840, 0x071f1b],
            [1821, 860, 0x0d3931],
            [1771, 881, 0x071f1b],
            [1611, 899, 0x071f1b],
            [1861, 917, 0x0d3931],
            [1781, 941, 0x071f1b],
            [1681, 963, 0x156355],
            [1580, 987, 0x0f443b],
            [1647, 1014, 0x156355],
            [1594, 1045, 0x041210],
        ],
    },
    ShadeRef {
        name: "sword_mid_swing",
        item: "diamond_sword",
        swing: 0.5,
        use_count: 0,
        samples: &[
            [973, 662, 0x0e3f36],
            [979, 687, 0x0a2b25],
            [1001, 713, 0x0a2b25],
            [1013, 736, 0x0e3f36],
            [1033, 759, 0x0a2b25],
            [1052, 785, 0x0a2b25],
            [1065, 806, 0x0e3f36],
            [1058, 828, 0x0a2b25],
            [1065, 849, 0x0e3f36],
            [1100, 868, 0x0a2b25],
            [1107, 888, 0x0e3f36],
            [1101, 908, 0x0a2b25],
            [1132, 926, 0x0e3f36],
            [1143, 944, 0x0a2b25],
            [1148, 958, 0x082520],
            [1164, 968, 0x082520],
            [1167, 979, 0x082520],
            [1157, 990, 0x082520],
            [1189, 1003, 0x051916],
            [1203, 1015, 0x051916],
            [1212, 1026, 0x051916],
            [1197, 1038, 0x051916],
            [1214, 1049, 0x051916],
            [1219, 1062, 0x051916],
        ],
    },
    ShadeRef {
        name: "sword_blocking",
        item: "diamond_sword",
        swing: 0.0,
        use_count: 71_995,
        samples: &[
            [943, 622, 0x0c362f],
            [910, 674, 0x25ac94],
            [1029, 705, 0x0c362f],
            [864, 732, 0x0c362f],
            [1031, 755, 0x2ccbaf],
            [921, 778, 0x0c362f],
            [1001, 796, 0x2ccbaf],
            [1464, 810, 0x07201c],
            [1093, 825, 0x25ac94],
            [1226, 839, 0x0c362f],
            [1026, 854, 0x0c362f],
            [1231, 866, 0x0c362f],
            [1424, 878, 0x07201c],
            [1297, 892, 0x0c362f],
            [1200, 907, 0x25ac94],
            [1197, 921, 0x25ac94],
            [1164, 935, 0x2ccbaf],
            [1513, 948, 0x07201c],
            [1174, 964, 0x0c362f],
            [1324, 979, 0x25ac94],
            [1520, 994, 0x07201c],
            [1311, 1014, 0x2ccbaf],
            [1416, 1036, 0x1a7767],
            [1546, 1056, 0x281e0b],
        ],
    },
    ShadeRef {
        name: "stone_block_idle",
        item: "stone",
        swing: 0.0,
        use_count: 0,
        samples: &[
            [1392, 789, 0x7f7f7f],
            [1570, 841, 0x747474],
            [1389, 860, 0x747474],
            [1298, 875, 0x747474],
            [1702, 886, 0x7f7f7f],
            [1762, 898, 0x747474],
            [1728, 910, 0x8f8f8f],
            [1698, 921, 0x7f7f7f],
            [1512, 932, 0x7f7f7f],
            [1675, 941, 0x7f7f7f],
            [1396, 951, 0x525252],
            [1296, 961, 0x494949],
            [1724, 970, 0x747474],
            [1438, 980, 0x656565],
            [1271, 989, 0x5a5a5a],
            [1889, 997, 0x8f8f8f],
            [1416, 1007, 0x525252],
            [1368, 1016, 0x5a5a5a],
            [1430, 1025, 0x525252],
            [1849, 1033, 0x7f7f7f],
            [1827, 1042, 0x7f7f7f],
            [1708, 1051, 0x525252],
            [1567, 1060, 0x525252],
            [1365, 1069, 0x5a5a5a],
        ],
    },
    ShadeRef {
        name: "bow_idle",
        item: "bow",
        swing: 0.0,
        use_count: 0,
        samples: &[
            [1896, 427, 0x423113],
            [1798, 492, 0x241b0a],
            [1852, 524, 0x34270f],
            [1802, 550, 0x34270f],
            [1803, 578, 0x34270f],
            [1718, 604, 0x443313],
            [1837, 624, 0x140f05],
            [1702, 643, 0x443313],
            [1692, 663, 0x443313],
            [1788, 687, 0x241b0a],
            [1785, 709, 0x241b0a],
            [1851, 729, 0x222222],
            [1535, 749, 0x241b0a],
            [1905, 777, 0x222222],
            [1559, 805, 0x353535],
            [1532, 831, 0x353535],
            [1579, 854, 0x4a4a4a],
            [1793, 883, 0x222222],
            [1553, 909, 0x34270f],
            [1841, 938, 0x3e3e3e],
            [1817, 966, 0x3e3e3e],
            [1607, 994, 0x241b0a],
            [1771, 1022, 0x3e3e3e],
            [1611, 1049, 0x140f05],
        ],
    },
    ShadeRef {
        name: "bow_drawn",
        item: "bow",
        swing: 0.0,
        use_count: 71_980,
        samples: &[
            [1789, 2, 0x281e0b],
            [1781, 77, 0x100c04],
            [1637, 139, 0x251b0b],
            [1822, 181, 0x222222],
            [1602, 220, 0x251b0b],
            [1615, 255, 0x35270f],
            [1728, 289, 0x100c04],
            [1891, 325, 0x444444],
            [1802, 361, 0x222222],
            [1494, 395, 0x251b0b],
            [1539, 427, 0x140f06],
            [1545, 464, 0x140f06],
            [1501, 511, 0x140f06],
            [1443, 589, 0x35270f],
            [1547, 656, 0x453414],
            [1607, 684, 0x453414],
            [1700, 709, 0x453414],
            [1506, 735, 0x140f06],
            [1471, 761, 0x6b6b6b],
            [1793, 787, 0x140f06],
            [1434, 820, 0x140f06],
            [1792, 862, 0x100c04],
            [1426, 928, 0x35270f],
            [1529, 1005, 0x281e0b],
        ],
    },
    ShadeRef {
        name: "apple_idle",
        item: "apple",
        swing: 0.0,
        use_count: 0,
        samples: &[
            [1637, 558, 0x6c2f0c],
            [1678, 633, 0x3e1b06],
            [1626, 703, 0x3b1a06],
            [1562, 739, 0x690b11],
            [1616, 759, 0x58090f],
            [1802, 778, 0x3e060a],
            [1908, 795, 0x3e060a],
            [1710, 812, 0x7f0e15],
            [1560, 829, 0x7f0e15],
            [1862, 843, 0x59090f],
            [1860, 858, 0x59090f],
            [1894, 875, 0x3e060a],
            [1613, 891, 0x7f4a4e],
            [1796, 906, 0x7f0e15],
            [1747, 923, 0x7f0e15],
            [1564, 939, 0x7f4f52],
            [1907, 953, 0x6e0b12],
            [1730, 969, 0x7f0e15],
            [1812, 984, 0x7f0e15],
            [1770, 1000, 0x7f0e15],
            [1666, 1016, 0x7f0e15],
            [1863, 1031, 0x770c14],
            [1914, 1046, 0x6e0b12],
            [1743, 1062, 0x7f0e15],
        ],
    },
    ShadeRef {
        name: "fishing_rod_idle",
        item: "fishing_rod",
        swing: 0.0,
        use_count: 0,
        samples: &[
            [1565, 469, 0x444444],
            [1632, 522, 0x222222],
            [1653, 564, 0x222222],
            [1558, 598, 0x443313],
            [1573, 629, 0x443313],
            [1497, 655, 0x222222],
            [1537, 681, 0x222222],
            [1646, 703, 0x222222],
            [1705, 726, 0x222222],
            [1474, 751, 0x222222],
            [1686, 774, 0x222222],
            [1470, 801, 0x222222],
            [1632, 825, 0x34270f],
            [1699, 848, 0x222222],
            [1713, 872, 0x222222],
            [1625, 898, 0x140f05],
            [1660, 921, 0x443313],
            [1763, 943, 0x222222],
            [1498, 967, 0xffffff],
            [1724, 986, 0x222222],
            [1601, 1003, 0x1a1a1a],
            [1504, 1019, 0x7e7e7e],
            [1542, 1035, 0x808080],
            [1677, 1055, 0x443313],
        ],
    },
];

/// First-person cases B1's table could not cover, from `javaref/SuffocationRef.java`'s eye-in-air
/// cases (the real `EntityRenderer.renderHand`): the empty right arm (`renderPlayerArm` ->
/// `RenderPlayer.renderRightArm`, Steve's skin, classic arms, all layers worn), an apple being
/// eaten (`performDrinking`), and a sword with the player looking 45 degrees right and 30 up
/// (`rotateArroundXAndY` turns the two item lights with the look).
struct ArmRef {
    name: &'static str,
    /// `None` is the empty hand; otherwise the item and its `itemInUseCount`.
    item: Option<(&'static str, i32)>,
    swing: f32,
    /// `rotationYaw`, `rotationPitch`; the arm has caught up with them.
    look: [f32; 2],
    /// Coverage against the magenta clear: count, `[min_x, max_x, min_y, max_y]`, centroid.
    mask: (u64, [u32; 4], [f64; 2]),
    samples: &'static [[u32; 3]],
}

// Generated by SuffocationRef.java (ArmRef rows) and the InteriorSamples scratch tool.
const ARMS: &[ArmRef] = &[
    ArmRef {
        name: "air_empty_hand",
        item: None,
        swing: 0.0,
        look: [0.0, 0.0],
        mask: (91_250, [1354, 1735, 760, 1079], [1520.1118, 947.2665]),
        samples: &[
            [1476, 764, 0x83614f],
            [1444, 818, 0x946d59],
            [1516, 840, 0x965f41],
            [1410, 858, 0x83614f],
            [1653, 874, 0x965f41],
            [1439, 891, 0x946d59],
            [1584, 906, 0x87553b],
            [1551, 921, 0x965f41],
            [1445, 936, 0x946d59],
            [1454, 951, 0x946d59],
            [1626, 965, 0x965f41],
            [1565, 980, 0x965f41],
            [1584, 995, 0x965f41],
            [1663, 1009, 0x965f41],
            [1512, 1024, 0x946d59],
            [1405, 1038, 0x946d59],
            [1582, 1051, 0x87553b],
            [1692, 1064, 0x87553b],
        ],
    },
    ArmRef {
        name: "air_empty_hand_swing",
        item: None,
        swing: 0.5,
        look: [0.0, 0.0],
        mask: (16_710, [956, 1178, 952, 1079], [1063.1338, 1034.1209]),
        samples: &[
            [1050, 957, 0x946e5a],
            [1036, 982, 0x946e5a],
            [1023, 994, 0xa87c65],
            [1011, 1004, 0xa87c65],
            [1078, 1011, 0xa87c65],
            [1039, 1018, 0xa87c65],
            [1038, 1024, 0xa87c65],
            [1119, 1029, 0x87553b],
            [992, 1035, 0x946e5a],
            [1139, 1039, 0x965f41],
            [1064, 1044, 0xa87c65],
            [1147, 1048, 0x965f41],
            [993, 1053, 0x946e5a],
            [996, 1057, 0x946e5a],
            [978, 1061, 0x946e5a],
            [987, 1065, 0x946e5a],
            [1021, 1069, 0x946e5a],
            [1084, 1073, 0xa87c65],
        ],
    },
    ArmRef {
        name: "air_eating_apple",
        item: Some(("apple", 20)),
        swing: 0.0,
        look: [0.0, 0.0],
        mask: (181_404, [669, 1238, 539, 1074], [948.6571, 858.1584]),
        samples: &[
            [911, 542, 0x77340d],
            [861, 657, 0x371806],
            [956, 718, 0x200e03],
            [1058, 747, 0x3a0609],
            [725, 771, 0xd31623],
            [859, 793, 0x271104],
            [1193, 811, 0x7d0d14],
            [882, 830, 0x760d14],
            [900, 849, 0x762c31],
            [838, 867, 0x764549],
            [1056, 885, 0x6f0c12],
            [1224, 902, 0x270406],
            [840, 920, 0x764549],
            [1078, 937, 0x6f0c12],
            [984, 955, 0x760d14],
            [923, 976, 0x760d14],
            [922, 998, 0x760d14],
            [870, 1025, 0x53090e],
        ],
    },
    ArmRef {
        name: "air_sword_looking",
        item: Some(("diamond_sword", 0)),
        swing: 0.0,
        look: [45.0, -30.0],
        mask: (177_607, [1473, 1919, 336, 1079], [1756.2761, 770.9449]),
        samples: &[
            [1917, 340, 0x08241f],
            [1795, 439, 0x08241f],
            [1879, 497, 0x197262],
            [1834, 547, 0x197262],
            [1892, 595, 0x1d8674],
            [1780, 641, 0x197262],
            [1754, 681, 0x197262],
            [1772, 720, 0x197262],
            [1861, 758, 0x08241f],
            [1861, 793, 0x092a24],
            [1659, 828, 0x1d8674],
            [1706, 852, 0x197262],
            [1771, 881, 0x08241f],
            [1536, 905, 0x0c3830],
            [1695, 934, 0x1d8674],
            [1681, 963, 0x197262],
            [1560, 996, 0x051512],
            [1817, 1034, 0x092a24],
        ],
    },
];

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let dir = root.join("MCP-919-main").join("jars");
    dir.is_dir().then_some(dir)
}

/// Bake and install the vanilla jar's models and atlas once for this test process.
fn baked_atlas() -> Option<Atlas> {
    use std::sync::OnceLock;
    static ATLAS: OnceLock<Option<Atlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| {
            let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
            if !jar.is_file() {
                return None;
            }
            let mut rm = ResourceManager::new();
            rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
            let (baked, _report) =
                mc_assets::bake_all(&rm, 4096).expect("bake_all against the real jar");
            let atlas = baked.atlas.clone();
            mc_assets::install(baked);
            Some(atlas)
        })
        .clone()
}

/// The model and use action `renderItemInFirstPerson` takes for this case.
fn held(case: &ShadeRef) -> (BakedModel, FirstPersonUseAction) {
    let id = mc_data::item_by_name(case.item).expect("a real 1.8.9 item");
    match (case.item, case.use_count) {
        (_, 0) => (
            mc_assets::item_model_for(id).clone(),
            FirstPersonUseAction::Idle,
        ),
        // 20 ticks drawn: bow_pulling_2, Java's variant 3.
        ("bow", _) => (
            mc_assets::item_model_for_variant(id, 3).clone(),
            FirstPersonUseAction::Bow,
        ),
        ("diamond_sword", _) => (
            mc_assets::item_model_for(id).clone(),
            FirstPersonUseAction::Block,
        ),
        (other, _) => panic!("{other}: no use action in this table"),
    }
}

/// Render one case through the real `FirstPersonPhase`, as `held_item_java_parity` does.
fn render(gpu: &mut Gpu, atlas: &Atlas, case: &ShadeRef) -> Vec<u8> {
    let (model, action) = held(case);
    let gui_3d = model.gui_3d;
    let mut phase = FirstPersonPhase::new(gpu);
    phase.set_atlas(gpu, atlas);
    phase.set_item_model(Some(model), gui_3d);
    phase.render_state.prev_equipped_progress = 1.0;
    phase.render_state.equipped_progress = 1.0;
    phase.render_state.prev_swing_progress = case.swing;
    phase.render_state.swing_progress = case.swing;
    phase.render_state.use_action = action;
    phase.render_state.item_use_count = case.use_count;
    phase.render_state.max_item_use_duration = MAX_ITEM_USE_DURATION;
    phase.partial_ticks = 0.0;
    phase.combined_light = mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;

    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    camera.update();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
fn held_items_are_lit_like_java() {
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let mut gpu = match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    };
    let mut failures = Vec::new();
    for case in REFS {
        let pixels = render(&mut gpu, &atlas, case);
        let mut worst = 0u8;
        for &[x, y, java] in case.samples {
            let i = ((y * W + x) * 4) as usize;
            let ours = [pixels[i], pixels[i + 1], pixels[i + 2]];
            let java = [(java >> 16) as u8, (java >> 8) as u8, java as u8];
            let difference = (0..3).map(|c| ours[c].abs_diff(java[c])).max().unwrap_or(0);
            worst = worst.max(difference);
            if difference > 2 {
                failures.push(format!(
                    "{} ({x}, {y}): mc-rs {ours:?}, Java {java:?}",
                    case.name
                ));
            }
        }
        eprintln!("{:18} worst sample channel difference {worst}", case.name);
    }
    assert!(
        failures.is_empty(),
        "held items are not lit like vanilla 1.8.9: {failures:#?}"
    );
}

fn steve_resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// The empty hand (idle and mid-swing), an apple being eaten, and a sword under a turned look:
/// the same pixels as vanilla, lit the same.
#[test]
fn empty_hand_eating_and_look_cover_and_are_lit_like_java() {
    let (Some(atlas), Some(rm)) = (baked_atlas(), steve_resources()) else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let mut gpu = match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    };
    let mut failures = Vec::new();
    for arm in ARMS {
        let mut phase = FirstPersonPhase::new(&gpu);
        phase.set_atlas(&gpu, &atlas);
        assert!(phase.load_skin_texture(&gpu, &rm, "textures/entity/steve.png"));
        phase.render_state.prev_equipped_progress = 1.0;
        phase.render_state.equipped_progress = 1.0;
        phase.render_state.prev_swing_progress = arm.swing;
        phase.render_state.swing_progress = arm.swing;
        phase.partial_ticks = 0.0;
        phase.combined_light = mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;
        if let Some((item, use_count)) = arm.item {
            let id = mc_data::item_by_name(item).expect("a real 1.8.9 item");
            let model = mc_assets::item_model_for(id).clone();
            let gui_3d = model.gui_3d;
            phase.set_item_model(Some(model), gui_3d);
            if use_count > 0 {
                // `ItemFood.getMaxItemUseDuration` is 32.
                phase.render_state.use_action = FirstPersonUseAction::EatDrink;
                phase.render_state.item_use_count = use_count;
                phase.render_state.max_item_use_duration = 32;
            }
        }
        let [yaw, pitch] = arm.look;
        phase.prev_render_arm_yaw = yaw;
        phase.render_arm_yaw = yaw;
        phase.prev_render_arm_pitch = pitch;
        phase.render_arm_pitch = pitch;
        let mut camera = Camera::default();
        camera.set_viewport(W, H);
        camera.yaw = yaw;
        camera.pitch = pitch;
        camera.update();
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };
        phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &camera, 0.0);
        let pixels = gpu.read_pixels().expect("readback");

        let (mut count, mut sx, mut sy) = (0_u64, 0.0_f64, 0.0_f64);
        let (mut x0, mut x1, mut y0, mut y1) = (W, 0, H, 0);
        for (index, p) in pixels.chunks_exact(4).enumerate() {
            if p[0] == 255 && p[1] == 0 && p[2] == 255 {
                continue;
            }
            let (x, y) = (index as u32 % W, index as u32 / W);
            count += 1;
            sx += f64::from(x) + 0.5;
            sy += f64::from(y) + 0.5;
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        let (java_count, [jx0, jx1, jy0, jy1], [jcx, jcy]) = arm.mask;
        let (cx, cy) = (sx / count.max(1) as f64, sy / count.max(1) as f64);
        eprintln!(
            "{:22} java: n {java_count} x {jx0}..{jx1} y {jy0}..{jy1} c ({jcx:.2}, {jcy:.2})   \
             mc-rs: n {count} x {x0}..{x1} y {y0}..{y1} c ({cx:.2}, {cy:.2})",
            arm.name
        );
        if (count as f64 - java_count as f64).abs() > java_count as f64 * 0.01
            || x0.abs_diff(jx0) > 1
            || x1.abs_diff(jx1) > 1
            || y0.abs_diff(jy0) > 1
            || y1.abs_diff(jy1) > 1
            || (cx - jcx).abs() > 1.0
            || (cy - jcy).abs() > 1.0
        {
            failures.push(format!(
                "{}: covers n {count} x {x0}..{x1} y {y0}..{y1} c ({cx:.2}, {cy:.2})",
                arm.name
            ));
        }
        for &[x, y, java] in arm.samples {
            let i = ((y * W + x) * 4) as usize;
            let ours = [pixels[i], pixels[i + 1], pixels[i + 2]];
            let java = [(java >> 16) as u8, (java >> 8) as u8, java as u8];
            if (0..3).any(|c| ours[c].abs_diff(java[c]) > 2) {
                failures.push(format!(
                    "{} ({x}, {y}): mc-rs {ours:?}, Java {java:?}",
                    arm.name
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "these first-person cases do not look like vanilla 1.8.9's: {failures:#?}"
    );
}
