//! `RenderFish` against the real Java renderer.
//!
//! The tables below were captured from the recompiled vanilla 1.8.9 `RenderFish.doRender` running
//! unmodified in an LWJGL 2 offscreen pbuffer (`crates/mc-client/javaref/RenderFishRef.java`).
//! The `Tessellator`'s uploader was swapped for one that copies each finished buffer, plus the
//! live modelview matrix, instead of drawing it:
//!
//! * `line` is the 17-vertex `GL_LINE_STRIP` exactly as `WorldRenderer.pos` stored it (float bit
//!   patterns; every vertex's colour was `0, 0, 0, 255`);
//! * `quad` is the bobber's four `POSITION_TEX_NORMAL` corners after that modelview, which the
//!   Rust billboard bakes into its camera-relative vertices.
//!
//! First-person cases point `Minecraft.thePlayer` at the angler, because `RenderFish` picks its
//! hand branch by reference comparison; remote cases use an ordinary `EntityPlayer`.

use glam::{DVec3, Vec3};
use mc_render::entity::objects::{emit_fishing_bobber, emit_fishing_line, FishingLineAngler};

struct LineRef {
    name: &'static str,
    local: bool,
    third_person: bool,
    sneaking: bool,
    eye_height: u32,
    angler_prev: [u64; 3],
    angler_pos: [u64; 3],
    yaw: [u32; 2],
    pitch: [u32; 2],
    body: [u32; 2],
    swing: [u32; 2],
    hook_prev: [u64; 3],
    hook_pos: [u64; 3],
    camera: [u64; 3],
    partial: u32,
    view: [u32; 2],
    line: [[u32; 3]; 17],
    quad: [([f32; 3], [f32; 2]); 4],
}

#[rustfmt::skip]
const REFS: &[LineRef] = &[
    LineRef {
        name: "first_person_looking_down",
        local: true, third_person: false, sneaking: false,
        eye_height: 0x3fcf5c29,
        angler_prev: [0x4024000000000000, 0x4050000000000000, 0xc00e000000000000],
        angler_pos: [0x4024800000000000, 0x4050000000000000, 0xc00c000000000000],
        yaw: [0x41f00000, 0x42200000], pitch: [0x42480000, 0x425c0000], body: [0x41a00000, 0x420c0000], swing: [0x3f000000, 0x3f2aaaab],
        hook_prev: [0x402c000000000000, 0x404f400000000000, 0x3ff8000000000000],
        hook_pos: [0x402d000000000000, 0x404f000000000000, 0x4000000000000000],
        camera: [0x4024200000000000, 0x405067ae147ae148, 0xc00d800000000000],
        partial: 0x3e800000, view: [0x42020000, 0x424d0000],
        // line mode 3, stride 16, 17 vertices; rgba of vertex 0 = 0,0,0,255
        line: [
            [0x40820000, 0xc03fae14, 0x40aa0000],
            [0x407225a8, 0xc039f5cf, 0x409f69fc],
            [0x40604b51, 0xc0339144, 0x4094d3f8],
            [0x404e70fa, 0xc02c8075, 0x408a3df5],
            [0x403c96a2, 0xc024c362, 0x407f4fe2],
            [0x402abc4a, 0xc01c5a0a, 0x406a23da],
            [0x4018e1f3, 0xc013446e, 0x4054f7d3],
            [0x4007079c, 0xc009828d, 0x403fcbcc],
            [0x3fea5a88, 0xbffe28d0, 0x402a9fc4],
            [0x3fc6a5d9, 0xbfe7f3fd, 0x401573bc],
            [0x3fa2f12a, 0xbfd066a1, 0x400047b5],
            [0x3f7e78f6, 0xbfb780bc, 0x3fd6375b],
            [0x3f370f98, 0xbf9d424e, 0x3fabdf4c],
            [0x3edf4c74, 0xbf81ab57, 0x3f81873d],
            [0x3e20f370, 0xbf4977af, 0x3f2e5e5c],
            [0xbdf96410, 0xbf0ce79d, 0x3eb35c7c],
            [0xbecd2bc0, 0xbe9b4cf4, 0x3c9fc400],
        ],
        // quad mode 7, stride 24, 4 vertices
        quad: [
            ([4.3781056, -3.4014807, 5.2823877], [0.0625, 0.1875]),
            ([3.9564102, -3.4014807, 5.013738], [0.125, 0.1875]),
            ([3.7468944, -3.088519, 5.3426123], [0.125, 0.125]),
            ([4.16859, -3.088519, 5.611262], [0.0625, 0.125]),
        ],
    },
    LineRef {
        name: "first_person_swing_wraps",
        local: true, third_person: false, sneaking: false,
        eye_height: 0x3fcf5c29,
        angler_prev: [0x3fe0000000000000, 0x4051800000000000, 0x3fe0000000000000],
        angler_pos: [0x3fe0000000000000, 0x4051800000000000, 0x3fe0000000000000],
        yaw: [0xc2f00000, 0xc2f00000], pitch: [0xc1200000, 0xc1200000], body: [0xc2f00000, 0xc2f00000], swing: [0x3f555555, 0x00000000],
        hook_prev: [0xc018000000000000, 0x4051000000000000, 0xc000000000000000],
        hook_pos: [0xc019000000000000, 0x4050f33333333333, 0xc000cccccccccccd],
        camera: [0x3fe0000000000000, 0x4051e7ae147ae148, 0x3fe0000000000000],
        partial: 0x3f000000, view: [0xc2f00000, 0xc1200000],
        // line mode 3, stride 16, 17 vertices; rgba of vertex 0 = 0,0,0,255
        line: [
            [0xc0d40000, 0xc05e147b, 0xc0233333],
            [0xc0c5c903, 0xc056897a, 0xc018870e],
            [0xc0b79206, 0xc04e1b4c, 0xc00ddae9],
            [0xc0a95b0a, 0xc044c9f0, 0xc0032ec4],
            [0xc09b240d, 0xc03a9567, 0xbff1053d],
            [0xc08ced10, 0xc02f7db1, 0xbfdbacf3],
            [0xc07d6c27, 0xc02382ce, 0xbfc654a9],
            [0xc060fe2e, 0xc016a4bd, 0xbfb0fc5f],
            [0xc0449034, 0xc008e37f, 0xbf9ba414],
            [0xc028223a, 0xbff47e27, 0xbf864bca],
            [0xc00bb441, 0xbfd56ef6, 0xbf61e700],
            [0xbfde8c8f, 0xbfb4996a, 0xbf37366b],
            [0xbfa5b09c, 0xbf91fd83, 0xbf0c85d7],
            [0xbf59a952, 0xbf5b3685, 0xbec3aa85],
            [0xbecfe2d8, 0xbf0ee54d, 0xbe5c92b7],
            [0x3d1c67a0, 0xbe7c1d84, 0xbd474195],
            [0x3ef6fcc0, 0x3da31a04, 0x3df1e3da],
        ],
        // quad mode 7, stride 24, 4 vertices
        quad: [
            ([-6.7124043, -3.966202, -2.7882123], [0.0625, 0.1875]),
            ([-6.4624043, -3.966202, -2.3551996], [0.125, 0.1875]),
            ([-6.5375957, -3.473798, -2.3117876], [0.125, 0.125]),
            ([-6.7875957, -3.473798, -2.7448003], [0.0625, 0.125]),
        ],
    },
    LineRef {
        name: "local_third_person",
        local: true, third_person: true, sneaking: false,
        eye_height: 0x3fcf5c29,
        angler_prev: [0x4024000000000000, 0x4050000000000000, 0xc00e000000000000],
        angler_pos: [0x4024800000000000, 0x4050000000000000, 0xc00c000000000000],
        yaw: [0x41f00000, 0x42200000], pitch: [0x42480000, 0x425c0000], body: [0x41a00000, 0x420c0000], swing: [0x00000000, 0x00000000],
        hook_prev: [0x402c000000000000, 0x404f400000000000, 0x3ff8000000000000],
        hook_pos: [0x402d000000000000, 0x404f000000000000, 0x4000000000000000],
        camera: [0x401c000000000000, 0x4050800000000000, 0xc01c000000000000],
        partial: 0x3f400000, view: [0x420c0000, 0x41a00000],
        // line mode 3, stride 16, 17 vertices; rgba of vertex 0 = 0,0,0,255
        line: [
            [0x40ec0000, 0xc0680000, 0x410e0000],
            [0x40e23255, 0xc0620f85, 0x4109109f],
            [0x40d864aa, 0xc05b6c29, 0x4104213d],
            [0x40ce96ff, 0xc05415eb, 0x40fe63b8],
            [0x40c4c954, 0xc04c0ccd, 0x40f484f5],
            [0x40bafba9, 0xc04350cd, 0x40eaa632],
            [0x40b12dfe, 0xc039e1eb, 0x40e0c76f],
            [0x40a76053, 0xc02fc029, 0x40d6e8ac],
            [0x409d92a8, 0xc024eb85, 0x40cd09ea],
            [0x4093c4fe, 0xc0196400, 0x40c32b27],
            [0x4089f753, 0xc00d2999, 0x40b94c64],
            [0x408029a8, 0xc0003c52, 0x40af6da1],
            [0x406cb7fa, 0xbfe53852, 0x40a58ede],
            [0x40591ca4, 0xbfc8923d, 0x409bb01b],
            [0x4045814e, 0xbfaa8666, 0x4091d159],
            [0x4031e5f8, 0xbf8b14cc, 0x4087f296],
            [0x401e4aa2, 0xbf547ae0, 0x407c27a6],
        ],
        // quad mode 7, stride 24, 4 vertices
        quad: [
            ([7.628832, -4.1099234, 8.948353], [0.0625, 0.1875]),
            ([7.2192554, -4.1099234, 8.661564], [0.125, 0.1875]),
            ([7.121168, -3.6400769, 8.801647], [0.125, 0.125]),
            ([7.5307446, -3.6400769, 9.088436], [0.0625, 0.125]),
        ],
    },
    LineRef {
        name: "remote_sneaking",
        local: false, third_person: false, sneaking: true,
        eye_height: 0x3fc51eb8,
        angler_prev: [0xc034000000000000, 0x4054000000000000, 0x4044000000000000],
        angler_pos: [0xc034333333333333, 0x4054000000000000, 0x4044266666666666],
        yaw: [0x432a0000, 0xc32f0000], pitch: [0x40a00000, 0x41000000], body: [0x43250000, 0x432c0000], swing: [0x00000000, 0x00000000],
        hook_prev: [0xc036000000000000, 0x4053800000000000, 0x403e000000000000],
        hook_pos: [0xc03619999999999a, 0x405379999999999a, 0x403d800000000000],
        camera: [0xc024000000000000, 0x405467ae147ae148, 0x4041800000000000],
        partial: 0x3ecccccd, view: [0x42c80000, 0x40400000],
        // line mode 3, stride 16, 17 vertices; rgba of vertex 0 = 0,0,0,255
        line: [
            [0xc140a3d7, 0xc05a3d71, 0xc0a66666],
            [0xc13e81c9, 0xc05484b8, 0xc09378c8],
            [0xc13c5fba, 0xc04e1fae, 0xc0808b29],
            [0xc13a3dac, 0xc0470e52, 0xc05b3b14],
            [0xc1381b9e, 0xc03f50a4, 0xc0355fd6],
            [0xc135f98f, 0xc036e6a4, 0xc00f8498],
            [0xc133d781, 0xc02dd052, 0xbfd352b5],
            [0xc131b573, 0xc0240dae, 0xbf879c39],
            [0xc12f9365, 0xc0199eb9, 0xbeef96f6],
            [0xc12d7156, 0xc00e8371, 0x3dfd0bde],
            [0xc12b4f48, 0xc002bbd7, 0x3f370e73],
            [0xc1292d3a, 0xbfec8fd8, 0x3fa73db5],
            [0xc1270b2b, 0xbfd24f5d, 0x3ff2f430],
            [0xc124e91d, 0xbfb6b63e, 0x401f5556],
            [0xc122c70f, 0xbf99c47c, 0x40453094],
            [0xc120a500, 0xbf76f42b, 0x406b0bd1],
            [0xc11e82f2, 0xbf37ae17, 0x40887388],
        ],
        // quad mode 7, stride 24, 4 vertices
        quad: [
            ([-12.070527, -3.9096575, -4.9515257], [0.0625, 0.1875]),
            ([-11.983703, -3.9096575, -5.4439297], [0.125, 0.1875]),
            ([-12.009473, -3.4103427, -5.448474], [0.125, 0.125]),
            ([-12.096297, -3.4103427, -4.95607], [0.0625, 0.125]),
        ],
    },
    LineRef {
        name: "remote_standing",
        local: false, third_person: false, sneaking: false,
        eye_height: 0x3fcf5c29,
        angler_prev: [0xc034000000000000, 0x4054000000000000, 0x4044000000000000],
        angler_pos: [0xc034333333333333, 0x4054000000000000, 0x4044266666666666],
        yaw: [0x432a0000, 0xc32f0000], pitch: [0x40a00000, 0x41000000], body: [0x43250000, 0x432c0000], swing: [0x3e800000, 0x3f000000],
        hook_prev: [0xc036000000000000, 0x4053800000000000, 0x403e000000000000],
        hook_pos: [0xc03619999999999a, 0x405379999999999a, 0x403d800000000000],
        camera: [0xc024000000000000, 0x405467ae147ae148, 0x4041800000000000],
        partial: 0x3ecccccd, view: [0x42c80000, 0x40400000],
        // line mode 3, stride 16, 17 vertices; rgba of vertex 0 = 0,0,0,255
        line: [
            [0xc140a3d7, 0xc05a3d71, 0xc0a66666],
            [0xc13e81c9, 0xc053f333, 0xc09378c8],
            [0xc13c5fba, 0xc04ceb85, 0xc0808b29],
            [0xc13a3dac, 0xc0452666, 0xc05b3b14],
            [0xc1381b9e, 0xc03ca3d7, 0xc0355fd6],
            [0xc135f98f, 0xc03363d7, 0xc00f8498],
            [0xc133d781, 0xc0296666, 0xbfd352b5],
            [0xc131b573, 0xc01eab85, 0xbf879c39],
            [0xc12f9365, 0xc0133333, 0xbeef96f6],
            [0xc12d7156, 0xc006fd71, 0x3dfd0bde],
            [0xc12b4f48, 0xbff4147b, 0x3f370e73],
            [0xc1292d3a, 0xbfd8b333, 0x3fa73db5],
            [0xc1270b2b, 0xbfbbd70a, 0x3ff2f430],
            [0xc124e91d, 0xbf9d8000, 0x401f5556],
            [0xc122c70f, 0xbf7b5c28, 0x40453094],
            [0xc120a500, 0xbf38c28f, 0x406b0bd1],
            [0xc11e82f2, 0xbee66665, 0x40887388],
        ],
        // quad mode 7, stride 24, 4 vertices
        quad: [
            ([-12.070527, -3.9096575, -4.9515257], [0.0625, 0.1875]),
            ([-11.983703, -3.9096575, -5.4439297], [0.125, 0.1875]),
            ([-12.009473, -3.4103427, -5.448474], [0.125, 0.125]),
            ([-12.096297, -3.4103427, -4.95607], [0.0625, 0.125]),
        ],
    },
];

fn dvec(bits: [u64; 3]) -> DVec3 {
    DVec3::new(
        f64::from_bits(bits[0]),
        f64::from_bits(bits[1]),
        f64::from_bits(bits[2]),
    )
}

/// `RenderManager.renderEntityStatic`'s interpolated bobber, as `TrackedEntity::render_pos`
/// computes it for the object pass.
fn bobber(reference: &LineRef) -> DVec3 {
    let previous = dvec(reference.hook_prev);
    let current = dvec(reference.hook_pos);
    previous + (current - previous) * f64::from(f32::from_bits(reference.partial))
}

fn angler(reference: &LineRef) -> FishingLineAngler {
    FishingLineAngler {
        prev_pos: dvec(reference.angler_prev),
        pos: dvec(reference.angler_pos),
        prev_yaw: f32::from_bits(reference.yaw[0]),
        yaw: f32::from_bits(reference.yaw[1]),
        prev_pitch: f32::from_bits(reference.pitch[0]),
        pitch: f32::from_bits(reference.pitch[1]),
        prev_body_yaw: f32::from_bits(reference.body[0]),
        body_yaw: f32::from_bits(reference.body[1]),
        eye_height: f32::from_bits(reference.eye_height),
        sneaking: reference.sneaking,
        prev_swing_progress: f32::from_bits(reference.swing[0]),
        swing_progress: f32::from_bits(reference.swing[1]),
        is_local: reference.local,
        third_person: reference.third_person,
    }
}

/// Every vertex of the line, bit for bit, in all three `RenderFish` branches (first-person hand,
/// local F5, remote player) including the sneak offset and a swing that wraps past 1.0.
#[test]
fn fishing_line_matches_the_vertices_vanilla_render_fish_uploads() {
    let mut failures = Vec::new();
    for reference in REFS {
        let mut out = Vec::new();
        emit_fishing_line(
            bobber(reference),
            dvec(reference.camera),
            angler(reference),
            f32::from_bits(reference.partial),
            &mut out,
        );
        assert_eq!(out.len(), 17, "{}", reference.name);
        for (index, (got, want)) in out.iter().zip(reference.line.iter()).enumerate() {
            assert_eq!(got.color, [0.0, 0.0, 0.0, 1.0], "{}", reference.name);
            if got.pos.map(f32::to_bits) != *want {
                failures.push(format!(
                    "{} vertex {index}: {:?} != {:?}",
                    reference.name,
                    got.pos,
                    want.map(f32::from_bits)
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} line vertices differ from RenderFish:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The bobber billboard's corners and UVs against `RenderFish`'s quad after the GL modelview.
#[test]
fn fishing_bobber_matches_the_render_fish_quad() {
    for reference in REFS {
        let relative = (bobber(reference) - dvec(reference.camera)).as_vec3();
        let mut out = Vec::new();
        emit_fishing_bobber(
            relative,
            f32::from_bits(reference.view[0]),
            f32::from_bits(reference.view[1]),
            &mut out,
        );
        assert_eq!(out.len(), 6, "{}", reference.name);
        // `emit_quad` fans the four corners as 0-1-2 / 0-2-3.
        let corners = [&out[0], &out[1], &out[2], &out[5]];
        for (corner, (want_pos, want_uv)) in corners.iter().zip(reference.quad.iter()) {
            let error = (Vec3::from(corner.pos) - Vec3::from(*want_pos)).length();
            assert!(
                error < 2e-5,
                "{}: corner {:?} != {:?}",
                reference.name,
                corner.pos,
                want_pos
            );
            assert_eq!(corner.uv, *want_uv, "{}", reference.name);
        }
    }
}
