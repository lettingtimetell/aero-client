//! The terrain vertex format and CPU-side vertex buffers filled by `mesh_section`.
//!
//! Vanilla 1.8.9 uses `DefaultVertexFormats.BLOCK` verbatim: `position(3f) | color(4ub) |
//! tex(2f) | lightmap(2s)`, a 28-byte record. Positions and UVs therefore retain the exact
//! `f32` values written by `WorldRenderer`; a fluid height such as `8/9`, a rotated model, or a
//! generated-model UV is not snapped to an invented grid before OpenGL sees it. The lightmap
//! remains four bytes because the 1.8.9 terrain writer only supplies `level * 16` in its two
//! meaningful channels.
//!
//! The previous 20-byte `Unorm16` representation is retained below as a regression control.
//! It is never the normal renderer path: `TerrainOptions::packed_vertices` explicitly packs a
//! faithful CPU buffer while uploading it, so tests can prove that a precision-sensitive scene
//! does not accidentally start accepting the lossy format again.
//!
//! java: `client/renderer/vertex/DefaultVertexFormats.java`, `WorldRenderer.java`

/// One terrain vertex, exactly as vanilla's `DefaultVertexFormats.BLOCK` supplies it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    /// Section-local position, written to the GPU without quantisation.
    pub pos: [f32; 3],
    /// Atlas UV, written to the GPU without quantisation.
    pub uv: [f32; 2],
    /// Biome tint × vanilla's per-face brightness × ambient occlusion.
    pub color: [u8; 4],
    /// `[sky, block, material, weight]`; light channels hold vanilla's `level * 16` byte.
    /// The last two bytes are zero, as vanilla's, except for a shader pack's material code and
    /// sway weight (`terrain::pack::material`, `PackQuad::light_meta`).
    pub light: [u8; 4],
}

/// Bytes per terrain vertex: vanilla's `DefaultVertexFormats.BLOCK` stride.
pub const VERTEX_SIZE: usize = std::mem::size_of::<TerrainVertex>();

impl TerrainVertex {
    /// Builds a terrain vertex without changing its position or UV bits.
    #[must_use]
    pub const fn new(pos: [f32; 3], uv: [f32; 2], color: [u8; 4], light: [u8; 4]) -> Self {
        Self {
            pos,
            uv,
            color,
            light,
        }
    }

    /// Returns the unmodified section-local position.
    #[must_use]
    pub const fn position(&self) -> [f32; 3] {
        self.pos
    }

    /// Returns the unmodified atlas coordinate.
    #[must_use]
    pub const fn texcoord(&self) -> [f32; 2] {
        self.uv
    }

    /// The `wgpu` attributes for the faithful Java layout, at `@location(0..=3)`.
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Unorm8x4,
        3 => Uint8x4,
    ];

    /// Returns the wgpu layout for this vertex.
    #[must_use]
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: VERTEX_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// The old lossy control format, used only with
/// [`TerrainOptions::packed_vertices`](super::phase::TerrainOptions::packed_vertices).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedTerrainVertex {
    /// Quantised section-local position; the fourth component is unused.
    pub pos: [u16; 4],
    /// Quantised atlas coordinate.
    pub uv: [u16; 2],
    /// Unmodified vertex colour.
    pub color: [u8; 4],
    /// Unmodified light channels.
    pub light: [u8; 4],
}

/// Bytes per old packed control vertex.
pub const PACKED_VERTEX_SIZE: usize = std::mem::size_of::<PackedTerrainVertex>();

const POS_MIN: f32 = -8.0;
const POS_UNITS_PER_BLOCK: f32 = 2048.0;
const UV_UNITS: f32 = 32768.0;

fn encode_pos(v: f32) -> u16 {
    let code = ((v - POS_MIN) * POS_UNITS_PER_BLOCK).round();
    assert!(
        (0.0..=65535.0).contains(&code),
        "section-local coordinate {v} is outside the old packed control range"
    );
    code as u16
}

fn encode_uv(v: f32) -> u16 {
    let code = (v * UV_UNITS).round();
    assert!(
        (0.0..=65535.0).contains(&code),
        "atlas uv {v} is outside the old packed control range"
    );
    code as u16
}

impl PackedTerrainVertex {
    /// Layout consumed by `terrain.wgsl`'s packed-control entry point.
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Unorm16x4,
        1 => Unorm16x2,
        2 => Unorm8x4,
        3 => Uint8x4,
    ];

    /// Returns the wgpu layout for the packed control vertex.
    #[must_use]
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: PACKED_VERTEX_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

impl From<TerrainVertex> for PackedTerrainVertex {
    fn from(v: TerrainVertex) -> Self {
        Self {
            pos: [
                encode_pos(v.pos[0]),
                encode_pos(v.pos[1]),
                encode_pos(v.pos[2]),
                0,
            ],
            uv: [encode_uv(v.uv[0]), encode_uv(v.uv[1])],
            color: v.color,
            light: v.light,
        }
    }
}

/// Pack a run of faithful vertex bytes into the regression-control layout.
#[must_use]
pub fn pack_bytes(bytes: &[u8]) -> Vec<u8> {
    assert_eq!(bytes.len() % VERTEX_SIZE, 0, "not a whole vertex run");
    let faithful: &[TerrainVertex] = bytemuck::cast_slice(bytes);
    let packed: Vec<PackedTerrainVertex> = faithful.iter().copied().map(Into::into).collect();
    bytemuck::cast_slice(&packed).to_vec()
}

/// One render layer's geometry for one section: a flat run of quads, four vertices each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Buffer {
    verts: Vec<TerrainVertex>,
}

impl Buffer {
    /// Makes an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes an empty buffer with room for `quads` quads.
    #[must_use]
    pub fn with_capacity(quads: usize) -> Self {
        Self {
            verts: Vec::with_capacity(quads * 4),
        }
    }

    /// Reserves room for `quads` additional quads.
    pub fn reserve(&mut self, quads: usize) {
        self.verts.reserve(quads * 4);
    }

    /// Appends one quad in baked-model winding order.
    pub fn push_quad(&mut self, quad: [TerrainVertex; 4]) {
        self.verts.extend_from_slice(&quad);
    }

    /// Returns all vertices in quad order.
    #[must_use]
    pub fn vertices(&self) -> &[TerrainVertex] {
        &self.verts
    }

    /// Returns all vertices for in-place translucent re-sorting.
    pub fn vertices_mut(&mut self) -> &mut [TerrainVertex] {
        &mut self.verts
    }

    /// Returns the number of stored quads.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.verts.len() / 4
    }

    /// Returns the byte count of the faithful upload.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.verts.len() * VERTEX_SIZE
    }

    /// Whether no geometry is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    /// Returns quad `i`.
    #[must_use]
    pub fn quad(&self, i: usize) -> &[TerrainVertex] {
        &self.verts[i * 4..i * 4 + 4]
    }

    /// Returns vertex bytes ready for a faithful GPU upload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.verts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vertex_matches_vanillas_twenty_eight_byte_block_layout() {
        assert_eq!(VERTEX_SIZE, 28);
        assert_eq!(
            PACKED_VERTEX_SIZE, 20,
            "the regression control must stay distinct"
        );
        let offsets: Vec<u64> = TerrainVertex::ATTRIBUTES.iter().map(|a| a.offset).collect();
        assert_eq!(offsets, vec![0, 12, 20, 24]);
        assert_eq!(TerrainVertex::layout().array_stride, 28);
        assert_eq!(PackedTerrainVertex::layout().array_stride, 20);
    }

    #[test]
    fn off_grid_position_and_uv_keep_their_exact_f32_bits() {
        // Fluid 8/9 and generated/rotated model coordinates are not authored on the old
        // 1/2048 / 1/32768 grids. This fails if `TerrainVertex::new` resumes quantising.
        let pos = [1.0 - 1.0 / 9.0, 0.123_456_7, 1.0 / 3.0];
        let uv = [0.123_456_7, 0.987_654_3];
        let v = TerrainVertex::new(pos, uv, [1, 2, 3, 4], [240, 16, 0, 0]);
        assert_eq!(v.position().map(f32::to_bits), pos.map(f32::to_bits));
        assert_eq!(v.texcoord().map(f32::to_bits), uv.map(f32::to_bits));

        let packed: PackedTerrainVertex = v.into();
        let decoded_pos = [
            f32::from(packed.pos[0]) / POS_UNITS_PER_BLOCK + POS_MIN,
            f32::from(packed.pos[1]) / POS_UNITS_PER_BLOCK + POS_MIN,
            f32::from(packed.pos[2]) / POS_UNITS_PER_BLOCK + POS_MIN,
        ];
        let decoded_uv = [
            f32::from(packed.uv[0]) / UV_UNITS,
            f32::from(packed.uv[1]) / UV_UNITS,
        ];
        assert_ne!(decoded_pos.map(f32::to_bits), pos.map(f32::to_bits));
        assert_ne!(decoded_uv.map(f32::to_bits), uv.map(f32::to_bits));
    }

    #[test]
    fn packing_preserves_the_non_geometric_channels_and_shrinks_the_upload() {
        let v = TerrainVertex::new(
            [1.5, 2.25, 15.9375],
            [0.25, 0.75],
            [1, 2, 3, 4],
            [240, 16, 0, 0],
        );
        let mut b = Buffer::new();
        b.push_quad([v; 4]);
        let packed = pack_bytes(b.as_bytes());
        assert_eq!(packed.len(), 4 * PACKED_VERTEX_SIZE);
        assert_eq!(b.as_bytes().len(), 4 * VERTEX_SIZE);
        let back: &[PackedTerrainVertex] = bytemuck::cast_slice(&packed);
        assert_eq!(back[0].color, v.color);
        assert_eq!(back[0].light, v.light);
    }

    #[test]
    fn buffer_counts_quads_not_vertices() {
        let mut b = Buffer::new();
        assert!(b.is_empty());
        b.push_quad([TerrainVertex::default(); 4]);
        b.push_quad([TerrainVertex::default(); 4]);
        assert_eq!(b.quad_count(), 2);
        assert_eq!(b.vertices().len(), 8);
        assert_eq!(b.as_bytes().len(), 8 * VERTEX_SIZE);
    }
}
