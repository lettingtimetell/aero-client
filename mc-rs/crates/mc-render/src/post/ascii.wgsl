// ASCII: the Super Secret Settings entry that redraws the frame as Minecraft-font characters.
// Rust side and the design notes: `ascii.rs`.
//
// Pass 1 (`fs_cells`) draws a target with one texel per character cell: the cell's average
// colour. Pass 2 (`fs_ascii`) draws every frame pixel: it looks up its cell, picks a glyph from the
// brightness ramp and writes the glyph's ink in the cell's colour over a dark background.

struct Ascii {
    // x, y: frame size in pixels; z: 1 when the frame format is sRGB (loads decode, stores
    // encode), else 0; w: unused.
    frame: vec4<f32>,
    // x: cell size in pixels; y: glyphs in the ramp; z, w: one glyph's size in ramp texels.
    cell: vec4<f32>,
    // x: colour mode (0 colour, 1 white, 2 green); y: sample step of the cells pass in pixels;
    // z: 1 when the ramp's four edge glyphs (| - / \) follow the brightness glyphs; w: unused.
    mode: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Ascii;
// The frame so far (cells pass only).
@group(0) @binding(1) var scene: texture_2d<f32>;
// One texel per cell: its average colour, display-encoded (composite only).
@group(0) @binding(2) var cells: texture_2d<f32>;
// The glyph ramp: `cell.y` glyphs side by side, coverage in `r` (composite only).
@group(0) @binding(3) var ramp: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

// One triangle that covers the target.
@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// sRGB transfer functions, used only when the frame format is an sRGB one: the effect then still
// measures brightness and picks colours on the values the screen shows.
fn to_display(linear: vec3<f32>) -> vec3<f32> {
    let low = linear * 12.92;
    let high = 1.055 * pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= vec3<f32>(0.0031308));
}

fn to_linear(display: vec3<f32>) -> vec3<f32> {
    let low = display / 12.92;
    let high = pow((max(display, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, display <= vec3<f32>(0.04045));
}

@fragment
fn fs_cells(in: VsOut) -> @location(0) vec4<f32> {
    let size = max(i32(u.cell.x), 1);
    let step = clamp(i32(u.mode.y), 1, size);
    let last = vec2<i32>(u.frame.xy) - vec2<i32>(1);
    let origin = vec2<i32>(in.pos.xy) * size;
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var y = step / 2; y < size; y += step) {
        for (var x = step / 2; x < size; x += step) {
            let at = min(origin + vec2<i32>(x, y), last);
            sum += textureLoad(scene, at, 0).rgb;
            count += 1.0;
        }
    }
    var average = sum / max(count, 1.0);
    if (u.frame.z > 0.5) {
        average = to_display(average);
    }
    return vec4<f32>(average, 1.0);
}

fn luma_of(c: vec3<f32>) -> f32 {
    return clamp(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
}

// The luma of the cell `offset` away, clamped to the grid.
fn luma_at(cell: vec2<i32>, offset: vec2<i32>, last: vec2<i32>) -> f32 {
    return luma_of(textureLoad(cells, clamp(cell + offset, vec2<i32>(0), last), 0).rgb);
}

// A brightness change between neighbouring cells at least this big draws an edge glyph.
const EDGE_THRESHOLD: f32 = 0.16;

@fragment
fn fs_ascii(in: VsOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.pos.xy);
    let size = max(i32(u.cell.x), 1);
    let cell = pixel / size;
    let last = vec2<i32>(textureDimensions(cells)) - vec2<i32>(1);
    let colour = textureLoad(cells, cell, 0).rgb;

    // Brightness picks the glyph: darker cells get sparser characters. A gentle contrast stretch
    // spreads a typical scene (sky, stone, grass are close in brightness) over the whole ramp.
    let luma = luma_of(colour);
    let stretched = clamp((luma - 0.06) / 0.8, 0.0, 1.0);
    let glyphs = max(i32(u.cell.y), 1);
    var level = clamp(i32(floor(pow(stretched, 0.85) * f32(glyphs))), 0, glyphs - 1);

    // Outlines: where brightness changes sharply across the cell, an edge glyph along the change
    // (`|`, `-`, `/`, `\`), drawn on the brighter side only so outlines are one cell thick.
    if (u.mode.z > 0.5 && luma > 0.06) {
        let left = luma_at(cell, vec2<i32>(-1, 0), last);
        let right = luma_at(cell, vec2<i32>(1, 0), last);
        let up = luma_at(cell, vec2<i32>(0, -1), last);
        let down = luma_at(cell, vec2<i32>(0, 1), last);
        let gx = right - left;
        let gy = down - up;
        let ax = abs(gx);
        let ay = abs(gy);
        let across = select(0.5 * (up + down), 0.5 * (left + right), ax >= ay);
        if (max(ax, ay) >= EDGE_THRESHOLD && luma >= across) {
            if (ax > 2.0 * ay) {
                level = glyphs; // `|`: brightness changes left to right
            } else if (ay > 2.0 * ax) {
                level = glyphs + 1; // `-`: it changes top to bottom
            } else if (gx * gy > 0.0) {
                level = glyphs + 2; // `/`: it grows towards the bottom right
            } else {
                level = glyphs + 3; // `\`
            }
        }
    }

    // The pixel's texel inside its glyph. Cells are whole multiples of the font's pixel, so every
    // font pixel covers the same number of screen pixels.
    let local = pixel - cell * size;
    let glyph_size = vec2<i32>(i32(u.cell.z), i32(u.cell.w));
    let texel = vec2<i32>(local.x * glyph_size.x / size, local.y * glyph_size.y / size);
    let ink = textureLoad(ramp, vec2<i32>(level * glyph_size.x + texel.x, texel.y), 0).r;

    var ink_colour: vec3<f32>;
    var paper: vec3<f32>;
    let mode = i32(u.mode.x + 0.5);
    if (mode == 1) {
        // White on black.
        ink_colour = vec3<f32>(0.93, 0.93, 0.93);
        paper = vec3<f32>(0.0);
    } else if (mode == 2) {
        // Green phosphor.
        ink_colour = vec3<f32>(0.33, 1.0, 0.42);
        paper = vec3<f32>(0.0, 0.03, 0.01);
    } else {
        // The cell's own colour, brightened towards full intensity so a thin glyph still carries
        // its hue, over a darker wash of the same colour that keeps the scene readable.
        let peak = max(max(colour.r, colour.g), colour.b);
        let vivid = colour / max(peak, 1.0 / 255.0);
        ink_colour = clamp(mix(colour, vivid, 0.4) * (0.8 + 0.5 * peak), vec3<f32>(0.0),
            vec3<f32>(1.0));
        paper = colour * 0.33;
    }
    var out = mix(paper, ink_colour, ink);
    if (u.frame.z > 0.5) {
        out = to_linear(out);
    }
    return vec4<f32>(out, 1.0);
}
