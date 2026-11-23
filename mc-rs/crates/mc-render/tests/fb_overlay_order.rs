//! The overlay pass draws the vignette after the water and fire quads, as vanilla 1.8.9 does.
//!
//! java: `EntityRenderer.renderWorldPass` → `renderHand` → `ItemRenderer.renderOverlays` draws the
//! suffocation, water and fire quads inside the world render. `GuiIngame.renderGameOverlay` runs
//! afterwards and begins with `renderVignette`, then `renderPortal`. So the vignette also darkens
//! the water and fire quads, and the portal swirl goes on top of the vignette.
//!
//! The test renders one `OverlayPhase` with water, fire and a dark vignette, then the same
//! effects split across two phases in a known order. Blending happens per draw against the stored
//! target either way, so the single-phase frame must equal the split frame in vanilla's order byte
//! for byte. It must also differ from the reverse order, the old mc-rs order, which proves the
//! comparison can see the difference.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::post::{OverlayKind, OverlayPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

const SIZE: u32 = 96;
const GREY: wgpu::Color = wgpu::Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

struct Handle(Rc<RefCell<OverlayPhase>>);

impl RenderPhase for Handle {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view, cam);
    }

    fn name(&self) -> &'static str {
        self.0.borrow().name()
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }
}

/// Which effects one `OverlayPhase` draws.
#[derive(Clone, Copy)]
struct Effects {
    water_and_fire: bool,
    vignette: bool,
}

fn phase(gpu: &Gpu, rm: &ResourceManager, effects: Effects) -> Box<dyn RenderPhase> {
    let mut overlay = OverlayPhase::new(gpu);
    overlay.load_vignette_texture(gpu, rm);
    overlay.load_water_texture(gpu, rm);
    overlay.load_fire_texture(gpu, rm);
    if effects.water_and_fire {
        overlay.kind = OverlayKind::Water;
        overlay.fire = true;
    }
    // A dark place: brightness 0 keeps the eased vignette at its fully dark start. `None` is
    // the Fast-graphics gate, which draws no vignette.
    overlay.set_vignette_brightness(effects.vignette.then_some(0.0));
    assert!(overlay.enabled(), "the phase draws something");
    Box::new(Handle(Rc::new(RefCell::new(overlay))))
}

fn render(gpu: &mut Gpu, phases: Vec<(PhaseOrder, Box<dyn RenderPhase>)>) -> Vec<u8> {
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let mut list = PhaseList::new();
    list.clear_color = GREY;
    for (order, phase) in phases {
        list.insert(order, phase);
    }
    let frame = gpu.acquire().expect("acquire");
    list.render(gpu, frame.view(), &cam, 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.chunks(4).zip(b.chunks(4)).filter(|(p, q)| p != q).count()
}

#[test]
fn the_vignette_darkens_the_water_and_fire_quads_like_vanilla() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let all = Effects {
        water_and_fire: true,
        vignette: true,
    };
    let media = Effects {
        water_and_fire: true,
        vignette: false,
    };
    let vignette = Effects {
        water_and_fire: false,
        vignette: true,
    };
    let later = PhaseOrder(PhaseOrder::POST.0 + 1);

    let phases = vec![(PhaseOrder::POST, phase(&gpu, &rm, all))];
    let combined = render(&mut gpu, phases);
    let phases = vec![
        (PhaseOrder::POST, phase(&gpu, &rm, media)),
        (later, phase(&gpu, &rm, vignette)),
    ];
    let vanilla_order = render(&mut gpu, phases);
    let phases = vec![
        (PhaseOrder::POST, phase(&gpu, &rm, vignette)),
        (later, phase(&gpu, &rm, media)),
    ];
    let old_order = render(&mut gpu, phases);

    let sensitivity = differing(&vanilla_order, &old_order);
    assert!(
        sensitivity > 0,
        "negative control: both orders render the same frame, so the test cannot tell them apart"
    );
    let diff = differing(&combined, &vanilla_order);
    eprintln!(
        "overlay order: {diff} pixels differ from vanilla's order; the two orders differ on \
         {sensitivity} of {} pixels",
        SIZE * SIZE
    );
    assert_eq!(
        diff, 0,
        "the single pass must draw water, fire, then the vignette"
    );
}
