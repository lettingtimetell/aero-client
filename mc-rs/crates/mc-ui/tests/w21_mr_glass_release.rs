//! The mod menu's glass drops its screen-sized blur chain when the
//! menu closes, and the frame after it opens again is byte-identical to the first open.
//!
//! One `ModMenuPhase` lives across frames (as the client keeps it for the session) behind a
//! shared handle, over the same striped scene the glass tests use. Skips (printing `SKIP:`)
//! without an adapter.

use std::cell::RefCell;
use std::rc::Rc;

use mc_render::w21_mr::glass_release_enabled;
use mc_render::SceneTextures;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};
use mc_ui::screens::modmenu::glass::{GlassFrame, GlassRect, ModMenuPhase};
use mc_ui::widget::{DrawList, UiPhase};

const W: u32 = 320;
const H: u32 = 240;
const CHAIN: &str = "modmenu glass scene";

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn scene() -> DrawList {
    let mut list = DrawList::new();
    list.rect(0, 0, W as i32, 40, 0xFFDC_B43C);
    for x in (0..W as i32).step_by(2) {
        list.rect(x, 40, x + 1, H as i32, 0xFFFF_FFFF);
    }
    list
}

fn glass() -> GlassFrame {
    GlassFrame {
        rects: vec![GlassRect {
            rect: [80.0, 80.0, 160.0, 120.0],
            radius: 16.0,
            alpha: 1.0,
        }],
        dim: 0.4,
        strength: 1.0,
        frost: 1.5,
        px_per_dip: 1.0,
        time: 0.0,
    }
}

/// A phase the list owns one handle of while the test keeps the other.
struct Shared<T>(Rc<RefCell<T>>);

impl<T: RenderPhase> RenderPhase for Shared<T> {
    fn name(&self) -> &'static str {
        self.0.borrow().name()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view, cam);
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.0.borrow().needs_scene_color()
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.borrow_mut().record_scene(gpu, enc, view, cam, scene);
    }
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList, ui: &Rc<RefCell<UiPhase>>) -> Vec<u8> {
    ui.borrow_mut().submit(scene());
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    drop(target);
    gpu.read_pixels().expect("readback")
}

fn chain_bytes(gpu: &Gpu) -> Option<u64> {
    let _ = gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let report = gpu.device.generate_allocator_report()?;
    Some(
        report
            .allocations
            .iter()
            .filter(|allocation| allocation.name == CHAIN)
            .map(|allocation| allocation.size)
            .sum(),
    )
}

#[test]
fn w21_mr_the_glass_chain_goes_with_the_menu_and_the_reopened_frame_is_identical() {
    let Some(mut gpu) = gpu() else { return };
    let ui = Rc::new(RefCell::new(UiPhase::new(&gpu, W, H, 1, false)));
    let menu = Rc::new(RefCell::new(ModMenuPhase::new(&gpu)));
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(Shared(Rc::clone(&ui))));
    phases.insert(
        PhaseOrder(PhaseOrder::UI.0 + 50),
        Box::new(Shared(Rc::clone(&menu))),
    );

    menu.borrow_mut().submit(DrawList::new(), glass());
    let first = frame(&mut gpu, &mut phases, &ui);
    assert!(menu.borrow().glass_chain_allocated());
    let open_bytes = chain_bytes(&gpu);
    if let Some(bytes) = open_bytes {
        assert!(bytes > 0, "the open menu holds its chain");
    }

    menu.borrow_mut().deactivate();
    let closed = frame(&mut gpu, &mut phases, &ui);
    assert_ne!(
        closed, first,
        "the glass changes the picture (negative control)"
    );
    assert_eq!(
        menu.borrow().glass_chain_allocated(),
        !glass_release_enabled(),
        "closing drops the chain (unless MC_W21_MR_GLASS_RELEASE=0)"
    );
    if open_bytes.is_some() && glass_release_enabled() {
        assert_eq!(chain_bytes(&gpu), Some(0), "the chain's memory is gone");
    }

    menu.borrow_mut().submit(DrawList::new(), glass());
    let reopened = frame(&mut gpu, &mut phases, &ui);
    assert!(menu.borrow().glass_chain_allocated());
    assert_eq!(
        reopened, first,
        "the reopened glass draws exactly what the first open drew"
    );
}
