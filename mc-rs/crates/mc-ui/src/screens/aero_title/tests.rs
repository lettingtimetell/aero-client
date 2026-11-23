//! Headless title-screen tests: the pop-in clock, the glass panes, hit testing and clicks, with
//! the deterministic 5-pixel test font and no GPU.

use std::time::{Duration, Instant};

use super::*;
use crate::widget::test_support::uniform_font;

const SCREEN: (f32, f32) = (1920.0, 1080.0);

struct Harness {
    text: MenuText,
    title: TitleFont,
    mc: FontRenderer,
    labels: [String; AeroButton::COUNT],
    screen: AeroTitleScreen,
    t0: Instant,
}

impl Harness {
    fn new() -> Harness {
        let t0 = Instant::now();
        Harness {
            text: MenuText::minecraft_only(),
            title: TitleFont::embedded(),
            mc: uniform_font(),
            labels: AeroButton::ALL.map(|b| b.english().to_owned()),
            screen: AeroTitleScreen::new(t0),
            t0,
        }
    }

    fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
        let mut list = DrawList::new();
        let mut env = AeroTitleEnv {
            text: &mut self.text,
            title_font: &mut self.title,
            mc_font: &mut self.mc,
            labels: &self.labels,
            version: "Minecraft 1.8.9",
            copyright: "Copyright Mojang AB. Do not distribute!",
            screen: SCREEN,
            now: self.t0 + Duration::from_millis(ms),
            time_s: ms as f32 / 1000.0,
        };
        let glass = self.screen.build(&mut env, &mut list);
        (list, glass)
    }
}

#[test]
fn the_title_and_buttons_pop_in_staggered_like_the_menus_windows() {
    let mut h = Harness::new();
    let (list, glass) = h.build(0);
    assert!(list.is_empty(), "nothing is visible at the first instant");
    assert!(glass.rects.is_empty());
    // The title starts first; button i starts STAGGER_MS * (i + 1) later.
    let (list, glass) = h.build(20);
    assert!(!list.is_empty(), "the title is fading in");
    assert!(glass.rects.is_empty(), "no button yet");
    let (_, glass) = h.build((anim::STAGGER_MS * 2.0) as u64 + 5);
    assert_eq!(glass.rects.len(), 2, "Multiplayer and Replays have started");
    assert!(glass.rects[0].alpha > glass.rects[1].alpha);
    // A rising button sits lower than where it lands.
    let landed = h.screen.button_rect(AeroButton::Multiplayer);
    assert!(glass.rects[0].rect[1] > landed.y);
    // Settled: one pane per button, fully opaque, in place.
    let done = (anim::OPEN_MS + AeroButton::COUNT as f32 * anim::STAGGER_MS) as u64;
    assert!(h.screen.settled(h.t0 + Duration::from_millis(done)));
    let (_, glass) = h.build(done);
    assert_eq!(glass.rects.len(), AeroButton::COUNT);
    for (pane, button) in glass.rects.iter().zip(AeroButton::ALL) {
        let r = h.screen.button_rect(button);
        assert_eq!(pane.alpha, 1.0);
        assert_eq!(pane.rect, [r.x, r.y, r.w, r.h]);
    }
    assert_eq!(glass.dim, 0.0, "the title screen never dims the greenhouse");
}

#[test]
fn a_click_activates_on_release_over_the_same_button_only() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let options = h.screen.button_rect(AeroButton::Options);
    let (x, y) = options.center();
    assert!(h.screen.mouse_pressed(x, y, 0));
    assert_eq!(h.screen.mouse_released(x, y, 0), Some(AeroButton::Options));
    // Released elsewhere: nothing.
    assert!(h.screen.mouse_pressed(x, y, 0));
    let quit = h.screen.button_rect(AeroButton::Quit);
    let (qx, qy) = quit.center();
    assert_eq!(h.screen.mouse_released(qx, qy, 0), None);
    // Right clicks and clicks between buttons do nothing.
    assert!(!h.screen.mouse_pressed(x, y, 1));
    assert_eq!(h.screen.mouse_released(x, y, 1), None);
    assert!(!h.screen.mouse_pressed(5.0, 5.0, 0));
    assert_eq!(h.screen.mouse_released(5.0, 5.0, 0), None);
}

#[test]
fn a_button_that_has_not_appeared_cannot_be_clicked() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let quit = h.screen.button_rect(AeroButton::Quit);
    // A fresh screen: Quit Game starts appearing last.
    h.screen.restart(h.t0 + Duration::from_millis(2000));
    let _ = h.build(2000 + 10);
    let (x, y) = quit.center();
    assert!(!h.screen.mouse_pressed(x, y, 0));
}

#[test]
fn hover_and_press_change_the_drawn_button() {
    let mut h = Harness::new();
    let (idle, _) = h.build(1000);
    let r = h.screen.button_rect(AeroButton::Multiplayer);
    h.screen.pointer_moved(r.x + 10.0, r.y + 10.0);
    // Two frames: the highlight eases in.
    let _ = h.build(1016);
    let (hovered, _) = h.build(1300);
    assert_ne!(
        idle.vertices().len(),
        hovered.vertices().len(),
        "the hover adds the left edge bar and the chevron"
    );
    let (x, y) = r.center();
    h.screen.pointer_moved(x, y);
    assert!(h.screen.mouse_pressed(x, y, 0));
    let _ = h.build(1316);
    let (_, glass) = h.build(1600);
    let pane = glass.rects[0].rect;
    assert!(
        pane[2] < r.w,
        "a pressed button sinks in: {} < {}",
        pane[2],
        r.w
    );
}

#[test]
fn the_crisp_layer_repeats_only_the_buttons_marks_at_part_strength() {
    let mut h = Harness::new();
    let (plain, _) = h.build(2000);
    let mut list = DrawList::new();
    let mut crisp = DrawList::new();
    let mut env = AeroTitleEnv {
        text: &mut h.text,
        title_font: &mut h.title,
        mc_font: &mut h.mc,
        labels: &h.labels,
        version: "Minecraft 1.8.9",
        copyright: "Copyright Mojang AB. Do not distribute!",
        screen: SCREEN,
        now: h.t0 + Duration::from_millis(2016),
        time_s: 2.0,
    };
    let glass = h.screen.build_layers(&mut env, &mut list, Some(&mut crisp));
    assert_eq!(glass.rects.len(), AeroButton::COUNT);
    assert_eq!(
        list.vertices().len(),
        plain.vertices().len(),
        "the main layer is unchanged"
    );
    assert!(!crisp.is_empty());
    assert!(
        crisp.vertices().len() < list.vertices().len(),
        "no title, shadows or glass fills in the crisp layer"
    );
    // Nothing above the first button: the title is not repeated.
    let top = h.screen.button_rect(AeroButton::Multiplayer).y - 4.0;
    assert!(crisp.vertices().iter().all(|v| v.pos[1] >= top));
    // Part strength: no vertex is more opaque than CRISP_ALPHA.
    assert!(crisp
        .vertices()
        .iter()
        .all(|v| v.color[3] <= CRISP_ALPHA + 1e-4));
}

#[test]
fn the_buttons_map_to_vanillas_actions() {
    assert_eq!(AeroButton::Multiplayer.action(), Action::OpenMultiplayer);
    // Replays has no vanilla action: the client opens the replay browser by variant.
    assert_eq!(AeroButton::Replays.action(), Action::None);
    // Mods opens the mod menu over the title screen.
    assert_eq!(AeroButton::Mods.action(), Action::OpenModMenu);
    assert_eq!(AeroButton::Options.action(), Action::OpenOptions);
    assert_eq!(
        AeroButton::Language.action(),
        Action::OpenOptionsPage(OptionsPage::Language)
    );
    assert_eq!(AeroButton::Quit.action(), Action::Quit);
}

#[test]
fn labels_come_from_the_language_without_vanillas_ellipsis() {
    let mut lang = mc_assets::Language::new("es_ES");
    lang.merge_lang_str(
        "menu.multiplayer=Multijugador\nmenu.options=Opciones...\noptions.language=Idioma...\n",
    );
    assert_eq!(
        labels(&lang),
        [
            "Multijugador",
            "Replays",
            "Mods",
            "Opciones",
            "Idioma",
            "Quit Game"
        ],
        "a missing key falls back to English"
    );
}

#[test]
fn the_stand_in_backdrop_covers_the_screen() {
    let mut list = DrawList::new();
    draw_stand_in_backdrop(&mut list, 640, 360);
    let max_y = list
        .vertices()
        .iter()
        .map(|v| v.pos[1])
        .fold(0.0_f32, f32::max);
    assert_eq!(max_y, 360.0);
}
