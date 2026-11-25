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
    /// The selection the client hands in (and saves a pick to).
    game_version: GameVersion,
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
            game_version: GameVersion::V1_8_9,
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
            game_version: self.game_version,
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
    // The title starts first; button i starts STAGGER_MS * (i + 1) later, and the version slider
    // with Multiplayer.
    let (list, glass) = h.build(20);
    assert!(!list.is_empty(), "the title is fading in");
    assert!(glass.rects.is_empty(), "no button yet");
    let (_, glass) = h.build((anim::STAGGER_MS * 2.0) as u64 + 5);
    // The buttons' panes come first, the slider's after them.
    assert_eq!(
        glass.rects.len(),
        3,
        "Multiplayer, Replays and the slider have started"
    );
    assert!(glass.rects[0].alpha > glass.rects[1].alpha);
    assert_eq!(
        glass.rects[2].alpha, glass.rects[0].alpha,
        "the slider with Multiplayer"
    );
    // A rising button sits lower than where it lands.
    let landed = h.screen.button_rect(AeroButton::Multiplayer);
    assert!(glass.rects[0].rect[1] > landed.y);
    // Settled: one pane per button and the slider's, fully opaque, in place, and the Discord
    // button's last.
    let done = (anim::OPEN_MS + AeroButton::COUNT as f32 * anim::STAGGER_MS) as u64;
    assert!(h.screen.settled(h.t0 + Duration::from_millis(done)));
    let (_, glass) = h.build(done);
    assert_eq!(glass.rects.len(), AeroButton::COUNT + 2);
    let discord = h.screen.discord.rect().expect("the Discord button is in");
    assert_eq!(
        glass.rects[AeroButton::COUNT + 1].rect,
        [discord.x, discord.y, discord.w, discord.h]
    );
    for (pane, button) in glass.rects.iter().zip(AeroButton::ALL) {
        let r = h.screen.button_rect(button);
        assert_eq!(pane.alpha, 1.0);
        assert_eq!(pane.rect, [r.x, r.y, r.w, r.h]);
    }
    let slider = h.screen.version_slider.rect().expect("on screen");
    let pane = glass.rects[AeroButton::COUNT];
    assert_eq!(pane.alpha, 1.0);
    assert_eq!(pane.rect, [slider.x, slider.y, slider.w, slider.h]);
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
        game_version: h.game_version,
        copyright: "Copyright Mojang AB. Do not distribute!",
        screen: SCREEN,
        now: h.t0 + Duration::from_millis(2016),
        time_s: 2.0,
    };
    let glass = h.screen.build_layers(&mut env, &mut list, Some(&mut crisp));
    assert_eq!(
        glass.rects.len(),
        AeroButton::COUNT + 2,
        "the buttons, the slider and the Discord button"
    );
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
    // Nothing above the column (the version slider heads it): the title is not repeated.
    let top = h.screen.version_slider.rect().expect("on screen").y - 4.0;
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

// ---- the version slider ----------------------------------------------------------------------

/// Where `version` sits on the slider in the last frame.
fn slot_centre(h: &Harness, version: GameVersion) -> (f32, f32) {
    h.screen.version_slider.version_rect(version).center()
}

/// A left click at `(x, y)`, routed to the slider as the client routes it; a pick is saved as
/// the client saves it, and the next frame (`ms`) is built.
fn click_slider(h: &mut Harness, (x, y): (f32, f32), ms: u64) -> Option<GameVersion> {
    h.screen.pointer_moved(x, y);
    let slider = &mut h.screen.version_slider;
    assert!(slider.mouse_pressed(x, y, 0), "the slider takes it");
    let picked = slider.mouse_released(x, y, 0);
    if let Some(version) = picked {
        h.game_version = version;
    }
    let _ = h.build(ms);
    picked
}

#[test]
fn the_version_slider_heads_the_column_and_pops_in_with_multiplayer() {
    let mut h = Harness::new();
    let _ = h.build(20);
    assert!(h.screen.version_slider.rect().is_none(), "the title alone");
    let _ = h.build(anim::STAGGER_MS as u64 + 5);
    let slider = h.screen.version_slider.rect().expect("with Multiplayer");
    let (x, y) = h.screen.button_rect(AeroButton::Multiplayer).center();
    assert_eq!(h.screen.hit(x, y), Some(AeroButton::Multiplayer));
    let _ = h.build(1000);
    let multiplayer = h.screen.button_rect(AeroButton::Multiplayer);
    assert!(slider.bottom() < multiplayer.y, "above Multiplayer");
    assert_eq!((slider.x, slider.w), (multiplayer.x, multiplayer.w));
    let slider = &h.screen.version_slider;
    assert_eq!(slider.selected(), GameVersion::V1_8_9, "1.8.9 by default");
    assert_eq!(slider.knob(), 1.0, "the knob rests on 1.8.9");
}

/// A press held on the knob whose release never comes (the window lost focus with the button
/// down) must not turn a later click elsewhere into a pick, nor swallow that click.
#[test]
fn a_press_left_on_the_knob_never_picks_on_a_later_click_elsewhere() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let knob = slot_centre(&h, GameVersion::V1_8_9);
    h.screen.pointer_moved(knob.0, knob.1);
    assert!(h.screen.version_slider.mouse_pressed(knob.0, knob.1, 0));
    // Later, a click on Multiplayer 120 dip to the left, routed as the client routes it: the
    // slider first, then the buttons.
    let y = h.screen.button_rect(AeroButton::Multiplayer).center().1;
    let x = knob.0 - 120.0;
    h.screen.pointer_moved(x, y);
    let _ = h.build(1016);
    assert!(
        !h.screen.version_slider.mouse_pressed(x, y, 0),
        "not its own"
    );
    assert!(
        !h.screen.version_slider.has_press(),
        "the old press is gone"
    );
    assert!(h.screen.mouse_pressed(x, y, 0));
    assert_eq!(h.screen.version_slider.mouse_released(x, y, 0), None);
    assert_eq!(
        h.screen.mouse_released(x, y, 0),
        Some(AeroButton::Multiplayer)
    );
    for ms in (1032..1600).step_by(16) {
        let _ = h.build(ms);
    }
    assert_eq!(h.screen.version_slider.selected(), GameVersion::V1_8_9);
    assert_eq!(h.screen.version_slider.knob(), 1.0, "back on 1.8.9");
}

#[test]
fn a_cancelled_press_drags_nothing_and_picks_nothing() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let knob = slot_centre(&h, GameVersion::V1_8_9);
    h.screen.pointer_moved(knob.0, knob.1);
    assert!(h.screen.version_slider.mouse_pressed(knob.0, knob.1, 0));
    // The client's response to losing focus.
    h.screen.version_slider.cancel_press();
    assert!(!h.screen.version_slider.has_press());
    // The pointer moves on with no button held: the knob stays put.
    let x = knob.0 - 120.0;
    h.screen.pointer_moved(x, knob.1);
    let _ = h.build(1016);
    assert_eq!(
        h.screen.version_slider.knob(),
        1.0,
        "no drag without a press"
    );
    assert_eq!(h.screen.version_slider.mouse_released(x, knob.1, 0), None);
}

#[test]
fn clicking_a_version_picks_it_and_the_knob_slides_there() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    // The selected version: nothing to pick.
    let current = slot_centre(&h, GameVersion::V1_8_9);
    assert_eq!(click_slider(&mut h, current, 1016), None);
    // The other one.
    let older = slot_centre(&h, GameVersion::V1_7_10);
    assert_eq!(
        click_slider(&mut h, older, 1032),
        Some(GameVersion::V1_7_10)
    );
    assert_eq!(h.screen.version_slider.selected(), GameVersion::V1_7_10);
    // The knob slides over a few frames rather than jumping.
    let moving = h.screen.version_slider.knob();
    assert!(moving > 0.0 && moving < 1.0, "{moving}");
    for ms in (1048..1600).step_by(16) {
        let _ = h.build(ms);
    }
    assert_eq!(h.screen.version_slider.knob(), 0.0, "it rests on 1.7.10");
    // And back.
    assert_eq!(
        click_slider(&mut h, current, 1616),
        Some(GameVersion::V1_8_9)
    );
}

#[test]
fn dragging_the_knob_picks_the_version_it_is_released_nearest_to() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let from = slot_centre(&h, GameVersion::V1_8_9);
    let slot = from.0 - slot_centre(&h, GameVersion::V1_7_10).0;
    // Short of the middle: it springs back, nothing picked.
    h.screen.pointer_moved(from.0, from.1);
    assert!(h.screen.version_slider.mouse_pressed(from.0, from.1, 0));
    let short = from.0 - slot * 0.4;
    h.screen.pointer_moved(short, from.1);
    let _ = h.build(1016);
    let knob = h.screen.version_slider.knob();
    assert!(
        (knob - 0.6).abs() < 0.01,
        "the knob follows the pointer: {knob}"
    );
    assert_eq!(
        h.screen.version_slider.mouse_released(short, from.1, 0),
        None
    );
    for ms in (1032..1600).step_by(16) {
        let _ = h.build(ms);
    }
    assert_eq!(h.screen.version_slider.knob(), 1.0, "back on 1.8.9");
    // Past the middle, released off the row: 1.7.10.
    assert!(h.screen.version_slider.mouse_pressed(from.0, from.1, 0));
    let (past, below) = (from.0 - slot * 0.7, from.1 + 200.0);
    h.screen.pointer_moved(past, below);
    let _ = h.build(1616);
    assert_eq!(
        h.screen.version_slider.mouse_released(past, below, 0),
        Some(GameVersion::V1_7_10)
    );
}

#[test]
fn only_a_left_click_on_a_shown_slider_picks() {
    let mut h = Harness::new();
    let _ = h.build(1000);
    let older = slot_centre(&h, GameVersion::V1_7_10);
    let quit = h.screen.button_rect(AeroButton::Quit).center();
    // The right button: taken, picks nothing.
    assert!(h.screen.version_slider.mouse_pressed(older.0, older.1, 1));
    assert!(!h.screen.version_slider.has_press());
    assert_eq!(
        h.screen.version_slider.mouse_released(older.0, older.1, 1),
        None
    );
    // Pressed on the slider, released on a button: nothing.
    assert!(h.screen.version_slider.mouse_pressed(older.0, older.1, 0));
    assert_eq!(
        h.screen.version_slider.mouse_released(quit.0, quit.1, 0),
        None
    );
    assert!(!h.screen.version_slider.has_press());
    // A press off the slider is not its own; one before it has appeared neither.
    assert!(!h.screen.version_slider.mouse_pressed(quit.0, quit.1, 0));
    h.screen.restart(h.t0 + Duration::from_millis(2000));
    let _ = h.build(2010);
    assert!(!h.screen.version_slider.mouse_pressed(older.0, older.1, 0));
}

#[test]
fn the_knob_follows_the_selection_wherever_it_was_changed() {
    let mut h = Harness::new();
    // A saved choice shows at once, without a slide.
    h.game_version = GameVersion::V1_7_10;
    let _ = h.build(1000);
    assert_eq!(h.screen.version_slider.knob(), 0.0);
    // Changed elsewhere (the mod menu's row): the knob slides there.
    h.game_version = GameVersion::V1_8_9;
    let _ = h.build(1016);
    let moving = h.screen.version_slider.knob();
    assert!(moving > 0.0 && moving < 1.0, "{moving}");
    for ms in (1032..1600).step_by(16) {
        let _ = h.build(ms);
    }
    assert_eq!(h.screen.version_slider.knob(), 1.0);
    assert_eq!(h.screen.version_slider.selected(), GameVersion::V1_8_9);
}

#[test]
fn the_knob_is_drawn_behind_the_selected_version() {
    let mut h = Harness::new();
    let (knob_on_new, _) = h.build(1000);
    let mut h_old = Harness::new();
    h_old.game_version = GameVersion::V1_7_10;
    let (knob_on_old, _) = h_old.build(1000);
    // The knob's bright fill is in the selected version's slot (the other holds a white label).
    let bright_in = |list: &DrawList, version: GameVersion| {
        let slot = h.screen.version_slider.version_rect(version);
        list.vertices()
            .iter()
            .filter(|v| v.color[0] > 0.9 && v.color[3] > 0.7)
            .filter(|v| slot.contains(v.pos[0], v.pos[1]))
            .count()
    };
    let (old, new) = (GameVersion::V1_7_10, GameVersion::V1_8_9);
    assert!(bright_in(&knob_on_new, new) > 2 * bright_in(&knob_on_new, old));
    assert!(bright_in(&knob_on_old, old) > 2 * bright_in(&knob_on_old, new));
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
