//! Headless tests of the Accounts screen: no window, no GPU, no network.

use std::time::{Duration, Instant};

use super::*;
use crate::screens::clipboard::NullClipboard;
use crate::widget::test_support::uniform_font;
use crate::widget::ScaledResolution;

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
}

fn lang() -> Language {
    let mut l = Language::new("en_US");
    l.insert("gui.cancel", "Cancel");
    l.insert("gui.done", "Done");
    l
}

const ALPHA: AccountKey = AccountKey::Stored(1);
const BETA: AccountKey = AccountKey::Stored(2);

fn row(key: AccountKey, name: &str, status: RowStatus) -> AccountRow {
    AccountRow {
        key,
        name: name.to_owned(),
        head: UiTexture::gui("textures/entity/steve.png"),
        head_sheet_height: 64.0,
        status,
    }
}

fn rows() -> Vec<AccountRow> {
    vec![
        row(AccountKey::Launcher, "FromLauncher", RowStatus::Ready),
        row(ALPHA, "Alpha", RowStatus::Ready),
        row(BETA, "Beta", RowStatus::NeedsSignIn),
    ]
}

fn screen(active: Option<AccountKey>) -> AccountsScreen {
    let mut s = AccountsScreen::new(rows(), active);
    s.init(&ctx(), &lang());
    s
}

fn button(s: &AccountsScreen, id: i32) -> &Button {
    s.buttons.iter().find(|b| b.id == id).expect("button")
}

fn click_button(s: &mut AccountsScreen, id: i32) -> Action {
    let b = button(s, id);
    let (x, y) = (b.rect.x + 1, b.rect.y + 1);
    s.mouse_clicked(x, y, 0, &mut NullClipboard)
}

fn click_row(s: &mut AccountsScreen, index: usize, now: Instant) -> Action {
    let r = s.list.slot_rect(index);
    s.mouse_clicked_at(r.x + 40, r.y + 10, 0, &mut NullClipboard, now)
}

#[test]
fn the_account_in_use_starts_selected_and_cannot_be_used_again() {
    let s = screen(Some(ALPHA));
    assert_eq!(s.selected_key(), Some(ALPHA));
    assert!(!button(&s, ids::USE).enabled, "already in use");
    assert!(
        button(&s, ids::REMOVE).enabled,
        "an added account can be removed"
    );
    assert!(button(&s, ids::ADD).enabled);
    assert!(button(&s, ids::DONE).enabled);
}

#[test]
fn selecting_another_account_and_pressing_use_switches_to_it() {
    let mut s = screen(Some(ALPHA));
    let t = Instant::now();
    assert_eq!(click_row(&mut s, 0, t), Action::None);
    assert_eq!(s.selected_key(), Some(AccountKey::Launcher));
    assert!(button(&s, ids::USE).enabled);
    assert!(
        !button(&s, ids::REMOVE).enabled,
        "the launcher's session is not stored, so it cannot be removed"
    );
    assert_eq!(
        click_button(&mut s, ids::USE),
        Action::Accounts(AccountsAction::Use(AccountKey::Launcher))
    );
}

#[test]
fn a_double_click_uses_the_row() {
    let mut s = screen(Some(ALPHA));
    let t = Instant::now();
    assert_eq!(click_row(&mut s, 0, t), Action::None);
    assert_eq!(
        click_row(&mut s, 0, t + Duration::from_millis(100)),
        Action::Accounts(AccountsAction::Use(AccountKey::Launcher))
    );
    // Too slow for a double-click: just a selection.
    let mut s = screen(Some(ALPHA));
    click_row(&mut s, 0, t);
    assert_eq!(
        click_row(&mut s, 0, t + Duration::from_millis(400)),
        Action::None
    );
}

#[test]
fn in_a_world_nothing_switches_and_the_status_line_says_why() {
    let mut s = screen(Some(ALPHA));
    s.set_world_loaded(true);
    click_row(&mut s, 0, Instant::now());
    assert!(!button(&s, ids::USE).enabled);
    assert_eq!(click_button(&mut s, ids::USE), Action::None);
    // The account in use cannot be removed while it is playing.
    s.select(ALPHA);
    assert!(!button(&s, ids::REMOVE).enabled);
    let mut d = DrawList::new();
    let mut font = uniform_font();
    s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang());
    assert!(!d.is_empty());
}

#[test]
fn an_account_that_needs_a_new_sign_in_offers_to_sign_in_again() {
    let mut s = screen(Some(ALPHA));
    s.select(BETA);
    s.init(&ctx(), &lang());
    assert_eq!(button(&s, ids::USE).text, "Sign in again");
    assert!(button(&s, ids::USE).enabled);
    assert_eq!(
        click_button(&mut s, ids::USE),
        Action::Accounts(AccountsAction::Add)
    );
}

#[test]
fn remove_asks_first() {
    let mut s = screen(Some(ALPHA));
    s.select(ALPHA);
    assert_eq!(click_button(&mut s, ids::REMOVE), Action::None);
    assert!(s.is_confirming());
    s.init(&ctx(), &lang());
    // Cancel: back to the list, nothing removed.
    assert_eq!(click_button(&mut s, ids::CONFIRM_CANCEL), Action::None);
    assert!(!s.is_confirming());
    // Confirm.
    s.init(&ctx(), &lang());
    click_button(&mut s, ids::REMOVE);
    s.init(&ctx(), &lang());
    let mut d = DrawList::new();
    s.draw(&mut d, &mut uniform_font(), &ctx(), (0, 0), &lang());
    assert_eq!(
        click_button(&mut s, ids::CONFIRM_REMOVE),
        Action::Accounts(AccountsAction::Remove(ALPHA))
    );
    assert!(!s.is_confirming());
    // Escape on the question goes back to the list too.
    s.init(&ctx(), &lang());
    click_button(&mut s, ids::REMOVE);
    assert_eq!(s.back(), Action::None);
    assert!(!s.is_confirming());
}

#[test]
fn a_build_without_sign_in_explains_instead_of_starting_one() {
    let mut s = screen(None);
    s.set_sign_in_available(false);
    assert_eq!(click_button(&mut s, ids::ADD), Action::None);
    assert!(s.says_not_set_up());
    assert!(!s.is_signing_in());
    assert_eq!(
        not_set_up_text(&lang()),
        "Microsoft sign-in isn't set up in this build of Aero Client."
    );
    s.set_sign_in_available(true);
    assert_eq!(
        click_button(&mut s, ids::ADD),
        Action::Accounts(AccountsAction::Add)
    );
}

#[test]
fn the_sign_in_page_shows_the_code_and_cancel_goes_back() {
    let mut s = screen(Some(ALPHA));
    s.begin_sign_in();
    assert!(s.is_signing_in());
    s.init(&ctx(), &lang());
    s.sign_in_prompt(DeviceCodePrompt {
        user_code: "ABCD1234".into(),
        verification_uri: "https://example.invalid/link".into(),
        message: "Open the page and enter the code".into(),
        expires_in: Duration::from_secs(900),
        interval: Duration::from_secs(5),
    });
    let lines = s.sign_in().expect("page").body_lines(&lang());
    assert!(lines.iter().any(|line| line.ends_with("ABCD1234")));
    assert!(lines
        .iter()
        .any(|line| line == "https://example.invalid/link"));
    let mut d = DrawList::new();
    s.draw(&mut d, &mut uniform_font(), &ctx(), (0, 0), &lang());
    assert!(!d.is_empty());
    // Escape cancels the sign-in and returns to the list.
    assert_eq!(s.back(), Action::Accounts(AccountsAction::CancelAdd));
    assert!(!s.is_signing_in());
}

#[test]
fn a_failed_sign_in_explains_mojangs_403_and_offers_a_retry() {
    let mut s = screen(None);
    s.begin_sign_in();
    s.init(&ctx(), &lang());
    let unapproved = AuthError::ServicesForbidden {
        body: "Invalid app registration, see https://aka.ms/AppRegInfo for more information".into(),
    };
    s.sign_in_failed(&unapproved, &lang());
    let lines = s.sign_in().expect("page").body_lines(&lang());
    assert!(
        lines[0].contains("hasn't approved"),
        "a clear message, not the raw body: {lines:?}"
    );
    // "Sign in with Microsoft" is the retry: it asks the client for a new sign-in.
    s.init(&ctx(), &lang());
    let retry = s
        .sign_in()
        .expect("page")
        .buttons_for_tests()
        .into_iter()
        .find(|(_, enabled)| *enabled)
        .map(|(rect, _)| rect)
        .expect("an enabled button");
    assert_eq!(
        s.mouse_clicked(retry.x + 1, retry.y + 1, 0, &mut NullClipboard),
        Action::Accounts(AccountsAction::Add)
    );
}

#[test]
fn finishing_the_sign_in_returns_to_the_list_with_the_new_account_selected() {
    let mut s = screen(None);
    s.begin_sign_in();
    let mut added = rows();
    added.push(row(AccountKey::Stored(9), "Gamma", RowStatus::Ready));
    s.set_rows(added, Some(AccountKey::Stored(9)));
    s.sign_in_finished(AccountKey::Stored(9));
    assert!(!s.is_signing_in());
    assert_eq!(s.selected_key(), Some(AccountKey::Stored(9)));
    assert_eq!(s.active(), Some(AccountKey::Stored(9)));
}

#[test]
fn rows_draw_heads_names_and_states_and_an_empty_list_says_what_to_do() {
    let mut s = screen(Some(ALPHA));
    let mut d = DrawList::new();
    s.draw(&mut d, &mut uniform_font(), &ctx(), (0, 0), &lang());
    let skin = UiTexture::gui("textures/entity/steve.png");
    assert!(d.batches().iter().any(|batch| batch.texture == skin));
    let mut empty = AccountsScreen::new(Vec::new(), None);
    empty.init(&ctx(), &lang());
    assert!(!button(&empty, ids::USE).enabled);
    assert!(!button(&empty, ids::REMOVE).enabled);
    let mut d = DrawList::new();
    empty.draw(&mut d, &mut uniform_font(), &ctx(), (0, 0), &lang());
    assert!(!d.is_empty());
    assert_eq!(empty.back(), Action::Accounts(AccountsAction::Done));
}

#[test]
fn the_selection_follows_its_account_when_the_rows_change() {
    let mut s = screen(Some(ALPHA));
    s.select(BETA);
    // Alpha removed: Beta moves up a row and stays selected.
    s.set_rows(
        vec![
            row(AccountKey::Launcher, "FromLauncher", RowStatus::Ready),
            row(BETA, "Beta", RowStatus::Ready),
        ],
        Some(AccountKey::Launcher),
    );
    assert_eq!(s.selected_key(), Some(BETA));
    // The selected account removed: nothing selected.
    s.set_rows(rows()[..1].to_vec(), Some(AccountKey::Launcher));
    assert_eq!(s.selected_key(), None);
}
