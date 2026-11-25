//! Minecraft 1.7.10's scoreboard sidebar. A child of `hud`, for [`Sidebar`]'s private state.
//!
//! java17 `GuiIngame.renderScoreboard` (`client/gui/GuiIngame.java:490-536`) draws the sidebar
//! only while the objective has at most 15 scores, and then draws every one of them: above 15 it
//! draws nothing at all, and it has no rule for names starting with `#`. (1.8.9 drops `#` names and
//! shows the top 15 of a longer list.) The objective is the sidebar slot's alone (java17 `:372`);
//! 1.7.10 has no team-colour slots.

use super::{Sidebar, SIDEBAR_MAX_ROWS};

impl Sidebar {
    /// The 1.7.10 sidebar for `getSortedScores` output (raw names, ascending by points), names
    /// formatted by the caller's `ScorePlayerTeam.formatPlayerName`; `None` above 15 scores.
    #[must_use]
    pub fn from_scores_v5(
        title: impl Into<String>,
        scores: &[(String, i32)],
        mut format: impl FnMut(&str) -> String,
    ) -> Option<Sidebar> {
        if scores.len() > SIDEBAR_MAX_ROWS {
            return None;
        }
        Some(Sidebar {
            title: title.into(),
            rows: scores
                .iter()
                .map(|(name, points)| (format(name), *points))
                .collect(),
            skip: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(names: &[&str]) -> Vec<(String, i32)> {
        names
            .iter()
            .enumerate()
            .map(|(points, name)| ((*name).to_owned(), points as i32))
            .collect()
    }

    #[test]
    fn sixteen_scores_hide_the_1_7_10_sidebar() {
        let names: Vec<String> = (0..16).map(|i| format!("p{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        assert!(Sidebar::from_scores_v5("t", &scores(&names), str::to_owned).is_none());
        // 1.8.9 shows the top 15 (negative control).
        let v47 = Sidebar::from_scores("t", &scores(&names), str::to_owned);
        assert_eq!(v47.visible_rows().len(), 15);
        // Fifteen are all shown in both.
        let fifteen = &names[..15];
        let v5 = Sidebar::from_scores_v5("t", &scores(fifteen), str::to_owned).unwrap();
        assert_eq!(v5.visible_rows().len(), 15);
    }

    #[test]
    fn names_starting_with_a_hash_are_shown_in_1_7_10() {
        let board = scores(&["a", "#x", "b"]);
        let v5 = Sidebar::from_scores_v5("t", &board, |name| format!("[{name}]")).unwrap();
        assert_eq!(
            v5.visible_rows(),
            [
                ("[a]".to_owned(), 0),
                ("[#x]".to_owned(), 1),
                ("[b]".to_owned(), 2)
            ]
        );
        assert_eq!(
            Sidebar::from_scores("t", &board, str::to_owned)
                .visible_rows()
                .len(),
            2
        );
    }
}
