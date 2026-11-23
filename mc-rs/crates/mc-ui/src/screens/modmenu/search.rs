//! The mod menu's search, menu side.
//!
//! The engine is [`mc_mods::menu_search`]: what matches (both spellings, optional spaces,
//! keywords, setting labels, descriptions and choices), the ranking and the golden test of plan
//! §5.5. [`MenuSearch`] keeps one search alive for the menu and runs it again only when the query
//! or the dynamic rows (a waypoint's name) change, so drawing a frame never searches. The list
//! (`list.rs`) holds one per menu.

use mc_mods::menu_search::{SearchIndex, SearchResults};
use mc_mods::{DynamicRows, ModDescriptor};

/// One live search: the catalog's index, the query its results belong to and the results.
#[derive(Debug)]
pub(super) struct MenuSearch {
    catalog: &'static [&'static ModDescriptor],
    /// Built at the first non-empty query (about 0.7 ms optimised), so a menu that is never
    /// searched never pays for it.
    index: Option<SearchIndex>,
    /// The query [`MenuSearch::results`] belongs to.
    query: String,
    /// [`rows_fingerprint`] of the dynamic rows the results saw.
    rows_print: u64,
    results: SearchResults,
}

impl MenuSearch {
    /// A search over `catalog` (one per menu). Builds nothing until the first query.
    pub(super) fn new(catalog: &'static [&'static ModDescriptor]) -> MenuSearch {
        MenuSearch {
            catalog,
            index: None,
            query: String::new(),
            rows_print: 0,
            results: SearchResults::default(),
        }
    }

    /// Bring the results up to date with `query` and this frame's dynamic `rows` (per catalog
    /// index). Searches only when one of them changed since the last call; returns whether it
    /// did. Allocation-free once the index is built and the result vectors have grown.
    pub(super) fn update(&mut self, query: &str, rows: &[DynamicRows]) -> bool {
        let searching = query.split_whitespace().next().is_some();
        let print = if searching { rows_fingerprint(rows) } else { 0 };
        if query == self.query && print == self.rows_print {
            return false;
        }
        self.query.clear();
        self.query.push_str(query);
        self.rows_print = print;
        if !searching && self.index.is_none() {
            // Nothing to search and nothing built: the results are already empty.
            return true;
        }
        let catalog = self.catalog;
        let index = self.index.get_or_insert_with(|| SearchIndex::new(catalog));
        index.search_into(query, rows, &mut self.results);
        true
    }

    /// The results of the last [`MenuSearch::update`] (empty when the query is).
    pub(super) fn results(&self) -> &SearchResults {
        &self.results
    }
}

/// A cheap fingerprint of every dynamic row's key and label (FNV-1a), to notice a renamed,
/// added or removed waypoint without keeping a copy of the rows.
fn rows_fingerprint(rows: &[DynamicRows]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    for (index, list) in rows.iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        eat(&(index as u64).to_le_bytes());
        for row in list.rows() {
            eat(&row.key.to_le_bytes());
            eat(row.label.as_bytes());
            eat(&[0xFF]);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_mods::menu_search::Hit;

    fn rows() -> Vec<DynamicRows> {
        vec![DynamicRows::default(); mc_mods::catalog::all().len()]
    }

    /// The search runs when the query changes, not every frame; the results are the engine's.
    #[test]
    fn the_search_runs_only_when_the_query_changes() {
        let mut search = MenuSearch::new(mc_mods::catalog::all());
        let rows = rows();
        assert!(!search.update("", &rows), "nothing to do yet");
        assert!(!search.update("   ", &rows) || search.results().is_empty());
        assert!(search.index.is_none(), "an unsearched menu builds no index");
        assert!(search.results().is_empty());
        assert!(search.update("gamma", &rows));
        assert!(search.index.is_some());
        assert!(!search.update("gamma", &rows), "same query, same rows");
        let fullbright = mc_mods::catalog::index_of("fullbright").unwrap();
        assert_eq!(
            search.results().best().map(Hit::mod_index),
            Some(fullbright)
        );
        assert_eq!(search.query, "gamma");
        assert!(
            search.update("", &rows),
            "clearing the query clears the results"
        );
        assert!(search.results().is_empty());
    }

    /// A waypoint renamed while the search is open shows up without retyping.
    #[test]
    fn a_changed_dynamic_row_runs_the_search_again() {
        let mut search = MenuSearch::new(mc_mods::catalog::all());
        let waypoints = mc_mods::catalog::index_of("waypoints").unwrap();
        let mut rows = rows();
        rows[waypoints].begin();
        rows[waypoints].push(1).label.push_str("Base");
        assert!(search.update("diamonds", &rows));
        assert!(!search.results().row_hit(waypoints, 1));
        rows[waypoints].begin();
        rows[waypoints].push(1).label.push_str("Diamonds");
        assert!(search.update("diamonds", &rows), "the row changed");
        assert!(search.results().row_hit(waypoints, 1));
        assert!(!search.update("diamonds", &rows));
    }
}
