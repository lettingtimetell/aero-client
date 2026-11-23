//! The generated DEBUG_WORLD sidecar must reproduce the recorded Java tuple list exactly.

use mc_data::block::debug_world_states;

const JAVA_DEBUG_GRID: &str = include_str!("../extractor/out/debug-grid.tsv");
const HEADER: &str =
    "index\tregistry_id\tregistry_name\traw_storage_id\tmetadata\tjava_state_id\tproperties";
const EXPECTED_ROWS: usize = 7806;

#[test]
fn debug_world_states_match_java_oracle_order_storage_and_properties() {
    let mut lines = JAVA_DEBUG_GRID.lines();
    assert_eq!(lines.next(), Some(HEADER));
    let actual: Vec<_> = debug_world_states().collect();
    assert_eq!(actual.len(), EXPECTED_ROWS);

    let mut checked = 0usize;
    for line in lines {
        let fields: Vec<_> = line.splitn(7, '\t').collect();
        assert_eq!(fields.len(), 7, "malformed Java oracle row {checked}");
        let index = fields[0].parse::<usize>().expect("oracle index");
        let block = fields[1].parse::<u16>().expect("oracle registry id");
        let raw_storage_id = fields[3].parse::<u16>().expect("oracle raw storage id");
        let metadata = fields[4].parse::<u8>().expect("oracle metadata");
        assert_eq!(index, checked);

        let expected_properties: Vec<_> = if fields[6].is_empty() {
            Vec::new()
        } else {
            fields[6]
                .split(';')
                .map(|property| {
                    let (name, value) = property.split_once('=').expect("name=value property");
                    (name, value)
                })
                .collect()
        };

        let state = &actual[checked];
        assert_eq!(
            state.stored().id,
            raw_storage_id,
            "row {index} raw storage id"
        );
        assert_eq!(state.stored().block().0, block, "row {index} block");
        assert_eq!(state.stored().meta(), metadata, "row {index} metadata");
        assert_eq!(
            state.properties(),
            expected_properties.as_slice(),
            "row {index} properties"
        );
        checked += 1;
    }
    assert_eq!(checked, EXPECTED_ROWS);
}
