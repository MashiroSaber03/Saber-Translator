//! Index lookups must return the same spec, in the same order, as the registry's public slice.
use photocraft_engine::commands;

#[test]
fn every_lookup_matches_the_first_linear_match() {
    let specs = commands::command_specs();
    for spec in specs {
        let linear = specs.iter().find(|s| s.id == spec.id).unwrap();
        assert!(std::ptr::eq(commands::find(spec.id).unwrap(), linear), "{}", spec.id);
        // The id need not be a static string: control/MCP dispatch owns its input.
        let owned = spec.id.to_string();
        assert!(std::ptr::eq(commands::find(&owned).unwrap(), linear));
    }
    for id in ["", "missing.command", "FILE.NEW", "file.new\0", "💾"] {
        assert!(commands::find(id).is_none(), "{id:?}");
    }
}
