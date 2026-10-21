//! The Contributors tab as drawn: the table's header stays put while its rows scroll, its columns
//! keep their places whatever the sort, counts line up on the right, and every column fits the
//! About window; the grab bag's rows are evenly spaced.

use egui::{Rect, accesskit::Role, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};

use super::*;

/// The About window's width (`dialogs.rs` gives its tabs 700 pt).
const WIDTH: f32 = 700.0;

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// `n` contributors with distinct, realistic-sized counts (six-digit line counts, a long login).
fn people(n: u64) -> Vec<Contributor> {
    (0..n)
        .map(|i| Contributor {
            login: if i == 0 { "nullpointerexception" } else { leak(format!("user{i:02}")) },
            display_name: None,
            real_name: None,
            prs: 100 + i,
            commits: 1000 + i,
            lines_added: 277_000 + i,
            lines_deleted: 22_000 + i,
            binary_added: 30 + i,
            binary_deleted: i,
            first_commit: leak(format!("2026-09-{:02}T00:00:00Z", 1 + i % 28)),
            last_commit: "2026-10-07T00:00:00Z",
        })
        .collect()
}

fn harness(list: Vec<Contributor>, table: bool) -> Harness<'static> {
    let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let mut h = Harness::builder().with_size(vec2(WIDTH + 20.0, 520.0)).build_ui(move |ui| {
        if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
            return;
        }
        let id = egui::Id::new("credits_view");
        if ui.data(|d| d.get_temp::<View>(id)).is_none() {
            ui.data_mut(|d| d.insert_temp(id, View { table, ..View::default() }));
        }
        ui.set_width(WIDTH);
        contributors_in(&mut app, ui, &list, 1);
    });
    crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

fn header(h: &Harness<'_>, k: SortKey) -> Rect {
    let label = k.label().1;
    // The old table put the sort arrow in the label ("First ▲").
    let named = |l: &str| l == label || l.strip_prefix(label).is_some_and(|rest| rest.trim().chars().all(|c| "▲▼".contains(c)));
    h.query_all_by_role(Role::Button)
        .find(|n| n.accesskit_node().label().is_some_and(|l| named(&l)))
        .map(|n| n.rect())
        .unwrap_or_else(|| panic!("no {label} header"))
}

fn headers(h: &Harness<'_>) -> Vec<Rect> {
    SortKey::ALL.iter().map(|&k| header(h, k)).collect()
}

fn text_rect(h: &Harness<'_>, text: &str) -> Option<Rect> {
    h.query_all_by_label(text).next().map(|n| n.rect())
}

fn view(h: &Harness<'_>) -> View {
    h.ctx.data(|d| d.get_temp::<View>(egui::Id::new("credits_view"))).unwrap_or_default()
}

#[test]
fn the_table_header_stays_put_while_the_rows_scroll() {
    let mut h = harness(people(60), true);
    let before = headers(&h);
    let first = text_rect(&h, "@user01").expect("the second row is shown");
    let over = first.center();
    h.event(egui::Event::PointerMoved(over));
    for _ in 0..6 {
        h.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: vec2(0.0, -200.0),
            modifiers: egui::Modifiers::NONE,
            phase: egui::TouchPhase::Move,
        });
        h.run_steps(2);
    }
    h.run_steps(4);
    assert_eq!(headers(&h), before, "the header moved with the rows");
    let moved = text_rect(&h, "@user01").map(|r| r.top());
    assert!(moved.is_none_or(|y| y < first.top() - 100.0), "the rows didn't scroll: {moved:?} vs {}", first.top());
}

#[test]
fn clicking_a_header_sorts_by_it_and_no_column_moves() {
    let mut h = harness(people(12), true);
    let before = headers(&h);
    assert_eq!((view(&h).key, view(&h).ascending), (SortKey::FirstCommit, true));
    for (k, ascending) in [(SortKey::Commits, false), (SortKey::Commits, true), (SortKey::Name, true), (SortKey::LastCommit, true)] {
        let at = header(&h, k).center();
        h.query_all_by_role(Role::Button).find(|n| n.rect().contains(at)).expect("the header").click();
        h.run_steps(3);
        assert_eq!((view(&h).key, view(&h).ascending), (k, ascending));
        assert_eq!(headers(&h), before, "sorting by {k:?} moved the columns");
    }
}

#[test]
fn counts_line_up_on_the_right_and_names_and_dates_on_the_left() {
    let h = harness(people(8), true);
    let (commits, first) = (header(&h, SortKey::Commits), header(&h, SortKey::FirstCommit));
    let cells = |col: Rect| -> Vec<Rect> {
        h.query_all_by_role(Role::Label)
            .map(|n| n.rect())
            .filter(|r| r.top() > col.bottom() && r.center().x > col.left() && r.center().x < col.right())
            .collect()
    };
    let nums = cells(commits);
    assert!(nums.len() >= 8, "{} commit cells", nums.len());
    assert!(nums.iter().all(|r| (r.right() - nums[0].right()).abs() < 0.5), "commit counts don't share a right edge: {nums:?}");
    let dates = cells(first);
    assert!(dates.len() >= 8, "{} date cells", dates.len());
    assert!(dates.iter().all(|r| (r.left() - dates[0].left()).abs() < 0.5), "dates don't share a left edge: {dates:?}");
    assert!(nums[0].right() <= commits.right() && dates[0].left() >= first.left());
}

#[test]
fn every_column_fits_the_about_window() {
    let h = harness(people(8), true);
    let hs = headers(&h);
    let (left, right) = (hs[0].left(), hs.last().map_or(0.0, |r| r.right()));
    assert!(right - left <= WIDTH + 0.5, "the table is {} pt wide in a {WIDTH} pt window", right - left);
    // The longest value of each column is shown whole, inside its column.
    for (k, col) in SortKey::ALL.iter().zip(&hs).skip(1) {
        let widest = people(8).iter().map(|c| k.cell(c)).max_by_key(|s| s.len()).unwrap_or_default();
        let r = text_rect(&h, &widest).unwrap_or_else(|| panic!("{k:?}: no {widest} cell"));
        assert!(r.left() >= col.left() && r.right() <= col.right(), "{k:?}: {widest} at {r:?} spills out of {col:?}");
    }
}

#[test]
fn the_grab_bag_rows_are_evenly_spaced() {
    let h = harness(people(60), false);
    let links = || h.query_all(egui_kittest::kittest::by().predicate(|n| n.label().is_some_and(|l| l.starts_with('@'))));
    let mut tops: Vec<f32> = links().map(|n| n.rect().top()).collect();
    tops.sort_by(f32::total_cmp);
    tops.dedup_by(|a, b| (*a - *b).abs() < 0.5);
    assert!(tops.len() >= 3, "{} rows", tops.len());
    let gaps: Vec<f32> = tops.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.iter().all(|g| (g - gaps[0]).abs() < 0.5), "uneven rows: {gaps:?}");
    // And spaced apart: more than the text's own line height.
    let line = links().next().map_or(0.0, |n| n.rect().height());
    assert!(gaps[0] > line + 4.0, "rows {} pt apart for {line} pt links", gaps[0]);
}

fn model(company: &'static str, name: &'static str, commits: u64) -> Model {
    Model { company, model: name, version: "1.0", commits, lines_added: 150_000 + commits, lines_deleted: 20_000 }
}

#[test]
fn a_model_with_commits_never_reads_zero_percent() {
    assert_eq!(percent(228, 482), "47%");
    assert_eq!(percent(2, 482), "<1%");
    assert_eq!(percent(0, 482), "0%");
    assert_eq!(percent(5, 0), "100%"); // a total that trails the model's own count
    assert_eq!(percent(3, 1), "100%");
}

#[test]
fn the_models_table_lines_counts_up_on_the_right_under_a_rule_of_headers() {
    let models: Vec<Model> = (0..8).map(|i| model("Anthropic", leak(format!("Model {i}")), [228, 45, 6, 2, 2, 1, 1, 1][i])).collect();
    let mut h = Harness::builder().with_size(vec2(WIDTH + 20.0, 520.0)).build_ui(move |ui| {
        if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
            return;
        }
        ui.set_width(WIDTH);
        models_in(ui, &models, 482);
    });
    crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    let head = |l: &str| h.query_all_by_label(l).next().map(|n| n.rect()).unwrap_or_else(|| panic!("no {l} header"));
    let (commits, lines) = (head("Commits"), head("Lines +/−"));
    assert!(lines.right() - head("Company").left() <= WIDTH + 0.5, "the table is wider than the window");
    let column = |col: Rect| -> Vec<Rect> {
        h.query_all_by_role(Role::Label)
            .map(|n| n.rect())
            .filter(|r| r.top() > col.bottom() && r.center().x > col.left() && r.center().x < col.right())
            .collect()
    };
    for (name, col) in [("Commits", commits), ("Lines", lines)] {
        let cells = column(col);
        assert_eq!(cells.len(), 8, "{name}: {} cells", cells.len());
        assert!(cells.iter().all(|r| (r.right() - cells[0].right()).abs() < 0.5), "{name} isn't right-aligned: {cells:?}");
    }
    assert!(h.query_all_by_label("<1%").count() == 5, "small shares read <1%");
}

/// The About window on `tab` (and the Contributors table when `table`): the top of its title and of
/// its OK button, and the bottom of the lowest thing between them.
fn about_window(tab: &str, table: bool) -> (f32, f32, f32) {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
        crate::PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
    });
    let mut fields = serde_json::Map::new();
    fields.insert("tab".into(), serde_json::json!(tab));
    h.state_mut().ui.open_dialog(crate::state::DialogKind::About, fields);
    h.ctx.data_mut(|d| d.insert_temp(egui::Id::new("credits_view"), View { table, ..View::default() }));
    h.run_steps(6);
    let ok = h.query_all_by_role(Role::Button).find(|n| n.accesskit_node().label().as_deref() == Some("OK")).map(|n| n.rect()).expect("the OK button");
    // The dialog's contents: everything between its title and the OK button.
    let title = h.query_all_by_label("About PhotoCraft").map(|n| n.rect()).find(|r| r.top() < ok.top()).expect("the dialog title");
    let lowest = h
        .query_all(egui_kittest::kittest::by().predicate(|n| n.bounding_box().is_some()))
        .map(|n| n.rect())
        .filter(|r| r.top() > title.bottom() && r.bottom() < ok.top() && r.left() >= title.left() - 1.0 && r.right() <= ok.right() + 1.0 && r.height() < 100.0)
        .map(|r| r.bottom())
        .fold(f32::MIN, f32::max);
    (title.top(), ok.top(), lowest)
}

#[test]
fn about_and_models_end_where_their_content_does() {
    for tab in ["about", "models"] {
        let (_, ok, lowest) = about_window(tab, false);
        assert!(lowest > f32::MIN, "{tab}: nothing above OK");
        assert!(ok - lowest < 40.0, "{tab}: {} pt of blank space above OK", ok - lowest);
    }
}

#[test]
fn switching_the_contributors_view_keeps_the_window_size() {
    // A new window is placed by its first frame, so compare the height, title to OK.
    let height = |table| {
        let (title, ok, _) = about_window("contributors", table);
        ok - title
    };
    let (bag, table) = (height(false), height(true));
    assert!((bag - table).abs() < 0.5, "{bag} pt tall on the grab bag, {table} pt on the table");
}
