//! Bounded arithmetic parser shared by every editable numeric widget.
//! No scripting, allocations derived from results, or non-finite values.

use photocraft_engine::prefs::Unit;

pub(crate) fn parse(text: &str) -> Option<f64> {
    parse_with_units(text, None)
}

/// What a unit typed after a number means in a field (Photoshop: `1in` in a px field is the
/// pixels of an inch at the document resolution).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Units<'s> {
    /// The field's own unit, `None` for a field that isn't a length or a percentage.
    pub unit: Option<Unit>,
    /// The suffix the field shows; typed after a number it means the field's unit (`45°`).
    pub suffix: &'s str,
    /// Document resolution, pixels per inch.
    pub dpi: f64,
    /// Points per inch (72 PostScript, 72.27 traditional).
    pub ppi: f64,
}

/// Unit names a number may carry, longest first so `pica` wins over `pc`.
const UNIT_NAMES: [(&str, Unit); 8] = [
    ("pica", Unit::Picas),
    ("px", Unit::Pixels),
    ("pt", Unit::Points),
    ("pc", Unit::Picas),
    ("in", Unit::Inches),
    ("cm", Unit::Centimeters),
    ("mm", Unit::Millimeters),
    ("%", Unit::Percent),
];

/// [`parse`] where a number may carry a unit (`2in`, `5 mm`, `50%`), converted into the field's
/// unit. A unit that can't convert (a length in a percent field, any unit in a unitless one) makes
/// the whole text invalid; without `units`, unit names are invalid as before.
pub(crate) fn parse_with_units(text: &str, units: Option<Units>) -> Option<f64> {
    if text.len() > 1024 {
        return None;
    }
    let normalized = text.replace('−', "-");
    let mut parser = Parser { rest: &normalized, units };
    let value = parser.expression(0, 0)?;
    parser.space();
    (parser.rest.is_empty() && value.is_finite()).then_some(value)
}

struct Parser<'a> {
    rest: &'a str,
    units: Option<Units<'a>>,
}
impl Parser<'_> {
    fn space(&mut self) {
        self.rest = self.rest.trim_start();
    }
    fn take(&mut self, token: &str) -> bool {
        self.space();
        if let Some(rest) = self.rest.strip_prefix(token) {
            self.rest = rest;
            true
        } else {
            false
        }
    }
    fn expression(&mut self, minimum: u8, depth: u8) -> Option<f64> {
        if depth >= 32 {
            return None;
        }
        self.space();
        let mut left = if self.take("+") {
            self.expression(3, depth + 1)?
        } else if self.take("-") {
            -self.expression(3, depth + 1)?
        } else if self.take("(") {
            let value = self.expression(0, depth + 1)?;
            if !self.take(")") {
                return None;
            }
            value
        } else if self.take("pi") {
            std::f64::consts::PI
        } else if self.take("tau") {
            std::f64::consts::TAU
        } else {
            self.number()?
        };
        loop {
            self.space();
            let (token, precedence, right_minimum) = if self.rest.starts_with("**") {
                ("**", 3, 3)
            } else {
                match self.rest.chars().next() {
                    Some('+') => ("+", 1, 2),
                    Some('-') => ("-", 1, 2),
                    Some('*') => ("*", 2, 3),
                    Some('/') => ("/", 2, 3),
                    Some('%') => ("%", 2, 3),
                    Some('^') => ("^", 3, 3),
                    _ => break,
                }
            };
            if precedence < minimum {
                break;
            }
            if !self.take(token) {
                return None;
            }
            let right = self.expression(right_minimum, depth + 1)?;
            left = match token {
                "+" => left + right,
                "-" => left - right,
                "*" => left * right,
                "/" if right != 0.0 => left / right,
                "%" if right != 0.0 => left % right,
                "^" | "**" => left.powf(right),
                _ => return None,
            };
            if !left.is_finite() {
                return None;
            }
        }
        left.is_finite().then_some(left)
    }
    fn number(&mut self) -> Option<f64> {
        let mut length = 0;
        let mut exponent = false;
        for (offset, ch) in self.rest.char_indices() {
            let accepted = ch.is_ascii_digit()
                || ch == '.'
                || (!exponent && matches!(ch, 'e' | 'E'))
                || (matches!(ch, '+' | '-') && self.rest.get(..offset).is_some_and(|s| s.ends_with(['e', 'E'])));
            if !accepted {
                break;
            }
            exponent |= matches!(ch, 'e' | 'E');
            length = offset + ch.len_utf8();
        }
        let value = self.rest.get(..length)?.parse::<f64>().ok()?;
        self.rest = self.rest.get(length..)?;
        let value = match self.units {
            Some(units) => self.unit(value, units)?,
            None => value,
        };
        value.is_finite().then_some(value)
    }
    /// The unit after a number, if any, converting `value` into the field's unit.
    fn unit(&mut self, value: f64, units: Units) -> Option<f64> {
        let rest = self.rest.trim_start();
        let typed = UNIT_NAMES.iter().find(|(name, _)| rest.starts_with(name)).filter(|(name, _)| {
            // `%` followed by an operand is the remainder operator (`10%3`), not percent.
            *name != "%" || rest.get(1..).map(str::trim_start).and_then(|r| r.chars().next()).is_none_or(|c| matches!(c, ')' | '+' | '-' | '*' | '/' | '^'))
        });
        let Some(&(name, typed)) = typed else {
            // The field's own suffix (`45°`), when it is no operator or number of its own.
            let own = units.suffix;
            let operand = |c: char| c.is_ascii_digit() || matches!(c, '.' | '(' | ')' | '+' | '-' | '*' | '/' | '%' | '^');
            if !own.is_empty() && !own.starts_with(operand) && rest.starts_with(own) {
                self.rest = rest.get(own.len()..)?;
            }
            return Some(value);
        };
        self.rest = rest.get(name.len()..)?;
        let dpi = if units.dpi.is_finite() && units.dpi > 0.0 { units.dpi } else { 72.0 };
        let ppi = if units.ppi.is_finite() && units.ppi > 0.0 { units.ppi } else { 72.0 };
        match units.unit {
            Some(field) if field == typed => Some(value),
            // Lengths convert through pixels; a percentage has no length to convert against.
            Some(field) if field != Unit::Percent && typed != Unit::Percent => Some(field.from_px(typed.to_px(value, dpi, 1.0, ppi), dpi, 1.0, ppi)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;
    #[test]
    fn arithmetic() {
        for (text, expected) in [
            ("1920/2", 960.0),
            ("30+15", 45.0),
            ("2+3*4", 14.0),
            ("(2+3)*4", 20.0),
            ("-2^2", -4.0),
            ("2**3**2", 512.0),
            ("2^-2", 0.25),
            ("10%3", 1.0),
            ("1e-3 * 1000", 1.0),
            ("−12 + .5", -11.5),
            ("pi*2", std::f64::consts::TAU),
            ("tau/2", std::f64::consts::PI),
        ] {
            assert_eq!(parse(text), Some(expected), "{text}");
        }
    }
    #[test]
    fn rejects_invalid_and_hostile_input() {
        for text in ["", "1/0", "0/0", "1%0", "NaN", "inf", "1e999", "2+", "(2*3", "2 3", "pi junk", "sqrt(4)", "2^(1e300)", "你好"] {
            assert_eq!(parse(text), None, "{text}");
        }
        assert_eq!(parse(&"(".repeat(1000)), None);
        assert_eq!(parse(&"1".repeat(1025)), None);
    }
    #[test]
    fn units_convert_into_the_field_unit_and_bad_input_is_rejected() {
        use super::{Unit, Units, parse_with_units};
        let px = Units { unit: Some(Unit::Pixels), suffix: "px", dpi: 300.0, ppi: 72.0 };
        let pt = Units { unit: Some(Unit::Points), suffix: "pt", dpi: 144.0, ppi: 72.0 };
        let pct = Units { unit: Some(Unit::Percent), suffix: "%", dpi: 72.0, ppi: 72.0 };
        let deg = Units { unit: None, suffix: "°", dpi: 72.0, ppi: 72.0 };
        for (text, units, want) in [
            ("1in", px, 300.0),
            ("2.54cm", px, 300.0),
            ("1in+10", px, 310.0),
            ("1 in * 2", px, 600.0),
            ("-25.4mm", px, -300.0),
            ("72pt", px, 300.0),
            ("1pica", px, 50.0),
            ("1pc", px, 50.0),
            ("100px", px, 100.0),
            ("(1in)/3", px, 100.0),
            ("144px", pt, 72.0),
            ("12", pt, 12.0),
            ("50%", pct, 50.0),
            ("50 % + 10", pct, 60.0),
            ("10%3", pct, 1.0),
            ("45°", deg, 45.0),
            ("45°*2", deg, 90.0),
        ] {
            let got = parse_with_units(text, Some(units));
            assert!(got.is_some_and(|v| (v - want).abs() < 1e-9), "{text}: {got:?}, want {want}");
        }
        let nan_dpi = Units { dpi: f64::NAN, ppi: 0.0, ..px };
        assert_eq!(parse_with_units("1in", Some(nan_dpi)), Some(72.0), "a bad resolution falls back to 72");
        for (text, units) in [
            ("", px),
            ("1/0", px),
            ("1e400", px),
            ("((((", px),
            ("你好", px),
            ("1in/0", px),
            ("50%", px),
            ("2in", pct),
            ("2in", deg),
            ("2 inches", px),
            ("2px3", px),
            ("in", px),
            ("1e308in", px),
            ("°", deg),
            ("2°", px),
        ] {
            assert_eq!(parse_with_units(text, Some(units)), None, "{text}");
        }
        assert_eq!(parse("1in"), None, "without a field unit, unit names stay invalid");
    }
}

#[cfg(test)]
mod widget_tests {
    use egui_kittest::{Harness, kittest::Queryable};
    #[test]
    fn typed_expression_commits_clamps_and_rejects_invalid() {
        let mut h = Harness::builder().build_ui_state(
            |ui, value| {
                ui.add(egui::DragValue::new(value).range(0.0..=1000.0).custom_parser(super::parse));
            },
            100.0_f64,
        );
        for (text, expected) in [("1920/2", 960.0), ("2000*2", 1000.0), ("1/0", 1000.0)] {
            let center = h.get_by_role(egui::accesskit::Role::SpinButton).rect().center();
            h.hover_at(center);
            h.run_steps(1);
            h.drag_at(center);
            h.run_steps(1);
            h.drop_at(center);
            h.run();
            h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
            h.run();
            h.event(egui::Event::Text(text.to_owned()));
            h.run();
            h.key_press(egui::Key::Enter);
            h.run();
            assert_eq!(*h.state(), expected);
        }
    }
}
