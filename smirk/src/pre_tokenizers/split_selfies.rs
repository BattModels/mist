use super::split_smiles::{ATOMIC_SYMBOLS, BONDS, BRACKETED, CHARGE_OR_COUNT};
use const_format::concatcp;
use once_cell::sync::Lazy;
use regex::Regex;

// Capture Rings and Branching and Stereochemistry
const STRUCTURE: &'static str = r"Ring|Branch|@{1,2}";

// Capture Hydrogen
const HCOUNT: &'static str = r"[H]\d+";

// Capture tokens outside of brackets
pub static MATCH_OUTER: Lazy<Regex> =
    Lazy::new(|| Regex::new(concatcp!(BRACKETED, "|", r"\.")).unwrap());

// Capture tokens within brackets
pub static MATCH_INNER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concatcp!(
        STRUCTURE,
        "|",
        BONDS,
        "|",
        ATOMIC_SYMBOLS,
        "|",
        CHARGE_OR_COUNT,
        "|",
        HCOUNT
    ))
    .unwrap()
});
