use const_format::concatcp;
use once_cell::sync::Lazy;
use regex::Regex;

// Capture the organic subset
const ORGANIC_SUBSET: &'static str = r"Cl?|Br?|N|P|S|O|I|F";

const AROMATIC_ORGANIC: &'static str = r"b|c|n|o|p|s";

// Capture elements (Generated with opt/element_regex.py)
pub const ATOMIC_SYMBOLS: &'static str = concatcp!(
    r"A[c|g|l|m|r|s|t|u]|",
    r"B[a|e|h|i|k|r]?|",
    r"C[a|d|e|f|l|m|n|o|r|s|u]?|",
    r"D[b|s|y]|",
    r"E[r|s|u]|",
    r"F[e|l|m|r]?|",
    r"G[a|d|e]|",
    r"H[e|f|g|o|s]?|",
    r"I[n|r]?|",
    r"K[r]?|",
    r"L[a|i|r|u|v]|",
    r"M[c|d|g|n|o|t]|",
    r"N[a|b|d|e|h|i|o|p]?|",
    r"O[g|s]?|",
    r"P[a|b|d|m|o|r|t|u]?|",
    r"R[a|b|e|f|g|h|n|u]|",
    r"S[b|c|e|g|i|m|n|r]?|",
    r"T[a|b|c|e|h|i|l|m|s]|",
    r"U|",
    r"V|",
    r"W|",
    r"X[e]|",
    r"Y[b]?",
);

// Capture bond symbols
pub const BONDS: &'static str = r"[\.\-=#\$:/\\]";

// Capture Rings, Branching, and Stereochemistry
pub const STRUCTURE: &'static str = r"%|[\(\)]|[/\\]|@{1,2}";

// Capture brackets
pub const BRACKETED: &'static str = r"\[.*?\]";

pub const CHARGE_OR_COUNT: &'static str = r"\d|\+|\-";

// Capture tokens outside of brackets
pub static MATCH_OUTER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concatcp!(
        ORGANIC_SUBSET,
        "|",
        AROMATIC_ORGANIC,
        "|",
        BONDS,
        "|",
        STRUCTURE,
        "|",
        BRACKETED,
        "|",
        CHARGE_OR_COUNT
    ))
    .unwrap()
});

// Capture tokens within brackets
pub static MATCH_INNER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concatcp!(
        ATOMIC_SYMBOLS,
        "|",
        AROMATIC_ORGANIC,
        "|",
        BONDS,
        "|",
        STRUCTURE,
        "|",
        CHARGE_OR_COUNT,
    ))
    .unwrap()
});
