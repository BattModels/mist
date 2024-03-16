use const_format::concatcp;
use once_cell::sync::Lazy;
use regex::Regex;

// Capture the organic subset
const ORGANIC_SUBSET: &'static str = r"Cl?|Br?|N|P|S|O|I|F";

const AROMATIC_ORGANIC: &'static str = r"b|c|n|o|p|s";

// Capture elements (Generated with opt/element_regex.py)
const ATOMIC_SYMBOLS: &'static str = concatcp!(
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
const BONDS: &'static str = r"[\.\-=#\$:/\\]";

// Capture Rings, Branching, and Sterochemistry
const STRUCTURE: &'static str = r"%|[\(\)]|[/\\]|@{1,2}";

// Capture brackets
const BRACKETED: &'static str = r"\[.*?\]";

const CHARGE_OR_COUNT: &'static str = r"\d|\+|\-";

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

pub fn split_chemically_consistent(a: &str) -> Vec<String> {
    // Iterate over string, splitting into tokens
    let mut tokens = Vec::new();
    for m in MATCH_OUTER.find_iter(a).map(|m| m.as_str()) {
        if m.starts_with("[") {
            tokens.push("[".to_string());
            for i in MATCH_INNER.find_iter(m) {
                tokens.push(i.as_str().to_string())
            }
            tokens.push("]".to_string());
        } else {
            tokens.push(m.to_string())
        }
    }
    return tokens;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_split(smile: &str) -> Vec<String> {
        let split = split_chemically_consistent(&smile);
        assert_eq!(split.join(""), smile);
        return split;
    }

    #[test]
    fn test_outer() {
        let split = check_split("CN=C=O");
        assert_eq!(split, ["C", "N", "=", "C", "=", "O"]);
    }

    #[test]
    fn chlorine_carbon() {
        let split = check_split("[2H]ClC");
        assert_eq!(split, ["[", "2", "H", "]", "Cl", "C"]);
    }

    #[test]
    fn boron_bromide() {
        let split = check_split("BrB3");
        assert_eq!(split, ["Br", "B", "3"]);
    }

    #[test]
    fn pyrrole() {
        let split = check_split("[nH]1cccc1");
        assert_eq!(split, ["[", "n", "H", "]", "1", "c", "c", "c", "c", "1"]);
    }

    #[test]
    fn test_brackets() {
        let split = check_split("[14c][o3]");
        assert_eq!(split, ["[", "1", "4", "c", "]", "[", "o", "3", "]"]);
    }

    #[test]
    fn test_wikipedia_examples() {
        check_split("N#N");
        check_split("CN=C=O");
        check_split("[Cu+2].[O-]S(=O)(=O)[O-]");
        check_split("O=Cc1ccc(O)c(OC)c1COc1cc(C=O)ccc1O");
        check_split("CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1");
        check_split("CN1CCC[C@H]1c2cccnc2");
        check_split(r"CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\COCCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO");
        check_split(r"CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C");
        check_split("O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5");
        check_split("OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1");
        check_split("OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2");
        check_split(r"CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C");
        check_split("CC[C@H](O1)CC[C@@]12CCCO2");
        check_split("CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2");
        check_split("OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N");
    }
}
